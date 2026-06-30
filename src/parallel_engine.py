import os
import math
import multiprocessing

from concurrent.futures import ProcessPoolExecutor, as_completed

# Ngưỡng chuyển sang tuần tự: filter nhẹ cần signal dài hơn mới có lợi khi song song
MIN_SAMPLES_LIGHTWEIGHT = 50_000   # Moving Average, Median, Kalman
MIN_SAMPLES_HEAVY       = 8_000    # Spectral Subtraction, Wiener (FFT-based)


# --- Tự động chọn số workers tối ưu theo độ dài signal và loại filter ---

def _optimal_workers(signal_len: int, requested: int, filter_type: str) -> int:
    max_cores = os.cpu_count() or 4
    n         = min(requested or max_cores, max_cores)

    if filter_type in ("moving_average", "median", "kalman"):
        if signal_len < MIN_SAMPLES_LIGHTWEIGHT:
            return 1
        return min(n, max(1, signal_len // 10_000))
    else:
        return 1 if signal_len < MIN_SAMPLES_HEAVY else n


# --- Worker functions (chạy trong subprocess) ---

def _worker_moving_average(args):
    chunk, window_size = args
    from src.filters import moving_average_filter
    return moving_average_filter(chunk, window_size)

def _worker_median(args):
    chunk, window_size = args
    from src.filters_advanced import median_filter
    return median_filter(chunk, window_size)

def _worker_spectral_subtraction(args):
    chunk, sample_rate, noise_duration_sec, alpha_max, alpha_min, beta, frame_size, hop_size, noise_profile_arr = args
    from src.filters import spectral_subtraction
    return spectral_subtraction(chunk, sample_rate,
                                 noise_duration_sec=noise_duration_sec,
                                 alpha_max=alpha_max, alpha_min=alpha_min, beta=beta,
                                 frame_size=frame_size, hop_size=hop_size,
                                 noise_profile_arr=noise_profile_arr)

def _worker_wiener(args):
    chunk, sample_rate, noise_duration_sec, alpha_dd, frame_size, hop_size, noise_profile_arr = args
    from src.filters import wiener_filter
    return wiener_filter(chunk, sample_rate,
                          noise_duration_sec=noise_duration_sec,
                          alpha_dd=alpha_dd,
                          frame_size=frame_size, hop_size=hop_size,
                          noise_profile_arr=noise_profile_arr)

def _worker_mmse_stsa(args):
    chunk, sample_rate, noise_duration_sec, alpha_dd, frame_size, hop_size = args
    from src.filters_advanced import mmse_stsa
    return mmse_stsa(chunk, sample_rate,
                      noise_duration_sec=noise_duration_sec,
                      alpha_dd=alpha_dd,
                      frame_size=frame_size, hop_size=hop_size)

def _worker_kalman(args):
    chunk, process_noise_var, measurement_noise_var = args
    from src.filters_advanced import kalman_filter
    return kalman_filter(chunk, process_noise_var, measurement_noise_var)

def _worker_adaptive_kalman(args):
    chunk, window_size, base_process_noise = args
    from src.filters_advanced import adaptive_kalman_filter
    return adaptive_kalman_filter(chunk, window_size, base_process_noise)


# --- Chia tín hiệu thành N chunks có overlap ---

def split_signal(signal: list, num_chunks: int, overlap: int = 512) -> list:
    n          = len(signal)
    chunk_size = math.ceil(n / num_chunks)
    chunks     = []
    for i in range(num_chunks):
        start = max(0, i * chunk_size - overlap)
        end   = min(n, (i + 1) * chunk_size + overlap)
        chunks.append((signal[start:end], start, end))
    return chunks


# --- Ghép chunks bằng linear cross-fade tại vùng overlap ---

def merge_chunks(chunks_output: list, chunk_info: list,
                 signal_len: int, overlap: int = 512) -> list:
    output = [0.0] * signal_len
    weight = [0.0] * signal_len

    for processed, info in zip(chunks_output, chunk_info):
        _, start, end = info
        chunk_len = end - start
        proc_len  = min(len(processed), chunk_len)

        for j in range(proc_len):
            gi = start + j
            if gi >= signal_len:
                break
            if j < overlap and start > 0:
                fade = j / overlap
            elif j >= proc_len - overlap and end < signal_len:
                fade = (proc_len - j) / overlap
            else:
                fade = 1.0
            output[gi] += processed[j] * fade
            weight[gi] += fade

    for i in range(signal_len):
        if weight[i] > 1e-8:
            output[i] /= weight[i]
    return output



# --- Parallel Moving Average ---

def parallel_moving_average(signal: list, window_size: int = 5, num_workers: int = None) -> list:
    from src.filters import moving_average_filter
    num_workers = _optimal_workers(len(signal), num_workers, "moving_average")
    if num_workers == 1:
        return moving_average_filter(signal, window_size)
    overlap    = window_size * 2
    chunk_info = split_signal(signal, num_workers, overlap)
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.map(_worker_moving_average, [(c, window_size) for c, _, _ in chunk_info])
    return merge_chunks(results, chunk_info, len(signal), overlap)


# --- Parallel Median Filter ---

def parallel_median(signal: list, window_size: int = 5, num_workers: int = None) -> list:
    from src.filters_advanced import median_filter
    num_workers = _optimal_workers(len(signal), num_workers, "median")
    if num_workers == 1:
        return median_filter(signal, window_size)
    overlap    = window_size * 2
    chunk_info = split_signal(signal, num_workers, overlap)
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.map(_worker_median, [(c, window_size) for c, _, _ in chunk_info])
    return merge_chunks(results, chunk_info, len(signal), overlap)


# --- Parallel Spectral Subtraction ---
# Tính global noise profile 1 lần rồi truyền vào tất cả workers

def parallel_spectral_subtraction(signal: list, sample_rate: int,
                                   noise_duration_sec: float = 0.5,
                                   alpha_max: float = 5.0,
                                   alpha_min: float = 1.0,
                                   beta: float = 0.01,
                                   frame_size: int = 1024,
                                   hop_size: int = 512,
                                   num_workers: int = None) -> list:
    num_workers = _optimal_workers(len(signal), num_workers, "spectral_subtraction")
    if num_workers == 1:
        from src.filters import spectral_subtraction
        return spectral_subtraction(signal, sample_rate,
                                    noise_duration_sec=noise_duration_sec,
                                    alpha_max=alpha_max, alpha_min=alpha_min, beta=beta,
                                    frame_size=frame_size, hop_size=hop_size)

    from src.filters import estimate_noise_profile
    global_noise = estimate_noise_profile(signal, sample_rate, noise_duration_sec, frame_size)
    overlap      = frame_size * 2
    chunk_info   = split_signal(signal, num_workers, overlap)
    task_args    = [
        (c, sample_rate, noise_duration_sec, alpha_max,
         alpha_min, beta, frame_size, hop_size, global_noise)
        for c, _, _ in chunk_info
    ]
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.map(_worker_spectral_subtraction, task_args)
    return merge_chunks(results, chunk_info, len(signal), overlap)


# --- Parallel Wiener Filter ---
# Tính global noise profile 1 lần rồi truyền vào tất cả workers

def parallel_wiener(signal: list, sample_rate: int,
                    noise_duration_sec: float = 0.5,
                    alpha_dd: float = 0.98,
                    frame_size: int = 1024,
                    hop_size: int = 512,
                    num_workers: int = None) -> list:
    num_workers = _optimal_workers(len(signal), num_workers, "wiener")
    if num_workers == 1:
        from src.filters import wiener_filter
        return wiener_filter(signal, sample_rate,
                             noise_duration_sec=noise_duration_sec,
                             alpha_dd=alpha_dd,
                             frame_size=frame_size, hop_size=hop_size)

    from src.filters import estimate_noise_profile
    global_noise = estimate_noise_profile(signal, sample_rate, noise_duration_sec, frame_size)
    overlap      = frame_size * 2
    chunk_info   = split_signal(signal, num_workers, overlap)
    task_args    = [
        (c, sample_rate, noise_duration_sec, alpha_dd, frame_size, hop_size, global_noise)
        for c, _, _ in chunk_info
    ]
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.map(_worker_wiener, task_args)
    return merge_chunks(results, chunk_info, len(signal), overlap)


# --- Parallel MMSE-STSA ---

def parallel_mmse_stsa(signal: list, sample_rate: int,
                       noise_duration_sec: float = 0.5,
                       alpha_dd: float = 0.98,
                       frame_size: int = 1024,
                       hop_size: int = 512,
                       num_workers: int = None) -> list:
    num_workers = _optimal_workers(len(signal), num_workers, "mmse_stsa")
    if num_workers == 1:
        from src.filters_advanced import mmse_stsa
        return mmse_stsa(signal, sample_rate,
                         noise_duration_sec=noise_duration_sec,
                         alpha_dd=alpha_dd,
                         frame_size=frame_size, hop_size=hop_size)
    overlap    = frame_size * 2
    chunk_info = split_signal(signal, num_workers, overlap)
    task_args  = [(c, sample_rate, noise_duration_sec, alpha_dd, frame_size, hop_size) for c, _, _ in chunk_info]
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.map(_worker_mmse_stsa, task_args)
    return merge_chunks(results, chunk_info, len(signal), overlap)


# --- Parallel Kalman Filter ---
# Overlap lớn (2000) để ổn định trạng thái tại biên chunk

def parallel_kalman(signal: list,
                    process_noise_var: float = 1e-5,
                    measurement_noise_var: float = 0.01,
                    num_workers: int = None) -> list:
    num_workers = _optimal_workers(len(signal), num_workers, "kalman")
    if num_workers == 1:
        from src.filters_advanced import kalman_filter
        return kalman_filter(signal, process_noise_var, measurement_noise_var)
    overlap    = 2000
    chunk_info = split_signal(signal, num_workers, overlap)
    task_args  = [(c, process_noise_var, measurement_noise_var) for c, _, _ in chunk_info]
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.map(_worker_kalman, task_args)
    return merge_chunks(results, chunk_info, len(signal), overlap)



