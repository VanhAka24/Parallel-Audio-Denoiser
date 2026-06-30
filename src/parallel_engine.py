import os
import math
import multiprocessing

from concurrent.futures import ProcessPoolExecutor, as_completed

# Ngưỡng chuyển sang tuần tự: filter nhẹ cần signal dài hơn mới có lợi khi song song
MIN_SAMPLES_HEAVY       = 8_000    # Spectral Subtraction, Wiener (FFT-based)


# --- Tự động chọn số workers tối ưu theo độ dài signal và loại filter ---

def _optimal_workers(signal_len: int, requested: int, filter_type: str) -> int:
    max_cores = os.cpu_count() or 4
    n         = min(requested or max_cores, max_cores)
    return 1 if signal_len < MIN_SAMPLES_HEAVY else n


# --- Worker functions (chạy trong subprocess) ---


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




