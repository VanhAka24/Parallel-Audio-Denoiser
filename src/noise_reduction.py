import os
import time
from src.audio_io import read_wav, write_wav, mono_mix
from src.filters import (
    moving_average_filter,
    spectral_subtraction,
    wiener_filter,
    compute_rms,
)
from src.filters_advanced import (
    median_filter,
    mmse_stsa,
    kalman_filter,
    adaptive_kalman_filter,
)
from src.parallel_engine import (
    parallel_moving_average,
    parallel_median,
    parallel_spectral_subtraction,
    parallel_wiener,
    parallel_mmse_stsa,
    parallel_kalman,
)

# --- Tham số mặc định cho từng bộ lọc ---
DEFAULT_CONFIG = {
    "moving_average":       {"window_size": 15},
    "spectral_subtraction": {"noise_duration_sec": 0.5, "alpha_max": 5.0,
                              "alpha_min": 1.0, "beta": 0.01,
                              "frame_size": 1024, "hop_size": 512},
    "wiener":               {"noise_duration_sec": 0.5, "alpha_dd": 0.98,
                              "frame_size": 1024, "hop_size": 512},
    "median":               {"window_size": 7},
    "mmse_stsa":            {"noise_duration_sec": 0.5, "alpha_dd": 0.98,
                              "frame_size": 1024, "hop_size": 512},
    "kalman":               {"process_noise_var": 1e-5, "measurement_noise_var": 0.01},
    "adaptive_kalman":      {"window_size": 1000, "base_process_noise": 1e-5},
}

FILTER_DISPLAY_NAMES = {
    "moving_average":       "Moving Average Filter",
    "spectral_subtraction": "Spectral Subtraction",
    "wiener":               "Wiener Filter",
    "median":               "Median Filter",
    "mmse_stsa":            "MMSE-STSA (Ephraim-Malah)",
    "kalman":               "Kalman Filter (1D)",
    "adaptive_kalman":      "Adaptive Kalman Filter",
}

ALL_FILTERS = list(FILTER_DISPLAY_NAMES.keys())


# --- Gọi filter theo chế độ tuần tự ---

def _run_sequential(signal: list, sample_rate: int, filter_name: str, cfg: dict) -> list:
    if filter_name == "moving_average":
        return moving_average_filter(signal, **cfg)
    elif filter_name == "spectral_subtraction":
        return spectral_subtraction(signal, sample_rate, **cfg)
    elif filter_name == "wiener":
        return wiener_filter(signal, sample_rate, **cfg)
    elif filter_name == "median":
        return median_filter(signal, **cfg)
    elif filter_name == "mmse_stsa":
        return mmse_stsa(signal, sample_rate, **cfg)
    elif filter_name == "kalman":
        return kalman_filter(signal, **cfg)
    elif filter_name == "adaptive_kalman":
        return adaptive_kalman_filter(signal, **cfg)
    raise ValueError(f"Unknown filter: {filter_name}")


# --- Gọi filter theo chế độ song song ---

def _run_parallel(signal: list, sample_rate: int, filter_name: str,
                  cfg: dict, num_workers: int) -> list:
    if filter_name == "moving_average":
        return parallel_moving_average(signal, num_workers=num_workers, **cfg)
    elif filter_name == "spectral_subtraction":
        return parallel_spectral_subtraction(signal, sample_rate,
                                              num_workers=num_workers, **cfg)
    elif filter_name == "wiener":
        return parallel_wiener(signal, sample_rate, num_workers=num_workers, **cfg)
    elif filter_name == "median":
        return parallel_median(signal, num_workers=num_workers, **cfg)
    elif filter_name == "mmse_stsa":
        return parallel_mmse_stsa(signal, sample_rate, num_workers=num_workers, **cfg)
    elif filter_name == "kalman":
        return parallel_kalman(signal, num_workers=num_workers, **cfg)
    elif filter_name == "adaptive_kalman":
        return adaptive_kalman_filter(signal, **cfg)
    raise ValueError(f"Unknown filter: {filter_name}")


# --- Xử lý 1 file WAV: đọc → lọc → normalize → ghi ---

def process_file(input_path: str, output_path: str,
                 filter_name: str = "spectral_subtraction",
                 mode: str = "parallel",
                 num_workers: int = None,
                 config: dict = None,
                 verbose: bool = True) -> dict:
    cfg = dict(DEFAULT_CONFIG.get(filter_name, {}))
    if config:
        cfg.update(config)

    if verbose:
        print(f"\n[INPUT]  {input_path}")

    samples, sample_rate, num_channels = read_wav(input_path)
    signal = mono_mix(samples, num_channels)

    if verbose:
        print(f"         {len(signal)} samples | {sample_rate} Hz | "
              f"{len(signal)/sample_rate:.2f}s | {num_channels}ch")
        print(f"[FILTER] {FILTER_DISPLAY_NAMES.get(filter_name, filter_name)} "
              f"| mode={mode} | workers={num_workers if mode == 'parallel' else 1}")

    t0 = time.perf_counter()
    if mode == "sequential":
        result = _run_sequential(signal, sample_rate, filter_name, cfg)
    else:
        if num_workers is None:
            num_workers = os.cpu_count() or 4
        result = _run_parallel(signal, sample_rate, filter_name, cfg, num_workers)
    elapsed = time.perf_counter() - t0

    # Normalize biên độ về 95% để tránh clipping
    max_val = max(abs(x) for x in result) if result else 0
    if max_val > 0:
        scale = min(0.95 / max_val, 5.0)
        if scale > 1.0:
            result = [x * scale for x in result]

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    write_wav(output_path, result, sample_rate, num_channels=1)

    input_rms  = compute_rms(signal)
    output_rms = compute_rms(result)

    if verbose:
        print(f"[OUTPUT] {output_path}")
        print(f"[TIME]   {elapsed:.4f}s")
        print(f"[RMS]    Input={input_rms:.6f} | Output={output_rms:.6f}")

    return {
        "input_path":   input_path,
        "output_path":  output_path,
        "filter":       filter_name,
        "mode":         mode,
        "num_workers":  num_workers if mode == "parallel" else 1,
        "time_sec":     round(elapsed, 6),
        "sample_rate":  sample_rate,
        "num_samples":  len(signal),
        "duration_sec": round(len(signal) / sample_rate, 3),
        "input_rms":    round(input_rms, 6),
        "output_rms":   round(output_rms, 6),
    }


# --- Xử lý toàn bộ thư mục input ---

def process_directory(input_dir: str = "input", output_dir: str = "output",
                      filter_name: str = "spectral_subtraction",
                      mode: str = "parallel",
                      num_workers: int = None,
                      verbose: bool = True) -> list:
    wav_files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(".wav")
    ]

    if not wav_files:
        print(f"[WARNING] Không tìm thấy file WAV trong '{input_dir}'")
        return []

    print(f"\n[INFO] Tìm thấy {len(wav_files)} file WAV.")
    results = []
    for wav in wav_files:
        basename = os.path.splitext(os.path.basename(wav))[0]
        out_path = os.path.join(output_dir, f"{basename}_{filter_name}_cleaned.wav")
        results.append(process_file(wav, out_path, filter_name, mode, num_workers, verbose=verbose))

    return results
