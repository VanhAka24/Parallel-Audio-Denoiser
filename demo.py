"""
demo.py — Script demo toàn bộ pipeline Parallel Audio Noise Reduction
Chạy: python demo.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def separator(title=""):
    print("\n" + "═" * 65)
    if title:
        print(f"  {title}")
        print("═" * 65)


def demo():
    separator("PARALLEL AUDIO NOISE REDUCTION — DEMO")
    print("  Dự án cuối kỳ | Python | Tính toán song song")
    print("  Máy: {} logical CPU cores".format(os.cpu_count()))

    # ─────────────────────────────────────────
    # BƯỚC 1: Tạo file test
    # ─────────────────────────────────────────
    separator("BƯỚC 1: Tạo file WAV mẫu")
    sys.path.insert(0, "samples")
    from samples.generate_sample import generate_test_samples
    generate_test_samples(input_dir="input")

    # ─────────────────────────────────────────
    # BƯỚC 2: Xử lý với từng bộ lọc
    # ─────────────────────────────────────────
    separator("BƯỚC 2: Khử nhiễu với 3 bộ lọc")

    from src.audio_io import read_wav, mono_mix
    from src.noise_reduction import process_file

    test_file = os.path.join("input", "short_5s.wav")

    filters = [
        ("moving_average",       {"window_size": 15}),
        ("spectral_subtraction", {"noise_duration_sec": 0.5, "alpha": 1.2, "beta": 0.001,
                                   "frame_size": 512, "hop_size": 256}),
        ("wiener",               {"noise_duration_sec": 0.5, "frame_size": 512, "hop_size": 256}),
    ]

    print(f"\n  File test: {test_file}")
    print(f"  {'Bộ lọc':<25} {'Sequential':>12} {'Parallel(all cores)':>20} {'Speedup':>10}")
    print(f"  {'-'*70}")

    for fname, fkwargs in filters:
        out_seq  = f"output/{fname}_seq_demo.wav"
        out_par  = f"output/{fname}_par_demo.wav"

        # Sequential
        res_seq = process_file(test_file, out_seq, fname, "sequential", config=fkwargs, verbose=False)

        # Parallel
        res_par = process_file(test_file, out_par, fname, "parallel", config=fkwargs, verbose=False)

        speedup = res_seq["time_sec"] / res_par["time_sec"] if res_par["time_sec"] > 0 else 1.0
        print(f"  {fname:<25} {res_seq['time_sec']:>10.4f}s  {res_par['time_sec']:>18.4f}s  {speedup:>8.2f}x")

    # ─────────────────────────────────────────
    # BƯỚC 3: Mini benchmark
    # ─────────────────────────────────────────
    separator("BƯỚC 3: Benchmark nhanh (Spectral Subtraction)")

    from src.benchmark import benchmark_filter, print_summary, save_results_csv

    samples, sr, ch = read_wav(os.path.join("input", "medium_15s.wav"))
    signal = mono_mix(samples, ch)

    max_cores = os.cpu_count() or 4
    worker_counts = [1, 2, 4]
    if max_cores >= 8:
        worker_counts.append(8)

    bm_result = benchmark_filter(
        signal, sr,
        filter_name="spectral_subtraction",
        worker_counts=worker_counts,
        filter_kwargs={"noise_duration_sec": 0.3, "frame_size": 512, "hop_size": 256},
    )

    print_summary([bm_result])

    os.makedirs("report/results", exist_ok=True)
    save_results_csv([bm_result], "report/results/demo_benchmark.csv")

    # ─────────────────────────────────────────
    # HOÀN THÀNH
    # ─────────────────────────────────────────
    separator("DEMO HOÀN THÀNH")
    print("  Output files:")
    for f in os.listdir("output"):
        fpath = os.path.join("output", f)
        size  = os.path.getsize(fpath)
        print(f"    output/{f}  ({size/1024:.1f} KB)")

    print("\n  Benchmark CSV: report/results/demo_benchmark.csv")
    print("\n  Chạy benchmark đầy đủ: python main.py benchmark")
    print("═" * 65)


if __name__ == "__main__":
    demo()
