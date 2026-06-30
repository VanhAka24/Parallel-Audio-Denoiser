import time
import os
import csv
import json
from datetime import datetime


# --- Context manager đo thời gian chạy ---

class Timer:
    def __init__(self, name: str = ""):
        self.name    = name
        self.elapsed = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self._start
        return False


# --- Đo 1 bộ lọc với nhiều mức worker (tuần tự + song song) ---

def benchmark_filter(signal: list, sample_rate: int,
                     filter_name: str = "spectral_subtraction",
                     worker_counts: list = None,
                     filter_kwargs: dict = None) -> dict:
    if worker_counts is None:
        max_cores     = os.cpu_count() or 4
        worker_counts = [w for w in [1, 2, 4, 8, 16] if w <= max_cores]
        if max_cores not in worker_counts:
            worker_counts.append(max_cores)

    if filter_kwargs is None:
        filter_kwargs = {}

    results = {
        "filter":       filter_name,
        "signal_len":   len(signal),
        "sample_rate":  sample_rate,
        "duration_sec": len(signal) / sample_rate,
        "timestamp":    datetime.now().isoformat(),
        "runs":         [],
    }

    print(f"\n[BENCHMARK] Filter: {filter_name}")
    print(f"  Signal: {len(signal)} samples ({len(signal)/sample_rate:.1f}s)")
    print(f"  {'Workers':<10} {'Time(s)':<12} {'Speedup':<10} {'Efficiency':<12}")
    print(f"  {'-'*46}")

    sequential_time = None
    NUM_RUNS = 10
    for nw in sorted(set(worker_counts)):
        total_time = 0.0
        for _ in range(NUM_RUNS):
            run_result = _run_single(signal, sample_rate, filter_name, nw, filter_kwargs)
            total_time += run_result["time_sec"]
        
        avg_time = round(total_time / NUM_RUNS, 6)
        run_result["time_sec"] = avg_time

        if nw == 1:
            sequential_time = avg_time
            speedup = efficiency = 1.0
        else:
            if sequential_time and sequential_time > 0:
                speedup    = sequential_time / avg_time
                efficiency = speedup / nw
            else:
                speedup = efficiency = 1.0

        run_result["speedup"]    = round(speedup, 4)
        run_result["efficiency"] = round(efficiency, 4)
        results["runs"].append(run_result)
        print(f"  {nw:<10} {run_result['time_sec']:<12.4f} {speedup:<10.3f} {efficiency:<12.3f}")

    return results


# --- Chạy 1 lần benchmark cho num_workers workers ---

def _run_single(signal, sample_rate, filter_name, num_workers, filter_kwargs):
    from src.filters import moving_average_filter, spectral_subtraction, wiener_filter
    from src.filters_advanced import median_filter, mmse_stsa, kalman_filter, adaptive_kalman_filter
    from src.parallel_engine import (
        parallel_moving_average, parallel_spectral_subtraction, parallel_wiener,
        parallel_median, parallel_mmse_stsa, parallel_kalman,
    )

    if num_workers == 1:
        with Timer() as t:
            if filter_name == "moving_average":
                _ = moving_average_filter(signal, **filter_kwargs)
            elif filter_name == "spectral_subtraction":
                _ = spectral_subtraction(signal, sample_rate, **filter_kwargs)
            elif filter_name == "wiener":
                _ = wiener_filter(signal, sample_rate, **filter_kwargs)
            elif filter_name == "median":
                _ = median_filter(signal, **filter_kwargs)
            elif filter_name == "mmse_stsa":
                _ = mmse_stsa(signal, sample_rate, **filter_kwargs)
            elif filter_name == "kalman":
                _ = kalman_filter(signal, **filter_kwargs)
            elif filter_name == "adaptive_kalman":
                _ = adaptive_kalman_filter(signal, **filter_kwargs)
    else:
        with Timer() as t:
            if filter_name == "moving_average":
                _ = parallel_moving_average(signal, num_workers=num_workers, **filter_kwargs)
            elif filter_name == "spectral_subtraction":
                _ = parallel_spectral_subtraction(signal, sample_rate,
                                                   num_workers=num_workers, **filter_kwargs)
            elif filter_name == "wiener":
                _ = parallel_wiener(signal, sample_rate,
                                    num_workers=num_workers, **filter_kwargs)
            elif filter_name == "median":
                _ = parallel_median(signal, num_workers=num_workers, **filter_kwargs)
            elif filter_name == "mmse_stsa":
                _ = parallel_mmse_stsa(signal, sample_rate,
                                       num_workers=num_workers, **filter_kwargs)
            elif filter_name == "kalman":
                _ = parallel_kalman(signal, num_workers=num_workers, **filter_kwargs)
            elif filter_name == "adaptive_kalman":
                _ = adaptive_kalman_filter(signal, **filter_kwargs)

    return {
        "num_workers": num_workers,
        "mode":        "sequential" if num_workers == 1 else "parallel",
        "time_sec":    round(t.elapsed, 6),
    }


# --- Benchmark toàn bộ pipeline (Spectral Subtraction + Wiener) ---

def benchmark_all_filters(signal: list, sample_rate: int, worker_counts: list = None) -> list:
    FILTER_KWARGS = {
        "spectral_subtraction": {"noise_duration_sec": 0.5, "alpha_max": 5.0,
                                  "alpha_min": 1.0, "beta": 0.01,
                                  "frame_size": 512, "hop_size": 256},
        "wiener":               {"noise_duration_sec": 0.5, "alpha_dd": 0.98,
                                  "frame_size": 512, "hop_size": 256},
    }

    all_results = []
    for fname, kwargs in FILTER_KWARGS.items():
        all_results.append(benchmark_filter(signal, sample_rate, fname, worker_counts, kwargs))

    # In báo cáo tổng kết
    print("\n" + "═"*60)
    print("  BÁO CÁO TỔNG KẾT HIỆU NĂNG (PERFORMANCE REPORT)")
    print("═"*60)
    for res in all_results:
        runs = res.get("runs", [])
        if not runs:
            continue
        seq_run  = next((r for r in runs if r["num_workers"] == 1), runs[0])
        best_run = min(runs, key=lambda x: x["time_sec"])
        print(f" 🔹 Thuật toán: {res['filter'].upper()}")
        print(f"    - Cấu hình thử nghiệm: {len(runs)} mức độ luồng (workers)")
        print(f"    - Thời gian Tuần tự (1 Core):     {seq_run['time_sec']:.4f} giây")
        print(f"    - Thời gian Song song ({best_run['num_workers']} Cores):   {best_run['time_sec']:.4f} giây")
        print(f"    => Tốc độ tăng gấp {best_run['speedup']:.2f} lần!")
        print("─"*60)

    print(" ✓ Đã hoàn tất. Kết quả lưu tại report/results.\n")
    return all_results


# --- Lưu kết quả ra CSV / JSON ---

def save_results_csv(results: list, output_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    rows = []
    for r in results:
        for run in r["runs"]:
            rows.append({
                "filter":       r["filter"],
                "duration_sec": r["duration_sec"],
                "num_workers":  run["num_workers"],
                "mode":         run["mode"],
                "time_sec":     run["time_sec"],
                "speedup":      run["speedup"],
                "efficiency":   run["efficiency"],
                "timestamp":    r["timestamp"],
            })
    if not rows:
        return
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[SAVED] CSV: {output_path}")


def save_results_json(results: list, output_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[SAVED] JSON: {output_path}")


# --- In bảng tổng hợp chi tiết ---

def print_summary(results: list) -> None:
    print("\n" + "=" * 70)
    print("  BENCHMARK SUMMARY -- Parallel Audio Noise Reduction")
    print("=" * 70)
    for r in results:
        print(f"\n  Filter: {r['filter'].upper()}")
        print(f"  Audio duration: {r['duration_sec']:.1f}s")
        print(f"  {'Workers':<10} {'Mode':<12} {'Time(s)':<10} {'Speedup':<10} {'Efficiency'}")
        print(f"  {'-'*55}")
        for run in r["runs"]:
            print(f"  {run['num_workers']:<10} {run['mode']:<12} "
                  f"{run['time_sec']:<10.4f} {run['speedup']:<10.3f} {run['efficiency']:.3f}")
    print("\n" + "=" * 70)
