"""
main.py -- Entry point chinh cua du an Parallel Audio Noise Reduction
Ho tro che do:
  1. process   -- xu ly 1 file hoac toan bo thu muc
  2. benchmark -- so sanh tuan tu vs song song
  3. generate  -- tao file WAV mau de test
  4. report    -- tao bieu do bao cao hieu nang (can matplotlib)
"""

import sys
import os
import io
import argparse

# Fix UTF-8 output on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except AttributeError:
        pass


# ─────────────────────────────────────────────
#  Thêm src vào path
# ─────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def cmd_process(args):
    """Xử lý file(s) âm thanh."""
    from src.noise_reduction import process_file, process_directory

    if args.input:
        # Xử lý 1 file
        if not args.output:
            name, ext = os.path.splitext(os.path.basename(args.input))
            args.output = os.path.join("output", f"{name}_{args.filter}_cleaned{ext}")

        process_file(
            input_path=args.input,
            output_path=args.output,
            filter_name=args.filter,
            mode=args.mode,
            num_workers=args.workers,
            verbose=True,
        )
    else:
        # Xử lý thư mục
        process_directory(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            filter_name=args.filter,
            mode=args.mode,
            num_workers=args.workers,
        )


def cmd_benchmark(args):
    """Chạy benchmark so sánh tuần tự vs song song."""
    from src.audio_io import read_wav, mono_mix
    from src.benchmark import benchmark_all_filters, save_results_csv, save_results_json, print_summary
    import os

    # Xác định file benchmark
    if args.input:
        wav_path = args.input
    else:
        # Dùng file test ngắn nhất trong input/
        candidates = [
            os.path.join("input", f) for f in os.listdir("input")
            if f.endswith(".wav")
        ]
        if not candidates:
            print("[ERROR] Không có file WAV trong input/. Chạy 'python main.py generate' trước.")
            sys.exit(1)
        wav_path = sorted(candidates)[0]

    print(f"\n[BENCHMARK] File: {wav_path}")
    samples, sr, ch = read_wav(wav_path)
    signal = mono_mix(samples, ch)

    # Xác định số workers để test
    max_cores = os.cpu_count() or 4
    worker_counts = [1, 2, 4]
    if max_cores >= 8:
        worker_counts.append(8)
    if max_cores >= 16:
        worker_counts.append(16)

    results = benchmark_all_filters(signal, sr, worker_counts=worker_counts)
    print_summary(results)

    # Luu ket qua
    os.makedirs("report/results", exist_ok=True)
    save_results_csv(results, "report/results/benchmark.csv")
    save_results_json(results, "report/results/benchmark.json")


def cmd_report(args):
    """Tao bao cao hieu nang voi bieu do matplotlib."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from report.generate_report import generate_full_report

    csv_path  = args.csv or "report/results/benchmark.csv"
    input_wav = args.input

    if not os.path.isfile(csv_path):
        print(f"[ERROR] Khong tim thay {csv_path}")
        print("        Chay 'python main.py benchmark' truoc.")
        sys.exit(1)

    if not input_wav:
        # Tu tim file WAV trong input/
        candidates = [
            os.path.join("input", f) for f in os.listdir("input")
            if f.endswith(".wav")
        ] if os.path.isdir("input") else []
        if candidates:
            input_wav = sorted(candidates)[0]

    generated = generate_full_report(
        csv_path=csv_path,
        input_wav=input_wav,
        output_dir="report/results",
    )
    print(f"\n[DONE] Generated {len(generated)} chart(s):")
    for p in generated:
        print(f"  {p}")


def cmd_generate(args):
    """Tạo file WAV mẫu để test."""
    sys.path.insert(0, "samples")
    from samples.generate_sample import generate_test_samples
    generate_test_samples(input_dir="input")


# ─────────────────────────────────────────────
#  CLI Parser
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Parallel Audio Noise Reduction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  # Tạo file test
  python main.py generate

  # Xử lý 1 file (song song, spectral subtraction)
  python main.py process --input input/short_5s.wav

  # Xử lý với moving average, tuần tự
  python main.py process --input input/short_5s.wav --filter moving_average --mode sequential

  # Xử lý toàn bộ thư mục
  python main.py process --input-dir input --output-dir output

  # Chạy benchmark
  python main.py benchmark

  # Benchmark trên file cụ thể
  python main.py benchmark --input input/medium_15s.wav
        """
    )

    sub = parser.add_subparsers(dest="command")

    # ── process ──
    p_proc = sub.add_parser("process", help="Xử lý file WAV")
    p_proc.add_argument("--input",      "-i",  type=str, default=None,
                        help="Đường dẫn file WAV đầu vào (nếu không có → xử lý thư mục)")
    p_proc.add_argument("--output",     "-o",  type=str, default=None,
                        help="Đường dẫn file WAV đầu ra")
    p_proc.add_argument("--input-dir",         type=str, default="input",
                        help="Thư mục đầu vào (mặc định: input/)")
    p_proc.add_argument("--output-dir",        type=str, default="output",
                        help="Thư mục đầu ra (mặc định: output/)")
    p_proc.add_argument("--filter", "-f",
                        choices=[
                            "moving_average", "spectral_subtraction", "wiener",
                            "median", "mmse_stsa", "kalman", "adaptive_kalman",
                        ],
                        default="spectral_subtraction",
                        help="Filter to use (default: spectral_subtraction)")
    p_proc.add_argument("--mode",       "-m",
                        choices=["sequential", "parallel"],
                        default="parallel",
                        help="Chế độ chạy (mặc định: parallel)")
    p_proc.add_argument("--workers",    "-w",  type=int, default=None,
                        help="Số worker processes (mặc định: số CPU core)")

    # ── benchmark ──
    p_bm = sub.add_parser("benchmark", help="Benchmark tuần tự vs song song")
    p_bm.add_argument("--input", "-i", type=str, default=None,
                      help="File WAV để benchmark")

    # ── generate ──
    sub.add_parser("generate", help="Tao file WAV test mau trong input/")

    # ── report ──
    p_rep = sub.add_parser("report", help="Tao bieu do bao cao hieu nang (can matplotlib)")
    p_rep.add_argument("--csv",   "-c", type=str, default=None,
                       help="Duong dan file benchmark.csv (mac dinh: report/results/benchmark.csv)")
    p_rep.add_argument("--input", "-i", type=str, default=None,
                       help="File WAV goc de ve waveform/spectrogram")

    args = parser.parse_args()

    if args.command == "process":
        cmd_process(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    elif args.command == "generate":
        cmd_generate(args)
    elif args.command == "report":
        cmd_report(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
