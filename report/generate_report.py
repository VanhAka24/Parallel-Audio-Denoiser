import os
import csv
import math
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
def _require_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg") 
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        return plt, gridspec
    except ImportError:
        print("[ERROR] matplotlib chua duoc cai dat.")
        print("        Chay: pip install matplotlib")
        sys.exit(1)

# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
def plot_speedup_chart(csv_path: str, output_path: str = None) -> str:
    plt, gridspec = _require_matplotlib()

    data = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname   = row["filter"]
            workers = int(row["num_workers"])
            speedup = float(row["speedup"])
            eff     = float(row["efficiency"])
            if fname not in data:
                data[fname] = {"workers": [], "speedup": [], "efficiency": []}
            data[fname]["workers"].append(workers)
            data[fname]["speedup"].append(speedup)
            data[fname]["efficiency"].append(eff)

    if not data:
        print("[WARNING] Khong co du lieu de ve bieu do.")
        return ""

    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(csv_path), "speedup_chart.png"
        )

    colors = {
        "moving_average":       "#FF6B6B",
        "median":               "#FF9F43",
        "spectral_subtraction": "#48CAE4",
        "wiener":               "#06D6A0",
        "mmse_stsa":            "#A855F7",
        "kalman":               "#FFD166",
        "adaptive_kalman":      "#EF476F",
    }
    display_names = {
        "moving_average":       "Moving Average",
        "spectral_subtraction": "Spectral Subtraction",
        "wiener":               "Wiener Filter",
        "median":               "Median Filter",
        "mmse_stsa":            "MMSE-STSA",
        "kalman":               "Kalman Filter",
        "adaptive_kalman":      "Adaptive Kalman",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Parallel Audio Noise Reduction -- Performance Benchmark",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.patch.set_facecolor("#1a1a2e")

    for ax in axes:
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    ax_speed, ax_eff = axes

    all_workers = sorted(set(
        w for d in data.values() for w in d["workers"]
    ))
    max_w = max(all_workers) if all_workers else 16
    ideal_x = [1] + all_workers
    ideal_y = ideal_x[:]

    ax_speed.plot(ideal_x, ideal_y, "--", color="#ffffff", alpha=0.4,
                  linewidth=1.5, label="Ideal (linear)")

    for fname, d in data.items():
        color = colors.get(fname, "#aaaaaa")
        label = display_names.get(fname, fname)
        ax_speed.plot(d["workers"], d["speedup"],
                      "o-", color=color, linewidth=2, markersize=7,
                      markerfacecolor=color, label=label)
        ax_eff.plot(d["workers"], d["efficiency"],
                    "s-", color=color, linewidth=2, markersize=7,
                    markerfacecolor=color, label=label)

    ax_speed.set_title("Speedup vs. Number of Workers", color="white", fontsize=12)
    ax_speed.set_xlabel("Number of Workers (Processes)")
    ax_speed.set_ylabel("Speedup (T_seq / T_par)")
    ax_speed.legend(loc="upper left", facecolor="#0f3460", edgecolor="#444",
                    labelcolor="white", fontsize=9)
    ax_speed.grid(True, alpha=0.2, color="#555")
    ax_speed.set_xticks(all_workers)

    ax_eff.axhline(y=1.0, linestyle="--", color="#ffffff", alpha=0.4,
                   linewidth=1.5, label="Ideal efficiency")
    ax_eff.set_title("Parallel Efficiency vs. Number of Workers", color="white", fontsize=12)
    ax_eff.set_xlabel("Number of Workers (Processes)")
    ax_eff.set_ylabel("Efficiency = Speedup / N_workers")
    ax_eff.legend(loc="upper right", facecolor="#0f3460", edgecolor="#444",
                  labelcolor="white", fontsize=9)
    ax_eff.grid(True, alpha=0.2, color="#555")
    ax_eff.set_xticks(all_workers)
    ax_eff.set_ylim(-0.1, 1.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor="#1a1a2e")
    plt.close()
    print(f"[CHART] Speedup chart: {output_path}")
    return output_path

# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
def plot_waveform_comparison(input_wav: str,
                              output_path: str = None,
                              max_sec: float = 3.0) -> str:
    plt, gridspec = _require_matplotlib()
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.audio_io import read_wav, mono_mix
    from src.filters import spectral_subtraction, wiener_filter

    samples_in, sr, ch = read_wav(input_wav)
    signal_in = mono_mix(samples_in, ch)
    max_n     = int(max_sec * sr)
    signal_in = signal_in[:max_n]
    t_in      = [i / sr for i in range(len(signal_in))]

    outputs = {}
    outputs["spectral_subtraction"] = spectral_subtraction(signal_in, sr)
    outputs["wiener"]               = wiener_filter(signal_in, sr)

    n_plots  = 1 + len(outputs)
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 2.5 * n_plots))
    fig.suptitle(f"Waveform Comparison (first {max_sec:.1f}s)",
                 fontsize=13, fontweight="bold", color="white")
    fig.patch.set_facecolor("#1a1a2e")

    if n_plots == 1:
        axes = [axes]

    display_names = {
        "spectral_subtraction": "Spectral Subtraction",
        "wiener":               "Wiener Filter",
    }
    colors = ["#48CAE4", "#06D6A0"]

    ax0 = axes[0]
    ax0.set_facecolor("#16213e")
    ax0.plot(t_in, signal_in, color="#aaaaff", linewidth=0.6, alpha=0.9)
    ax0.set_title("Input (Noisy)", color="white", fontsize=10)
    ax0.set_ylabel("Amplitude", color="white")
    ax0.tick_params(colors="white")
    ax0.set_ylim(-1.1, 1.1)
    ax0.grid(True, alpha=0.15, color="#555")
    for spine in ax0.spines.values():
        spine.set_edgecolor("#444")

    for i, (fname, mono) in enumerate(outputs.items()):
        t_out = [j / sr for j in range(len(mono))]
        ax    = axes[i + 1]
        ax.set_facecolor("#16213e")
        ax.plot(t_out, mono, color=colors[i % len(colors)], linewidth=0.6, alpha=0.9)
        ax.set_title(display_names.get(fname, fname), color="white", fontsize=10)
        ax.set_ylabel("Amplitude", color="white")
        ax.tick_params(colors="white")
        ax.set_ylim(-1.1, 1.1)
        ax.grid(True, alpha=0.15, color="#555")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    axes[-1].set_xlabel("Time (seconds)", color="white")

    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(os.path.dirname(input_wav)),
            "report", "results", "waveform_comparison.png"
        )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()
    print(f"[CHART] Waveform: {output_path}")
    return output_path

# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
def compute_spectrogram(signal: list, frame_size: int = 512,
                         hop_size: int = 128) -> list:
    from src.fft import stfft, magnitude

    frames = stfft(signal, frame_size=frame_size, hop_size=hop_size)
    spec   = []
    n_bins = frame_size

    for spectrum, _ in frames:
        mags = magnitude(spectrum)[:n_bins // 2]
        log_mags = [20 * math.log10(max(m, 1e-10)) for m in mags]
        spec.append(log_mags)

    return spec

def plot_spectrogram_comparison(input_wav: str,
                                 output_path: str = None) -> str:
    plt, gridspec = _require_matplotlib()
    from src.audio_io import read_wav, mono_mix
    from src.filters import wiener_filter

    samples_in,  sr, ch = read_wav(input_wav)
    signal_in  = mono_mix(samples_in, ch)
    
    max_n = int(3.0 * sr)
    signal_in = signal_in[:max_n]
    signal_out = wiener_filter(signal_in, sr)

    frame_size = 512
    hop_size   = 128

    spec_in  = compute_spectrogram(signal_in,  frame_size, hop_size)
    spec_out = compute_spectrogram(signal_out, frame_size, hop_size)

    n_frames = min(len(spec_in), len(spec_out))
    n_bins   = len(spec_in[0]) if spec_in else 1

    def to_2d(spec):
        arr = []
        for row in spec[:n_frames]:
            arr.append(row[:n_bins])
        return arr

    img_in  = to_2d(spec_in)
    img_out = to_2d(spec_out)

    vmin = min(min(r) for r in img_in + img_out)
    vmax = max(max(r) for r in img_in + img_out)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Spectrogram Comparison -- Wiener Filter",
        fontsize=13, fontweight="bold", color="white"
    )
    fig.patch.set_facecolor("#1a1a2e")

    duration_in  = len(signal_in)  / sr
    duration_out = len(signal_out) / sr
    freq_max     = sr / 2 / 1000 

    def draw_spec(ax, img, title, duration):
        ax.set_facecolor("#16213e")
        im = ax.imshow(
            list(zip(*img)), 
            aspect="auto",
            origin="lower",
            extent=[0, duration, 0, freq_max],
            cmap="magma",
            vmin=vmin, vmax=vmax,
            interpolation="bilinear",
        )
        ax.set_title(title, color="white", fontsize=11)
        ax.set_xlabel("Time (s)", color="white")
        ax.set_ylabel("Frequency (kHz)", color="white")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
        return im

    im1 = draw_spec(ax1, img_in,  "Input (Noisy)",   duration_in)
    im2 = draw_spec(ax2, img_out, "Output (Cleaned)", duration_out)

    cbar = fig.colorbar(im2, ax=[ax1, ax2], shrink=0.8, pad=0.02)
    cbar.set_label("Magnitude (dB)", color="white")
    cbar.ax.tick_params(colors="white")

    if output_path is None:
        output_path = os.path.join(
            "report", "results",
            f"spectrogram_wiener.png"
        )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()
    print(f"[CHART] Spectrogram: {output_path}")
    return output_path

# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
def plot_execution_time_bar(csv_path: str, output_path: str = None) -> str:
    plt, _ = _require_matplotlib()

    data = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname   = row["filter"]
            workers = int(row["num_workers"])
            t       = float(row["time_sec"])
            if fname not in data:
                data[fname] = {"seq": None, "best_par": float("inf"), "best_w": 1}
            if workers == 1:
                data[fname]["seq"] = t
            else:
                if t < data[fname]["best_par"]:
                    data[fname]["best_par"] = t
                    data[fname]["best_w"]   = workers

    if not data:
        return ""

    if output_path is None:
        output_path = os.path.join(os.path.dirname(csv_path), "execution_time_bar.png")

    display_names = {
        "moving_average":       "Moving\nAverage",
        "spectral_subtraction": "Spectral\nSubtraction",
        "wiener":               "Wiener\nFilter",
        "median":               "Median\nFilter",
        "mmse_stsa":            "MMSE-STSA",
        "kalman":               "Kalman\nFilter",
        "adaptive_kalman":      "Adaptive\nKalman",
    }

    filters = list(data.keys())
    labels  = [display_names.get(f, f) for f in filters]
    seq_t   = [data[f]["seq"] or 0 for f in filters]
    par_t   = [data[f]["best_par"] if data[f]["best_par"] < float("inf") else 0
               for f in filters]

    x     = list(range(len(filters)))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    bars1 = ax.bar([xi - width/2 for xi in x], seq_t, width,
                   label="Sequential (1 worker)", color="#FF6B6B", alpha=0.9)
    bars2 = ax.bar([xi + width/2 for xi in x], par_t, width,
                   label="Best Parallel", color="#48CAE4", alpha=0.9)

    for bar in bars1:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.01,
                    f"{h:.3f}s", ha="center", va="bottom",
                    color="white", fontsize=8)

    for bar in bars2:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.01,
                    f"{h:.3f}s", ha="center", va="bottom",
                    color="white", fontsize=8)

    ax.set_xlabel("Filter", color="white", fontsize=11)
    ax.set_ylabel("Execution Time (seconds)", color="white", fontsize=11)
    ax.set_title("Execution Time: Sequential vs Best Parallel",
                 color="white", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color="white", fontsize=9)
    ax.tick_params(colors="white")
    ax.legend(facecolor="#0f3460", edgecolor="#444", labelcolor="white")
    ax.grid(True, alpha=0.15, color="#555", axis="y")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()
    print(f"[CHART] Execution time bar: {output_path}")
    return output_path

# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
def generate_full_report(csv_path: str = "report/results/benchmark.csv",
                          input_wav: str = None,
                          output_dir: str = "report/results") -> list:
    os.makedirs(output_dir, exist_ok=True)
    generated = []

    if os.path.isfile(csv_path):
        p = plot_speedup_chart(csv_path, os.path.join(output_dir, "speedup_chart.png"))
        if p:
            generated.append(p)

        p = plot_execution_time_bar(csv_path, os.path.join(output_dir, "execution_time_bar.png"))
        if p:
            generated.append(p)

    if input_wav and os.path.isfile(input_wav):
        p = plot_waveform_comparison(
            input_wav,
            os.path.join(output_dir, "waveform_comparison.png")
        )
        if p:
            generated.append(p)

        p = plot_spectrogram_comparison(
            input_wav,
            output_path=os.path.join(output_dir, f"spectrogram_wiener.png"),
        )
        if p:
            generated.append(p)

    print(f"\n[REPORT] Da tao {len(generated)} bieu do trong '{output_dir}/'")
    return generated

if __name__ == "__main__":
    generate_full_report()
