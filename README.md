<h1 align="center">🎙️ Parallel Audio Noise Reduction</h1>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python 3.8+">
  <img src="https://img.shields.io/badge/Multiprocessing-Enabled-success.svg" alt="Multiprocessing">
  <img src="https://img.shields.io/badge/DSP-Pure%20Python-orange.svg" alt="Pure Python DSP">
</p>

<p align="center">
  <b>Parallel Audio Noise Reduction System built with Pure Python.</b><br>
  This project implements Digital Signal Processing (DSP) algorithms from scratch, including FFT, STFT, and complex filters, <b>WITHOUT relying on</b> third-party libraries like <code>numpy</code> or <code>scipy</code>.
</p>

---

## ✨ Features

- 🚫 **Pure Python DSP Engine**: Custom implementation of the **Fast Fourier Transform (FFT)** using the Cooley-Tukey algorithm, Hann window function, and Overlap-Add method.
- 🚀 **Parallel Processing (Multiprocessing)**: Bypasses Python's GIL by splitting the audio signal into smaller chunks (with overlap) and processing them concurrently using multi-threading/multi-processing. Automatically merges chunks using Linear Cross-Fade to prevent clicking or popping artifacts.
- 🎛️ **Diverse Noise Reduction Algorithms**:
  - `moving_average`: Time-domain Moving Average Filter.
  - `median`: Time-domain Median Filter.
  - `spectral_subtraction`: Frequency-domain Adaptive Spectral Subtraction.
  - `wiener`: Frequency-domain SNR-optimized Wiener Filter.
  - `mmse_stsa`: Minimum Mean-Square Error Short-Time Spectral Amplitude estimator (Ephraim-Malah).
  - `kalman` & `adaptive_kalman`: 5-step Kalman Filter (Time-domain/State-space).
- 🖥️ **Modern GUI**: Includes an intuitive Graphical User Interface (built with `tkinter` + `matplotlib`) to test algorithms, compare pre/post processing waveforms, and observe real-time RMS reduction.
- 📊 **CLI & Benchmark Tool**: Built-in powerful Command Line Interface and automated benchmark engine to evaluate speedup and Amdahl's Law.

---

## 📂 Project Structure

```text
ckTTSS/
├── input/                   # Directory for input WAV files
├── output/                  # Directory for processed WAV files
├── src/                     # Core DSP Engine
│   ├── audio_io.py          # Read/Write WAV files (using Python struct)
│   ├── fft.py               # Cooley-Tukey FFT / IFFT algorithms
│   ├── filters.py           # Spectral Subtraction, Wiener, Moving Average
│   ├── filters_advanced.py  # MMSE-STSA, Kalman, Median Filter
│   ├── parallel_engine.py   # Multiprocessing Pool & Chunking Manager
│   ├── noise_reduction.py   # Orchestrator pipeline
│   └── benchmark.py         # Parallel processing performance benchmarking
├── samples/                 # Scripts to automatically generate test files (real human voice & noise)
├── report/                  # Directory for benchmark results and charts
├── main.py                  # CLI Entry point
├── gui.py                   # Graphical User Interface (GUI)
└── demo.py                  # Full pipeline demo script
```

---

## 🛠️ Installation

The project requires Python 3.8 or higher. You only need to install `matplotlib` for the GUI and plotting charts (the Core DSP runs entirely on Python's standard library).

```bash
git clone https://github.com/your-username/Parallel-Audio-Noise-Reduction.git
cd Parallel-Audio-Noise-Reduction
pip install -r requirements.txt
```

---

## 🚀 Usage Guide

### 1. Using the Graphical User Interface (GUI)
The easiest way to experience the project is to launch the UI:
```bash
python gui.py
```
*(You can load a WAV file, select a filter, processing mode (Sequential / Parallel), choose the number of workers, and click "Start Noise Reduction" to visually see the waveform)*

### 2. Using the Command Line Interface (CLI)

#### Auto-generate test data:
Generate noisy human voice audio files (using Text-to-Speech):
```bash
python main.py generate
```

#### Process a single WAV file:
```bash
# Use default config (Spectral Subtraction, Parallel mode, auto-workers)
python main.py process --input input/hello_en_noisy.wav

# Specify the Wiener algorithm and a specific number of workers (e.g., 8 cores)
python main.py process --input input/hello_en_noisy.wav --filter wiener --mode parallel --workers 8

# Process using the MMSE-STSA algorithm in sequential mode (1 thread)
python main.py process --input input/hello_en_noisy.wav --filter mmse_stsa --mode sequential
```

#### Process an entire directory (Batch Processing):
```bash
python main.py process --input-dir input --output-dir output --filter kalman
```

#### Run System Benchmark:
Test the multi-core acceleration capabilities on your machine:
```bash
python main.py benchmark
python main.py benchmark --input input/hello_en_noisy.wav
```
*The system will run with 1, 2, 4, 8, and 16 workers respectively and save the evaluation charts in the `report/results/` directory.*

### 3. Quick Demo
```bash
python demo.py
```

---

## 🧠 Parallelization Mechanism (Multiprocessing)

The chunking and parallel processing mechanism is specifically designed for Audio DSP:
1. **Chunking**: Long audio signals are sliced into smaller `chunks`, leaving an `overlap` margin at both ends to preserve frequency domain integrity during STFT (Short-Time Fourier Transform).
2. **Pool Initialization**: `multiprocessing.Pool` bypasses the Python GIL by mapping each chunk to multiple separate Processes.
3. **Merge & Cross-Fade**: Results are returned to the main process and stitched back together. A Linear window (*Linear Cross-Fade*) is applied to the `overlap` sections for smooth merging, preventing "Clicking/Popping artifacts" at the connection points.

---

## 📊 Expected Performance

By distributing the heavy FFT computations and complex matrix operations across multiple CPU cores, the application demonstrates a processing speedup of **1.5x to 3x** (depending on the WAV file length and OS overhead).

---

## 📜 License
This project is distributed under the MIT License. It serves as an educational project for studying Parallel Computing and Digital Signal Processing.
