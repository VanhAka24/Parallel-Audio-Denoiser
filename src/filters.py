import math
from src.fft import (
    fft, ifft, stfft, istfft,
    magnitude, phase, power_spectrum,
    reconstruct_from_mag_phase,
)


# --- Moving Average Filter (miền thời gian) ---

def moving_average_filter(signal: list, window_size: int = 5) -> list:
    n      = len(signal)
    output = [0.0] * n
    half_w = window_size // 2
    for i in range(n):
        start = max(0, i - half_w)
        end   = min(n, i + half_w + 1)
        vals  = signal[start:end]
        output[i] = sum(vals) / len(vals)
    return output


# --- Ước lượng nhiễu nền từ 0.5s đầu file ---

def estimate_noise_profile(signal: list, sample_rate: int,
                            noise_duration_sec: float = 0.5,
                            frame_size: int = 1024) -> list:
    from src.fft import stfft, power_spectrum

    noise_n   = min(int(noise_duration_sec * sample_rate), len(signal))
    frames    = stfft(signal[:noise_n], frame_size=frame_size, hop_size=frame_size // 2)

    if not frames:
        return [1e-8] * frame_size

    fft_size  = len(frames[0][0])
    avg_power = [0.0] * fft_size
    for spectrum, _ in frames:
        for k, pw in enumerate(power_spectrum(spectrum)):
            avg_power[k] += pw

    # Nhân 1.5x để bao phủ toàn bộ năng lượng nhiễu (Overestimation)
    return [p / len(frames) * 1.5 for p in avg_power]


# --- Adaptive Spectral Subtraction (Berouti 1979) ---
# alpha thay đổi theo SNR từng frame: frame nhiễu -> alpha lớn, frame giọng -> alpha nhỏ

def spectral_subtraction(signal: list, sample_rate: int,
                          noise_duration_sec: float = 0.5,
                          alpha_max: float = 5.0,
                          alpha_min: float = 1.0,
                          beta: float = 0.01,
                          frame_size: int = 1024,
                          hop_size: int = 512,
                          noise_profile_arr: list = None) -> list:
    noise_profile = (noise_profile_arr if noise_profile_arr is not None
                     else estimate_noise_profile(signal, sample_rate, noise_duration_sec, frame_size))

    frames         = stfft(signal, frame_size=frame_size, hop_size=hop_size)
    cleaned_frames = []
    prev_pw        = None
    mean_noise_pw  = sum(noise_profile) / len(noise_profile) if noise_profile else 1e-10

    for spectrum, start in frames:
        pw       = power_spectrum(spectrum)
        mag      = magnitude(spectrum)
        pha      = phase(spectrum)
        fft_size = len(spectrum)

        if prev_pw is None:
            prev_pw = pw

        # Tính alpha thích ứng theo SNR
        mean_sig_pw  = sum(pw) / fft_size
        frame_snr_db = 10 * math.log10(max(mean_sig_pw / mean_noise_pw, 1e-10))
        if frame_snr_db >= 20.0:
            alpha = alpha_min
        elif frame_snr_db <= -5.0:
            alpha = alpha_max
        else:
            alpha = alpha_max - (frame_snr_db + 5.0) * (alpha_max - alpha_min) / 25.0

        new_mag = []
        for k in range(fft_size):
            smooth_pw_k = 0.3 * pw[k] + 0.7 * prev_pw[k]
            prev_pw[k]  = smooth_pw_k
            noise_k     = noise_profile[k] if k < len(noise_profile) else 0.0
            clean_pw    = smooth_pw_k - alpha * noise_k
            floor_mag   = beta * mag[k]
            clean_mag   = math.sqrt(clean_pw) if clean_pw > 0 else floor_mag
            gain        = max(clean_mag, floor_mag) / max(mag[k], 1e-10)
            new_mag.append(mag[k] * gain)

        cleaned_frames.append((reconstruct_from_mag_phase(new_mag, pha), start))

    return istfft(cleaned_frames, len(signal), frame_size, hop_size)


# --- Decision-Directed Wiener Filter (Scalart & Filho 1996) ---
# Ước lượng a priori SNR từ lịch sử frame trước, loại bỏ musical noise

def wiener_filter(signal: list, sample_rate: int,
                  noise_duration_sec: float = 0.5,
                  alpha_dd: float = 0.98,
                  frame_size: int = 1024,
                  hop_size: int = 512,
                  noise_profile_arr: list = None) -> list:
    noise_profile = (noise_profile_arr if noise_profile_arr is not None
                     else estimate_noise_profile(signal, sample_rate, noise_duration_sec, frame_size))

    frames         = stfft(signal, frame_size=frame_size, hop_size=hop_size)
    cleaned_frames = []
    fft_size       = len(frames[0][0]) if frames else frame_size
    prev_clean_pw  = None

    for spectrum, start in frames:
        pw = power_spectrum(spectrum)
        if prev_clean_pw is None:
            prev_clean_pw = pw

        new_spectrum = []
        new_clean_pw = []
        mag_spec     = magnitude(spectrum)

        for k in range(fft_size):
            noise_k = max(noise_profile[k] if k < len(noise_profile) else 1e-10, 1e-10)
            gamma_k = pw[k] / noise_k  # a posteriori SNR
            xi_k    = (alpha_dd * prev_clean_pw[k] / noise_k   # a priori SNR (Decision-Directed)
                       + (1.0 - alpha_dd) * max(gamma_k - 1.0, 0.0))
            gain    = max(xi_k / (1.0 + xi_k), 0.02)  # Wiener gain, floor = 0.02

            new_clean_pw.append((gain * mag_spec[k]) ** 2)
            new_spectrum.append(spectrum[k] * gain)

        prev_clean_pw = new_clean_pw
        cleaned_frames.append((new_spectrum, start))

    return istfft(cleaned_frames, len(signal), frame_size, hop_size)


# --- Đo chất lượng SNR và RMS ---


def compute_rms(signal: list) -> float:
    if not signal:
        return 0.0
    return math.sqrt(sum(x ** 2 for x in signal) / len(signal))
