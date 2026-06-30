"""
filters_advanced.py -- Cac bo loc khu nhieu nang cao (tu cai dat)
Bao gom:
  1. Median Filter        -- tot hon MA voi nhieu xung (impulse noise)
  2. MMSE-STSA           -- Minimum Mean Square Error Short-Time Spectral Amplitude
                            Su dung Decision-Directed a priori SNR estimator
  3. Kalman Filter (1D)  -- loc thich nghi theo thoi gian, tu cap nhat noise estimate

Khong su dung scipy, librosa hay bat ky thu vien DSP nao.
"""

import math
import cmath
from src.fft import (
    fft, ifft, stfft, istfft,
    magnitude, phase, power_spectrum,
    reconstruct_from_mag_phase,
)
from src.filters import estimate_noise_profile


# ================================================================
#  1. MEDIAN FILTER -- mien thoi gian
# ================================================================
def _insertion_sort(arr: list) -> list:
    """Selection sort cho list nho (dung cho window median)."""
    a = arr[:]
    for i in range(1, len(a)):
        key = a[i]
        j   = i - 1
        while j >= 0 and a[j] > key:
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = key
    return a


def median_filter(signal: list, window_size: int = 5) -> list:
    """
    Median Filter -- loc trung vi truot (Median Smoothing Filter).

    Nguyen ly:
        y[n] = median( x[n - W//2], ..., x[n], ..., x[n + W//2] )

        Thay vi lay trung binh (Moving Average), lay gia tri TRUNG VI
        cua cua so W phan tu. Hieu qua hon MA voi:
        - Nhieu xung (impulse noise / salt-and-pepper noise)
        - Nhieu click trong audio

    Do phuc tap: O(N * W * log W) voi sort-based approach.

    So sanh voi Moving Average:
        MA:     y[n] = mean(window)   -- bi keo lech boi outlier
        Median: y[n] = median(window) -- khang nhieu voi outlier

    Args:
        signal:      Tin hieu dau vao (list of float).
        window_size: Kich thuoc cua so W (nen dung so le). Mac dinh=5.

    Returns:
        Tin hieu da loc (list of float, cung do dai voi input).
    """
    n      = len(signal)
    output = [0.0] * n
    half_w = window_size // 2

    for i in range(n):
        start = max(0, i - half_w)
        end   = min(n, i + half_w + 1)
        window_vals = signal[start:end]

        # Sort va lay phan tu giua
        sorted_vals = _insertion_sort(window_vals)
        mid         = len(sorted_vals) // 2
        output[i]   = sorted_vals[mid]

    return output


# ================================================================
#  2. MMSE-STSA -- Minimum Mean Square Error Short-Time Spectral Amplitude
# ================================================================
def _bessel_i0(x: float) -> float:
    """
    Xap xi ham Bessel bac 0 loai 1: I_0(x).
    Dung chuoi Taylor: I_0(x) = sum_{k=0}^{inf} (x/2)^{2k} / (k!)^2

    Dung trong cong thuc MMSE-STSA de tinh gain chinh xac.
    """
    if x == 0.0:
        return 1.0

    # Voi x lon, dung xap xi: I_0(x) ~ exp(x) / sqrt(2*pi*x)
    if abs(x) > 20.0:
        return math.exp(abs(x)) / math.sqrt(2 * math.pi * abs(x))

    result = 1.0
    term   = 1.0
    half_x = x / 2.0
    for k in range(1, 30):
        term   = term * (half_x / k) * (half_x / k)
        result += term
        if term < 1e-12:
            break
    return result


def _gamma_half_int(n: float) -> float:
    """
    Tinh Gamma(n) voi n la so nguyen hoac ban nguyen.
    Gamma(1/2) = sqrt(pi), Gamma(1) = 1, Gamma(n+1) = n*Gamma(n)
    """
    if abs(n - 0.5) < 1e-9:
        return math.sqrt(math.pi)
    if abs(n - 1.0) < 1e-9:
        return 1.0
    if n > 1.0:
        return (n - 1.0) * _gamma_half_int(n - 1.0)
    return math.sqrt(math.pi)   # fallback


def _mmse_gain(nu: float) -> float:
    """
    Tinh MMSE-STSA gain cho mot bin tan so.

    Cong thuc Ephraim-Malah (1984):
        H[k] = (sqrt(pi)/2) * (sqrt(nu[k]) / gamma_k) * exp(-nu[k]/2)
                * [(1 + nu[k]) * I_0(nu[k]/2) + nu[k] * I_1(nu[k]/2)]

    Xap xi thuc te (tranh overflow):
        nu = SNR_a * SNR_p / (1 + SNR_a)
        Khi nu lon: gain ~ 1
        Khi nu nho: gain ~ 0

    Args:
        nu: Tham so Bessel (nu = xi * gamma / (1 + xi), xi la a priori SNR)

    Returns:
        Gain H[k] trong [0, 1]
    """
    if nu <= 0:
        return 0.0

    try:
        exp_term = math.exp(-nu / 2.0)
        i0       = _bessel_i0(nu / 2.0)

        # I_1(x) xap xi = I_0(x) - 2/x * sum -- dung cong thuc don gian
        # I_1(x) ~ I_0(x) * (1 - 1/(2x)) cho x lon
        if nu > 0.1:
            i1 = i0 * (1.0 - 1.0 / max(nu, 0.01))
        else:
            i1 = nu / 2.0   # xap xi bac 1 khi nu nho

        gain = (math.sqrt(math.pi) / 2.0) * math.sqrt(nu) * exp_term * (
            (1.0 + nu) * i0 + nu * i1
        )

        # Clamp ve [0, 1.5] de tranh artifact
        return max(0.0, min(1.5, gain))
    except (OverflowError, ZeroDivisionError):
        return 1.0


def mmse_stsa(signal: list, sample_rate: int,
              noise_duration_sec: float = 0.5,
              alpha_dd: float = 0.98,
              frame_size: int = 1024,
              hop_size: int = 512) -> list:
    """
    Khu nhieu bang MMSE-STSA voi Decision-Directed SNR estimator.

    Nguyen ly:
        MMSE-STSA (Ephraim & Malah, 1984) tim ham gain H[k] de minimize
        Mean Square Error cua bien do pho tin hieu.

        1. A posteriori SNR:  gamma[k] = |X[k]|^2 / N[k]
        2. A priori SNR (Decision-Directed):
              xi[k] = alpha * |S_prev[k]|^2/N[k] + (1-alpha) * max(gamma[k]-1, 0)
           Lam tron theo thoi gian (alpha=0.98) de tranh musical noise.
        3. Nu:    nu[k] = xi[k] * gamma[k] / (1 + xi[k])
        4. Gain:  H[k] = MMSE_gain(nu[k])  (Bessel function estimator)
        5. S_hat[k] = H[k] * X[k]

    Uu diem so voi Wiener:
        - Wiener minimize MSE cua cong suat (power domain)
        - MMSE-STSA minimize MSE cua bien do (amplitude domain) -> tu nhien hon
        - Decision-Directed tranh musical noise tot hon

    Args:
        signal:           Tin hieu dau vao.
        sample_rate:      Tan so lay mau.
        noise_duration_sec: Thoi gian lay mau nhieu.
        alpha_dd:         He so lam tron Decision-Directed (0.95~0.99). Mac dinh=0.98.
        frame_size:       Kich thuoc FFT frame.
        hop_size:         Buoc nhay.

    Returns:
        Tin hieu da khu nhieu (list of float).
    """
    # Buoc 1: Uoc tinh noise profile
    noise_profile = estimate_noise_profile(
        signal, sample_rate, noise_duration_sec, frame_size
    )

    # Buoc 2: STFFT
    frames = stfft(signal, frame_size=frame_size, hop_size=hop_size)

    cleaned_frames = []
    fft_size       = None

    # Khoi tao a priori SNR truoc (cua frame truoc)
    prev_clean_pw  = None

    for spectrum, start in frames:
        pw  = power_spectrum(spectrum)   # |X[k]|^2
        mag = magnitude(spectrum)        # |X[k]|
        pha = phase(spectrum)            # angle X[k]

        if fft_size is None:
            fft_size      = len(spectrum)
            prev_clean_pw = [0.0] * fft_size

        new_spectrum = []
        new_prev     = []

        for k in range(fft_size):
            noise_k = max(noise_profile[k] if k < len(noise_profile) else 1e-10, 1e-10)

            # A posteriori SNR
            gamma_k = pw[k] / noise_k

            # A priori SNR (Decision-Directed)
            xi_k = (alpha_dd * prev_clean_pw[k] / noise_k
                    + (1.0 - alpha_dd) * max(gamma_k - 1.0, 0.0))
            xi_k = max(xi_k, 1e-6)   # tranh chia 0

            # Tinh nu
            nu_k = xi_k * gamma_k / (1.0 + xi_k)

            # Gain MMSE
            gain = _mmse_gain(nu_k)

            # Uoc tinh bien do sach
            clean_mag = gain * mag[k]
            new_prev.append(clean_mag ** 2)
            new_spectrum.append(cmath.rect(clean_mag, pha[k]))

        prev_clean_pw = new_prev
        cleaned_frames.append((new_spectrum, start))

    return istfft(cleaned_frames, len(signal), frame_size, hop_size)


# ================================================================
#  3. KALMAN FILTER (1D) -- mien thoi gian
# ================================================================
def kalman_filter(signal: list,
                  process_noise_var: float = 1e-5,
                  measurement_noise_var: float = 0.01) -> list:
    """
    Kalman Filter 1D thich nghi cho khu nhieu audio.

    Mo hinh he thong:
        x[n] = x[n-1] + w[n]    (process model: tin hieu bien doi cham)
        z[n] = x[n] + v[n]      (measurement: quan sat co nhieu)

        w[n] ~ N(0, Q)   : nhieu qua trinh (process noise)
        v[n] ~ N(0, R)   : nhieu do luong (measurement noise)

    Thuat toan Kalman (5 buoc):
        1. Predict:
             x_pred[n] = x_est[n-1]
             P_pred[n] = P[n-1] + Q

        2. Update (Kalman Gain):
             K[n] = P_pred[n] / (P_pred[n] + R)

        3. Correct:
             x_est[n] = x_pred[n] + K[n] * (z[n] - x_pred[n])
             P[n]     = (1 - K[n]) * P_pred[n]

    Khi K -> 0: tin tuong prediction hon (it nhieu qua trinh)
    Khi K -> 1: tin tuong measurement hon (it nhieu do luong)

    Dieu chinh tham so:
        - process_noise_var (Q) nho  --> filter lam min nhieu hon (lag nhieu hon)
        - measurement_noise_var (R) lon --> tin tuong model nhieu hon, filter manh

    Args:
        signal:               Tin hieu dau vao (list of float).
        process_noise_var:    Q -- phuong sai nhieu qua trinh. Mac dinh=1e-5.
        measurement_noise_var: R -- phuong sai nhieu do luong. Mac dinh=0.01.

    Returns:
        Tin hieu da loc (list of float).
    """
    n      = len(signal)
    output = [0.0] * n

    # Khoi tao
    x_est = signal[0] if n > 0 else 0.0   # Uoc tinh ban dau
    P     = 1.0                            # Uoc tinh sai so ban dau

    Q = process_noise_var
    R = measurement_noise_var

    for i in range(n):
        z = signal[i]   # measurement

        # --- Predict ---
        x_pred = x_est
        P_pred = P + Q

        # --- Kalman Gain ---
        K = P_pred / (P_pred + R)

        # --- Update ---
        x_est    = x_pred + K * (z - x_pred)
        P        = (1.0 - K) * P_pred
        output[i] = x_est

    return output


def adaptive_kalman_filter(signal: list,
                           window_size: int = 1000,
                           base_process_noise: float = 1e-5) -> list:
    """
    Kalman Filter thich nghi: tu dong uoc tinh measurement noise
    tu phuong sai cuc bo cua tin hieu.

    R[n] = var( signal[n-W : n] )   (uoc tinh truc tuyen)

    Cho ket qua tot hon khi tin hieu co tinh chat thay doi theo thoi gian.

    Args:
        signal:             Tin hieu dau vao.
        window_size:        Kich thuoc cua so uoc tinh R.
        base_process_noise: Q ban dau.

    Returns:
        Tin hieu da loc (list of float).
    """
    n      = len(signal)
    output = [0.0] * n

    x_est  = signal[0] if n > 0 else 0.0
    P      = 1.0
    Q      = base_process_noise

    for i in range(n):
        z = signal[i]

        # Uoc tinh R tu cua so cuc bo
        start_w = max(0, i - window_size)
        window  = signal[start_w:i + 1]
        if len(window) > 1:
            mean_w = sum(window) / len(window)
            R      = sum((x - mean_w) ** 2 for x in window) / len(window)
            R      = max(R, 1e-8)   # tranh R = 0
        else:
            R = 0.01

        # Kalman steps
        x_pred   = x_est
        P_pred   = P + Q
        K        = P_pred / (P_pred + R)
        x_est    = x_pred + K * (z - x_pred)
        P        = (1.0 - K) * P_pred
        output[i] = x_est

    return output
