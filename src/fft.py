import cmath
import math


# --- Các hàm hỗ trợ nội bộ ---

def _next_power_of_2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def _zero_pad(signal: list, size: int) -> list:
    return signal + [0.0] * (size - len(signal))


def _bit_reverse_copy(a: list) -> list:
    n      = len(a)
    bits   = int(math.log2(n))
    result = [None] * n
    for i in range(n):
        rev = 0
        x   = i
        for _ in range(bits):
            rev = (rev << 1) | (x & 1)
            x >>= 1
        result[rev] = a[i]
    return result


# --- Cooley-Tukey Radix-2 FFT (O(N log N)) ---

def fft(signal: list) -> list:
    n      = _next_power_of_2(len(signal))
    padded = _zero_pad([complex(x) for x in signal], n)
    a      = _bit_reverse_copy(padded)

    s = 1
    while s <= int(math.log2(n)):
        m   = 1 << s
        w_m = cmath.exp(-2j * cmath.pi / m)
        for k in range(0, n, m):
            w = complex(1, 0)
            for j in range(m // 2):
                t                 = w * a[k + j + m // 2]
                u                 = a[k + j]
                a[k + j]          = u + t
                a[k + j + m // 2] = u - t
                w *= w_m
        s += 1
    return a


# --- Inverse FFT: IFFT(X) = conj(FFT(conj(X))) / N ---

def ifft(spectrum: list) -> list:
    n         = len(spectrum)
    conj_spec = [x.conjugate() for x in spectrum]
    result    = fft(conj_spec)
    return [x.conjugate() / n for x in result]


# --- Short-Time FFT: chia frame + áp Hann window ---

def stfft(signal: list, frame_size: int = 1024, hop_size: int = 512) -> list:
    frames = []
    n      = len(signal)
    start  = 0
    while start < n:
        frame    = signal[start:start + frame_size]
        windowed = apply_hann_window(frame, frame_size)
        frames.append((fft(windowed), start))
        start += hop_size
    return frames


# --- Inverse STFFT: ghép frame bằng Overlap-Add ---

def istfft(frames: list, signal_len: int, frame_size: int = 1024, hop_size: int = 512) -> list:
    output     = [0.0] * (signal_len + frame_size)
    weight_sum = [0.0] * (signal_len + frame_size)
    hann_win   = _make_hann_window(frame_size)

    for spectrum, start in frames:
        time_frame = ifft(spectrum)
        real_frame = [x.real for x in time_frame[:frame_size]]
        for i in range(len(real_frame)):
            idx = start + i
            if idx < len(output):
                output[idx]     += real_frame[i] * hann_win[i]
                weight_sum[idx] += hann_win[i] * hann_win[i]

    result = []
    for i in range(signal_len):
        if weight_sum[i] > 1e-2:
            result.append(output[i] / weight_sum[i])
        elif weight_sum[i] > 0:
            result.append(output[i] / 1e-2)
        else:
            result.append(0.0)
    return result


# --- Hann window: w[n] = 0.5 * (1 - cos(2π*n/(N-1))) ---

def _make_hann_window(size: int) -> list:
    if size <= 1:
        return [1.0]
    return [0.5 * (1.0 - math.cos(2.0 * math.pi * n / (size - 1))) for n in range(size)]


def apply_hann_window(frame: list, frame_size: int) -> list:
    win    = _make_hann_window(frame_size)
    padded = frame + [0.0] * (frame_size - len(frame))
    return [padded[i] * win[i] for i in range(frame_size)]


# --- Tiện ích phổ tần số ---

def magnitude(spectrum: list) -> list:
    return [abs(x) for x in spectrum]

def phase(spectrum: list) -> list:
    return [cmath.phase(x) for x in spectrum]

def power_spectrum(spectrum: list) -> list:
    return [abs(x) ** 2 for x in spectrum]

def reconstruct_from_mag_phase(mag: list, pha: list) -> list:
    return [cmath.rect(mag[k], pha[k]) for k in range(len(mag))]
