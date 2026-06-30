"""
generate_sample.py -- Tao file WAV mau phong phu hon
Bao gom: am thanh tong hop gia lap giong noi, hoi thoai, nhieu nen van phong
"""

import struct
import math
import random
import os


def _write_wav(filepath, samples_float, sample_rate, num_channels=1):
    """Ghi WAV PCM 16-bit."""
    int_s = [max(-32767, min(32767, int(s * 32767))) for s in samples_float]
    data  = struct.pack(f"<{len(int_s)}h", *int_s)
    br    = sample_rate * num_channels * 2
    ba    = num_channels * 2
    fmt   = struct.pack("<HHIIHH", 1, num_channels, sample_rate, br, ba, 16)
    rsz   = 4 + (8 + 16) + (8 + len(data))
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "wb") as f:
        f.write(b"RIFF"); f.write(struct.pack("<I", rsz))
        f.write(b"WAVE")
        f.write(b"fmt "); f.write(struct.pack("<I", 16)); f.write(fmt)
        f.write(b"data"); f.write(struct.pack("<I", len(data))); f.write(data)


def _gaussian(rng):
    """Box-Muller Gaussian noise."""
    u1 = max(rng.random(), 1e-12)
    u2 = rng.random()
    return math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)


def _envelope(t, total, attack=0.02, decay=0.05, release=0.05):
    """ADSR envelope don gian."""
    if t < attack:
        return t / attack
    if t > total - release:
        return max(0.0, (total - t) / release)
    return 1.0


# ─────────────────────────────────────────────
#  1. Giong noi tong hop (formant-based)
# ─────────────────────────────────────────────
def _voiced_segment(sr, duration, pitch, formants, amp, rng):
    """
    Tao mot doan 'am thanh giong noi' tong hop:
    - Nguon thanh: chuoi xung tuan hoan (pitch period)
    - Loc bo: tong hop cac formant (sine waves tai tan so formant)
    """
    n = int(duration * sr)
    out = []
    pitch_period = sr / pitch  # so sample / chu ky

    for i in range(n):
        t = i / sr
        # Glottal pulse: xung rang cua
        phase_in_period = (i % pitch_period) / pitch_period
        if phase_in_period < 0.3:
            glottal = math.sin(math.pi * phase_in_period / 0.3) ** 2
        else:
            glottal = 0.0

        # Formant resonance
        sig = 0.0
        for k, (freq, bw, gain) in enumerate(formants):
            sig += gain * math.sin(2 * math.pi * freq * t + k * 0.3)

        # Nhan voi glottal source
        sample = sig * glottal * amp * _envelope(t, duration)
        out.append(sample)
    return out


def _unvoiced_segment(sr, duration, amp, rng):
    """Am vo thanh: tieng fricative (s, f, sh) = filtered noise."""
    n = int(duration * sr)
    out = []
    # High-pass noise (simulate fricatives)
    prev = 0.0
    for i in range(n):
        noise = _gaussian(rng) * amp * 0.3
        hp    = noise - prev * 0.7   # simple 1st-order high-pass
        prev  = noise
        t     = i / sr
        out.append(hp * _envelope(t, duration, attack=0.01, release=0.02))
    return out


def _silence(sr, duration):
    return [0.0] * int(duration * sr)


def generate_speech_like(filepath, duration_sec=8.0, sample_rate=16000,
                          noise_level=0.08, noise_only_sec=0.5, seed=42):
    """
    Tao file WAV gia lap giong noi don (monologue):
    Xen ke cac am huong co thanh, am fricative, khoang lang.
    """
    rng = random.Random(seed)
    out = []

    # Doan dau: noise thuan de uoc tinh nhieu
    for _ in range(int(noise_only_sec * sample_rate)):
        out.append(_gaussian(rng) * noise_level)

    remaining = duration_sec - noise_only_sec
    t_used    = 0.0

    # Cac "am tiet" voi pitch va formant khac nhau
    syllable_templates = [
        # (pitch Hz, formants [(freq, bw, gain), ...], duration, voiced)
        (140, [(700,  90, 0.4), (1100, 110, 0.25), (2600, 160, 0.1)], 0.15, True),   # /a/
        (145, [(350,  60, 0.45),(2000,  90, 0.2),  (2800, 140, 0.08)], 0.12, True),  # /u/
        (135, [(280,  60, 0.4), (2700, 100, 0.25), (3300, 160, 0.08)], 0.10, True),  # /i/
        (150, [(400,  80, 0.4), (1000, 100, 0.22), (2500, 150, 0.1)],  0.13, True),  # /o/
        (0,   [],                                                        0.06, False), # fricative
        (0,   [],                                                        0.08, True),  # pause
    ]

    while t_used < remaining - 0.1:
        # Chon am tiet ngau nhien
        idx = rng.randint(0, len(syllable_templates) - 1)
        pitch, formants, base_dur, voiced = syllable_templates[idx]

        dur = base_dur * (0.8 + rng.random() * 0.6)
        dur = min(dur, remaining - t_used)

        if voiced and formants:
            seg = _voiced_segment(sample_rate, dur, pitch + rng.randint(-5, 5),
                                   formants, 0.55, rng)
        elif not voiced and not formants:
            # Khoang lang ngan
            seg = _silence(sample_rate, dur * 0.5)
        else:
            seg = _unvoiced_segment(sample_rate, dur, 0.5, rng)

        # Them nhieu nen
        for s in seg:
            out.append(s + _gaussian(rng) * noise_level)
        t_used += dur

        # Khoang nghi giua am tiet
        gap = rng.uniform(0.02, 0.06)
        gap = min(gap, remaining - t_used)
        for _ in range(int(gap * sample_rate)):
            out.append(_gaussian(rng) * noise_level)
        t_used += gap

    _write_wav(filepath, out, sample_rate)
    print(f"[GENERATED] {filepath} -- speech-like | {len(out)/sample_rate:.1f}s | noise={noise_level}")


def generate_dialogue(filepath, duration_sec=15.0, sample_rate=16000,
                       noise_level=0.10, seed=7):
    """
    Gia lap hoi thoai 2 nguoi:
    - Nguoi A: pitch thap (nam, 110-130 Hz)
    - Nguoi B: pitch cao (nu, 180-220 Hz)
    Xen ke nhau voi khoang nghi o giua.
    """
    rng  = random.Random(seed)
    out  = []

    # Noise dau de uoc tinh
    for _ in range(int(0.5 * sample_rate)):
        out.append(_gaussian(rng) * noise_level)

    remaining = duration_sec - 0.5
    t_used    = 0.0

    # Dinh nghia 2 giong noi
    speakers = {
        "A": {  # Nam
            "pitch_range": (110, 130),
            "formants": [(650, 80, 0.45), (1100, 100, 0.22), (2500, 150, 0.09)],
            "turn_dur": (1.5, 3.0),
        },
        "B": {  # Nu
            "pitch_range": (190, 220),
            "formants": [(800, 90, 0.42), (1500, 110, 0.24), (2800, 160, 0.10)],
            "turn_dur": (1.2, 2.5),
        },
    }

    speaker_order = ["A", "B", "A", "B", "A", "B", "A", "B"]
    s_idx = 0

    while t_used < remaining - 0.2:
        spk  = speaker_order[s_idx % len(speaker_order)]
        info = speakers[spk]
        s_idx += 1

        turn_dur = rng.uniform(*info["turn_dur"])
        turn_dur = min(turn_dur, remaining - t_used)
        t_turn   = 0.0

        # Mot luot noi gom nhieu am tiet
        while t_turn < turn_dur - 0.05:
            syl_dur = rng.uniform(0.08, 0.18)
            syl_dur = min(syl_dur, turn_dur - t_turn)
            pitch   = rng.randint(*info["pitch_range"])
            seg = _voiced_segment(sample_rate, syl_dur, pitch,
                                   info["formants"], 0.5, rng)
            for s in seg:
                out.append(s + _gaussian(rng) * noise_level)
            t_turn += syl_dur

            # Khoang nghi nho giua am tiet
            gap = rng.uniform(0.01, 0.04)
            gap = min(gap, turn_dur - t_turn)
            for _ in range(int(gap * sample_rate)):
                out.append(_gaussian(rng) * noise_level * 0.8)
            t_turn += gap

        t_used += turn_dur

        # Khoang cach giua 2 nguoi noi
        pause = rng.uniform(0.15, 0.45)
        pause = min(pause, remaining - t_used)
        for _ in range(int(pause * sample_rate)):
            out.append(_gaussian(rng) * noise_level * 0.6)
        t_used += pause

    _write_wav(filepath, out, sample_rate)
    print(f"[GENERATED] {filepath} -- dialogue | {len(out)/sample_rate:.1f}s | noise={noise_level}")


def generate_office_noise(filepath, duration_sec=10.0, sample_rate=16000, seed=99):
    """
    Gia lap tieng on nen van phong: tieng may lanh, tieng go phim,
    tieng nguoi noi xa + nhieu trang bai.
    Dung de test khu nhieu nen.
    """
    rng = random.Random(seed)
    n   = int(duration_sec * sample_rate)
    out = []

    # Tan so may lanh / quat (low-freq hum)
    hum_freqs = [50, 100, 150, 200]

    for i in range(n):
        t = i / sample_rate

        # Hum may lanh
        hum = sum(0.03 * math.sin(2 * math.pi * f * t + rng.random() * 0.01)
                  for f in hum_freqs)

        # Tieng go phim (random clicks)
        click = 0.0
        if rng.random() < 0.002:
            click = rng.uniform(0.05, 0.15) * math.exp(-rng.uniform(50, 200))

        # Nhieu trang
        white = _gaussian(rng) * 0.04

        # Tieng nguoi noi xa (very attenuated speech-like)
        distant = 0.02 * math.sin(2 * math.pi * rng.uniform(200, 500) * t)

        out.append(hum + click + white + distant)

    _write_wav(filepath, out, sample_rate)
    print(f"[GENERATED] {filepath} -- office noise | {duration_sec}s")


# ─────────────────────────────────────────────
#  Ham tong hop: tao tat ca file mau
# ─────────────────────────────────────────────
def generate_test_samples(input_dir: str = "input") -> None:
    """Tao bo file WAV mau phong phu."""
    os.makedirs(input_dir, exist_ok=True)

    print("\n[*] Dang tao cac file WAV mau...")
    print("-" * 50)

    # --- Giong noi tong hop ---
    generate_speech_like(
        os.path.join(input_dir, "speech_clean_8s.wav"),
        duration_sec=8.0, noise_level=0.06, seed=1)

    generate_speech_like(
        os.path.join(input_dir, "speech_noisy_8s.wav"),
        duration_sec=8.0, noise_level=0.18, seed=1)

    # --- Hoi thoai ---
    generate_dialogue(
        os.path.join(input_dir, "dialogue_15s.wav"),
        duration_sec=15.0, noise_level=0.10, seed=7)

    generate_dialogue(
        os.path.join(input_dir, "dialogue_noisy_15s.wav"),
        duration_sec=15.0, noise_level=0.25, seed=7)

    # --- Nhieu van phong (test khu nhieu nen) ---
    generate_office_noise(
        os.path.join(input_dir, "office_noise_10s.wav"),
        duration_sec=10.0, seed=99)

    # --- Giong noi + nhieu van phong (thuc te nhat) ---
    _mix_speech_and_noise(
        os.path.join(input_dir, "speech_office_mix_10s.wav"),
        duration_sec=10.0, speech_noise=0.08, bg_noise=0.12)

    print("-" * 50)
    print(f"[DONE] Tao xong {6} file WAV trong '{input_dir}/'")


def _mix_speech_and_noise(filepath, duration_sec=10.0, sample_rate=16000,
                           speech_noise=0.08, bg_noise=0.12):
    """Tron giong noi tong hop voi nhieu nen van phong."""
    rng = random.Random(42)
    n   = int(duration_sec * sample_rate)
    out = []

    # Noise profile dau
    for _ in range(int(0.5 * sample_rate)):
        # Nhieu van phong thuan
        hum   = sum(0.03 * math.sin(2 * math.pi * f * _ / sample_rate)
                    for f in [50, 100]) 
        out.append(hum + _gaussian(rng) * bg_noise)

    # Giong noi xen voi nhieu
    pitch_vals = [130, 140, 125, 145, 135]
    formants_A = [(650, 80, 0.45), (1100, 100, 0.22), (2500, 150, 0.09)]
    t_used = 0.5

    while t_used < duration_sec - 0.1:
        pitch    = pitch_vals[rng.randint(0, 4)]
        syl_dur  = rng.uniform(0.1, 0.22)
        syl_dur  = min(syl_dur, duration_sec - t_used)
        seg      = _voiced_segment(sample_rate, syl_dur, pitch, formants_A, 0.5, rng)

        for i, s in enumerate(seg):
            gi    = int(t_used * sample_rate) + i
            hum   = sum(0.025 * math.sin(2 * math.pi * f * gi / sample_rate)
                        for f in [50, 100])
            out.append(s + _gaussian(rng) * speech_noise + hum + _gaussian(rng) * bg_noise)
        t_used += syl_dur

        gap = rng.uniform(0.03, 0.1)
        gap = min(gap, duration_sec - t_used)
        for gi_off in range(int(gap * sample_rate)):
            gi  = int(t_used * sample_rate) + gi_off
            hum = sum(0.025 * math.sin(2 * math.pi * f * gi / sample_rate)
                      for f in [50, 100])
            out.append(hum + _gaussian(rng) * bg_noise)
        t_used += gap

    out = out[:n]
    _write_wav(filepath, out, sample_rate)
    print(f"[GENERATED] {filepath} -- speech+office mix | {len(out)/sample_rate:.1f}s")


if __name__ == "__main__":
    generate_test_samples()
