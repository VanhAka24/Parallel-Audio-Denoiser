import struct
import os

# --- Hằng số WAV header ---
RIFF_ID    = b"RIFF"
WAVE_ID    = b"WAVE"
FMT_ID     = b"fmt "
DATA_ID    = b"data"
PCM_FORMAT = 1   # Linear PCM
BIT_DEPTH  = 16  # 16-bit per sample


# --- Đọc file WAV (PCM 16-bit / 8-bit, mono / stereo) ---

def read_wav(filepath: str) -> tuple:
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Không tìm thấy file: {filepath}")

    with open(filepath, "rb") as f:
        raw = f.read()

    pos = 0
    riff_id    = raw[pos:pos+4]; pos += 4
    _          = struct.unpack_from("<I", raw, pos)[0]; pos += 4
    wave_id    = raw[pos:pos+4]; pos += 4

    if riff_id != RIFF_ID:
        raise ValueError("Không phải file RIFF.")
    if wave_id != WAVE_ID:
        raise ValueError("Không phải file WAVE.")

    fmt_found    = False
    data_chunk   = None
    sample_rate  = None
    num_channels = None
    bit_depth    = None
    audio_format = None

    while pos < len(raw) - 8:
        chunk_id  = raw[pos:pos+4]; pos += 4
        chunk_len = struct.unpack_from("<I", raw, pos)[0]; pos += 4

        if chunk_id == FMT_ID:
            audio_format = struct.unpack_from("<H", raw, pos)[0]
            num_channels = struct.unpack_from("<H", raw, pos+2)[0]
            sample_rate  = struct.unpack_from("<I", raw, pos+4)[0]
            bit_depth    = struct.unpack_from("<H", raw, pos+14)[0]
            fmt_found    = True
            pos += chunk_len
        elif chunk_id == DATA_ID:
            data_chunk = raw[pos:pos+chunk_len]
            pos += chunk_len
        else:
            pos += chunk_len

    if not fmt_found:
        raise ValueError("Thiếu chunk 'fmt '.")
    if data_chunk is None:
        raise ValueError("Thiếu chunk 'data'.")
    if audio_format != PCM_FORMAT:
        raise ValueError(f"Chỉ hỗ trợ PCM (format=1), file này có format={audio_format}.")

    return _decode_samples(data_chunk, bit_depth, num_channels), sample_rate, num_channels


def _decode_samples(data: bytes, bit_depth: int, num_channels: int) -> list:
    if bit_depth == 16:
        fmt, step, norm = "<h", 2, 32768.0
    elif bit_depth == 8:
        fmt, step, norm = "<B", 1, 128.0
    else:
        raise ValueError(f"Bit depth {bit_depth} không được hỗ trợ.")

    total  = len(data) // step
    values = [struct.unpack_from(fmt, data, i * step)[0] for i in range(total)]
    if bit_depth == 8:
        values = [v - 128 for v in values]
    normalized = [v / norm for v in values]

    if num_channels == 1:
        return normalized
    elif num_channels == 2:
        return list(zip(normalized[0::2], normalized[1::2]))
    else:
        raise ValueError(f"Số kênh {num_channels} không được hỗ trợ.")


# --- Ghi file WAV PCM 16-bit ---

def write_wav(filepath: str, samples: list, sample_rate: int, num_channels: int = 1) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    flat = []
    if num_channels == 2:
        for L, R in samples:
            flat.append(L); flat.append(R)
    else:
        flat = samples

    int_samples = [int(max(-1.0, min(1.0, s)) * 32767) for s in flat]
    data_bytes  = struct.pack(f"<{len(int_samples)}h", *int_samples)
    data_size   = len(data_bytes)

    byte_rate   = sample_rate * num_channels * (BIT_DEPTH // 8)
    block_align = num_channels * (BIT_DEPTH // 8)
    fmt_chunk   = struct.pack("<HHIIHH", PCM_FORMAT, num_channels, sample_rate,
                              byte_rate, block_align, BIT_DEPTH)
    riff_size   = 4 + (8 + 16) + (8 + data_size)

    with open(filepath, "wb") as f:
        f.write(RIFF_ID)
        f.write(struct.pack("<I", riff_size))
        f.write(WAVE_ID)
        f.write(FMT_ID)
        f.write(struct.pack("<I", 16))
        f.write(fmt_chunk)
        f.write(DATA_ID)
        f.write(struct.pack("<I", data_size))
        f.write(data_bytes)


# --- Tiện ích ---


def mono_mix(samples: list, num_channels: int) -> list:
    # Chuyển stereo sang mono bằng trung bình 2 kênh
    if num_channels == 1:
        return samples
    return [(L + R) / 2.0 for L, R in samples]
