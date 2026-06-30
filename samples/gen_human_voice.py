"""gen_human_voice.py -- Tao file WAV giong nguoi that bang Windows SAPI (TTS)"""
import subprocess, os, struct, math, random, sys

os.makedirs("input", exist_ok=True)

PHRASES = [
    ("hello_en",
     "Hello, this is a test of the noise reduction system. "
     "The quick brown fox jumps over the lazy dog."),
    ("meeting_en",
     "Good morning everyone. Today we will discuss the quarterly report "
     "and our plans for next year. Please take a seat and we will begin shortly."),
    ("news_en",
     "Breaking news: Scientists have discovered a new method to improve "
     "audio quality in noisy environments using parallel computing techniques."),
]

def gen_one(filepath, text, noise_level=0.0):
    tmp = filepath.replace(".wav", "_tmp.wav")
    ps  = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "$s.Rate = -1; $s.Volume = 80; "
        f"$s.SetOutputToWaveFile('{tmp}'); "
        f"$s.Speak('{text}'); $s.Dispose()"
    )
    r = subprocess.run(["powershell", "-Command", ps],
                       capture_output=True, text=True)
    if not os.path.isfile(tmp):
        print(f"  [FAIL] TTS: {r.stderr[:120]}")
        return False

    import wave as wv
    with wv.open(tmp, "rb") as wf:
        sr0 = wf.getframerate()
        ch  = wf.getnchannels()
        raw = wf.readframes(wf.getnframes())

    n = len(raw) // (2 * ch)
    s = list(struct.unpack(f"<{n*ch}h", raw))
    if ch > 1:
        s = [sum(s[i*ch:(i+1)*ch])//ch for i in range(n)]
    s = [x/32767.0 for x in s]

    if sr0 != 16000:
        ratio = sr0 / 16000
        n2    = int(len(s)/ratio)
        out   = []
        for i in range(n2):
            p  = i*ratio; lo = int(p); hi = min(lo+1, len(s)-1)
            out.append(s[lo]*(1-(p-lo)) + s[hi]*(p-lo))
        s = out

    if noise_level > 0:
        rng = random.Random(42)
        ns  = []
        # Prepend 0.5s NOISE THUAN de noise estimator co doan tham chieu chinh xac
        for _ in range(int(0.5 * 16000)):
            u1 = max(rng.random(), 1e-12)
            g  = math.sqrt(-2*math.log(u1)) * math.cos(2*math.pi*rng.random())
            ns.append(max(-1.0, min(1.0, g * noise_level)))
        # Phan con lai: speech + noise
        for x in s:
            u1 = max(rng.random(), 1e-12)
            g  = math.sqrt(-2*math.log(u1)) * math.cos(2*math.pi*rng.random())
            ns.append(max(-1.0, min(1.0, x + g*noise_level)))
        s = ns

    SR = 16000
    ints = [max(-32767, min(32767, int(x*32767))) for x in s]
    data = struct.pack(f"<{len(ints)}h", *ints)
    fmt  = struct.pack("<HHIIHH", 1, 1, SR, SR*2, 2, 16)
    rsz  = 4 + 24 + 8 + len(data)
    with open(filepath, "wb") as f:
        f.write(b"RIFF"); f.write(struct.pack("<I", rsz))
        f.write(b"WAVE")
        f.write(b"fmt "); f.write(struct.pack("<I", 16)); f.write(fmt)
        f.write(b"data"); f.write(struct.pack("<I", len(data))); f.write(data)
    os.remove(tmp)
    print(f"  [OK] {filepath} | {len(s)/SR:.1f}s | noise={noise_level}")
    return True


if __name__ == "__main__":
    print("\n[*] Tao file giong noi nguoi that (Windows TTS)...")
    for name, text in PHRASES:
        gen_one(f"input/{name}_clean.wav",  text, noise_level=0.0)
        gen_one(f"input/{name}_noisy.wav",  text, noise_level=0.10)
    print("[DONE]")
