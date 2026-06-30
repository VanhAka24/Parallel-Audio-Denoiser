"""gui.py — Giao dien Parallel Audio Noise Reduction (Tkinter)"""

import sys, os, threading, time, tkinter as tk
from tkinter import ttk, filedialog, messagebox

import pygame
pygame.mixer.init(frequency=16000, size=-16, channels=1, buffer=512)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

BG     = "#12141C"
PANEL  = "#1C1F2E"
CARD   = "#252840"
LINE   = "#2E3250"
ACC    = "#4F8EF7"
ACC2   = "#34D399"
ACC3   = "#F472B6"
WARN   = "#FBBF24"
ERR    = "#F87171"
TXT    = "#E2E8F0"
TXT2   = "#94A3B8"
TXT3   = "#64748B"
WHITE  = "#FFFFFF"

FILTERS = {
    "Spectral Subtraction": "spectral_subtraction",
    "Wiener Filter":        "wiener",
}

FONT      = ("Segoe UI", 10)
FONT_SM   = ("Segoe UI", 9)
FONT_LG   = ("Segoe UI", 12, "bold")
FONT_TITLE= ("Segoe UI", 16, "bold")
FONT_MONO = ("Consolas",  9)

def _style(root):
    s = ttk.Style(root); s.theme_use("clam")
    s.configure(".", background=PANEL, foreground=TXT,
                 fieldbackground=CARD, troughcolor=CARD,
                 font=FONT, borderwidth=0)
    s.configure("TNotebook", background=BG, tabmargins=[0,0,0,0])
    s.configure("TNotebook.Tab", background=PANEL, foreground=TXT2,
                 padding=[20,9], font=("Segoe UI",10,"bold"))
    s.map("TNotebook.Tab",
          background=[("selected",ACC)], foreground=[("selected",WHITE)])
    s.configure("TCombobox", arrowcolor=ACC, relief="flat")
    s.map("TCombobox", fieldbackground=[("readonly",CARD)])
    s.configure("custom.Horizontal.TProgressbar", troughcolor=CARD, background=ACC, thickness=5)

def _btn(p, txt, cmd, bg=ACC, fg=WHITE, **kw):
    b = tk.Button(p, text=txt, command=cmd, bg=bg, fg=fg,
                  activebackground=bg, activeforeground=fg,
                  relief="flat", bd=0, font=("Segoe UI",10,"bold"),
                  cursor="hand2", **kw)
    b.bind("<Enter>", lambda e: b.config(bg=_mix(bg, 40)))
    b.bind("<Leave>", lambda e: b.config(bg=bg))
    return b

def _mix(h, amt=30):
    try:
        r=min(255,int(h[1:3],16)+amt)
        g=min(255,int(h[3:5],16)+amt)
        b=min(255,int(h[5:7],16)+amt)
        return f"#{r:02x}{g:02x}{b:02x}"
    except: return h

def _lbl(p, txt, font=FONT, fg=TXT, bg=None, **kw):
    return tk.Label(p, text=txt, font=font, fg=fg,
                    bg=bg or p["bg"] if hasattr(p,"__getitem__") else PANEL,
                    **kw)

def _sep(p):
    tk.Frame(p, height=1, bg=LINE).pack(fill="x", pady=6)

class WavePanel(tk.Frame):
    def __init__(self, p, title="", height=2.2, **kw):
        super().__init__(p, bg=PANEL, **kw)
        self.fig  = Figure(figsize=(1,height), dpi=100, facecolor=CARD)
        self.ax   = self.fig.add_subplot(111)
        self._ph  = None
        self._xmax = 0
        self._init_ax(title)
        self.cv   = FigureCanvasTkAgg(self.fig, master=self)
        self.cv.get_tk_widget().pack(fill="both", expand=True)
        self.cv.draw()

    def _init_ax(self, title=""):
        self.ax.set_facecolor(CARD)
        self.ax.tick_params(colors=TXT3, labelsize=8)
        self.ax.set_title(title, color=TXT2, fontsize=9, pad=5, fontfamily="Segoe UI")
        for sp in self.ax.spines.values(): sp.set_color(LINE)
        self.ax.set_ylim(-1.1, 1.1)
        self.ax.grid(True, alpha=0.1, color=LINE)

    def plot(self, sig, sr, color=ACC, title=""):
        self.ax.clear(); self._ph = None
        self._init_ax(title)
        if sig:
            n = len(sig)
            step = max(1, n // 4000)
            t    = [i / sr for i in range(0, n, step)]
            s    = sig[::step]
            self.ax.plot(t, s, color=color, lw=0.6, alpha=0.9)
            self._xmax = n / sr
        else:
            self._xmax = 0
        self.fig.tight_layout(pad=0.4)
        self.cv.draw()

    def move_playhead(self, t_sec):
        if self._ph is not None:
            try: self._ph.remove()
            except Exception: pass
            self._ph = None
        if t_sec >= 0 and self._xmax > 0:
            x = min(t_sec, self._xmax)
            self._ph = self.ax.axvline(x=x, color=WARN, lw=1.5, alpha=0.85, zorder=5)
        self.cv.draw_idle()

class _AudioPlayer:
    _state       = "stopped"
    _path        = ""
    _dur         = 0.0
    _paused_pos  = 0.0
    _t0          = 0.0
    _stop_evt    = threading.Event()
    _tick_thread = None

    # --- Public API ---
    @classmethod
    def play_or_resume(cls, path, dur, on_tick=None, on_end=None):
        if cls._state == "paused" and cls._path == path:
            cls._resume(on_tick, on_end)
        else:
            cls._play_new(path, dur, on_tick, on_end)

    @classmethod
    def pause(cls):
        if cls._state == "playing":
            pygame.mixer.music.pause()
            cls._paused_pos = time.perf_counter() - cls._t0
            cls._state = "paused"
            cls._stop_evt.set()

    @classmethod
    def restart(cls, on_tick=None, on_end=None):
        cls._stop_evt.set()
        try: pygame.mixer.music.stop()
        except Exception: pass
        cls._paused_pos = 0.0
        cls._state = "stopped"
        if cls._path:
            cls._play_new(cls._path, cls._dur, on_tick, on_end)

    @classmethod
    def stop(cls):
        cls._stop_evt.set()
        cls._state = "stopped"
        cls._paused_pos = 0.0
        try: pygame.mixer.music.stop()
        except Exception: pass

    @classmethod
    def get_elapsed(cls):
        if cls._state == "playing":
            return time.perf_counter() - cls._t0
        if cls._state == "paused":
            return cls._paused_pos
        return 0.0

    # --- Internal ---
    @classmethod
    def _play_new(cls, path, dur, on_tick, on_end):
        cls.stop()
        cls._path = path
        cls._dur  = dur
        try:
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
        except Exception as e:
            print("[pygame] load error:", e)
            return
        cls._t0    = time.perf_counter()
        cls._state = "playing"
        cls._stop_evt.clear()
        cls._start_tick(on_tick, on_end)

    @classmethod
    def _resume(cls, on_tick, on_end):
        pygame.mixer.music.unpause()
        cls._t0    = time.perf_counter() - cls._paused_pos
        cls._state = "playing"
        cls._stop_evt.clear()
        cls._start_tick(on_tick, on_end)

    @classmethod
    def _start_tick(cls, on_tick, on_end):
        def _run():
            while not cls._stop_evt.is_set():
                elapsed = time.perf_counter() - cls._t0
                if elapsed >= cls._dur:
                    if on_tick: on_tick(-1)
                    if on_end:  on_end()
                    cls._state = "stopped"
                    break
                if on_tick: on_tick(elapsed)
                time.sleep(0.05)
        cls._tick_thread = threading.Thread(target=_run, daemon=True)
        cls._tick_thread.start()

def _play_bar(parent, get_path_fn, get_dur_fn=None, wave=None):
    bar     = tk.Frame(parent, bg=PANEL)
    btn_pp  = _btn(bar, "▶  Phát",       lambda: None, padx=12, pady=5)
    btn_rst = _btn(bar, "↺  Lại từ đầu", lambda: None,
                   bg=CARD, fg=TXT2, padx=10, pady=5)
    time_lbl = _lbl(bar, "—", FONT_SM, TXT3, PANEL)

    btn_pp.pack(side="left")
    btn_rst.pack(side="left", padx=(6,0))
    time_lbl.pack(side="left", padx=12)

    _ctx = {"path": "", "dur": 0}

    def _fmt(s): m=int(s)//60; return f"{m}:{int(s)%60:02d}"

    def _upd_ui(elapsed):
        d = _ctx["dur"]
        if elapsed < 0:
            time_lbl.config(text="—", fg=TXT3)
            if wave: wave.move_playhead(-1)
            btn_pp.config(text="▶  Phát", bg=ACC)
        else:
            time_lbl.config(text=f"{_fmt(elapsed)} / {_fmt(d)}", fg=TXT2)
            if wave: wave.move_playhead(elapsed)

    def _on_tick(elapsed):
        try: bar.after(0, lambda e=elapsed: _upd_ui(e))
        except Exception: pass

    def _on_end():
        try:
            bar.after(0, lambda: [
                btn_pp.config(text="▶  Phát", bg=ACC),
                time_lbl.config(text="—", fg=TXT3),
                wave.move_playhead(-1) if wave else None,
            ])
        except Exception: pass

    def _on_play_pause():
        p   = get_path_fn()
        dur = get_dur_fn() if get_dur_fn else 30
        if not p or not os.path.isfile(p):
            time_lbl.config(text="Chưa có file", fg=ERR); return
        _ctx["path"] = p; _ctx["dur"] = dur

        if _AudioPlayer._state == "playing" and _AudioPlayer._path == p:
            _AudioPlayer.pause()
            btn_pp.config(text="▶  Tiếp tục", bg=ACC2)
        elif _AudioPlayer._state == "paused" and _AudioPlayer._path == p:
            _AudioPlayer.play_or_resume(p, dur, _on_tick, _on_end)
            btn_pp.config(text="⏸  Tạm dừng", bg=WARN)
        else:
            _AudioPlayer.play_or_resume(p, dur, _on_tick, _on_end)
            btn_pp.config(text="⏸  Tạm dừng", bg=WARN)

    def _on_restart():
        p   = get_path_fn()
        dur = get_dur_fn() if get_dur_fn else 30
        if not p or not os.path.isfile(p):
            time_lbl.config(text="Chưa có file", fg=ERR); return
        _ctx["path"] = p; _ctx["dur"] = dur
        _AudioPlayer.restart(_on_tick, _on_end)
        btn_pp.config(text="⏸  Tạm dừng", bg=WARN)

    btn_pp.config(command=_on_play_pause)
    btn_rst.config(command=_on_restart)
    return bar

class _Scrollable(tk.Frame):
    def __init__(self, parent, bg=PANEL, **kw):
        super().__init__(parent, bg=bg, **kw)
        self._cv  = tk.Canvas(self, bg=bg, highlightthickness=0)
        self._sb  = ttk.Scrollbar(self, orient="vertical", command=self._cv.yview)
        self.inner = tk.Frame(self._cv, bg=bg)
        self._win  = self._cv.create_window((0,0), window=self.inner, anchor="nw")
        self._cv.configure(yscrollcommand=self._sb.set)
        self._sb.pack(side="right", fill="y")
        self._cv.pack(side="left", fill="both", expand=True)
        self.inner.bind("<Configure>", self._on_configure)
        self._cv.bind("<Configure>",   self._on_canvas)
        self._cv.bind_all("<MouseWheel>", self._on_wheel)

    def _on_configure(self, e):
        self._cv.configure(scrollregion=self._cv.bbox("all"))

    def _on_canvas(self, e):
        self._cv.itemconfig(self._win, width=e.width)

    def _on_wheel(self, e):
        self._cv.yview_scroll(int(-1*(e.delta/120)), "units")

class TabXuLy(tk.Frame):
    def __init__(self, nb, app):
        super().__init__(nb, bg=PANEL)
        self.app = app
        self._scroll = _Scrollable(self)
        self._scroll.pack(fill="both", expand=True)
        self._inner  = self._scroll.inner
        self._build()

    def _build(self):
        p = self._inner

        g1 = tk.LabelFrame(p, text="  Tệp âm thanh đầu vào  ",
                            bg=PANEL, fg=TXT2, font=FONT_SM,
                            bd=1, relief="solid", highlightbackground=LINE)
        g1.pack(fill="x", padx=18, pady=(14,6))
        fr = tk.Frame(g1, bg=PANEL); fr.pack(fill="x", padx=10, pady=8)
        self.path_var = tk.StringVar()
        tk.Entry(fr, textvariable=self.path_var, bg=CARD, fg=TXT,
                 insertbackground=TXT, relief="flat", font=FONT, bd=6
                 ).pack(side="left", fill="x", expand=True)
        _btn(fr,"  Chọn file  ",self._browse,pady=6).pack(side="left",padx=(8,4))
        _btn(fr,"  Tạo mẫu  ",self._gen,bg=CARD,fg=TXT2,pady=6).pack(side="left")

        self.wav_in = WavePanel(p, "Dạng sóng đầu vào", height=1.7)
        self.wav_in.pack(fill="x", padx=18, pady=(2,0))
        self._in_path  = ""
        self._in_dur   = 0
        pb_in = _play_bar(p,
                          lambda: self._in_path,
                          lambda: self._in_dur,
                          wave=self.wav_in)
        pb_in.pack(anchor="w", padx=20, pady=(2,6))

        mid = tk.Frame(p, bg=PANEL); mid.pack(fill="x", padx=18, pady=6)

        # --- Settings ---
        g2 = tk.LabelFrame(mid, text="  Cài đặt  ",
                            bg=PANEL, fg=TXT2, font=FONT_SM,
                            bd=1, relief="solid", highlightbackground=LINE)
        g2.pack(side="left", fill="both", expand=True, padx=(0,10))
        row = tk.Frame(g2, bg=PANEL); row.pack(fill="x", padx=12, pady=10)

        fc = tk.Frame(row, bg=PANEL); fc.pack(side="left", padx=(0,24))
        _lbl(fc,"Thuật toán",FONT_SM,TXT2,PANEL).pack(anchor="w")
        self.filter_var = tk.StringVar(value="Spectral Subtraction")
        ttk.Combobox(fc, textvariable=self.filter_var,
                     values=list(FILTERS.keys()),
                     state="readonly", width=20, font=FONT
                     ).pack(anchor="w", pady=(3,0))

        mc = tk.Frame(row, bg=PANEL); mc.pack(side="left", padx=(0,24))
        _lbl(mc,"Chế độ",FONT_SM,TXT2,PANEL).pack(anchor="w")
        self.mode_var = tk.StringVar(value="parallel")
        for val, txt in [("sequential","Tuần tự"),("parallel","Song song")]:
            tk.Radiobutton(mc, text=txt, variable=self.mode_var, value=val,
                           bg=PANEL, fg=TXT, selectcolor=ACC,
                           activebackground=PANEL, font=FONT).pack(anchor="w")

        wc = tk.Frame(row, bg=PANEL); wc.pack(side="left")
        _lbl(wc,"Số luồng",FONT_SM,TXT2,PANEL).pack(anchor="w")
        self.w_var = tk.IntVar(value=min(8,os.cpu_count() or 4))
        self.w_lbl = _lbl(wc, str(self.w_var.get()),("Segoe UI",13,"bold"),ACC,PANEL)
        self.w_lbl.pack(anchor="w")
        ttk.Scale(wc, from_=1, to=os.cpu_count() or 16,
                  variable=self.w_var, orient="horizontal", length=120,
                  command=lambda v: self.w_lbl.config(text=str(int(float(v)))
                  )).pack(anchor="w")

        # --- Nut xu ly ---
        ac = tk.Frame(mid, bg=PANEL); ac.pack(side="left", fill="y")
        self.pbtn = _btn(ac,"▶  XỬ LÝ",self._process,padx=20,pady=16)
        self.pbtn.pack(fill="x")
        self.prog = ttk.Progressbar(ac, style="custom.Horizontal.TProgressbar",
                                     mode="indeterminate", length=120)
        self.prog.pack(fill="x", pady=(8,0))
        self.stat_lbl = _lbl(ac,"",FONT_SM,TXT2,PANEL)
        self.stat_lbl.pack(pady=(4,0))

        sf = tk.Frame(p, bg=CARD); sf.pack(fill="x", padx=18, pady=(0,6))
        self._stats = {}
        for key, lbl in [("time","⏱ Thời gian"),("rms","📊 RMS in → out"),("out","💾 Tệp xuất")]:
            col = tk.Frame(sf, bg=CARD); col.pack(side="left", padx=18, pady=8)
            _lbl(col, lbl, FONT_SM, TXT3, CARD).pack(anchor="w")
            v = _lbl(col, "—", ("Segoe UI",10,"bold"), ACC2, CARD)
            v.pack(anchor="w")
            self._stats[key] = v

        self.wav_out = WavePanel(p, "Dạng sóng đầu ra (sau xử lý)", height=1.7)
        self.wav_out.pack(fill="x", padx=18, pady=(2,0))
        self._out_path = ""
        self._out_dur  = 0
        pb_out = _play_bar(p,
                           lambda: self._out_path,
                           lambda: self._out_dur,
                           wave=self.wav_out)
        pb_out.pack(anchor="w", padx=20, pady=(2,18))

    def _browse(self):
        p = filedialog.askopenfilename(filetypes=[("WAV","*.wav"),("Tất cả","*.*")])
        if p:
            self.path_var.set(p)
            self._load_wave(p)

    def _load_wave(self, p):
        try:
            from src.audio_io import read_wav, mono_mix
            s,sr,ch = read_wav(p); sig = mono_mix(s,ch)
            self._in_path = p
            self._in_dur  = len(sig) / sr
            self.wav_in.plot(sig, sr, ACC, "Dạng sóng đầu vào")
        except Exception as e:
            messagebox.showerror("Lỗi", str(e))

    def _gen(self):
        def run():
            try:
                from samples.generate_sample import generate_test_samples
                generate_test_samples("input")
                self.after(0,lambda:self.stat_lbl.config(
                    text="✓ Đã tạo file mẫu trong input/",fg=ACC2))
            except Exception as e:
                self.after(0,lambda:messagebox.showerror("Lỗi",str(e)))
        self.stat_lbl.config(text="Đang tạo...", fg=WARN)
        threading.Thread(target=run, daemon=True).start()

    def _process(self):
        p = self.path_var.get().strip()
        if not p or not os.path.isfile(p):
            messagebox.showwarning("Chưa chọn file","Vui lòng chọn file WAV đầu vào.")
            return
        self.pbtn.config(state="disabled")
        self.prog.start(12)
        self.stat_lbl.config(text="Đang xử lý...", fg=WARN)
        threading.Thread(target=self._do, args=(p,), daemon=True).start()

    def _do(self, p):
        try:
            fk   = FILTERS[self.filter_var.get()]
            mode = self.mode_var.get()
            nw   = int(self.w_var.get())
            name = os.path.splitext(os.path.basename(p))[0]
            out  = os.path.join("output", f"{name}_{fk}.wav")
            os.makedirs("output", exist_ok=True)
            from src.noise_reduction import process_file
            res = process_file(p, out, fk, mode, num_workers=nw, verbose=False)
            from src.audio_io import read_wav, mono_mix
            s,sr,ch = read_wav(out); sig = mono_mix(s,ch)
            out_dur = len(sig) / sr
            def upd():
                _AudioPlayer.stop()
                self._out_path = out
                self._out_dur  = out_dur
                self.wav_out.plot(sig, sr, ACC2, "Dạng sóng đầu ra (sau xử lý)")
                self._stats["time"].config(text=f"{res['time_sec']:.4f}s")
                self._stats["rms"].config(
                    text=f"{res['input_rms']:.4f} → {res['output_rms']:.4f}")
                self._stats["out"].config(text=os.path.basename(out))
                self.prog.stop(); self.pbtn.config(state="normal")
                self.stat_lbl.config(text="✓ Hoàn thành! Nhấn ▶ Phát để nghe.", fg=ACC2)
            self.after(0, upd)
        except Exception as e:
            def err():
                self.prog.stop(); self.pbtn.config(state="normal")
                self.stat_lbl.config(text="✗ Lỗi", fg=ERR)
                messagebox.showerror("Lỗi xử lý", str(e))
            self.after(0, err)

class TabBenchmark(tk.Frame):
    def __init__(self, nb, app):
        super().__init__(nb, bg=PANEL)
        self.app = app
        self._build()

    def _build(self):
        top = tk.Frame(self, bg=PANEL); top.pack(fill="x", padx=18, pady=14)

        fc = tk.Frame(top, bg=PANEL); fc.pack(side="left")
        _lbl(fc,"Tệp WAV để đo",FONT_SM,TXT2,PANEL).pack(anchor="w")
        fr2 = tk.Frame(fc, bg=PANEL); fr2.pack(anchor="w", pady=(3,0))
        self.f_var = tk.StringVar()
        tk.Entry(fr2, textvariable=self.f_var, bg=CARD, fg=TXT,
                 insertbackground=TXT, relief="flat", font=FONT,
                 bd=6, width=36).pack(side="left")
        _btn(fr2,"Chọn",self._browse,pady=5,padx=10).pack(side="left",padx=(6,0))

        wc = tk.Frame(top, bg=PANEL); wc.pack(side="right")
        _lbl(wc,"Số luồng tối đa",FONT_SM,TXT2,PANEL).pack(anchor="w")
        self.mw = tk.IntVar(value=8)
        self.mw_lbl = _lbl(wc,str(self.mw.get()),("Segoe UI",13,"bold"),ACC,PANEL)
        self.mw_lbl.pack(anchor="w")
        ttk.Scale(wc, from_=2, to=os.cpu_count() or 16,
                  variable=self.mw, orient="horizontal", length=130,
                  command=lambda v: self.mw_lbl.config(
                      text=str(int(float(v))))).pack(anchor="w")

        br = tk.Frame(self, bg=PANEL); br.pack(fill="x", padx=18, pady=(0,10))
        self.rbtn = _btn(br,"  ▶  ĐO HIỆU NĂNG  ",self._run,padx=20,pady=12)
        self.rbtn.pack(side="left")
        self.prog = ttk.Progressbar(br,style="custom.Horizontal.TProgressbar",
                                     mode="indeterminate",length=180)
        self.prog.pack(side="left",padx=12)
        self.slbl = _lbl(br,"",FONT_SM,TXT2,PANEL)
        self.slbl.pack(side="left")

        res = tk.Frame(self, bg=CARD); res.pack(fill="both", expand=True, padx=18, pady=(0,18))
        _lbl(res,"Kết quả",FONT_SM,TXT3,CARD).pack(anchor="w", padx=10, pady=(8,2))
        self.txt = tk.Text(res, bg=CARD, fg=TXT, font=FONT_MONO,
                           relief="flat", bd=0, insertbackground=TXT)
        sb = ttk.Scrollbar(res, command=self.txt.yview)
        self.txt.config(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.txt.pack(fill="both", expand=True, padx=8, pady=(0,8))

    def _browse(self):
        p = filedialog.askopenfilename(filetypes=[("WAV","*.wav")])
        if p: self.f_var.set(p)

    def _run(self):
        p = self.f_var.get().strip()
        if not p:
            cands = ([os.path.join("input",f) for f in os.listdir("input")
                      if f.endswith(".wav")] if os.path.isdir("input") else [])
            if not cands:
                messagebox.showwarning("Không có file","Tạo file mẫu trước."); return
            p = sorted(cands)[0]; self.f_var.set(p)
        self.rbtn.config(state="disabled")
        self.prog.start(12)
        self.slbl.config(text="Đang đo...", fg=WARN)
        self.txt.delete("1.0","end")
        threading.Thread(target=self._do, args=(p,), daemon=True).start()

    def _do(self, p):
        import io, contextlib
        try:
            from src.audio_io import read_wav, mono_mix
            from src.benchmark import benchmark_all_filters, save_results_csv, save_results_json
            s,sr,ch = read_wav(p); sig = mono_mix(s,ch)
            mw = int(self.mw.get())
            counts = [w for w in [1,2,4,8,16] if w<=mw]
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                results = benchmark_all_filters(sig, sr, worker_counts=counts)
            os.makedirs("report/results", exist_ok=True)
            save_results_csv(results,"report/results/benchmark.csv")
            save_results_json(results,"report/results/benchmark.json")
            out = buf.getvalue()
            def upd():
                self.txt.insert("end", out); self.txt.see("end")
                self.prog.stop(); self.rbtn.config(state="normal")
                self.slbl.config(text="✓ Xong! Đã lưu CSV.", fg=ACC2)
                self.app.tab_charts.refresh()
            self.after(0, upd)
        except Exception as e:
            def err():
                self.prog.stop(); self.rbtn.config(state="normal")
                self.slbl.config(text="✗ Lỗi", fg=ERR)
                messagebox.showerror("Lỗi",str(e))
            self.after(0, err)

class TabCharts(tk.Frame):
    def __init__(self, nb, app):
        super().__init__(nb, bg=PANEL)
        self.app = app
        self._build()

    def _build(self):
        top = tk.Frame(self, bg=PANEL); top.pack(fill="x", padx=18, pady=10)
        _btn(top,"↻ Làm mới biểu đồ",self.refresh,bg=CARD,fg=TXT2,pady=8).pack(side="left")
        _btn(top,"📁 Mở thư mục kết quả",self._open,bg=CARD,fg=TXT2,pady=8).pack(side="left",padx=8)
        self.slbl = _lbl(top,"",FONT_SM,TXT2,PANEL)
        self.slbl.pack(side="left",padx=10)

        self._scroll = _Scrollable(self)
        self._scroll.pack(fill="both", expand=True, padx=18, pady=(0, 18))
        inner = self._scroll.inner

        self._cells = {}
        labels = [("speedup_chart.png", "Biểu đồ Tốc độ (Speedup vs Số luồng)"),
                  ("execution_time_bar.png", "Biểu đồ Thời gian thực thi"),
                  ("waveform_comparison.png", "Dạng sóng trước/sau khi khử nhiễu"),
                  ("spectrogram_wiener.png", "Phổ tần số (Spectrogram - Wiener Filter)")]

        for fname, title in labels:
            cell = tk.LabelFrame(inner, text=f"  {title}  ",
                                  bg=CARD, fg=TXT2, font=FONT_LG,
                                  bd=1, relief="solid", highlightbackground=LINE)
            cell.pack(fill="x", expand=True, padx=6, pady=10)

            canvas_frame = tk.Frame(cell, bg=CARD, height=400)
            canvas_frame.pack(fill="both", expand=True, padx=10, pady=10)
            canvas_frame.pack_propagate(False) # Giu chieu cao co dinh de render anh khong bi be

            self._cells[fname] = canvas_frame

        self.refresh()

    def _show(self, frame, img_path):
        for w in frame.winfo_children(): w.destroy()
        if not os.path.isfile(img_path):
            _lbl(frame,"Chưa có dữ liệu.\nHãy chạy Đo hiệu năng trước.",
                 FONT_SM, TXT3, CARD).pack(expand=True)
            return
        try:
            fig = Figure(facecolor=CARD)
            ax  = fig.add_subplot(111); ax.axis("off"); ax.set_facecolor(CARD)
            img = plt.imread(img_path)
            ax.imshow(img); fig.tight_layout(pad=0)
            cv = FigureCanvasTkAgg(fig, master=frame)
            cv.draw(); cv.get_tk_widget().pack(fill="both",expand=True)
        except Exception as e:
            _lbl(frame,str(e),FONT_SM,ERR,CARD).pack(expand=True)

    def refresh(self):
        csv = "report/results/benchmark.csv"
        if os.path.isfile(csv):
            try:
                from report.generate_report import generate_full_report
                wavs = ([os.path.join("input",f) for f in os.listdir("input")
                          if f.endswith(".wav")] if os.path.isdir("input") else [])
                generate_full_report(csv, sorted(wavs)[0] if wavs else None,
                                     "report/results")
            except Exception: pass
        for fname, cell in self._cells.items():
            self._show(cell, os.path.join("report/results", fname))
        self.slbl.config(text="✓ Đã cập nhật", fg=ACC2)

    def _open(self):
        p = os.path.abspath("report/results")
        os.makedirs(p, exist_ok=True)
        os.startfile(p)

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Khử Nhiễu Âm Thanh Song Song")
        self.geometry("980x800"); self.minsize(820,660)
        self.configure(bg=BG)
        _style(self)
        self._header()
        self._tabs()

    def _header(self):
        h = tk.Frame(self, bg=BG, pady=16); h.pack(fill="x")
        _lbl(h,"🎵",("Segoe UI",22),ACC,BG).pack(side="left",padx=(20,8))
        tc = tk.Frame(h,bg=BG); tc.pack(side="left")
        _lbl(tc,"Khử Nhiễu Âm Thanh Song Song",FONT_TITLE,WHITE,BG).pack(anchor="w")
        _lbl(tc,"Parallel Audio Noise Reduction  —  Python",FONT_SM,TXT3,BG).pack(anchor="w")
        bg_b = tk.Frame(h, bg=ACC, padx=10, pady=4)
        bg_b.pack(side="right", padx=20)
        _lbl(bg_b,f"{os.cpu_count()} CPU Cores",FONT_SM,WHITE,ACC).pack()
        tk.Frame(self, height=1, bg=LINE).pack(fill="x")

    def _tabs(self):
        nb = ttk.Notebook(self); nb.pack(fill="both",expand=True,padx=0,pady=0)
        self.tab_xu_ly    = TabXuLy(nb, self)
        self.tab_benchmark= TabBenchmark(nb, self)
        self.tab_charts   = TabCharts(nb, self)
        nb.add(self.tab_xu_ly,     text="  Xử Lý  ")
        nb.add(self.tab_benchmark, text="  Đo Hiệu Năng  ")
        nb.add(self.tab_charts,    text="  Biểu Đồ  ")

if __name__ == "__main__":
    app = App(); app.mainloop()
