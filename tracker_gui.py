"""
DHM Particle Tracker - GUI v3
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading, queue, sys, os, cv2, numpy as np
from PIL import Image, ImageTk

DEFAULTS = {
    'Brightfield': dict(blob_color=0,   min_area=400, max_area=5000, min_circ=0.5,
        filter_type='bilateral', clahe_clip=3.0, use_dog=False,
        dog_sigma1=2.0, dog_sigma2=8.0, tophat_ksize=21,
        cam_pixel=3.75, magnification=40, max_dist=25, max_skips=5, min_track=5,
        P_init=100, Q_val=1, R_val=50,
        filter_by_color=True, filter_by_circularity=True,
        filter_by_inertia=False, filter_by_convexity=False),
    'Amplitude':   dict(blob_color=255, min_area=100, max_area=3000, min_circ=0.5,
        filter_type='bilateral', clahe_clip=3.0, use_dog=False,
        dog_sigma1=2.0, dog_sigma2=8.0, tophat_ksize=21,
        cam_pixel=3.75, magnification=20, max_dist=50, max_skips=53, min_track=3,
        P_init=100, Q_val=1, R_val=50,
        filter_by_color=True, filter_by_circularity=True,
        filter_by_inertia=False, filter_by_convexity=False),
    'Hologram':    dict(blob_color=255, min_area=150, max_area=2500, min_circ=0.3,
        filter_type='bilateral', clahe_clip=3.0, use_dog=False,
        dog_sigma1=2.0, dog_sigma2=8.0, tophat_ksize=21,
        cam_pixel=3.75, magnification=40, max_dist=40, max_skips=5,  min_track=5,
        P_init=100, Q_val=1, R_val=30,
        filter_by_color=True, filter_by_circularity=True,
        filter_by_inertia=False, filter_by_convexity=False),
    'Phase':       dict(blob_color=255, min_area=100, max_area=2500, min_circ=0.3,
        filter_type='bilateral', clahe_clip=3.0, use_dog=True,
        dog_sigma1=2.0, dog_sigma2=8.0, tophat_ksize=21,
        cam_pixel=3.75, magnification=40, max_dist=40, max_skips=8,  min_track=3,
        P_init=100, Q_val=1, R_val=30,
        filter_by_color=True, filter_by_circularity=True,
        filter_by_inertia=False, filter_by_convexity=False),
}

MODE_TO_IMAGE = {
    'Brightfield': 'standard',
    'Amplitude':   'amplitude',
    'Hologram':    'hologram',
    'Phase':       'phase',
}


# ══════════════════════════════════════════════════════════════════
#  PREVIEW
# ══════════════════════════════════════════════════════════════════

class VideoWindow:
    def __init__(self, parent, video_w, video_h, scale=0.6, on_stop=None):
        self.win = tk.Toplevel(parent)
        self.win.title('Tracking — Live Preview')
        self.win.resizable(True, True)
        self.win.protocol('WM_DELETE_WINDOW', self._on_close)
        self.closed  = False
        self.on_stop = on_stop

        self.disp_w = max(int(video_w * scale), 640)
        self.disp_h = max(int(video_h * scale), 360)

        info = tk.Frame(self.win, bg='#2C3E50', pady=4)
        info.pack(fill='x')
        self.info_var = tk.StringVar(value='Starting...')
        tk.Label(info, textvariable=self.info_var, font=('Arial', 9),
                 bg='#2C3E50', fg='white').pack(side='left', padx=10)

        self.canvas = tk.Canvas(self.win, width=self.disp_w, height=self.disp_h,
                                bg='black', highlightthickness=0)
        self.canvas.pack()

        self.progress = ttk.Progressbar(self.win, length=self.disp_w, mode='determinate')
        self.progress.pack(fill='x')

        btn_frame = tk.Frame(self.win, bg='#ECF0F1', pady=6)
        btn_frame.pack(fill='x')
        tk.Button(btn_frame, text='⏹  Stop',
                  font=('Arial', 9, 'bold'), bg='#C0392B', fg='white',
                  relief='flat', padx=12, pady=4, cursor='hand2',
                  command=self._on_stop_click).pack(side='left', padx=10)

        self._photo = None

    def show_frame(self, frame_bgr, fi, total, n_tracks):
        if self.closed: return
        rgb = cv2.cvtColor(cv2.resize(frame_bgr, (self.disp_w, self.disp_h)), cv2.COLOR_BGR2RGB)
        self._photo = ImageTk.PhotoImage(image=Image.fromarray(rgb))
        self.canvas.create_image(0, 0, anchor='nw', image=self._photo)
        pct = int(fi / max(total, 1) * 100)
        self.progress['value'] = pct
        self.info_var.set(f'Frame {fi}/{total}  |  Tracks: {n_tracks}  |  {pct}%')

    def close(self):
        if not self.closed:
            self.closed = True
            try: self.win.destroy()
            except: pass

    def _on_stop_click(self):
        if self.on_stop: self.on_stop()
        self.close()

    def _on_close(self):
        self._on_stop_click()


# ══════════════════════════════════════════════════════════════════
#  GUI
# ══════════════════════════════════════════════════════════════════

class TrackerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title('DHM Particle Tracker')
        self.root.resizable(False, False)
        self.root.protocol('WM_DELETE_WINDOW', self._on_quit)

        BG     = '#F4F6F8'
        HDR_BG = '#2C3E50'
        HDR_FG = '#FFFFFF'
        BTN_BG = '#2980B9'
        BTN_FG = '#FFFFFF'

        self.root.configure(bg=BG)
        style = ttk.Style(); style.theme_use('clam')
        for s, cfg in [
            ('TFrame',      {'background': BG}),
            ('TLabel',      {'background': BG, 'font': ('Arial', 9)}),
            ('TEntry',      {'font': ('Arial', 9)}),
            ('TCheckbutton',{'background': BG, 'font': ('Arial', 9)}),
            ('TRadiobutton',{'background': BG, 'font': ('Arial', 9)}),
        ]:
            style.configure(s, **cfg)

        # Header
        hdr = tk.Frame(root, bg=HDR_BG, pady=10)
        hdr.grid(row=0, column=0, columnspan=2, sticky='ew')
        tk.Label(hdr, text='DHM Particle Tracker', font=('Arial', 14, 'bold'),
                 bg=HDR_BG, fg=HDR_FG).pack()
        tk.Label(hdr, text='Digital Holographic Microscopy Tracking Tool',
                 font=('Arial', 9), bg=HDR_BG, fg='#BDC3C7').pack()

        # Scrollable canvas
        cv_main = tk.Canvas(root, bg=BG, highlightthickness=0, width=540, height=600)
        sb = ttk.Scrollbar(root, orient='vertical', command=cv_main.yview)
        cv_main.configure(yscrollcommand=sb.set)
        cv_main.grid(row=1, column=0, sticky='nsew')
        sb.grid(row=1, column=1, sticky='ns')
        root.grid_rowconfigure(1, weight=1); root.grid_columnconfigure(0, weight=1)

        self.main = ttk.Frame(cv_main, padding=10)
        wid = cv_main.create_window((0, 0), window=self.main, anchor='nw')

        def _cfg(e):
            cv_main.configure(scrollregion=cv_main.bbox('all'))
            cv_main.itemconfig(wid, width=cv_main.winfo_width())
        self.main.bind('<Configure>', _cfg)
        cv_main.bind('<Configure>', lambda e: cv_main.itemconfig(wid, width=e.width))
        cv_main.bind_all('<MouseWheel>', lambda e: cv_main.yview_scroll(-1*(e.delta//120), 'units'))

        row = 0

        # ── VIDEO TYPE ────────────────────────────────────────────────────────
        row = self._sec(row, '  VIDEO TYPE')
        self.mode_var = tk.StringVar(value='Brightfield')
        for m in ['Brightfield', 'Amplitude', 'Hologram', 'Phase']:
            ttk.Radiobutton(self.main, text=m, variable=self.mode_var,
                            value=m, command=self._on_mode).grid(
                row=row, column=0, columnspan=2, sticky='w', padx=20, pady=1)
            row += 1

        # ── VIDEO FILE ────────────────────────────────────────────────────────
        row = self._sec(row, '  VIDEO FILE')
        self.video_path = self._file_row(row, 'Video path:')
        row += 1

        # ── OPTICAL SYSTEM ────────────────────────────────────────────────────
        row = self._sec(row, '  OPTICAL SYSTEM')
        self.cam_pixel = self._erow(row, 'Camera pixel size (µm):'); row += 1
        self.magnif    = self._erow(row, 'Magnification (x):');      row += 1

        # FPS: auto-detected, with manual override option
        fps_lbl = ttk.Frame(self.main)
        fps_lbl.grid(row=row, column=0, sticky='w', padx=20, pady=1)
        ttk.Label(fps_lbl, text='FPS:').pack(side='left')
        self.fps_auto_label = ttk.Label(fps_lbl, text='(auto)', foreground='gray')
        self.fps_auto_label.pack(side='left', padx=4)

        fps_ctrl = ttk.Frame(self.main)
        fps_ctrl.grid(row=row, column=1, sticky='w')
        self.fps_override_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(fps_ctrl, text='Override:', variable=self.fps_override_var,
                        command=self._toggle_fps_override).pack(side='left')
        self.fps_entry = ttk.Entry(fps_ctrl, width=7, state='disabled')
        self.fps_entry.pack(side='left', padx=4)
        self.fps_entry.insert(0, '15')
        row += 1

        # ── DETECTION ─────────────────────────────────────────────────────────
        row = self._sec(row, '  DETECTION')

        # Blob color
        self.filter_color_var = tk.BooleanVar(value=True)
        fc_row = ttk.Frame(self.main)
        fc_row.grid(row=row, column=0, columnspan=2, sticky='w', padx=20, pady=1)
        ttk.Checkbutton(fc_row, text='Filter by blob color:',
                        variable=self.filter_color_var,
                        command=self._toggle_color_filter).pack(side='left')
        self.blob_color_var = tk.IntVar(value=255)
        self.bc_frame = ttk.Frame(fc_row)
        self.bc_frame.pack(side='left', padx=8)
        ttk.Radiobutton(self.bc_frame, text='Dark (0)',     variable=self.blob_color_var, value=0  ).pack(side='left', padx=3)
        ttk.Radiobutton(self.bc_frame, text='Bright (255)', variable=self.blob_color_var, value=255).pack(side='left')
        row += 1

        self.min_area = self._erow(row, 'Min area (px²):'); row += 1
        self.max_area = self._erow(row, 'Max area (px²):'); row += 1

        # Circularity
        self.filter_circ_var = tk.BooleanVar(value=True)
        fcirc_row = ttk.Frame(self.main)
        fcirc_row.grid(row=row, column=0, columnspan=2, sticky='w', padx=20, pady=1)
        ttk.Checkbutton(fcirc_row, text='Filter by circularity  min:',
                        variable=self.filter_circ_var,
                        command=self._toggle_circ_filter).pack(side='left')
        self.min_circ = ttk.Entry(fcirc_row, width=7)
        self.min_circ.pack(side='left', padx=4)
        row += 1

        # Inertia
        self.filter_inertia_var = tk.BooleanVar(value=False)
        fi_row = ttk.Frame(self.main)
        fi_row.grid(row=row, column=0, columnspan=2, sticky='w', padx=20, pady=1)
        ttk.Checkbutton(fi_row, text='Filter by inertia  min:',
                        variable=self.filter_inertia_var,
                        command=self._toggle_inertia_filter).pack(side='left')
        self.min_inertia = ttk.Entry(fi_row, width=7, state='disabled')
        self.min_inertia.insert(0, '0.1')
        self.min_inertia.pack(side='left', padx=4)
        row += 1

        # Convexity
        self.filter_convex_var = tk.BooleanVar(value=False)
        fco_row = ttk.Frame(self.main)
        fco_row.grid(row=row, column=0, columnspan=2, sticky='w', padx=20, pady=1)
        ttk.Checkbutton(fco_row, text='Filter by convexity  min:',
                        variable=self.filter_convex_var,
                        command=self._toggle_convex_filter).pack(side='left')
        self.min_convexity = ttk.Entry(fco_row, width=7, state='disabled')
        self.min_convexity.insert(0, '0.8')
        self.min_convexity.pack(side='left', padx=4)
        row += 1

        # Preprocessing filter
        ttk.Label(self.main, text='Preprocessing filter:').grid(row=row, column=0, sticky='w', padx=20)
        self.filter_var = tk.StringVar(value='bilateral')
        ff = ttk.Frame(self.main); ff.grid(row=row, column=1, sticky='w')
        ttk.Radiobutton(ff, text='Bilateral', variable=self.filter_var, value='bilateral').pack(side='left', padx=5)
        ttk.Radiobutton(ff, text='Gaussian',  variable=self.filter_var, value='gaussian' ).pack(side='left')
        row += 1

        self.clahe_clip = self._erow(row, 'CLAHE clip limit:'); row += 1

        # DoG + Top-hat
        self.use_dog_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.main, text='Use DoG + Top-hat (Phase enhancement)',
                        variable=self.use_dog_var,
                        command=self._toggle_dog).grid(
            row=row, column=0, columnspan=2, sticky='w', padx=20, pady=2)
        row += 1
        self.dog_frame = ttk.Frame(self.main)
        self.dog_frame.grid(row=row, column=0, columnspan=2, sticky='ew', padx=30)
        row += 1
        ttk.Label(self.dog_frame, text='DoG sigma 1:').grid(row=0, column=0, sticky='w', padx=5)
        self.dog_s1 = ttk.Entry(self.dog_frame, width=8); self.dog_s1.grid(row=0, column=1, padx=5)
        ttk.Label(self.dog_frame, text='DoG sigma 2:').grid(row=0, column=2, sticky='w', padx=5)
        self.dog_s2 = ttk.Entry(self.dog_frame, width=8); self.dog_s2.grid(row=0, column=3, padx=5)
        ttk.Label(self.dog_frame, text='Top-hat kernel:').grid(row=1, column=0, sticky='w', padx=5, pady=2)
        self.tophat = ttk.Entry(self.dog_frame, width=8); self.tophat.grid(row=1, column=1, padx=5)
        self.dog_s1.insert(0, '2.0'); self.dog_s2.insert(0, '8.0'); self.tophat.insert(0, '21')
        self._toggle_dog()

        # ── KALMAN ────────────────────────────────────────────────────────────
        row = self._sec(row, '  KALMAN FILTER')
        ttk.Label(self.main, text='P — Initial covariance', font=('Arial', 8),
                  foreground='gray').grid(row=row, column=0, columnspan=2, sticky='w', padx=20); row += 1
        self.P_init = self._erow(row, 'P init:'); row += 1
        ttk.Label(self.main, text='Q — Process noise (higher = more agile)', font=('Arial', 8),
                  foreground='gray').grid(row=row, column=0, columnspan=2, sticky='w', padx=20); row += 1
        self.Q_val  = self._erow(row, 'Q value:'); row += 1
        ttk.Label(self.main, text='R — Measurement noise (higher = trust Kalman more)', font=('Arial', 8),
                  foreground='gray').grid(row=row, column=0, columnspan=2, sticky='w', padx=20); row += 1
        self.R_val  = self._erow(row, 'R value:'); row += 1

        # ── TRACKING ─────────────────────────────────────────────────────────
        row = self._sec(row, '  TRACKING')
        self.max_dist  = self._erow(row, 'Max distance (px):');         row += 1
        self.max_skips = self._erow(row, 'Max skips (frames):');        row += 1
        self.min_track = self._erow(row, 'Min track length (frames):'); row += 1

        # ── OUTPUT ────────────────────────────────────────────────────────────
        row = self._sec(row, '  OUTPUT')
        self.show_plot_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.main, text='Show trajectory plot after tracking',
                        variable=self.show_plot_var).grid(
            row=row, column=0, columnspan=2, sticky='w', padx=20); row += 1

        ttk.Label(self.main, text='Plot style:').grid(row=row, column=0, sticky='w', padx=20)
        self.plot_style_var = tk.StringVar(value='markers')
        ps = ttk.Frame(self.main); ps.grid(row=row, column=1, sticky='w')
        ttk.Radiobutton(ps, text='Markers', variable=self.plot_style_var, value='markers').pack(side='left', padx=5)
        ttk.Radiobutton(ps, text='Dots',    variable=self.plot_style_var, value='dots'   ).pack(side='left')
        row += 1

        # CSV
        self.save_csv_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.main, text='Save CSV', variable=self.save_csv_var,
                        command=self._toggle_csv).grid(
            row=row, column=0, columnspan=2, sticky='w', padx=20); row += 1
        self.csv_frame = ttk.Frame(self.main)
        self.csv_frame.grid(row=row, column=0, columnspan=2, sticky='ew', padx=20); row += 1
        ttk.Label(self.csv_frame, text='CSV mode:').grid(row=0, column=0, sticky='w')
        self.csv_mode_var = tk.StringVar(value='single')
        ttk.Radiobutton(self.csv_frame, text='Single file', variable=self.csv_mode_var, value='single'   ).grid(row=0, column=1, padx=5)
        ttk.Radiobutton(self.csv_frame, text='Per track',   variable=self.csv_mode_var, value='per_track').grid(row=0, column=2)
        ttk.Label(self.csv_frame, text='CSV path:').grid(row=1, column=0, sticky='w', pady=2)
        self.csv_path_var = tk.StringVar()
        ttk.Entry(self.csv_frame, textvariable=self.csv_path_var, width=28).grid(row=1, column=1, columnspan=2)
        ttk.Button(self.csv_frame, text='📂', width=3,
                   command=lambda: self._bsave(self.csv_path_var, 'trajectories.csv')).grid(row=1, column=3)
        self._toggle_csv()

        # Output video
        self.save_video_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.main, text='Save output video',
                        variable=self.save_video_var,
                        command=self._toggle_video).grid(
            row=row, column=0, columnspan=2, sticky='w', padx=20); row += 1
        self.vid_frame = ttk.Frame(self.main)
        self.vid_frame.grid(row=row, column=0, columnspan=2, sticky='ew', padx=20); row += 1
        ttk.Label(self.vid_frame, text='Output video path:').grid(row=0, column=0, sticky='w')
        self.vid_path_var = tk.StringVar()
        ttk.Entry(self.vid_frame, textvariable=self.vid_path_var, width=28).grid(row=0, column=1)
        ttk.Button(self.vid_frame, text='📂', width=3,
                   command=lambda: self._bsave(self.vid_path_var, 'output.mp4')).grid(row=0, column=2)
        self._toggle_video()

        # ── BUTTONS ───────────────────────────────────────────────────────────
        row += 1
        btn_bar = ttk.Frame(self.main)
        btn_bar.grid(row=row, column=0, columnspan=2, pady=15)

        self.run_btn = tk.Button(btn_bar, text='▶  RUN TRACKING',
            font=('Arial', 11, 'bold'), bg=BTN_BG, fg=BTN_FG,
            activebackground='#1A6FA8', relief='flat', cursor='hand2',
            padx=20, pady=8, command=self._run)
        self.run_btn.pack(side='left', padx=6)

        tk.Button(btn_bar, text='✕  Exit',
            font=('Arial', 11, 'bold'), bg='#7F8C8D', fg='white',
            activebackground='#636E72', relief='flat', cursor='hand2',
            padx=16, pady=8, command=self._on_quit).pack(side='left', padx=6)

        # Status bar
        self.status_var = tk.StringVar(value='Ready')
        tk.Label(root, textvariable=self.status_var, font=('Arial', 8),
                 bg='#ECF0F1', fg='#555', anchor='w', padx=10).grid(
            row=2, column=0, columnspan=2, sticky='ew')

        self.frame_queue = queue.Queue(maxsize=4)
        self.video_win   = None
        self._stop_flag  = False

        self._on_mode()

    # ── Widget helpers ────────────────────────────────────────────────────────

    def _sec(self, row, title):
        f = tk.Frame(self.main, bg='#2C3E50', pady=3)
        f.grid(row=row, column=0, columnspan=2, sticky='ew', pady=(8, 2))
        tk.Label(f, text=title, font=('Arial', 9, 'bold'),
                 bg='#2C3E50', fg='white').pack(side='left', padx=8)
        return row + 1

    def _erow(self, row, label, width=10):
        ttk.Label(self.main, text=label).grid(row=row, column=0, sticky='w', padx=20, pady=1)
        e = ttk.Entry(self.main, width=width)
        e.grid(row=row, column=1, sticky='w', pady=1)
        return e

    def _file_row(self, row, label):
        ttk.Label(self.main, text=label).grid(row=row, column=0, sticky='w', padx=20, pady=1)
        f = ttk.Frame(self.main); f.grid(row=row, column=1, sticky='w')
        var = tk.StringVar()
        e = ttk.Entry(f, textvariable=var, width=28); e.grid(row=0, column=0)
        ttk.Button(f, text='📂', width=3,
                   command=lambda: self._bopen(var)).grid(row=0, column=1)
        return var

    def _bopen(self, var):
        p = filedialog.askopenfilename(
            filetypes=[('Video', '*.mp4 *.avi *.mkv *.mov'), ('All', '*.*')])
        if p:
            var.set(p)
            self._read_video_fps(p)

    def _bsave(self, var, default):
        p = filedialog.asksaveasfilename(
            initialfile=default,
            filetypes=[('CSV', '*.csv'), ('MP4', '*.mp4'), ('All', '*.*')])
        if p: var.set(p)

    def _set(self, w, v):
        w.delete(0, 'end'); w.insert(0, str(v))

    def _read_video_fps(self, path):
        """Read video FPS and update the info label."""
        try:
            cap = cv2.VideoCapture(path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            if fps > 0:
                self.fps_auto_label.config(text=f'(auto: {fps:.2f})', foreground='#27AE60')
            else:
                self.fps_auto_label.config(text='(auto: N/A)', foreground='gray')
        except Exception:
            self.fps_auto_label.config(text='(auto: error)', foreground='red')

    # ── Toggles ───────────────────────────────────────────────────────────────

    def _toggle_fps_override(self):
        s = 'normal' if self.fps_override_var.get() else 'disabled'
        self.fps_entry.configure(state=s)

    def _toggle_dog(self):
        s = 'normal' if self.use_dog_var.get() else 'disabled'
        for w in self.dog_frame.winfo_children():
            try: w.configure(state=s)
            except: pass

    def _toggle_color_filter(self):
        s = 'normal' if self.filter_color_var.get() else 'disabled'
        for w in self.bc_frame.winfo_children():
            try: w.configure(state=s)
            except: pass

    def _toggle_circ_filter(self):
        s = 'normal' if self.filter_circ_var.get() else 'disabled'
        try: self.min_circ.configure(state=s)
        except: pass

    def _toggle_inertia_filter(self):
        s = 'normal' if self.filter_inertia_var.get() else 'disabled'
        try: self.min_inertia.configure(state=s)
        except: pass

    def _toggle_convex_filter(self):
        s = 'normal' if self.filter_convex_var.get() else 'disabled'
        try: self.min_convexity.configure(state=s)
        except: pass

    def _toggle_csv(self):
        s = 'normal' if self.save_csv_var.get() else 'disabled'
        for w in self.csv_frame.winfo_children():
            try: w.configure(state=s)
            except: pass

    def _toggle_video(self):
        s = 'normal' if self.save_video_var.get() else 'disabled'
        for w in self.vid_frame.winfo_children():
            try: w.configure(state=s)
            except: pass

    # ── Mode defaults ─────────────────────────────────────────────────────────

    def _on_mode(self):
        d = DEFAULTS[self.mode_var.get()]
        self.blob_color_var.set(d['blob_color'])
        self._set(self.min_area,   d['min_area']);    self._set(self.max_area, d['max_area'])
        self._set(self.min_circ,   d['min_circ']);    self.filter_var.set(d['filter_type'])
        self._set(self.clahe_clip, d['clahe_clip'])
        self.use_dog_var.set(d['use_dog'])
        self._set(self.dog_s1,  d['dog_sigma1']); self._set(self.dog_s2, d['dog_sigma2'])
        self._set(self.tophat,  d['tophat_ksize'])
        self._set(self.cam_pixel, d['cam_pixel']); self._set(self.magnif, d['magnification'])
        self._set(self.max_dist,   d['max_dist']);    self._set(self.max_skips, d['max_skips'])
        self._set(self.min_track,  d['min_track'])
        self._set(self.P_init, d['P_init']); self._set(self.Q_val, d['Q_val']); self._set(self.R_val, d['R_val'])
        self.filter_color_var.set(d['filter_by_color'])
        self.filter_circ_var.set(d['filter_by_circularity'])
        self.filter_inertia_var.set(d['filter_by_inertia'])
        self.filter_convex_var.set(d['filter_by_convexity'])
        self._toggle_dog(); self._toggle_color_filter()
        self._toggle_circ_filter(); self._toggle_inertia_filter(); self._toggle_convex_filter()

    # ── Helpers ────────────────────────────────────────────────────

    def _gf(self, w, n):
        try:    return float(w.get())
        except: raise ValueError(f'"{n}" must be a number.')

    def _gi(self, w, n):
        try:    return int(w.get())
        except: raise ValueError(f'"{n}" must be an integer.')

    # ── Run / Stop / Quit ─────────────────────────────────────────────────────

    def _on_quit(self):
        self._stop_flag = True
        if self.video_win: self.video_win.close()
        self.root.destroy()

    def _on_stop(self):
        self._stop_flag = True
        self.run_btn.configure(state='normal', text='▶  RUN TRACKING')
        self.status_var.set('Stopped — ready to restart')

    def _run(self):
        try:    p = self._collect()
        except ValueError as e:
            messagebox.showerror('Invalid parameters', str(e)); return

        cap = cv2.VideoCapture(p['video_path'])
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        self._stop_flag = False
        while not self.frame_queue.empty():
            try: self.frame_queue.get_nowait()
            except: break

        if self.video_win:
            try: self.video_win.close()
            except: pass
        self.video_win = VideoWindow(self.root, w, h, scale=0.6, on_stop=self._on_stop)

        self.run_btn.configure(state='disabled', text='⏳  Processing...')
        self.status_var.set('Running tracking...')
        threading.Thread(target=self._thread, args=(p,), daemon=True).start()
        self.root.after(30, self._poll)

    def _collect(self):
        v = self.video_path.get()
        if not v:               raise ValueError('Select a video file.')
        if not os.path.exists(v): raise ValueError(f'Video not found:\n{v}')

        # FPS
        fps_auto = 0.0
        try:
            cap      = cv2.VideoCapture(v)
            fps_auto = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
        except Exception:
            pass

        if self.fps_override_var.get():
            fps = self._gf(self.fps_entry, 'FPS override')
        else:
            fps = fps_auto if fps_auto > 0 else 15.0

        return {
            'video_path':        v,
            'fps':               fps,
            'image_mode':        MODE_TO_IMAGE[self.mode_var.get()],
            'pixel_size':        self._gf(self.cam_pixel, 'Camera pixel') / self._gf(self.magnif, 'Magnification'),
            'blob_color':        self.blob_color_var.get(),
            'filter_by_color':   self.filter_color_var.get(),
            'min_area':          self._gi(self.min_area,  'Min area'),
            'max_area':          self._gi(self.max_area,  'Max area'),
            'filter_by_circ':    self.filter_circ_var.get(),
            'min_circ':          self._gf(self.min_circ, 'Min circularity') if self.filter_circ_var.get() else 0.0,
            'filter_by_inertia': self.filter_inertia_var.get(),
            'min_inertia':       self._gf(self.min_inertia, 'Min inertia') if self.filter_inertia_var.get() else 0.1,
            'filter_by_convex':  self.filter_convex_var.get(),
            'min_convexity':     self._gf(self.min_convexity, 'Min convexity') if self.filter_convex_var.get() else 0.8,
            'filter_type':       self.filter_var.get(),
            'clahe_clip':        self._gf(self.clahe_clip, 'CLAHE clip'),
            'use_dog':           self.use_dog_var.get(),
            'dog_sigma1':        self._gf(self.dog_s1, 'DoG sigma 1'),
            'dog_sigma2':        self._gf(self.dog_s2, 'DoG sigma 2'),
            'tophat_ksize':      self._gi(self.tophat,  'Top-hat kernel'),
            'P_init':            self._gf(self.P_init,  'P init'),
            'Q_val':             self._gf(self.Q_val,   'Q value'),
            'R_val':             self._gf(self.R_val,   'R value'),
            'max_dist':          self._gf(self.max_dist,   'Max distance'),
            'max_skips':         self._gi(self.max_skips,  'Max skips'),
            'min_track':         self._gi(self.min_track,  'Min track length'),
            'show_plot':         self.show_plot_var.get(),
            'plot_style':        self.plot_style_var.get(),
            'save_csv':          self.save_csv_var.get(),
            'csv_mode':          self.csv_mode_var.get(),
            'csv_path':          self.csv_path_var.get() or 'trajectories.csv',
            'save_video':        self.save_video_var.get(),
            'video_out':         self.vid_path_var.get() or None,
        }

    # ── Worker thread (uses _tracking_loop from function_tracking_improved) ──────

    def _thread(self, p):
        writer = None
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from function_tracking_improved import (
                KalmanFilter2D, detectar_centroides, _tracking_loop,
                _video_iter, _save_csv, _plot
            )

            cap   = cv2.VideoCapture(p['video_path'])
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps   = p['fps']
            w_v   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h_v   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            det_kw = dict(
                image_mode  = p['image_mode'],
                dog_sigma1  = p['dog_sigma1'],  dog_sigma2  = p['dog_sigma2'],
                clahe_clip  = p['clahe_clip'],  tophat_ksize= p['tophat_ksize'],
                use_hough   = False,
            )

            def detect(gray):
                return detectar_centroides(
                    gray,
                    p['min_area'], p['max_area'], p['blob_color'], p['filter_type'],
                    p['filter_by_circ'],  p['min_circ'],
                    p['filter_by_inertia'], p['filter_by_convex'],
                    p['filter_by_color'],
                    **det_kw
                )

            # VideoWriter (created when the first frame is received)
            if p['save_video'] and p['video_out']:
                writer = cv2.VideoWriter(
                    p['video_out'], cv2.VideoWriter_fourcc(*'mp4v'),
                    fps, (w_v, h_v))
                if not writer.isOpened():
                    raise ValueError(
                        f"Could not create the output video at:\n{p['video_out']}\n\n"
                        "Check that the folder exists, you have permissions, and the path ends in .mp4"
                    )

            mask = None

            def on_frame(frame, _mask, trajs, fi, total, n_active):
                nonlocal mask
                if mask is None:
                    mask = np.zeros_like(frame)

                # Draw tracks
                disp = frame.copy()
                for t in trajs:
                    if len(t) > 1:
                        pts = np.array(t[-2:], dtype=int)
                        cv2.line(mask, tuple(pts[0]), tuple(pts[1]), (0, 255, 0), 2)
                        cv2.circle(disp, tuple(pts[1]), 4, (0, 0, 255), -1)
                cv2.putText(disp, f'Tracks: {n_active}  Frame: {fi}/{total}',
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                result = cv2.add(disp, mask)

                if writer:
                    writer.write(result)

                try:
                    self.frame_queue.put(
                        {'type': 'frame', 'frame': result,
                         'fi': fi, 'total': total, 'n_tracks': n_active},
                        timeout=0.1)
                except queue.Full:
                    pass

                return self._stop_flag  # True = stop tracking

            det_pos, trajs = _tracking_loop(
                _video_iter(cap), detect, total,
                p['P_init'], p['Q_val'], p['R_val'],
                p['max_dist'], p['max_skips'], p['min_track'],
                frame_callback=on_frame
            )

        except Exception as e:
            import traceback
            self.frame_queue.put({'type': 'error', 'error': traceback.format_exc()})
            return
        finally:
            try: cap.release()
            except: pass
            if writer:
                try:
                    writer.release()
                    print(f"Saved video: {p['video_out']}")
                except: pass

        if not self._stop_flag:
            self.frame_queue.put({
                'type': 'done',
                'det_pos': det_pos, 'trajs': trajs,
                'pixel_size': p['pixel_size'],
                'n': len(det_pos), 'p': p,
            })

    # ── Poll / Done / Error ───────────────────────────────────────────────────

    def _poll(self):
        try:
            while True:
                msg = self.frame_queue.get_nowait()
                if msg['type'] == 'frame':
                    if self.video_win and not self.video_win.closed:
                        self.video_win.show_frame(
                            msg['frame'], msg['fi'], msg['total'], msg['n_tracks'])
                elif msg['type'] == 'done':
                    self._on_done(msg); return
                elif msg['type'] == 'error':
                    self._on_error(msg['error']); return
        except queue.Empty:
            pass
        if not self._stop_flag:
            self.root.after(30, self._poll)
        else:
            self.run_btn.configure(state='normal', text='▶  RUN TRACKING')

    def _on_done(self, msg):
        self.run_btn.configure(state='normal', text='▶  RUN TRACKING')
        n = msg['n']
        self.status_var.set(f'Done — {n} tracks found.')
        if self.video_win: self.video_win.close()
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        if msg['p']['show_plot']:
            from function_tracking_improved import _plot
            _plot(msg['det_pos'], msg['trajs'], msg['pixel_size'],
                  style=msg['p']['plot_style'])
        if msg['p']['save_csv']:
            from function_tracking_improved import _save_csv
            _save_csv(msg['det_pos'], msg['pixel_size'],
                      msg['p']['csv_mode'], msg['p']['csv_path'])
        messagebox.showinfo('Tracking complete', f'{n} tracks found.')

    def _on_error(self, err):
        self.run_btn.configure(state='normal', text='▶  RUN TRACKING')
        self.status_var.set('Error.')
        if self.video_win: self.video_win.close()
        messagebox.showerror('Error', err)


# ════════════ main ══════════════════════════════════════════════════════
if __name__ == '__main__':
    root = tk.Tk()
    TrackerGUI(root)
    root.mainloop()
