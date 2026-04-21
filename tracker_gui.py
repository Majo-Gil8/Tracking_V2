"""
DHM Particle Tracker - GUI v2
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading, queue, sys, os, cv2, numpy as np
from PIL import Image, ImageTk

DEFAULTS = {
    'Brightfield': dict(blob_color=0, min_area=400, max_area=5000, min_circ=0.3,
        filter_type='bilateral', clahe_clip=3.0, use_dog=False,
        dog_sigma1=2.0, dog_sigma2=8.0, tophat_ksize=21,
        cam_pixel=3.75, magnification=40, max_dist=25, max_skips=5, min_track=5,
        P_init=100, Q_val=1, R_val=50,
        filter_by_color=True, filter_by_circularity=True),
    'Amplitude': dict(blob_color=255, min_area=100, max_area=3000, min_circ=0.5,
        filter_type='bilateral', clahe_clip=3.0, use_dog=False,
        dog_sigma1=2.0, dog_sigma2=8.0, tophat_ksize=21,
        cam_pixel=3.75, magnification=20, max_dist=50, max_skips=53, min_track=3,
        P_init=100, Q_val=1, R_val=50,
        filter_by_color=True, filter_by_circularity=True),
    'Hologram': dict(blob_color=255, min_area=150, max_area=2500, min_circ=0.3,
        filter_type='bilateral', clahe_clip=3.0, use_dog=False,
        dog_sigma1=2.0, dog_sigma2=8.0, tophat_ksize=21,
        cam_pixel=3.75, magnification=40, max_dist=40, max_skips=5, min_track=5,
        P_init=100, Q_val=1, R_val=30,
        filter_by_color=True, filter_by_circularity=True),
    'Phase': dict(blob_color=255, min_area=100, max_area=2500, min_circ=0.3,
        filter_type='bilateral', clahe_clip=3.0, use_dog=True,
        dog_sigma1=2.0, dog_sigma2=8.0, tophat_ksize=21,
        cam_pixel=3.75, magnification=40, max_dist=40, max_skips=8, min_track=3,
        P_init=100, Q_val=1, R_val=30,
        filter_by_color=True, filter_by_circularity=True),
}

MODE_TO_IMAGE = {'Brightfield':'standard','Amplitude':'amplitude',
                 'Hologram':'hologram','Phase':'phase'}


class VideoWindow:
    """Segunda ventana: preview del tracking en tiempo real."""
    def __init__(self, parent, video_w, video_h, scale=0.6, on_stop=None):
        self.win = tk.Toplevel(parent)
        self.win.title('Tracking — Live Preview')
        self.win.resizable(True, True)
        self.win.protocol('WM_DELETE_WINDOW', self._on_close)
        self.closed  = False
        self.on_stop = on_stop

        self.disp_w = max(int(video_w * scale), 640)
        self.disp_h = max(int(video_h * scale), 360)

        # Info bar
        info = tk.Frame(self.win, bg='#2C3E50', pady=4)
        info.pack(fill='x')
        self.info_var = tk.StringVar(value='Iniciando...')
        tk.Label(info, textvariable=self.info_var, font=('Arial',9),
                 bg='#2C3E50', fg='white').pack(side='left', padx=10)

        # Canvas
        self.canvas = tk.Canvas(self.win, width=self.disp_w, height=self.disp_h,
                                bg='black', highlightthickness=0)
        self.canvas.pack()

        # Progress bar
        self.progress = ttk.Progressbar(self.win, length=self.disp_w, mode='determinate')
        self.progress.pack(fill='x')

        # Control buttons
        btn_frame = tk.Frame(self.win, bg='#ECF0F1', pady=6)
        btn_frame.pack(fill='x')
        tk.Button(btn_frame, text='⏹  Stop',
            font=('Arial',9,'bold'), bg='#C0392B', fg='white',
            relief='flat', padx=12, pady=4, cursor='hand2',
            command=self._on_stop_click).pack(side='left', padx=10)

        self._photo = None

    # ── Public ───────────────────────────────────────────────────────────────
    def show_frame(self, frame_bgr, fi, total, n_tracks):
        if self.closed: return
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.disp_w, self.disp_h))
        self._photo = ImageTk.PhotoImage(image=Image.fromarray(rgb))
        self.canvas.create_image(0, 0, anchor='nw', image=self._photo)
        pct = int(fi / max(total,1) * 100)
        self.progress['value'] = pct
        self.info_var.set(f'Frame {fi}/{total}  |  Tracks: {n_tracks}  |  {pct}%')

    def close(self):
        if not self.closed:
            self.closed = True
            try: self.win.destroy()
            except: pass

    # ── Internal ─────────────────────────────────────────────────────────────
    def _on_stop_click(self):
        if self.on_stop: self.on_stop()
        self.close()

    def _on_close(self):
        self._on_stop_click()


class TrackerGUI:
    def __init__(self, root):
        self.root      = root
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
        for s,cfg in [('TFrame',{'background':BG}),
                      ('TLabel',{'background':BG,'font':('Arial',9)}),
                      ('TEntry',{'font':('Arial',9)}),
                      ('TCheckbutton',{'background':BG,'font':('Arial',9)}),
                      ('TRadiobutton',{'background':BG,'font':('Arial',9)})]:
            style.configure(s, **cfg)

        # Header
        hdr = tk.Frame(root, bg=HDR_BG, pady=10)
        hdr.grid(row=0, column=0, columnspan=2, sticky='ew')
        tk.Label(hdr, text='DHM Particle Tracker', font=('Arial',14,'bold'),
                 bg=HDR_BG, fg=HDR_FG).pack()
        tk.Label(hdr, text='Digital Holographic Microscopy Tracking Tool',
                 font=('Arial',9), bg=HDR_BG, fg='#BDC3C7').pack()

        # Scrollable canvas
        cv = tk.Canvas(root, bg=BG, highlightthickness=0, width=520, height=590)
        sb = ttk.Scrollbar(root, orient='vertical', command=cv.yview)
        cv.configure(yscrollcommand=sb.set)
        cv.grid(row=1, column=0, sticky='nsew'); sb.grid(row=1, column=1, sticky='ns')
        root.grid_rowconfigure(1, weight=1); root.grid_columnconfigure(0, weight=1)
        self.main = ttk.Frame(cv, padding=10)
        wid = cv.create_window((0,0), window=self.main, anchor='nw')
        def _cfg(e): cv.configure(scrollregion=cv.bbox('all')); cv.itemconfig(wid, width=cv.winfo_width())
        self.main.bind('<Configure>', _cfg)
        cv.bind('<Configure>', lambda e: cv.itemconfig(wid, width=e.width))
        cv.bind_all('<MouseWheel>', lambda e: cv.yview_scroll(-1*(e.delta//120),'units'))

        row = 0

        # ── VIDEO TYPE ────────────────────────────────────────────────────────
        row = self._sec(row, '  VIDEO TYPE')
        self.mode_var = tk.StringVar(value='Brightfield')
        for m in ['Brightfield','Amplitude','Hologram','Phase']:
            ttk.Radiobutton(self.main, text=m, variable=self.mode_var,
                            value=m, command=self._on_mode).grid(
                row=row, column=0, columnspan=2, sticky='w', padx=20, pady=1); row+=1

        # ── VIDEO FILES ───────────────────────────────────────────────────────
        row = self._sec(row, '  VIDEO FILES')
        self.video_path = self._file_row(row, 'Video path:'); row+=1

        # ── OPTICAL SYSTEM ────────────────────────────────────────────────────
        row = self._sec(row, '  OPTICAL SYSTEM')
        self.cam_pixel = self._erow(row, 'Camera pixel size (µm):'); row+=1
        self.magnif    = self._erow(row, 'Magnification (x):');      row+=1
        self.fps_val   = self._erow(row, 'FPS:');                    row+=1
        self.fps_val.insert(0,'15')

        # ── DETECTION ─────────────────────────────────────────────────────────
        row = self._sec(row, '  DETECTION')

        # Blob color with enable toggle
        self.filter_color_var = tk.BooleanVar(value=True)
        fc_row = ttk.Frame(self.main); fc_row.grid(row=row, column=0, columnspan=2, sticky='w', padx=20, pady=1)
        ttk.Checkbutton(fc_row, text='Filter by blob color:', variable=self.filter_color_var,
                        command=self._toggle_color_filter).pack(side='left')
        self.blob_color_var = tk.IntVar(value=255)
        self.bc_frame = ttk.Frame(fc_row)
        self.bc_frame.pack(side='left', padx=8)
        ttk.Radiobutton(self.bc_frame, text='Dark (0)',     variable=self.blob_color_var, value=0  ).pack(side='left', padx=3)
        ttk.Radiobutton(self.bc_frame, text='Bright (255)', variable=self.blob_color_var, value=255).pack(side='left')
        row+=1

        self.min_area = self._erow(row, 'Min area (px²):');  row+=1
        self.max_area = self._erow(row, 'Max area (px²):');  row+=1

        # Circularity with enable toggle
        self.filter_circ_var = tk.BooleanVar(value=True)
        fcirc_row = ttk.Frame(self.main); fcirc_row.grid(row=row, column=0, columnspan=2, sticky='w', padx=20, pady=1)
        ttk.Checkbutton(fcirc_row, text='Filter by circularity  min:', variable=self.filter_circ_var,
                        command=self._toggle_circ_filter).pack(side='left')
        self.min_circ = ttk.Entry(fcirc_row, width=7); self.min_circ.pack(side='left', padx=4)
        row+=1

        ttk.Label(self.main, text='Preprocessing filter:').grid(row=row, column=0, sticky='w', padx=20)
        self.filter_var = tk.StringVar(value='bilateral')
        ff = ttk.Frame(self.main); ff.grid(row=row, column=1, sticky='w')
        ttk.Radiobutton(ff, text='Bilateral', variable=self.filter_var, value='bilateral').pack(side='left', padx=5)
        ttk.Radiobutton(ff, text='Gaussian',  variable=self.filter_var, value='gaussian' ).pack(side='left')
        row+=1

        self.clahe_clip = self._erow(row, 'CLAHE clip limit:'); row+=1

        self.use_dog_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.main, text='Use DoG + Top-hat (Phase enhancement)',
                        variable=self.use_dog_var, command=self._toggle_dog).grid(
            row=row, column=0, columnspan=2, sticky='w', padx=20, pady=2); row+=1
        self.dog_frame = ttk.Frame(self.main)
        self.dog_frame.grid(row=row, column=0, columnspan=2, sticky='ew', padx=30); row+=1
        ttk.Label(self.dog_frame, text='DoG sigma 1:').grid(row=0,column=0,sticky='w',padx=5)
        self.dog_s1 = ttk.Entry(self.dog_frame, width=8); self.dog_s1.grid(row=0,column=1,padx=5)
        ttk.Label(self.dog_frame, text='DoG sigma 2:').grid(row=0,column=2,sticky='w',padx=5)
        self.dog_s2 = ttk.Entry(self.dog_frame, width=8); self.dog_s2.grid(row=0,column=3,padx=5)
        ttk.Label(self.dog_frame, text='Top-hat kernel:').grid(row=1,column=0,sticky='w',padx=5,pady=2)
        self.tophat = ttk.Entry(self.dog_frame, width=8); self.tophat.grid(row=1,column=1,padx=5)
        self.dog_s1.insert(0,'2.0'); self.dog_s2.insert(0,'8.0'); self.tophat.insert(0,'21')
        self._toggle_dog()

        # ── KALMAN ────────────────────────────────────────────────────────────
        row = self._sec(row, '  KALMAN FILTER')
        ttk.Label(self.main, text='P — Initial covariance', font=('Arial',8), foreground='gray').grid(row=row,column=0,columnspan=2,sticky='w',padx=20); row+=1
        self.P_init = self._erow(row,'P init:'); row+=1
        ttk.Label(self.main, text='Q — Process noise (higher = more agile)', font=('Arial',8), foreground='gray').grid(row=row,column=0,columnspan=2,sticky='w',padx=20); row+=1
        self.Q_val  = self._erow(row,'Q value:'); row+=1
        ttk.Label(self.main, text='R — Measurement noise (higher = trust Kalman more)', font=('Arial',8), foreground='gray').grid(row=row,column=0,columnspan=2,sticky='w',padx=20); row+=1
        self.R_val  = self._erow(row,'R value:'); row+=1

        # ── TRACKING ─────────────────────────────────────────────────────────
        row = self._sec(row, '  TRACKING')
        self.max_dist  = self._erow(row,'Max distance (px):');         row+=1
        self.max_skips = self._erow(row,'Max skips (frames):');        row+=1
        self.min_track = self._erow(row,'Min track length (frames):'); row+=1

        # ── OUTPUT ────────────────────────────────────────────────────────────
        row = self._sec(row, '  OUTPUT')
        self.show_plot_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.main, text='Show trajectory plot after tracking',
                        variable=self.show_plot_var).grid(row=row,column=0,columnspan=2,sticky='w',padx=20); row+=1
        ttk.Label(self.main, text='Plot style:').grid(row=row,column=0,sticky='w',padx=20)
        self.plot_style_var = tk.StringVar(value='markers')
        ps = ttk.Frame(self.main); ps.grid(row=row,column=1,sticky='w')
        ttk.Radiobutton(ps, text='Markers', variable=self.plot_style_var, value='markers').pack(side='left',padx=5)
        ttk.Radiobutton(ps, text='Dots',    variable=self.plot_style_var, value='dots'   ).pack(side='left')
        row+=1

        self.save_csv_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.main, text='Save CSV', variable=self.save_csv_var,
                        command=self._toggle_csv).grid(row=row,column=0,columnspan=2,sticky='w',padx=20); row+=1
        self.csv_frame = ttk.Frame(self.main)
        self.csv_frame.grid(row=row,column=0,columnspan=2,sticky='ew',padx=20); row+=1
        ttk.Label(self.csv_frame, text='CSV mode:').grid(row=0,column=0,sticky='w')
        self.csv_mode_var = tk.StringVar(value='single')
        ttk.Radiobutton(self.csv_frame, text='Single file', variable=self.csv_mode_var, value='single'   ).grid(row=0,column=1,padx=5)
        ttk.Radiobutton(self.csv_frame, text='Per track',   variable=self.csv_mode_var, value='per_track').grid(row=0,column=2)
        ttk.Label(self.csv_frame, text='CSV path:').grid(row=1,column=0,sticky='w',pady=2)
        self.csv_path_var = tk.StringVar()
        ttk.Entry(self.csv_frame, textvariable=self.csv_path_var, width=28).grid(row=1,column=1,columnspan=2)
        ttk.Button(self.csv_frame, text='📂', width=3,
                   command=lambda: self._bsave(self.csv_path_var,'trajectories.csv')).grid(row=1,column=3)
        self._toggle_csv()

        self.save_video_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.main, text='Save output video',
                        variable=self.save_video_var, command=self._toggle_video).grid(
            row=row,column=0,columnspan=2,sticky='w',padx=20); row+=1
        self.vid_frame = ttk.Frame(self.main)
        self.vid_frame.grid(row=row,column=0,columnspan=2,sticky='ew',padx=20); row+=1
        ttk.Label(self.vid_frame, text='Output video path:').grid(row=0,column=0,sticky='w')
        self.vid_path_var = tk.StringVar()
        ttk.Entry(self.vid_frame, textvariable=self.vid_path_var, width=28).grid(row=0,column=1)
        ttk.Button(self.vid_frame, text='📂', width=3,
                   command=lambda: self._bsave(self.vid_path_var,'output.mp4')).grid(row=0,column=2)
        self._toggle_video()

        # ── BUTTONS BAR ───────────────────────────────────────────────────────
        row+=1
        btn_bar = ttk.Frame(self.main)
        btn_bar.grid(row=row, column=0, columnspan=2, pady=15)

        self.run_btn = tk.Button(btn_bar, text='▶  RUN TRACKING',
            font=('Arial',11,'bold'), bg=BTN_BG, fg=BTN_FG,
            activebackground='#1A6FA8', relief='flat', cursor='hand2',
            padx=20, pady=8, command=self._run)
        self.run_btn.pack(side='left', padx=6)

        tk.Button(btn_bar, text='✕  Exit',
            font=('Arial',11,'bold'), bg='#7F8C8D', fg='white',
            activebackground='#636E72', relief='flat', cursor='hand2',
            padx=16, pady=8, command=self._on_quit).pack(side='left', padx=6)

        # Status bar
        self.status_var = tk.StringVar(value='Ready')
        tk.Label(root, textvariable=self.status_var, font=('Arial',8),
                 bg='#ECF0F1', fg='#555', anchor='w', padx=10).grid(
            row=2, column=0, columnspan=2, sticky='ew')

        # Threading state
        self.frame_queue = queue.Queue(maxsize=4)
        self.video_win   = None
        self._stop_flag  = False

        self._on_mode()

    # ── Widget helpers ────────────────────────────────────────────────────────
    def _sec(self, row, title):
        f = tk.Frame(self.main, bg='#2C3E50', pady=3)
        f.grid(row=row, column=0, columnspan=2, sticky='ew', pady=(8,2))
        tk.Label(f, text=title, font=('Arial',9,'bold'),
                 bg='#2C3E50', fg='white').pack(side='left', padx=8)
        return row+1

    def _erow(self, row, label, width=10):
        ttk.Label(self.main, text=label).grid(row=row, column=0, sticky='w', padx=20, pady=1)
        e = ttk.Entry(self.main, width=width); e.grid(row=row, column=1, sticky='w', pady=1)
        return e

    def _file_row(self, row, label):
        ttk.Label(self.main, text=label).grid(row=row, column=0, sticky='w', padx=20, pady=1)
        f = ttk.Frame(self.main); f.grid(row=row, column=1, sticky='w')
        var = tk.StringVar()
        ttk.Entry(f, textvariable=var, width=28).grid(row=0, column=0)
        ttk.Button(f, text='📂', width=3, command=lambda: self._bopen(var)).grid(row=0, column=1)
        return var

    def _bopen(self, var):
        p = filedialog.askopenfilename(
            filetypes=[('Video','*.avi *.mp4 *.mov *.tif *.tiff'),('All','*.*')])
        if p: var.set(p)

    def _bsave(self, var, default):
        p = filedialog.asksaveasfilename(initialfile=default,
            filetypes=[('CSV','*.csv'),('MP4','*.mp4'),('All','*.*')])
        if p: var.set(p)

    def _set(self, w, v): w.delete(0,'end'); w.insert(0,str(v))

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

    def _on_mode(self):
        d = DEFAULTS[self.mode_var.get()]
        self.blob_color_var.set(d['blob_color'])
        self._set(self.min_area, d['min_area']); self._set(self.max_area, d['max_area'])
        self._set(self.min_circ, d['min_circ']); self.filter_var.set(d['filter_type'])
        self._set(self.clahe_clip, d['clahe_clip'])
        self.use_dog_var.set(d['use_dog'])
        self._set(self.dog_s1, d['dog_sigma1']); self._set(self.dog_s2, d['dog_sigma2'])
        self._set(self.tophat, d['tophat_ksize'])
        self._set(self.cam_pixel, d['cam_pixel']); self._set(self.magnif, d['magnification'])
        self._set(self.max_dist, d['max_dist']); self._set(self.max_skips, d['max_skips'])
        self._set(self.min_track, d['min_track'])
        self._set(self.P_init, d['P_init']); self._set(self.Q_val, d['Q_val'])
        self._set(self.R_val, d['R_val'])
        self.filter_color_var.set(d['filter_by_color'])
        self.filter_circ_var.set(d['filter_by_circularity'])
        self._toggle_dog(); self._toggle_color_filter(); self._toggle_circ_filter()

    def _gf(self,w,n):
        try: return float(w.get())
        except: raise ValueError(f'"{n}" debe ser un numero.')
    def _gi(self,w,n):
        try: return int(w.get())
        except: raise ValueError(f'"{n}" debe ser un entero.')

    # ── Run / Pause / Stop / Quit ─────────────────────────────────────────────
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
        except ValueError as e: messagebox.showerror('Parametros invalidos', str(e)); return

        cap = cv2.VideoCapture(p['video_path'])
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Reset flags
        self._stop_flag = False

        # Clear queue
        while not self.frame_queue.empty():
            try: self.frame_queue.get_nowait()
            except: break

        # Open video window
        if self.video_win:
            try: self.video_win.close()
            except: pass
        self.video_win = VideoWindow(self.root, w, h, scale=0.6,
                                     on_stop=self._on_stop)

        self.run_btn.configure(state='disabled', text='⏳  Processing...')
        self.status_var.set('Running tracking...')
        threading.Thread(target=self._thread, args=(p,), daemon=True).start()
        self.root.after(30, self._poll)

    def _collect(self):
        v = self.video_path.get()
        if not v: raise ValueError('Selecciona un archivo de video.')
        if not os.path.exists(v): raise ValueError(f'Video no encontrado:\n{v}')
        return {
            'video_path':           v,
            'image_mode':           MODE_TO_IMAGE[self.mode_var.get()],
            'pixel_size':           self._gf(self.cam_pixel,'Camera pixel') / self._gf(self.magnif,'Magnification'),
            'fps':                  self._gf(self.fps_val,'FPS'),
            'blob_color':           self.blob_color_var.get(),
            'filter_by_color':      self.filter_color_var.get(),
            'min_area':             self._gi(self.min_area,'Min area'),
            'max_area':             self._gi(self.max_area,'Max area'),
            'filter_by_circ':       self.filter_circ_var.get(),
            'min_circ':             self._gf(self.min_circ,'Min circularity'),
            'filter_type':          self.filter_var.get(),
            'clahe_clip':           self._gf(self.clahe_clip,'CLAHE clip'),
            'use_dog':              self.use_dog_var.get(),
            'dog_sigma1':           self._gf(self.dog_s1,'DoG sigma 1'),
            'dog_sigma2':           self._gf(self.dog_s2,'DoG sigma 2'),
            'tophat_ksize':         self._gi(self.tophat,'Top-hat kernel'),
            'P_init':               self._gf(self.P_init,'P init'),
            'Q_val':                self._gf(self.Q_val,'Q value'),
            'R_val':                self._gf(self.R_val,'R value'),
            'max_dist':             self._gf(self.max_dist,'Max distance'),
            'max_skips':            self._gi(self.max_skips,'Max skips'),
            'min_track':            self._gi(self.min_track,'Min track length'),
            'show_plot':            self.show_plot_var.get(),
            'plot_style':           self.plot_style_var.get(),
            'save_csv':             self.save_csv_var.get(),
            'csv_mode':             self.csv_mode_var.get(),
            'csv_path':             self.csv_path_var.get() or 'trajectories.csv',
            'save_video':           self.save_video_var.get(),
            'video_out':            self.vid_path_var.get() or None,
        }

    def _poll(self):
        try:
            while True:
                msg = self.frame_queue.get_nowait()
                if msg['type'] == 'frame':
                    if self.video_win and not self.video_win.closed:
                        self.video_win.show_frame(msg['frame'], msg['fi'],
                                                   msg['total'], msg['n_tracks'])
                elif msg['type'] == 'done':
                    self._on_done(msg); return
                elif msg['type'] == 'error':
                    self._on_error(msg['error']); return
        except queue.Empty: pass
        if not self._stop_flag:
            self.root.after(30, self._poll)
        else:
            self.run_btn.configure(state='normal', text='▶  RUN TRACKING')

    def _thread(self, p):
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from function_tracking_improved import KalmanFilter2D, detectar_centroides, _save_csv, _plot
            from scipy.optimize import linear_sum_assignment

            cap   = cv2.VideoCapture(p['video_path'])
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # ── FIX 1: usar FPS real del video; solo recurrir al GUI si el video no lo reporta ──
            real_fps = cap.get(cv2.CAP_PROP_FPS)
            fps = real_fps if real_fps > 0 else (p['fps'] if p['fps'] > 0 else 15.0)

            det_kw = dict(image_mode=p['image_mode'],
                          dog_sigma1=p['dog_sigma1'], dog_sigma2=p['dog_sigma2'],
                          clahe_clip=p['clahe_clip'], tophat_ksize=p['tophat_ksize'],
                          use_hough=False)

            def detect(gray):
                return detectar_centroides(
                    gray, p['min_area'], p['max_area'], p['blob_color'],
                    p['filter_type'], p['filter_by_circ'], p['min_circ'],
                    False, False, p['filter_by_color'], **det_kw)

            ret, first = cap.read()
            if not ret: raise ValueError('No se pudo leer el video.')
            p0 = detect(cv2.cvtColor(first, cv2.COLOR_BGR2GRAY))
            if len(p0)==0:
                raise ValueError('No se detectaron particulas en el primer frame.\n'
                                  'Ajusta los parametros de deteccion.')

            NAN = np.array([np.nan,np.nan])
            kfs     = [KalmanFilter2D(x,y,p['P_init'],p['Q_val'],p['R_val']) for x,y in p0]
            det_pos = [[p0[i].copy()] for i in range(len(kfs))]
            trajs   = [[kfs[i].state[:2].copy()] for i in range(len(kfs))]
            skips   = [0]*len(kfs)
            done_dp, done_tr = [], []

            def _save_t(i):
                real = sum(1 for pt in det_pos[i] if not np.isnan(pt[0]))
                if real >= p['min_track']:
                    done_dp.append([pt.copy() for pt in det_pos[i]])
                    done_tr.append([pt.copy() for pt in trajs[i]])

            # ── FIX 2: crear VideoWriter con FPS correcto y verificar que se abrio bien ──
            writer = None
            if p['save_video'] and p['video_out']:
                h_v, w_v = first.shape[:2]
                writer = cv2.VideoWriter(
                    p['video_out'],
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    fps,
                    (w_v, h_v)
                )
                if not writer.isOpened():
                    raise ValueError(
                        f"No se pudo crear el video de salida en:\n{p['video_out']}\n\n"
                        "Verifica que:\n"
                        "  • La carpeta de destino exista\n"
                        "  • Tengas permisos de escritura\n"
                        "  • La ruta termine en .mp4"
                    )

            mask = np.zeros_like(first)

            # ── FIX 3: escribir frame 0 en el video (antes no se escribia) ──
            d0 = first.copy()
            for pt in p0: cv2.circle(d0,(int(pt[0]),int(pt[1])),5,(0,255,0),-1)
            if writer: writer.write(d0)
            try: self.frame_queue.put({'type':'frame','frame':d0,'fi':0,'total':total,'n_tracks':len(kfs)},timeout=0.1)
            except queue.Full: pass

            for fi in range(1, total):
                if self._stop_flag: break

                ret, frame = cap.read()
                if not ret: break

                p1 = detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                preds = np.array([kf.predict() for kf in kfs])

                if len(p1)==0:
                    for i in range(len(kfs)):
                        trajs[i].append(kfs[i].state[:2].copy())
                        det_pos[i].append(NAN.copy()); skips[i]+=1
                else:
                    D = np.linalg.norm(preds[:,None,:]-p1[None,:,:],axis=2)
                    ri,ci = linear_sum_assignment(D)
                    ap,ad = set(),set()
                    for r,c in zip(ri,ci):
                        if D[r,c]<p['max_dist']:
                            kfs[r].update(p1[c]); ap.add(r); ad.add(c)
                            det_pos[r].append(p1[c].copy()); skips[r]=0
                        else:
                            det_pos[r].append(NAN.copy()); skips[r]+=1
                    for i in range(len(kfs)):
                        if i not in ap: det_pos[i].append(NAN.copy()); skips[i]+=1
                    for i in range(len(p1)):
                        if i not in ad:
                            kfs.append(KalmanFilter2D(p1[i][0],p1[i][1],p['P_init'],p['Q_val'],p['R_val']))
                            trajs.append([p1[i].copy()]); det_pos.append([p1[i].copy()]); skips.append(0)

                for i,kf in enumerate(kfs): trajs[i].append(kf.state[:2].copy())
                for i in reversed(range(len(kfs))):
                    if skips[i]>p['max_skips']: _save_t(i); del kfs[i],trajs[i],det_pos[i],skips[i]

                disp = frame.copy()
                for t in trajs:
                    if len(t)>1:
                        pts = np.array(t[-2:],dtype=int)
                        cv2.line(mask,tuple(pts[0]),tuple(pts[1]),(0,255,0),2)
                        cv2.circle(disp,tuple(pts[1]),4,(0,0,255),-1)
                cv2.putText(disp,f'Tracks: {len(kfs)}  Frame: {fi}/{total}',
                            (10,28),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2)
                result = cv2.add(disp,mask)
                if writer: writer.write(result)
                try: self.frame_queue.put({'type':'frame','frame':result,'fi':fi,'total':total,'n_tracks':len(kfs)},timeout=0.1)
                except queue.Full: pass

            cap.release()
            # ── FIX 4: siempre liberar el writer con finally para evitar 0 bytes ──
            if writer:
                writer.release()
                print(f"Video guardado: {p['video_out']}")

            if not self._stop_flag:
                for i in range(len(kfs)): _save_t(i)
                self.frame_queue.put({'type':'done','det_pos':done_dp,'trajs':done_tr,
                                      'pixel_size':p['pixel_size'],'n':len(done_dp),'p':p})

        except Exception as e:
            import traceback
            # Asegurar que el writer se libere aunque haya error
            try:
                if 'writer' in dir() and writer is not None:
                    writer.release()
            except:
                pass
            self.frame_queue.put({'type':'error','error':traceback.format_exc()})

    def _on_done(self, msg):
        self.run_btn.configure(state='normal', text='▶  RUN TRACKING')
        n = msg['n']
        self.status_var.set(f'Done — {n} tracks found.')
        if self.video_win: self.video_win.close()
        if msg['p']['show_plot']:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from function_tracking_improved import _plot
            _plot(msg['det_pos'], msg['trajs'], msg['pixel_size'], style=msg['p']['plot_style'])
        if msg['p']['save_csv']:
            from function_tracking_improved import _save_csv
            _save_csv(msg['det_pos'], msg['pixel_size'], msg['p']['csv_mode'], msg['p']['csv_path'])
        messagebox.showinfo('Tracking completo', f'{n} trayectorias encontradas.')

    def _on_error(self, err):
        self.run_btn.configure(state='normal', text='▶  RUN TRACKING')
        self.status_var.set('Error.')
        if self.video_win: self.video_win.close()
        messagebox.showerror('Error', err)


if __name__ == '__main__':
    root = tk.Tk()
    TrackerGUI(root)
    root.mainloop()
