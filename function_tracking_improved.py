import csv
import os
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment


# ══════════════════════════════════════════════════════════════════
#  KALMAN FILTER
# ══════════════════════════════════════════════════════════════════

class KalmanFilter2D:
    def __init__(self, x, y, P_init, Q, R):
        self.state = np.array([x, y, 0.0, 0.0], dtype=np.float32)
        self.F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], dtype=np.float32)
        self.H = np.array([[1,0,0,0],[0,1,0,0]], dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * P_init
        self.Q = np.eye(4, dtype=np.float32) * Q
        self.R = np.eye(2, dtype=np.float32) * R

    def predict(self):
        self.state = self.F @ self.state
        self.P     = self.F @ self.P @ self.F.T + self.Q
        return self.state[:2]

    def update(self, z):
        z  = np.array(z, dtype=np.float32)
        y  = z - self.H @ self.state
        S  = self.H @ self.P @ self.H.T + self.R
        K  = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.P     = (np.eye(4) - K @ self.H) @ self.P


# ══════════════════════════════════════════════════════════════════
#  PREPROCESSING
# ══════════════════════════════════════════════════════════════════

def _preprocess_standard(frame_gray, filter_type):
    if filter_type == "gaussian":
        return cv2.GaussianBlur(frame_gray, (5, 5), 0)
    elif filter_type == "bilateral":
        return cv2.bilateralFilter(frame_gray, 9, 75, 75)
    raise ValueError("filter_type must be 'gaussian' or 'bilateral'")


def _preprocess_phase(frame_gray, dog_sigma1=2.0, dog_sigma2=8.0,
                      clahe_clip=3.0, tophat_ksize=21):
    f      = frame_gray.astype(np.float64)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tophat_ksize, tophat_ksize))
    tophat = cv2.morphologyEx(frame_gray, cv2.MORPH_TOPHAT, kernel).astype(np.float64)
    dog    = np.clip(cv2.GaussianBlur(f,(0,0),dog_sigma1) - cv2.GaussianBlur(f,(0,0),dog_sigma2), 0, None)
    combined = cv2.normalize(0.5*tophat + 0.5*dog, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8,8)).apply(combined)


def _preprocess_amplitude(frame_gray, clahe_clip=3.0):
    """CLAHE + bilateral for amplitude videos with fluctuating intensity."""
    return cv2.bilateralFilter(
        cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8,8)).apply(frame_gray),
        9, 75, 75)


# ══════════════════════════════════════════════════════════════════
#  CENTROID DETECTION
# ══════════════════════════════════════════════════════════════════

def detectar_centroides(frame_gray, minArea, maxArea, blobColor, filter_type,
                        filterByCircularity, minCircularity,
                        filterByInertia, filterByConvexity, filterByColor,
                        image_mode="standard",
                        dog_sigma1=2.0, dog_sigma2=8.0,
                        clahe_clip=3.0, tophat_ksize=21,
                        use_hough=False,
                        hough_dp=1.5, hough_min_dist=25,
                        hough_param1=40, hough_param2=18,
                        hough_min_radius=8, hough_max_radius=30):

    if image_mode == "phase":
        processed = _preprocess_phase(frame_gray, dog_sigma1, dog_sigma2, clahe_clip, tophat_ksize)
        if use_hough:
            circles = cv2.HoughCircles(
                cv2.GaussianBlur(processed, (7,7), 1.5), cv2.HOUGH_GRADIENT,
                dp=hough_dp, minDist=hough_min_dist,
                param1=hough_param1, param2=hough_param2,
                minRadius=hough_min_radius, maxRadius=hough_max_radius)
            if circles is not None:
                return np.round(circles[0]).astype(np.float32)[:, :2]
            return np.empty((0, 2), dtype=np.float32)
        img = processed
    elif image_mode == "amplitude":
        img = _preprocess_amplitude(frame_gray, clahe_clip)
    elif image_mode == "hologram":
        img = (_preprocess_phase(frame_gray, dog_sigma1, dog_sigma2, clahe_clip, tophat_ksize)
               if use_hough else _preprocess_standard(frame_gray, filter_type))
    else:  # standard / brightfield
        img = _preprocess_standard(frame_gray, filter_type)

    bp = cv2.SimpleBlobDetector_Params()
    bp.filterByArea        = True; bp.minArea        = minArea; bp.maxArea = maxArea
    bp.filterByColor       = filterByColor;  bp.blobColor = blobColor
    bp.filterByCircularity = filterByCircularity
    if filterByCircularity: bp.minCircularity = minCircularity
    bp.filterByInertia     = filterByInertia
    bp.filterByConvexity   = filterByConvexity

    kps = cv2.SimpleBlobDetector_create(bp).detect(img)
    return (np.array([[k.pt[0], k.pt[1]] for k in kps], dtype=np.float32)
            if kps else np.empty((0, 2), dtype=np.float32))


# ══════════════════════════════════════════════════════════════════
#  TRACKING CORE
# ══════════════════════════════════════════════════════════════════

def _tracking_loop(frames_iter, detect_fn, total,
                   P_init, Q_val, R_val,
                   max_dist, max_skips, min_track_length,
                   frame_callback=None):
    """
    Pure tracking core: iterates over BGR frames and a detection function.
    Returns (det_pos, trajs).

    frame_callback(frame_bgr, mask, trajs, fi, total, n_active) -> bool
        The GUI passes its preview function here and checks the stop flag.
        Returning True stops tracking.
    """
    NAN = np.array([np.nan, np.nan])

    first = next(frames_iter)
    p0    = detect_fn(cv2.cvtColor(first, cv2.COLOR_BGR2GRAY))
    if len(p0) == 0:
        raise ValueError(
            "No particles were detected in the first frame.\n"
            "  -> Brightfield: image_mode='standard',  blobColor=0\n"
            "  -> Amplitude:   image_mode='amplitude', blobColor=255\n"
            "  -> Hologram:    image_mode='hologram',  blobColor=255\n"
            "  -> Phase:       image_mode='phase',     blobColor=255\n"
            "  Check minArea/maxArea and the video type."
        )
    print(f"[Frame 0] {len(p0)} particles.")

    kfs     = [KalmanFilter2D(x, y, P_init, Q_val, R_val) for x, y in p0]
    det_pos = [[pt.copy()] for pt in p0]
    trajs   = [[kf.state[:2].copy()] for kf in kfs]
    skips   = [0] * len(kfs)
    done_dp, done_tr = [], []

    def _commit(i):
        if sum(1 for pt in det_pos[i] if not np.isnan(pt[0])) >= min_track_length:
            done_dp.append([pt.copy() for pt in det_pos[i]])
            done_tr.append([pt.copy() for pt in trajs[i]])

    mask = np.zeros_like(first)
    if frame_callback and frame_callback(first, mask, trajs, 0, total, len(kfs)):
        return done_dp, done_tr

    for fi, frame in enumerate(frames_iter, start=1):
        p1    = detect_fn(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        preds = np.array([kf.predict() for kf in kfs])

        if len(p1) == 0:
            for i in range(len(kfs)):
                trajs[i].append(kfs[i].state[:2].copy())
                det_pos[i].append(NAN.copy())
                skips[i] += 1
        else:
            D = np.linalg.norm(preds[:, None, :] - p1[None, :, :], axis=2)
            ri, ci = linear_sum_assignment(D)
            ap, ad = set(), set()
            for r, c in zip(ri, ci):
                if D[r, c] < max_dist:
                    kfs[r].update(p1[c]); ap.add(r); ad.add(c)
                    det_pos[r].append(p1[c].copy()); skips[r] = 0
                else:
                    det_pos[r].append(NAN.copy()); skips[r] += 1
            for i in range(len(kfs)):
                if i not in ap:
                    det_pos[i].append(NAN.copy()); skips[i] += 1
            for i in range(len(p1)):
                if i not in ad:
                    kfs.append(KalmanFilter2D(p1[i][0], p1[i][1], P_init, Q_val, R_val))
                    trajs.append([p1[i].copy()])
                    det_pos.append([p1[i].copy()])
                    skips.append(0)

        for i, kf in enumerate(kfs):
            trajs[i].append(kf.state[:2].copy())

        for i in reversed(range(len(kfs))):
            if skips[i] > max_skips:
                _commit(i)
                del kfs[i], trajs[i], det_pos[i], skips[i]

        if frame_callback and frame_callback(frame, mask, trajs, fi, total, len(kfs)):
            break

    for i in range(len(kfs)):
        _commit(i)

    print(f"Total tracks: {len(done_dp)}")
    return done_dp, done_tr


def _video_iter(cap):
    """Generator of BGR frames from an open VideoCapture."""
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame


# ══════════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════════

def kalman_tracking_video(
    video_path, pixel_size,
    minArea, maxArea, blobColor, filter_type,
    filterByCircularity, minCircularity,
    filterByInertia, filterByConvexity, filterByColor,
    max_dist, max_skips, P_init, Q_val, R_val,
    scale=0.7, show_window=True, show_plot=True,
    save_video_path=None, image_mode="standard",
    use_hough=False, dog_sigma1=2.0, dog_sigma2=8.0,
    clahe_clip=3.0, tophat_ksize=21,
    hough_dp=1.5, hough_min_dist=25, hough_param1=40,
    hough_param2=18, hough_min_radius=8, hough_max_radius=30,
    min_track_length=3,
    save_csv=False, csv_mode="single", csv_path="trajectories.csv",
    plot_style="dots",
):
    """Kalman tracking on a video. Visualization in an OpenCV window."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 15.0
    w_cap = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_cap = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    det_kw = dict(image_mode=image_mode, dog_sigma1=dog_sigma1, dog_sigma2=dog_sigma2,
                  clahe_clip=clahe_clip, tophat_ksize=tophat_ksize, use_hough=use_hough,
                  hough_dp=hough_dp, hough_min_dist=hough_min_dist,
                  hough_param1=hough_param1, hough_param2=hough_param2,
                  hough_min_radius=hough_min_radius, hough_max_radius=hough_max_radius)

    def detect(gray):
        return detectar_centroides(gray, minArea, maxArea, blobColor, filter_type,
                                   filterByCircularity, minCircularity,
                                   filterByInertia, filterByConvexity, filterByColor,
                                   **det_kw)

    writer   = None
    WIN_NAME = "Kalman Tracking  [ESC = exit]"

    if show_window:
        cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_NAME, int(w_cap*scale), int(h_cap*scale))

    def on_frame(frame, mask, trajs, fi, total, n_active):
        nonlocal writer
        if fi == 0 and save_video_path:
            writer = cv2.VideoWriter(save_video_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                     fps, (int(w_cap*scale), int(h_cap*scale)))
        if not show_window and not save_video_path:
            return False

        disp = frame.copy()
        for t in trajs:
            if len(t) > 1:
                pts = np.array(t[-2:], dtype=int)
                cv2.line(mask, tuple(pts[0]), tuple(pts[1]), (0, 255, 0), 2)
                cv2.circle(disp, tuple(pts[1]), 4, (0, 0, 255), -1)
        cv2.putText(disp, f"Tracks: {n_active}  Frame: {fi}/{total}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        out = cv2.resize(cv2.add(disp, mask), None, fx=scale, fy=scale)
        if writer:
            writer.write(out)
        if show_window:
            cv2.imshow(WIN_NAME, out)
            if cv2.waitKey(30) & 0xFF == 27:
                return True  # ESC -> stop
        return False

    try:
        det_pos, trajs = _tracking_loop(
            _video_iter(cap), detect, total,
            P_init, Q_val, R_val, max_dist, max_skips, min_track_length,
            frame_callback=on_frame)
    finally:
        cap.release()
        if writer:
            writer.release()
            print(f"Saved video: {save_video_path}")
        if show_window:
            cv2.destroyAllWindows()
            cv2.waitKey(1)

    if show_plot:
        _plot(det_pos, trajs, pixel_size, style=plot_style)
    if save_csv:
        _save_csv(det_pos, pixel_size, csv_mode, csv_path)
    return det_pos, trajs


# ══════════════════════════════════════════════════════════════════
#  PLOTS
# ══════════════════════════════════════════════════════════════════

def _get_colors(n):
    maps = ['tab20', 'tab20b', 'tab20c', 'Set1', 'Set2', 'Set3']
    if n <= 20:
        return [matplotlib.colormaps['tab20'](i / 20) for i in range(n)]
    return [matplotlib.colormaps[maps[(i // 20) % len(maps)]]((i % 20) / 20) for i in range(n)]


def _plot(det_pos, trajs, pixel_size, style='dots'):
    """
    style:
      'dots'    -> line + small dots (ground truth style)
      'markers' -> line with circle at start and square at end
    """
    n = len(det_pos)
    if n == 0:
        print("No tracks to plot.")
        return
    colors = _get_colors(n)

    fig, ax = plt.subplots(figsize=(9, 7))
    for i, traj in enumerate(det_pos):
        t     = np.array(traj, dtype=float) * pixel_size
        valid = t[~np.isnan(t[:, 0])]
        if len(valid) == 0:
            continue
        ax.plot(t[:, 0], t[:, 1], '-', lw=1.5, alpha=0.85, color=colors[i])
        if style == 'dots':
            ax.plot(t[:, 0], t[:, 1], '.', ms=3, alpha=0.6, color=colors[i])
        else:
            ax.plot(valid[0,  0], valid[0,  1], 'o', ms=7, color=colors[i], zorder=5)
            ax.plot(valid[-1, 0], valid[-1, 1], 's', ms=7, color=colors[i], zorder=5)
    ax.invert_yaxis()
    ax.set_xlabel('X (µm)'); ax.set_ylabel('Y (µm)')
    ax.set_title('Trajectories'); ax.grid(True, alpha=0.4)
    plt.tight_layout(); plt.show(block=False)

    if n <= 40:
        fig2, ax2 = plt.subplots(figsize=(4, max(n * 0.3 + 1, 2)))
        ax2.axis('off')
        for i in range(n):
            ax2.plot([], [], 'o', color=colors[i], label=f'Track {i}')
        ax2.legend(loc='center left', frameon=True)
        plt.title("Legend"); plt.tight_layout(); plt.show(block=False)

    fig3, ax3 = plt.subplots(figsize=(10, 6))
    for i, traj in enumerate(trajs):
        t = np.array(traj, dtype=float) * pixel_size
        if len(t) < 2:
            continue
        speed = np.linalg.norm(np.diff(t, axis=0), axis=1)
        ax3.plot(np.arange(1, len(speed) + 1), speed, color=colors[i], label=f'Track {i}')
    ax3.set_xlabel('Frame'); ax3.set_ylabel('Speed (µm/frame)')
    ax3.set_title('Speed Profiles'); ax3.grid(True)
    if n <= 20:
        ax3.legend(loc='upper right', bbox_to_anchor=(1.15, 1.0))
    plt.tight_layout(); plt.show()


# ══════════════════════════════════════════════════════════════════
#  EXPORTAR CSV
# ══════════════════════════════════════════════════════════════════

def _save_csv(det_pos, pixel_size, csv_mode, csv_path):
    """
    csv_mode:
      'single'    -> un archivo. Columnas: track_id, frame, x_px, y_px, x_um, y_um
      'per_track' -> un CSV por track en la carpeta csv_path.
    """
    if csv_mode == 'single':
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['track_id', 'frame', 'x_px', 'y_px', 'x_um', 'y_um'])
            for tid, positions in enumerate(det_pos):
                for frame_i, pt in enumerate(positions):
                    if np.isnan(pt[0]):
                        w.writerow([tid, frame_i, '', '', '', ''])
                    else:
                        w.writerow([tid, frame_i,
                                    f'{pt[0]:.3f}', f'{pt[1]:.3f}',
                                    f'{pt[0]*pixel_size:.4f}', f'{pt[1]*pixel_size:.4f}'])
        print(f"CSV saved to: {csv_path}  ({len(det_pos)} tracks)")

    elif csv_mode == 'per_track':
        os.makedirs(csv_path, exist_ok=True)
        for tid, positions in enumerate(det_pos):
            with open(os.path.join(csv_path, f'track_{tid:03d}.csv'), 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['frame', 'x_px', 'y_px', 'x_um', 'y_um'])
                for frame_i, pt in enumerate(positions):
                    if np.isnan(pt[0]):
                        w.writerow([frame_i, '', '', '', ''])
                    else:
                        w.writerow([frame_i,
                                    f'{pt[0]:.3f}', f'{pt[1]:.3f}',
                                    f'{pt[0]*pixel_size:.4f}', f'{pt[1]*pixel_size:.4f}'])
        print(f"CSVs saved to: {csv_path}  ({len(det_pos)} files)")
    else:
        raise ValueError("csv_mode must be 'single' or 'per_track'")


# ══════════════════════════════════════════════════════════════════
#  MODE CONFIGURATIONS
#  Default values for each video type.
#  Edit here or use the GUI (tracker_gui.py).
# ══════════════════════════════════════════════════════════════════

def run_brightfield():
    """Dark particles on bright background (e.g. erythrocytes, cells)."""
    kalman_tracking_video(
        video_path='video.mp4', pixel_size=3.75/40,
        image_mode='standard', filter_type='bilateral',
        blobColor=0, minArea=400, maxArea=5000,
        filterByCircularity=True, minCircularity=0.3,
        filterByInertia=False, filterByConvexity=False, filterByColor=True,
        max_dist=25, max_skips=5, min_track_length=5,
        P_init=100, Q_val=1, R_val=50,
        scale=0.7, show_window=True, show_plot=True, plot_style='markers',
    )


def run_amplitude():
    """DHM amplitude video. CLAHE normalizes fluctuating contrast."""
    kalman_tracking_video(
        video_path='video.avi', pixel_size=3.75/20,
        image_mode='amplitude', filter_type='bilateral',
        blobColor=255, minArea=100, maxArea=3000,
        filterByCircularity=True, minCircularity=0.5,
        filterByInertia=False, filterByConvexity=False, filterByColor=True,
        clahe_clip=3.0, max_dist=50, max_skips=53, min_track_length=3,
        P_init=100, Q_val=1, R_val=50,
        scale=0.7, show_window=True, show_plot=True, plot_style='markers',
    )


def run_hologram():
    """Hologram video without prior reconstruction."""
    kalman_tracking_video(
        video_path='video.avi', pixel_size=3.75/40,
        image_mode='hologram', filter_type='bilateral',
        blobColor=255, minArea=150, maxArea=2500,
        filterByCircularity=True, minCircularity=0.3,
        filterByInertia=False, filterByConvexity=False, filterByColor=True,
        max_dist=40, max_skips=5, min_track_length=5,
        P_init=100, Q_val=1, R_val=30,
        scale=0.7, show_window=True, show_plot=True, plot_style='markers',
    )


def run_phase():
    """DHM phase video. DoG + Top-hat + CLAHE enhance phase rings."""
    kalman_tracking_video(
        video_path='video.avi', pixel_size=3.75/40,
        image_mode='phase', filter_type='bilateral',
        blobColor=255, minArea=100, maxArea=2500,
        filterByCircularity=True, minCircularity=0.3,
        filterByInertia=False, filterByConvexity=False, filterByColor=True,
        use_hough=False, dog_sigma1=2.0, dog_sigma2=8.0,
        clahe_clip=3.0, tophat_ksize=21,
        max_dist=40, max_skips=8, min_track_length=3,
        P_init=100, Q_val=1, R_val=30,
        scale=0.7, show_window=True, show_plot=True, plot_style='markers',
    )
