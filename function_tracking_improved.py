import cv2
import numpy as np
import os
from scipy.optimize import linear_sum_assignment
import matplotlib
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════════════════════════
#  KALMAN FILTER
# ══════════════════════════════════════════════════════════════════

class KalmanFilter2D:
    def __init__(self, x, y, P_init, Q, R):
        self.state = np.array([x, y, 0, 0], dtype=np.float32)
        self.F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], dtype=np.float32)
        self.H = np.array([[1,0,0,0],[0,1,0,0]], dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * P_init
        self.Q = np.eye(4, dtype=np.float32) * Q
        self.R = np.eye(2, dtype=np.float32) * R

    def predict(self):
        self.state = self.F @ self.state
        self.P     = self.F @ self.P @ self.F.T + self.Q
        return self.state[:2]

    def update(self, measurement):
        z = np.array(measurement, dtype=np.float32)
        y = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
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
    else:
        raise ValueError("filter_type must be 'gaussian' or 'bilateral'")


def _preprocess_phase(frame_gray, dog_sigma1=2.0, dog_sigma2=8.0,
                      clahe_clip=3.0, tophat_ksize=21):
    f      = frame_gray.astype(np.float64)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tophat_ksize, tophat_ksize))
    tophat = cv2.morphologyEx(frame_gray, cv2.MORPH_TOPHAT, kernel).astype(np.float64)
    g1     = cv2.GaussianBlur(f, (0, 0), dog_sigma1)
    g2     = cv2.GaussianBlur(f, (0, 0), dog_sigma2)
    dog    = np.clip(g1 - g2, 0, None)
    combined = cv2.normalize(0.5*tophat + 0.5*dog, None, 0, 255,
                             cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8,8)).apply(combined)


def _preprocess_slm_amp(frame_gray, clahe_clip=3.0):
    """CLAHE + bilateral para videos de amplitud SLM con intensidad fluctuante."""
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8,8))
    return cv2.bilateralFilter(clahe.apply(frame_gray), 9, 75, 75)


# ══════════════════════════════════════════════════════════════════
#  DETECCION DE CENTROIDES (imagen microscopio)
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
        # Phase: DoG + Top-hat + CLAHE para realzar anillos de fase
        processed = _preprocess_phase(frame_gray, dog_sigma1, dog_sigma2,
                                      clahe_clip, tophat_ksize)
        if use_hough:
            blurred = cv2.GaussianBlur(processed, (7,7), 1.5)
            circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT,
                                       dp=hough_dp, minDist=hough_min_dist,
                                       param1=hough_param1, param2=hough_param2,
                                       minRadius=hough_min_radius,
                                       maxRadius=hough_max_radius)
            if circles is not None:
                return np.round(circles[0,:]).astype(np.float32)[:,:2]
            return np.empty((0,2), dtype=np.float32)
        img_to_detect = processed
    elif image_mode in ("amplitude", "slm_amp"):
        # Amplitude: CLAHE + bilateral para contraste fluctuante
        img_to_detect = _preprocess_slm_amp(frame_gray, clahe_clip)
    elif image_mode == "hologram":
        # Hologram: estandar; si use_dog=True aplica mejora de fase
        if use_hough:
            img_to_detect = _preprocess_phase(frame_gray, dog_sigma1, dog_sigma2,
                                              clahe_clip, tophat_ksize)
        else:
            img_to_detect = _preprocess_standard(frame_gray, filter_type)
    else:
        # Brightfield / standard: particulas oscuras sobre fondo claro
        img_to_detect = _preprocess_standard(frame_gray, filter_type)

    p = cv2.SimpleBlobDetector_Params()
    p.filterByArea=True; p.minArea=minArea; p.maxArea=maxArea
    p.filterByColor=filterByColor; p.blobColor=blobColor
    p.filterByCircularity=filterByCircularity
    if filterByCircularity: p.minCircularity=minCircularity
    p.filterByInertia=filterByInertia; p.filterByConvexity=filterByConvexity

    kps = cv2.SimpleBlobDetector_create(p).detect(img_to_detect)
    return np.array([[k.pt[0],k.pt[1]] for k in kps], dtype=np.float32) \
           if kps else np.empty((0,2), dtype=np.float32)


# ══════════════════════════════════════════════════════════════════
#  DETECCION DE TRAMPAS EN VIDEO SLM
# ══════════════════════════════════════════════════════════════════

def _extraer_posiciones_slm(slm_video_path, threshold=50, min_area=50):
    """Extrae centroides de las trampas en cada frame del video SLM."""
    cap = cv2.VideoCapture(slm_video_path)
    if not cap.isOpened():
        raise ValueError(f"No se pudo abrir video SLM: {slm_video_path}")
    positions = []
    while True:
        ret, f = cap.read()
        if not ret: break
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        _, b = cv2.threshold(g, threshold, 255, cv2.THRESH_BINARY)
        n, _, stats, cents = cv2.connectedComponentsWithStats(b)
        pts = [cents[i] for i in range(1,n) if stats[i,cv2.CC_STAT_AREA]>min_area]
        positions.append(np.array(pts, dtype=np.float32) if pts
                        else np.empty((0,2), dtype=np.float32))
    cap.release()
    return positions, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


def _calibrar_transformacion(slm_pts_f0, amp_pts_f0, slm_h, mirror_axis='y'):
    """
    Encuentra la transformacion afin entre coordenadas SLM y camara.

    El SLM y la imagen de la camara estan en espejo debido al camino optico.
    mirror_axis='y' invierte el eje Y del SLM antes de calcular la transformacion
    (flip_y fue el que dio mejor resultado: 11/15 matches con error < 80px).

    Retorna la matriz afin M (2x3) para usar con cv2.transform().
    """
    # Aplicar espejo
    if mirror_axis == 'y':
        slm_m = np.column_stack([slm_pts_f0[:,0], slm_h - slm_pts_f0[:,1]])
    elif mirror_axis == 'x':
        slm_w = slm_pts_f0[:,0].max() + slm_pts_f0[:,0].min()
        slm_m = np.column_stack([slm_w - slm_pts_f0[:,0], slm_pts_f0[:,1]])
    else:
        slm_m = slm_pts_f0.copy()

    # Normalizar y hacer matching por forma del patron
    slm_norm = (slm_m - slm_m.min(0)) / (slm_m.max(0) - slm_m.min(0) + 1e-9)
    amp_norm = (amp_pts_f0 - amp_pts_f0.min(0)) / (amp_pts_f0.max(0) - amp_pts_f0.min(0) + 1e-9)

    D = np.linalg.norm(slm_norm[:,None,:] - amp_norm[None,:,:], axis=2)
    ri, ci = linear_sum_assignment(D)
    good = [(r,c) for r,c in zip(ri,ci) if D[r,c] < 0.12]

    if len(good) < 4:
        raise ValueError(
            f"Solo {len(good)} correspondencias encontradas (necesitas >= 4).\n"
            "Verifica que ambos videos correspondan al mismo experimento."
        )

    src = np.array([slm_m[r] for r,c in good], dtype=np.float32)
    dst = np.array([amp_pts_f0[c] for r,c in good], dtype=np.float32)
    M, inliers = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC,
                                       ransacReprojThreshold=30)
    n_inliers = inliers.sum() if inliers is not None else 0
    print(f"Calibracion: {len(good)} matches, {n_inliers} inliers RANSAC")
    return M, mirror_axis


def _transformar_slm_a_camara(slm_pts, M, mirror_axis, slm_h, cam_w, cam_h):
    """Aplica espejo + transformacion afin y filtra puntos fuera del campo."""
    if len(slm_pts) == 0:
        return np.empty((0,2), dtype=np.float32)
    if mirror_axis == 'y':
        slm_m = np.column_stack([slm_pts[:,0], slm_h - slm_pts[:,1]])
    elif mirror_axis == 'x':
        slm_w = 800
        slm_m = np.column_stack([slm_w - slm_pts[:,0], slm_pts[:,1]])
    else:
        slm_m = slm_pts.copy()

    transformed = cv2.transform(slm_m.reshape(-1,1,2), M).reshape(-1,2)
    valid = ((transformed[:,0] > 0) & (transformed[:,0] < cam_w) &
             (transformed[:,1] > 0) & (transformed[:,1] < cam_h))
    return transformed[valid]


# ══════════════════════════════════════════════════════════════════
#  NUCLEO DE TRACKING (compartido)
# ══════════════════════════════════════════════════════════════════

def _run_tracking(all_detections, pixel_size, max_dist, max_skips,
                  P_init, Q_val, R_val, min_track_length,
                  video_path=None, scale=0.7, show_window=False,
                  save_video_path=None):

    NAN_POS = np.array([np.nan, np.nan])
    cap, writer, mask = None, None, None

    if video_path is not None and show_window:
        cap = cv2.VideoCapture(video_path)
        ret, first = cap.read()
        mask = np.zeros_like(first)
        if save_video_path:
            h, w = first.shape[:2]
            fps  = cap.get(cv2.CAP_PROP_FPS) or 8.0
            writer = cv2.VideoWriter(save_video_path,
                                     cv2.VideoWriter_fourcc(*'mp4v'),
                                     fps, (int(w*scale), int(h*scale)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    p0 = all_detections[0]
    if len(p0) == 0:
        raise ValueError("No se detectaron particulas en el primer frame.")
    print(f"[Frame 0] {len(p0)} particulas.")

    kfs     = [KalmanFilter2D(x, y, P_init, Q_val, R_val) for x,y in p0]
    det_pos = [[p0[i].copy()] for i in range(len(kfs))]
    trajs   = [[kfs[i].state[:2].copy()] for i in range(len(kfs))]
    skips   = [0]*len(kfs)
    completed_det_pos, completed_trajs = [], []

    def _save(idx):
        real = sum(1 for pt in det_pos[idx] if not np.isnan(pt[0]))
        if real >= min_track_length:
            completed_det_pos.append([pt.copy() for pt in det_pos[idx]])
            completed_trajs.append([pt.copy() for pt in trajs[idx]])

    for p1 in all_detections[1:]:
        predictions = np.array([kf.predict() for kf in kfs])

        if len(p1) == 0:
            for i in range(len(kfs)):
                trajs[i].append(kfs[i].state[:2].copy())
                det_pos[i].append(NAN_POS.copy())
                skips[i] += 1
        else:
            D = np.linalg.norm(predictions[:,None,:] - p1[None,:,:], axis=2)
            ri, ci = linear_sum_assignment(D)
            ap, ad = set(), set()
            for r,c in zip(ri,ci):
                if D[r,c] < max_dist:
                    kfs[r].update(p1[c]); ap.add(r); ad.add(c)
                    det_pos[r].append(p1[c].copy()); skips[r]=0
                else:
                    det_pos[r].append(NAN_POS.copy()); skips[r]+=1
            for i in range(len(kfs)):
                if i not in ap:
                    det_pos[i].append(NAN_POS.copy()); skips[i]+=1
            for i in range(len(p1)):
                if i not in ad:
                    nkf = KalmanFilter2D(p1[i][0],p1[i][1],P_init,Q_val,R_val)
                    kfs.append(nkf)
                    trajs.append([p1[i].copy()])
                    det_pos.append([p1[i].copy()])
                    skips.append(0)

        for i,kf in enumerate(kfs):
            trajs[i].append(kf.state[:2].copy())

        for i in reversed(range(len(kfs))):
            if skips[i] > max_skips:
                _save(i)
                del kfs[i], trajs[i], det_pos[i], skips[i]

        if cap and show_window:
            ret, frame = cap.read()
            if ret:
                for t in trajs:
                    if len(t)>1:
                        pts = np.array(t[-2:], dtype=int)
                        cv2.line(mask, tuple(pts[0]), tuple(pts[1]), (0,255,0), 2)
                        cv2.circle(frame, tuple(pts[1]), 3, (0,0,255), -1)
                out = cv2.resize(cv2.add(frame, mask), None, fx=scale, fy=scale)
                if writer: writer.write(out)
                cv2.imshow("Kalman Tracking", out)
                if cv2.waitKey(30) & 0xFF == 27:
                    break

    if cap: cap.release()
    if writer: writer.release(); print(f"Video guardado: {save_video_path}")
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # flush eventos pendientes de Windows

    for i in range(len(kfs)):
        _save(i)

    print(f"Trayectorias totales: {len(completed_det_pos)}")
    return completed_det_pos, completed_trajs


# ══════════════════════════════════════════════════════════════════
#  API PUBLICA
# ══════════════════════════════════════════════════════════════════

def kalman_tracking_video(
    video_path, pixel_size,
    minArea, maxArea, blobColor, filter_type,
    filterByCircularity, minCircularity,
    filterByInertia, filterByConvexity, filterByColor,
    max_dist, max_skips, P_init, Q_val, R_val,
    scale, show_window, show_plot,
    save_video_path=None, image_mode="standard",
    use_hough=False, dog_sigma1=2.0, dog_sigma2=8.0,
    clahe_clip=3.0, tophat_ksize=21,
    hough_dp=1.5, hough_min_dist=25, hough_param1=40,
    hough_param2=18, hough_min_radius=8, hough_max_radius=30,
    min_track_length=3,
    # -- Exportar CSV --
    save_csv   = False,      # True para guardar posiciones en CSV
    csv_mode   = "single",   # "single" -> un archivo | "per_track" -> un archivo por track
    csv_path   = "trajectories.csv",  # ruta del CSV (o carpeta si csv_mode="per_track")
    plot_style = "dots",     # "dots" -> estilo ground truth | "markers" -> circulo/cuadrado
):
    """Tracking en tiempo real: deteccion + Kalman + visualizacion frame a frame."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"No se pudo abrir: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 15.0
    w_cap = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_cap = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    det_kwargs = dict(
        image_mode=image_mode, dog_sigma1=dog_sigma1, dog_sigma2=dog_sigma2,
        clahe_clip=clahe_clip, tophat_ksize=tophat_ksize, use_hough=use_hough,
        hough_dp=hough_dp, hough_min_dist=hough_min_dist,
        hough_param1=hough_param1, hough_param2=hough_param2,
        hough_min_radius=hough_min_radius, hough_max_radius=hough_max_radius,
    )

    # ── Leer primer frame para inicializar ──────────────────────────────────
    ret, first = cap.read()
    if not ret:
        raise ValueError("No se pudo leer el video.")
    first_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    p0 = detectar_centroides(first_gray, minArea, maxArea, blobColor,
                             filter_type, filterByCircularity, minCircularity,
                             filterByInertia, filterByConvexity, filterByColor,
                             **det_kwargs)
    if len(p0) == 0:
        cap.release()
        raise ValueError(
            "No se detectaron particulas en el primer frame.\n"
            "  -> Brightfield: image_mode='standard',  blobColor=0\n"
            "  -> Amplitude:   image_mode='amplitude', blobColor=255\n"
            "  -> Hologram:    image_mode='hologram',  blobColor=255\n"
            "  -> Phase:       image_mode='phase',     blobColor=255\n"
            "  Verifica tambien minArea/maxArea y el tipo de video."
        )
    print(f"[Frame 0] {len(p0)} particulas.")

    # ── Inicializar Kalman ───────────────────────────────────────────────────
    NAN_POS  = np.array([np.nan, np.nan])
    kfs      = [KalmanFilter2D(x, y, P_init, Q_val, R_val) for x,y in p0]
    det_pos  = [[p0[i].copy()] for i in range(len(kfs))]
    trajs    = [[kfs[i].state[:2].copy()] for i in range(len(kfs))]
    skips    = [0]*len(kfs)
    next_id  = len(kfs)
    completed_det_pos, completed_trajs = [], []

    def _save_track(idx):
        real = sum(1 for pt in det_pos[idx] if not np.isnan(pt[0]))
        if real >= min_track_length:
            completed_det_pos.append([pt.copy() for pt in det_pos[idx]])
            completed_trajs.append([pt.copy() for pt in trajs[idx]])

    # ── Video writer ─────────────────────────────────────────────────────────
    writer = None
    if save_video_path:
        writer = cv2.VideoWriter(save_video_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                 fps, (int(w_cap*scale), int(h_cap*scale)))

    # ── Mostrar primer frame ─────────────────────────────────────────────────
    mask = np.zeros_like(first)
    WIN_NAME = "Kalman Tracking  [ESC = salir]"
    if show_window:
        cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)  # WINDOW_NORMAL permite redimensionar
        cv2.resizeWindow(WIN_NAME, int(w_cap*scale), int(h_cap*scale))
        disp = first.copy()
        for pt in p0:
            cv2.circle(disp, (int(pt[0]),int(pt[1])), 4, (0,255,0), -1)
        out = cv2.resize(cv2.add(disp, mask), None, fx=scale, fy=scale)
        cv2.imshow(WIN_NAME, out)
        if writer: writer.write(out)
        cv2.waitKey(30)  # 30ms da tiempo al sistema para renderizar

    # ── Loop principal ───────────────────────────────────────────────────────
    stop = False
    for fi in range(1, total):
        ret, frame = cap.read()
        if not ret: break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        p1   = detectar_centroides(gray, minArea, maxArea, blobColor,
                                   filter_type, filterByCircularity, minCircularity,
                                   filterByInertia, filterByConvexity, filterByColor,
                                   **det_kwargs)

        # Prediccion Kalman
        predictions = np.array([kf.predict() for kf in kfs])

        if len(p1) == 0:
            for i in range(len(kfs)):
                trajs[i].append(kfs[i].state[:2].copy())
                det_pos[i].append(NAN_POS.copy())
                skips[i] += 1
        else:
            D = np.linalg.norm(predictions[:,None,:] - p1[None,:,:], axis=2)
            from scipy.optimize import linear_sum_assignment
            ri, ci = linear_sum_assignment(D)
            ap, ad = set(), set()
            for r,c in zip(ri,ci):
                if D[r,c] < max_dist:
                    kfs[r].update(p1[c]); ap.add(r); ad.add(c)
                    det_pos[r].append(p1[c].copy()); skips[r]=0
                else:
                    det_pos[r].append(NAN_POS.copy()); skips[r]+=1
            for i in range(len(kfs)):
                if i not in ap:
                    det_pos[i].append(NAN_POS.copy()); skips[i]+=1
            for i in range(len(p1)):
                if i not in ad:
                    nkf = KalmanFilter2D(p1[i][0],p1[i][1],P_init,Q_val,R_val)
                    kfs.append(nkf); trajs.append([p1[i].copy()])
                    det_pos.append([p1[i].copy()]); skips.append(0)

        for i,kf in enumerate(kfs):
            trajs[i].append(kf.state[:2].copy())

        # Eliminar tracks muertos
        for i in reversed(range(len(kfs))):
            if skips[i] > max_skips:
                _save_track(i)
                del kfs[i], trajs[i], det_pos[i], skips[i]

        # Visualizacion en tiempo real
        if show_window:
            disp = frame.copy()
            for t in trajs:
                if len(t) > 1:
                    pts = np.array(t[-2:], dtype=int)
                    cv2.line(mask, tuple(pts[0]), tuple(pts[1]), (0,255,0), 2)
                    cv2.circle(disp, tuple(pts[1]), 4, (0,0,255), -1)
            # Mostrar numero de tracks activos
            cv2.putText(disp, f"Tracks: {len(kfs)}  Frame: {fi}/{total}",
                        (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
            out = cv2.resize(cv2.add(disp, mask), None, fx=scale, fy=scale)
            cv2.imshow(WIN_NAME, out)
            if writer: writer.write(out)
            key = cv2.waitKey(30) & 0xFF  # 30ms mantiene la ventana activa
            if key == 27:  # ESC
                stop = True; break

    cap.release()
    if writer: writer.release(); print(f"Video guardado: {save_video_path}")
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # flush eventos pendientes de Windows

    # Guardar tracks activos al finalizar
    for i in range(len(kfs)):
        _save_track(i)

    print(f"Trayectorias totales: {len(completed_det_pos)}")
    if show_plot:
        _plot(completed_det_pos, completed_trajs, pixel_size, style=plot_style)
    if save_csv:
        _save_csv(completed_det_pos, pixel_size, csv_mode, csv_path)
    return completed_det_pos, completed_trajs


def slm_guided_tracking(
    amplitude_video_path,
    slm_video_path,
    # Escala camara
    cam_pixel_um      = 3.75,
    cam_magnification = 20,
    # FPS de cada video (para sincronizar)
    amp_fps           = 8.0,
    slm_fps           = 15.0,
    # Deteccion en video de amplitud
    minArea           = 100,
    maxArea           = 3000,
    blobColor         = 255,
    minCircularity    = 0.5,
    clahe_clip        = 3.0,
    # Tracking
    max_dist          = 20,    # px camara — estricto para evitar swaps entre particulas cercanas
    max_skips         = 53,    # = total frames -> track nunca muere durante el video
    P_init=100, Q_val=1, R_val=50,
    min_track_length  = 3,
    # SLM
    slm_threshold     = 50,
    slm_min_area      = 50,
    mirror_axis       = 'y',   # eje de espejo SLM vs camara ('x', 'y', o None)
    # Salida
    show_plot         = True,
    show_window       = False,
    scale             = 0.7,
    save_video_path   = None,
    # -- Exportar CSV --
    save_csv          = False,     # True para guardar posiciones en CSV
    csv_mode          = "single",  # "single" -> un archivo | "per_track" -> un archivo por track
    csv_path          = "trajectories.csv",  # ruta del CSV (o carpeta si csv_mode="per_track")
    plot_style        = "dots",    # "dots" -> estilo ground truth | "markers" -> circulo/cuadrado
    # -- Modo ground truth --
    use_all_slm_frames = False,    # True -> usa todos los frames del SLM (ground truth completo)
                                   # False -> solo los frames sincronizados con la camara
):
    """
    Tracking guiado por SLM sobre el video de amplitud.

    Estrategia:
      - Cada frame de amplitud se sincroniza con el frame SLM correspondiente
        usando la relacion de FPS (amp_fps / slm_fps).
      - Las posiciones de las trampas del SLM se transforman al sistema de
        coordenadas de la camara usando una transformacion afin calibrada
        automaticamente en el frame 0 (incluyendo la correccion de espejo).
      - El Kalman recibe como deteccion:
          * La posicion de amplitud si hay match con la trampa SLM (< max_dist)
          * La posicion SLM transformada cuando la amplitud no detecta nada
        Esto permite mantener el track aunque la particula desaparezca en la imagen.

    Parametros clave
    ----------------
    mirror_axis   : eje de espejo entre SLM y camara.
                    'y' (por defecto) — encontrado empiricamente para este setup.
    max_dist      : distancia maxima (px camara) para asociar detecciones.
                    50px es un buen balance — suficiente para cubrir el error
                    de la transformacion afin pero estricto para no saltar
                    entre particulas vecinas.
    max_skips     : igual al total de frames del video (53) para que ningun
                    track se elimine durante la grabacion. Los tracks que
                    realmente son ruido se filtran con min_track_length.
    R_val         : 50 — balance entre seguir detecciones y suavizar Kalman.
    """
    pixel_size = cam_pixel_um / cam_magnification
    print(f"Pixel size camara: {pixel_size:.4f} um/px")

    # 1. Extraer posiciones SLM
    print("Extrayendo posiciones del SLM...")
    slm_positions, slm_h = _extraer_posiciones_slm(
        slm_video_path, slm_threshold, slm_min_area)
    print(f"SLM: {len(slm_positions)} frames, {len(slm_positions[0])} trampas en frame 0")

    # 1b. Si use_all_slm_frames=True, hacer tracking directo en SLM sin camara
    if use_all_slm_frames:
        print("Modo ground truth: usando todos los frames del SLM...")
        tracks_slm = {}
        current_slm = {i: slm_positions[0][i].copy() for i in range(len(slm_positions[0]))}
        next_id_slm = len(slm_positions[0])
        for i in range(len(slm_positions[0])):
            tracks_slm[i] = [slm_positions[0][i].copy()]

        for fi in range(1, len(slm_positions)):
            pts = slm_positions[fi]
            if len(pts) == 0: continue
            active_ids = list(current_slm.keys())
            active_pts = np.array([current_slm[i] for i in active_ids])
            D_slm = np.linalg.norm(active_pts[:,None,:] - pts[None,:,:], axis=2)
            ri_s, ci_s = linear_sum_assignment(D_slm)
            matched_a_s, matched_c_s = set(), set()
            for r,c in zip(ri_s, ci_s):
                if D_slm[r,c] < 15:
                    tid = active_ids[r]
                    tracks_slm[tid].append(pts[c].copy())
                    current_slm[tid] = pts[c].copy()
                    matched_a_s.add(r); matched_c_s.add(c)
            for r,tid in enumerate(active_ids):
                if r not in matched_a_s: del current_slm[tid]
            for c in range(len(pts)):
                if c not in matched_c_s:
                    tracks_slm[next_id_slm] = [pts[c].copy()]
                    current_slm[next_id_slm] = pts[c].copy()
                    next_id_slm += 1

        slm_pixel_size = slm_pixel_um / (demag_lens / demag_MO)
        det_pos_slm = [[pt.copy() for pt in pts]
                       for pts in tracks_slm.values() if len(pts) >= min_track_length]
        trajs_slm   = det_pos_slm  # same positions, no Kalman smoothing needed
        print(f"Ground truth: {len(det_pos_slm)} trayectorias")
        if show_plot:
            _plot(det_pos_slm, trajs_slm, slm_pixel_size, style=plot_style)
        if save_csv:
            _save_csv(det_pos_slm, slm_pixel_size, csv_mode, csv_path)
        return det_pos_slm, trajs_slm

    # 2. Cargar frames de amplitud y detectar
    print("Detectando en video de amplitud...")
    cap = cv2.VideoCapture(amplitude_video_path)
    if not cap.isOpened():
        raise ValueError(f"No se pudo abrir: {amplitude_video_path}")
    amp_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cam_w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    amp_detections = []
    for _ in range(amp_total):
        ret, f = cap.read()
        if not ret:
            amp_detections.append(np.empty((0,2), dtype=np.float32))
            continue
        g = _preprocess_slm_amp(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), clahe_clip)
        p = cv2.SimpleBlobDetector_Params()
        p.filterByArea=True; p.minArea=minArea; p.maxArea=maxArea
        p.filterByColor=True; p.blobColor=blobColor
        p.filterByCircularity=True; p.minCircularity=minCircularity
        p.filterByInertia=False; p.filterByConvexity=False
        kps = cv2.SimpleBlobDetector_create(p).detect(g)
        pts = np.array([[k.pt[0],k.pt[1]] for k in kps],dtype=np.float32) \
              if kps else np.empty((0,2),dtype=np.float32)
        amp_detections.append(pts)
    cap.release()

    # 3. Calibrar transformacion SLM -> camara usando frame 0
    print("Calibrando transformacion SLM -> camara...")
    if len(amp_detections[0]) < 4 or len(slm_positions[0]) < 4:
        raise ValueError(
            "Necesitas al menos 4 particulas detectadas en el frame 0 de ambos videos."
        )
    M, mirror = _calibrar_transformacion(
        slm_positions[0], amp_detections[0], slm_h, mirror_axis)

    # 4. Construir lista de detecciones combinadas (SLM + amplitud)
    print("Combinando detecciones SLM y amplitud...")
    amp_to_slm = lambda i: min(round(i * slm_fps / amp_fps), len(slm_positions)-1)

    combined_detections = []
    for i in range(amp_total):
        slm_idx  = amp_to_slm(i)
        slm_cam  = _transformar_slm_a_camara(
            slm_positions[slm_idx], M, mirror, slm_h, cam_w, cam_h)
        amp_dets = amp_detections[i]

        if len(slm_cam) == 0:
            # Sin guia SLM -> usar solo amplitud
            combined_detections.append(amp_dets)
            continue

        if len(amp_dets) == 0:
            # Sin deteccion en amplitud -> usar SLM como deteccion
            combined_detections.append(slm_cam)
            continue

        # Hay ambas: para cada trampa SLM, buscar deteccion de amplitud cercana
        # Si hay match -> usar posicion de amplitud (mas precisa localmente)
        # Si no hay match -> usar posicion SLM (trampa optica es ground truth)
        D = np.linalg.norm(slm_cam[:,None,:] - amp_dets[None,:,:], axis=2)
        best_combined = []
        used_amp = set()
        for s in range(len(slm_cam)):
            dists = D[s]
            best_a = np.argmin(dists)
            if dists[best_a] < max_dist and best_a not in used_amp:
                # Match con amplitud -> usar posicion de amplitud
                best_combined.append(amp_dets[best_a])
                used_amp.add(best_a)
            else:
                # Sin match -> usar posicion SLM
                best_combined.append(slm_cam[s])

        combined_detections.append(
            np.array(best_combined, dtype=np.float32) if best_combined
            else np.empty((0,2), dtype=np.float32)
        )

    # 5. Tracking con Kalman sobre detecciones combinadas
    print("Ejecutando tracking Kalman...")
    det_pos, trajs = _run_tracking(
        combined_detections, pixel_size, max_dist, max_skips,
        P_init, Q_val, R_val, min_track_length,
        video_path=amplitude_video_path if show_window else None,
        scale=scale, show_window=show_window,
        save_video_path=save_video_path,
    )

    if show_plot:
        _plot(det_pos, trajs, pixel_size, style=plot_style)
    if save_csv:
        _save_csv(det_pos, pixel_size, csv_mode, csv_path)

    return det_pos, trajs


# ══════════════════════════════════════════════════════════════════
#  PLOTS
# ══════════════════════════════════════════════════════════════════

def _get_colors(n):
    if n <= 20:
        return [matplotlib.colormaps['tab20'](i/20) for i in range(n)]
    colors = []
    maps = ['tab20','tab20b','tab20c','Set1','Set2','Set3']
    for i in range(n):
        cm = matplotlib.colormaps[maps[(i//20) % len(maps)]]
        colors.append(cm((i%20)/20))
    return colors



def _save_csv(det_pos, pixel_size, csv_mode, csv_path):
    """
    Guarda las trayectorias en formato CSV.

    csv_mode:
      'single'    -> un archivo con todas las trayectorias.
                     Columnas: track_id, frame, x_px, y_px, x_um, y_um
      'per_track' -> un archivo CSV por track.
                     Guardados en: <csv_path>/track_000.csv, track_001.csv, ...
    """
    import csv, os

    if csv_mode == 'single':
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['track_id', 'frame', 'x_px', 'y_px', 'x_um', 'y_um'])
            for tid, positions in enumerate(det_pos):
                for frame, pt in enumerate(positions):
                    if np.isnan(pt[0]):
                        w.writerow([tid, frame, '', '', '', ''])
                    else:
                        w.writerow([tid, frame,
                                    f'{pt[0]:.3f}', f'{pt[1]:.3f}',
                                    f'{pt[0]*pixel_size:.4f}', f'{pt[1]*pixel_size:.4f}'])
        print(f"CSV guardado en: {csv_path}  ({len(det_pos)} tracks)")

    elif csv_mode == 'per_track':
        os.makedirs(csv_path, exist_ok=True)
        for tid, positions in enumerate(det_pos):
            with open(os.path.join(csv_path, f'track_{tid:03d}.csv'), 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['frame', 'x_px', 'y_px', 'x_um', 'y_um'])
                for frame, pt in enumerate(positions):
                    if np.isnan(pt[0]):
                        w.writerow([frame, '', '', '', ''])
                    else:
                        w.writerow([frame,
                                    f'{pt[0]:.3f}', f'{pt[1]:.3f}',
                                    f'{pt[0]*pixel_size:.4f}', f'{pt[1]*pixel_size:.4f}'])
        print(f"CSVs guardados en: {csv_path}  ({len(det_pos)} archivos)")

    else:
        raise ValueError("csv_mode debe ser 'single' o 'per_track'")


def _plot(det_pos, trajs, pixel_size, style='dots'):
    """
    style opciones:
      'dots'    -> linea + puntos pequenos a lo largo (estilo ground truth / referencia)
      'markers' -> linea con circulo al inicio y cuadrado al final
    """
    n = len(det_pos)
    if n == 0:
        print("No hay trayectorias para graficar.")
        return
    colors = _get_colors(n)

    fig, ax = plt.subplots(figsize=(9, 7))
    for i in range(n):
        t = np.array(det_pos[i], dtype=float) * pixel_size
        valid = t[~np.isnan(t[:,0])]
        if len(valid) == 0:
            continue
        if style == 'dots':
            # Linea + puntos pequenos a lo largo (igual que el ground truth)
            ax.plot(t[:,0], t[:,1], '-', linewidth=1.5, alpha=0.85, color=colors[i])
            ax.plot(t[:,0], t[:,1], '.', markersize=3, alpha=0.6, color=colors[i])
        else:
            # Linea con circulo inicio y cuadrado final
            ax.plot(t[:,0], t[:,1], '-', linewidth=1.5, alpha=0.85, color=colors[i])
            ax.plot(valid[0,0],  valid[0,1],  'o', ms=7, color=colors[i], zorder=5)
            ax.plot(valid[-1,0], valid[-1,1], 's', ms=7, color=colors[i], zorder=5)
    ax.invert_yaxis()
    ax.set_xlabel('X (um)'); ax.set_ylabel('Y (um)')
    ax.set_title('Trajectories'); ax.grid(True, alpha=0.4)
    plt.tight_layout(); plt.show(block=False)

    if n <= 40:
        fig2, ax2 = plt.subplots(figsize=(4, max(n*0.3+1, 2)))
        ax2.axis('off')
        for i in range(n):
            ax2.plot([], [], marker='o', color=colors[i], label=f'Track {i}')
        ax2.legend(loc='center left', frameon=True)
        plt.title("Legend"); plt.tight_layout(); plt.show(block=False)

    fig3, ax3 = plt.subplots(figsize=(10, 6))
    for i in range(n):
        t = np.array(trajs[i], dtype=float) * pixel_size
        if len(t) < 2: continue
        speed = np.linalg.norm(np.diff(t, axis=0), axis=1)
        ax3.plot(np.arange(1, len(speed)+1), speed, color=colors[i], label=f'Track {i}')
    ax3.set_xlabel('Frame'); ax3.set_ylabel('Speed (um/frame)')
    ax3.set_title('Speed Profiles'); ax3.grid(True)
    if n <= 20:
        ax3.legend(loc='upper right', bbox_to_anchor=(1.15, 1.0))
    plt.tight_layout(); plt.show()


# ══════════════════════════════════════════════════════════════════
#  CONFIGURACIONES
# ══════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════
#  CONFIGURACIONES POR MODO
#  Valores por defecto para cada tipo de video.
#  Modifica aqui o usa la GUI (tracker_gui.py).
# ══════════════════════════════════════════════════════════════

def run_brightfield():
    """
    Modo: BRIGHTFIELD
    Particulas oscuras sobre fondo claro (ej: eritrocitos, celulas).
    blobColor=0 detecta objetos mas oscuros que el fondo.
    """
    kalman_tracking_video(
        video_path          = r'video.mp4',
        pixel_size          = 3.75 / 40,
        image_mode          = 'standard',
        filter_type         = 'bilateral',
        blobColor           = 0,
        minArea             = 400,
        maxArea             = 5000,
        filterByCircularity = True,
        minCircularity      = 0.3,
        filterByInertia     = False,
        filterByConvexity   = False,
        filterByColor       = True,
        max_dist            = 25,
        max_skips           = 5,
        min_track_length    = 5,
        P_init=100, Q_val=1, R_val=50,
        scale=0.7, show_window=True, show_plot=True,
        plot_style          = 'markers',
        # save_csv=True, csv_mode='single', csv_path=r'trajectories.csv',
    )


def run_amplitude():
    """
    Modo: AMPLITUDE
    Video de amplitud reconstruido desde hologramas DHM.
    Particulas brillantes sobre fondo variable.
    Se aplica CLAHE para normalizar el contraste fluctuante entre frames.
    """
    kalman_tracking_video(
        video_path          = r'video.avi',
        pixel_size          = 3.75 / 20,
        image_mode          = 'amplitude',
        filter_type         = 'bilateral',
        blobColor           = 255,
        minArea             = 100,
        maxArea             = 3000,
        filterByCircularity = True,
        minCircularity      = 0.5,
        filterByInertia     = False,
        filterByConvexity   = False,
        filterByColor       = True,
        clahe_clip          = 3.0,
        max_dist            = 50,
        max_skips           = 53,
        min_track_length    = 3,
        P_init=100, Q_val=1, R_val=50,
        scale=0.7, show_window=True, show_plot=True,
        plot_style          = 'markers',
        # save_csv=True, csv_mode='single', csv_path=r'trajectories.csv',
    )


def run_hologram():
    """
    Modo: HOLOGRAM
    Video holografico sin reconstruccion previa.
    NOTA: en el futuro este modo incluira reconstruccion en tiempo real
    mediante espectro angular antes de la deteccion.
    """
    kalman_tracking_video(
        video_path          = r'video.avi',
        pixel_size          = 3.75 / 40,
        image_mode          = 'hologram',
        filter_type         = 'bilateral',
        blobColor           = 255,
        minArea             = 150,
        maxArea             = 2500,
        filterByCircularity = True,
        minCircularity      = 0.3,
        filterByInertia     = False,
        filterByConvexity   = False,
        filterByColor       = True,
        max_dist            = 40,
        max_skips           = 5,
        min_track_length    = 5,
        P_init=100, Q_val=1, R_val=30,
        scale=0.7, show_window=True, show_plot=True,
        plot_style          = 'markers',
        # save_csv=True, csv_mode='single', csv_path=r'trajectories.csv',
    )


def run_phase():
    """
    Modo: PHASE
    Video de fase reconstruido desde hologramas DHM.
    Aplica DoG + Top-hat + CLAHE para realzar los anillos de fase
    caracteristicos de las particulas en microscopia holografica.
    """
    kalman_tracking_video(
        video_path          = r'video.avi',
        pixel_size          = 3.75 / 40,
        image_mode          = 'phase',
        filter_type         = 'bilateral',
        blobColor           = 255,
        minArea             = 100,
        maxArea             = 2500,
        filterByCircularity = True,
        minCircularity      = 0.3,
        filterByInertia     = False,
        filterByConvexity   = False,
        filterByColor       = True,
        use_hough           = False,
        dog_sigma1          = 2.0,
        dog_sigma2          = 8.0,
        clahe_clip          = 3.0,
        tophat_ksize        = 21,
        max_dist            = 40,
        max_skips           = 8,
        min_track_length    = 3,
        P_init=100, Q_val=1, R_val=30,
        scale=0.7, show_window=True, show_plot=True,
        plot_style          = 'markers',
        # save_csv=True, csv_mode='single', csv_path=r'trajectories.csv',
    )



