# Tracking_V2

# DHM Particle Tracker — 2D
Kalman-filter-based particle tracker for Digital Holographic Microscopy (DHM) videos. Supports four imaging modes — Brightfield, Amplitude, Hologram, and Phase — and provides both a graphical interface and a scripting API.

---
 
## Repository structure
 
```
├── function_tracking_improved.py   # Core tracking library
├── tracker_gui.py                  # Graphical user interface
├── Videos                          # Examples (videos) with the tracking parameters
└── README.md
```
 
---

## Requirements
 
| Package | Tested version |
|---------|---------------|
| Python  | ≥ 3.9 |
| OpenCV (`opencv-python`) | ≥ 4.7 |
| NumPy | ≥ 1.24 |
| SciPy | ≥ 1.10 |
| Matplotlib | ≥ 3.7 |
| Pillow | ≥ 9.0 |
| scikit-image | ≥ 0.20 |
 
Install all dependencies at once:
 
```bash
pip install opencv-python numpy scipy matplotlib pillow scikit-image
```
 
---

## Quick start — GUI
 
```bash
python tracker_gui.py
```
 
1. Select the **video type** (Brightfield, Amplitude, Hologram, or Phase). Default detection parameters are loaded automatically.
2. Open your video file with the 📂 button.
3. Fill in the **optical system** parameters (camera pixel size and magnification). FPS is read automatically from the video; enable *Override* only if needed.
4. Adjust detection and Kalman parameters if necessary.
5. Click **▶ RUN TRACKING**.
A live preview window shows the tracking in real time. Press **Stop** or **ESC** to interrupt at any time.
 
---

## Imaging modes
 
| Mode | `image_mode` | Typical sample | `blobColor` | Preprocessing |
|------|-------------|----------------|-------------|---------------|
| Brightfield | `'standard'` | Cells, erythrocytes | `0` (dark) | Gaussian or Bilateral |
| DHM Amplitude | `'amplitude'` | Reconstructed amplitude | `255` (bright) | CLAHE + Bilateral |
| DHM Hologram | `'hologram'` | Raw hologram | `255` (bright) | Bilateral (or DoG) |
| DHM Phase | `'phase'` | Reconstructed phase | `255` (bright) | DoG + Top-hat + CLAHE |
 
---

## Parameter reference
 
### Optical system
 
| Parameter | Description |
|-----------|-------------|
| Camera pixel size (µm) | Physical pixel pitch of the sensor (e.g. `3.75` µm for a typical CMOS) |
| Magnification (x) | Objective magnification (e.g. `20`, `40`) |
 
The effective pixel size (`cam_pixel_um / magnification`) is computed automatically by the code — the user never needs to calculate it manually.
 
### Detection (`detect_centroids`)
 
| Parameter | Description |
|-----------|-------------|
| `minArea` / `maxArea` | Blob area range in px² |
| `blobColor` | `0` = dark blobs, `255` = bright blobs |
| `filterByCircularity` | Enable circularity filter |
| `minCircularity` | Minimum circularity \[0–1\] (1 = perfect circle) |
| `filterByInertia` | Enable inertia ratio filter (elongated vs. round) |
| `filterByConvexity` | Enable convexity filter |
| `filter_type` | `'bilateral'` (recommended) or `'gaussian'` |
| `clahe_clip` | CLAHE clip limit — relevant for Amplitude and Phase modes |
| `use_hough` | Use Hough circles instead of blob detector (Phase mode only) |
| `dog_sigma1/2` | Inner/outer σ for Difference-of-Gaussians (Phase mode) |
| `tophat_ksize` | Morphological top-hat kernel size (Phase mode) |
 
### Kalman filter
 
| Parameter | Description |
|-----------|-------------|
| `P_init` | Initial state covariance. Higher = less trust in initial position |
| `Q_val` | Process noise. Higher = filter follows detections more aggressively |
| `R_val` | Measurement noise. Higher = filter trusts its own prediction more |
 
### Tracking
 
| Parameter | Description |
|-----------|-------------|
| `max_dist` | Maximum distance (px) to associate a detection to a track |
| `max_skips` | Frames a track can persist without a detection before being terminated |
| `min_track_length` | Minimum number of detected frames to keep a track in the output |
 
### Output
 
| Parameter | Description |
|-----------|-------------|
| `show_plot` | Show XY trajectory and speed-profile plots after tracking |
| `plot_style` | `'markers'` (circle = start, square = end) or `'dots'` |
| `save_csv` | Export trajectories to CSV |
| `csv_mode` | `'single'` — one file with all tracks; `'per_track'` — one file per track |
| `csv_path` | Output CSV path (or folder path for `per_track`) |
| `save_video_path` | If set, saves an annotated output video (`.mp4`) |
 
### CSV output format
 
**Single file** (`csv_mode='single'`):
 
| Column | Description |
|--------|-------------|
| `track_id` | Track index |
| `frame` | Frame number |
| `x_px`, `y_px` | Position in pixels |
| `x_um`, `y_um` | Position in micrometres |

---

## How it works
 
### Detection pipeline
 
Each frame goes through a mode-specific preprocessing step before blob detection:
 
- **Standard / Brightfield** — Gaussian or Bilateral smoothing.
- **Amplitude** — CLAHE equalisation + Bilateral filter to normalise contrast fluctuations between frames.
- **Hologram** — Bilateral filter (or optionally DoG + Top-hat when `use_hough=True`).
- **Phase** — Difference-of-Gaussians + morphological Top-hat + CLAHE to enhance the characteristic phase rings of DHM particles.
Centroid detection is performed with OpenCV's `SimpleBlobDetector`.
 
### Tracking pipeline
 
1. **Predict** — each active Kalman filter predicts the position of its particle in the next frame.
2. **Associate** — Hungarian algorithm (`linear_sum_assignment`) finds the globally optimal assignment between predictions and detections, rejecting pairs farther than `max_dist`.
3. **Update** — matched filters are updated with the detected position; unmatched filters accumulate a skip count.
4. **Manage** — tracks exceeding `max_skips` are terminated and saved; new tracks are spawned for unmatched detections.
---
 
## Output plots
 
After tracking, three figures are displayed:
 
1. **XY Trajectories** — all tracks in the focal plane, in µm.
2. **Legend** — color key for up to 40 tracks.
3. **Speed Profiles** — instantaneous speed (µm/frame) over time for each track.
---
 
## Tips
 
- Start with the default parameters for your mode and adjust `minArea`/`maxArea` first — they have the largest impact on detection quality.
- If tracks are lost during fast motion, increase `max_dist` and `max_skips`.
- If unrelated blobs are merged into the same track, decrease `max_dist`.
- For noisy amplitude videos, increasing `R_val` (e.g. to 100–200) makes the Kalman filter smoother and more robust.
- `min_track_length` is the simplest way to discard spurious short tracks (e.g. set to 5–10 for cleaner output).
