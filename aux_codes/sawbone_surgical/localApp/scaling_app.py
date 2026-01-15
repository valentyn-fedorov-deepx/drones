import os
import sys
from pathlib import Path
import json
import numpy as np
import cv2
import torch

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QLabel,
    QSpinBox, QTextEdit, QVBoxLayout, QSplitter,
    QListWidget, QInputDialog, QSplashScreen, QDoubleSpinBox, QHBoxLayout, QComboBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt6.QtGui import QImage, QPixmap, QSurfaceFormat, QColor, QShortcut, QKeySequence
from PyQt6.QtWebEngineWidgets import QWebEngineView

# ------------------------
# UI / display constants
# ------------------------
SCREEN_RATIO = float(2448)/float(2048)
PREVIEW_H = 210
PREVIEW_W = int(PREVIEW_H * SCREEN_RATIO)
TARGET_SHORT = 1024
USE_PYQTGRAPH = True
ADJUST_SCALE = 1.1
NEEDED_SCALE = 350.353

# ------------------------
# Paths & project imports
# ------------------------
root_dir = Path(__file__).parents[4]
data_dir = root_dir / "data"; data_dir.mkdir(exist_ok=True, parents=True)
TEMP_DATA_DIR = data_dir / "temp_data"
if not TEMP_DATA_DIR.exists():
    raise FileNotFoundError(f"Missing {TEMP_DATA_DIR}; run preprocessing first")

# Make repo modules importable
for sub in [
    "vggt", "sam2_root",
    "dx_vyzai_python",
    "dx_vyzai_python/aux_codes/sawbone_surgical",
    "dx_vyzai_python/aux_codes/sawbone_surgical/code"
]:
    sys.path.append(str(root_dir / sub))

from frame_loader   import FrameLoader  # not used directly, but kept for parity
from vggt_processor import VGGTProcessor
from sam2_model     import SegmentAnything2
from visualization  import (
    plot_pointcloud_plotly, append_pointcloud,
    create_gl_view, plot_pointcloud_pyqtgraph_into, set_view_to_points
)
from geometry_processor import most_collinear, DirectedAxisFitter

# Some projects expose reconstr helpers via VGGTProcessor; others as a module.
# We access them through an instance method to avoid import ambiguity.

# ------------------------
# Small UI helpers
# ------------------------
class ClickableLabel(QLabel):
    clicked = pyqtSignal(int, int, int)  # x, y, btn (1=left, 2=right)
    def __init__(self):
        super().__init__()
        self.setFixedSize(PREVIEW_W, PREVIEW_H)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
    def mousePressEvent(self, ev):
        if ev.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            which = 1 if ev.button() == Qt.MouseButton.LeftButton else 2
            self.clicked.emit(int(ev.position().x()), int(ev.position().y()), which)

# ------------------------
# Core math helpers (width & points extraction)
# ------------------------

def extract_pts_single_frame(reconstr_proc,
                              mask: np.ndarray,
                              img: np.ndarray,
                              preds: dict,
                              frame_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Mirror of the user's extract routine; works with preds.npz + reconstr proc.
    Returns (seg_pts[R^3], seg_cols[uint8x3]).
    """
    ex_all, in_all = reconstr_proc.get_camera_params(preds=preds, as_numpy=True, squeezed=False)
    E_b = ex_all[:, frame_idx, :, :]  # [1, 3, 4]
    K_b = in_all[:, frame_idx, :, :]  # [1, 3, 3]

    depth_all = reconstr_proc.get_depth(preds, as_numpy=True, squeezed=False)
    depth_b = depth_all[:, frame_idx, :, :, :]  # [1,1,H,W]

    world_pts_idx = reconstr_proc.depth_to_point_cloud(
        depth=depth_b,
        extrinsic=E_b,
        intrinsic=K_b,
        squeezed=False,
    )  # [1,H,W,3]
    _, H, W, _ = world_pts_idx.shape

    mask_resized = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)

    pts_flat = world_pts_idx.reshape(-1, 3)
    cols_flat = reconstr_proc.get_points_colors(
        points=world_pts_idx[None, ...],
        imgs=np.asarray([img]),
        squeezed=True,
        flatten=True,
    )  # [H*W, 3]

    seg_mask = mask_resized.flatten()
    seg_pts = pts_flat[seg_mask]
    seg_cols = cols_flat[seg_mask]
    return seg_pts, seg_cols


def get_sawbone_blade_width(saw_pts: np.ndarray, x_axis: np.ndarray) -> float:
    """Fit a directed line along provided axis; return the span (extent)."""
    line_fit = DirectedAxisFitter.fit(saw_pts, x_axis, 0.05)
    return float(line_fit["extent"])  # units: scene units (VGGT world units)


def analyze_sawbone_unit_blade_width(reconstr_proc,
                                     mask: np.ndarray,
                                     img: np.ndarray,
                                     x_axis: np.ndarray,
                                     preds: dict,
                                     frame_idx: int) -> float:
    saw_pts, _ = extract_pts_single_frame(reconstr_proc, mask, img, preds, frame_idx)
    return get_sawbone_blade_width(saw_pts, x_axis)

def to_torch_preds(preds_np: dict, device: torch.device | str = "cpu") -> dict:
    """Convert np arrays in preds to torch tensors (float32 for floats)."""
    device = torch.device(device)
    out = {}
    for k, v in preds_np.items():
        if isinstance(v, np.ndarray):
            t = torch.from_numpy(v)
            if np.issubdtype(v.dtype, np.floating):
                t = t.float()
            out[k] = t.to(device)
        else:
            out[k] = v
    return out


def calculate_pixel_width_at_rows(mask: np.ndarray, rows) -> float:
    """Average mask width (px) across the given row(s)."""
    if isinstance(rows, int):
        rows = [rows]
    widths = []
    H = mask.shape[0]
    for row_idx in rows:
        if row_idx < 0 or row_idx >= H:
            continue
        row = mask[row_idx, :]
        cols = row.nonzero()[0]
        if len(cols) > 1:
            widths.append(cols.max() - cols.min())
    return float(np.mean(widths)) if widths else 0.0


def convert_pixel_width_between_resolutions(
    pixel_width: float,
    src_shape: tuple,  # (W, H)
    dst_shape: tuple,  # (W, H)
) -> float:
    """Rescale a pixel width measured at src resolution to dst resolution."""
    src_w, _ = src_shape
    dst_w, _ = dst_shape
    if src_w <= 0:
        return 0.0
    return float(pixel_width) * (float(dst_w) / float(src_w))


def compute_scale_from_width(
    pixel_width: float,
    intrinsics: np.ndarray,   # 3x3
    blade_width_real: float,  # mm (or your real units)
    depth: float | None = None
) -> float:
    """
    Returns mm per scene-unit (global scale).
    If depth is given, uses metric projection; otherwise angular width scaling.
    """
    fx = float(intrinsics[0, 0])
    if fx <= 0 or pixel_width <= 0 or blade_width_real <= 0:
        return 1.0
    if depth is not None:
        predicted_width = (pixel_width * depth) / fx
    else:
        predicted_width = pixel_width / fx
    if predicted_width == 0:
        return 1.0
    return float(blade_width_real) / float(predicted_width)



# ------------------------
# Worker: track, compute scale, save scaled clouds
# ------------------------
class ScaleAndSaveWorker(QObject):
    progress = pyqtSignal(str)
    finished = pyqtSignal(object)

    def __init__(self, sam, vggt, sam_frames_dir, preds,
                 imgs_orig_rgb, imgs_1024_rgb,
                 saw_pos_1024: list[dict],
                 saw_neg_1024: list[dict],
                 frame_idx_clicks: int,
                 real_width_mm: float,
                 adjust_scale: float = 1.0,
                 mode: str = "multi",
                 n_rows_bottom: int = 100):
        super().__init__()
        self.sam = sam
        self.vggt = vggt
        self.sam_frames_dir = sam_frames_dir
        self.preds = preds
        self.imgs_orig_rgb = imgs_orig_rgb
        self.imgs_1024_rgb = imgs_1024_rgb
        self.saw_pos_1024 = saw_pos_1024
        self.saw_neg_1024 = saw_neg_1024
        self.frame_idx_clicks = frame_idx_clicks
        self.real_width_mm = float(real_width_mm)
        self.adjust_scale = float(adjust_scale)
        self.mode = mode
        self.n_rows_bottom = int(max(1, n_rows_bottom))
        
        try:
            # preds["images"] is usually [B, T, C, H, W]; using [0]
            im0 = preds["images"][0]
            self._model_dst_shape = (int(im0.shape[-1]), int(im0.shape[-2]))  # (W, H)
        except Exception:
            # Fallback: use 1024 image dims (will still be consistent)
            h, w = imgs_1024_rgb[0].shape[:2]
            self._model_dst_shape = (w, h)
    
    
    def _pixel_width_from_mask(self, mask_1024: np.ndarray) -> float:
        """Compute average width (px) across bottom N rows, then convert to model resolution."""
        H, W = mask_1024.shape[:2]
        n = min(self.n_rows_bottom, H)
        rows = range(H - n, H)
        px_w_1024 = calculate_pixel_width_at_rows(mask_1024.astype(bool), rows=rows)
        if px_w_1024 <= 0:
            return 0.0
        src_shape = (W, H)  # (W, H) for 1024-resized frame
        dst_shape = self._model_dst_shape  # model input (W, H)
        return convert_pixel_width_between_resolutions(px_w_1024, src_shape, dst_shape)



    def run(self):
        ##### TRACKING ON POINTS MODE
        # if self.mode == "single":
        #     i = int(self.frame_idx_clicks)
        #     self.progress.emit(f"Single-frame: computing mask on frame {i}...")

        #     clicks_labeled = (
        #         [{'x': int(p['x']), 'y': int(p['y']), 'label': 1} for p in self.saw_pos_1024] +
        #         [{'x': int(p['x']), 'y': int(p['y']), 'label': 0} for p in self.saw_neg_1024]
        #     )
        #     mask_1024 = self.sam.segment_with_clicks(self.imgs_1024_rgb[i], clicks_labeled)


        #     x_axis = None
        #     try:
        #         _, x_axes_raw, _, _ = self.vggt.get_camera_poses(self.preds)
        #         xa = np.asarray(x_axes_raw)
        #         if xa.ndim == 3: xa = xa[0]
        #         x_axis = xa[i].astype(np.float32)
        #     except Exception as e:
        #         self.progress.emit(f"(warn) camera axis unavailable: {e}")

        #     img_orig = self.imgs_orig_rgb[i]
        #     if x_axis is None:
        #         pts_tmp, _ = extract_pts_single_frame(self.vggt, mask_1024, img_orig, self.preds, i)
        #         x_axis = most_collinear(pts_tmp)

        #     width_i = analyze_sawbone_unit_blade_width(self.vggt, mask_1024, img_orig, x_axis, self.preds, i)
        #     widths = np.array([width_i], dtype=float)
        #     width_mean = float(width_i)
        #     self.progress.emit(f"Blade width (scene units) on frame {i}: {width_i:.3f}")

        # else:
        #     self.progress.emit("Tracking Sawblade across frames...")
        #     masks_1024 = self.sam.track_video_single_object(
        #         folder_path=str(self.sam_frames_dir),
        #         positive_clicks=self.saw_pos_1024,
        #         negative_clicks=self.saw_neg_1024,
        #         frame_idx=self.frame_idx_clicks,
        #     )


        #     x_axes = None
        #     try:
        #         cam_positions, x_axes_raw, y_axes, z_axes = self.vggt.get_camera_poses(self.preds)
        #         x_axes_np = np.asarray(x_axes_raw)
        #         if x_axes_np.ndim == 3: x_axes_np = x_axes_np[0]
        #         if x_axes_np.shape[-1] != 3: raise ValueError(f"Unexpected x_axes shape: {x_axes_np.shape}")
        #         if x_axes_np.shape[0] != len(masks_1024): raise ValueError("x_axes length mismatch")
        #         x_axes = x_axes_np.astype(np.float32)
        #         np.save(TEMP_DATA_DIR / "x_axes.npy", x_axes)
        #         self.progress.emit(f"Loaded Y-axes from camera poses (T={len(x_axes)}) and saved to x_axes.npy")
        #     except Exception as e:
        #         self.progress.emit(f"(warn) Could not get Y-axes from camera poses: {e}")
        #         x_axes_path = TEMP_DATA_DIR / "x_axes.npy"
        #         if x_axes_path.exists():
        #             try:
        #                 x_axes = np.load(x_axes_path)
        #                 if x_axes.shape[0] != len(masks_1024):
        #                     self.progress.emit("(warn) Saved x_axes length mismatch; will derive per-frame.")
        #                     x_axes = None
        #                 else:
        #                     self.progress.emit("Loaded Y-axes from x_axes.npy")
        #             except Exception as e2:
        #                 self.progress.emit(f"(warn) Could not read x_axes.npy: {e2}")
        #                 x_axes = None

        #     widths = []
        #     self.progress.emit("Estimating Sawblade width (per frame)...")
        #     for i, mask_1024 in enumerate(masks_1024):
        #         img_orig = self.imgs_orig_rgb[i]
        #         if x_axes is not None:
        #             x_axis = x_axes[i]
        #         else:
        #             pts_tmp, _ = extract_pts_single_frame(self.vggt, mask_1024, img_orig, self.preds, i)
        #             x_axis = most_collinear(pts_tmp)
        #         width_i = analyze_sawbone_unit_blade_width(self.vggt, mask_1024, img_orig, x_axis, self.preds, i)
        #         widths.append(width_i)
        #     widths = np.asarray(widths, dtype=float)
        #     width_mean = float(np.mean(widths)) if len(widths) else 1.0
        #     self.progress.emit(f"Mean blade width (scene units): {width_mean:.3f}")

        # scale_mm_per_unit = (self.real_width_mm / max(width_mean, 1e-9)) * self.adjust_scale
        # scale_mm_per_unit = NEEDED_SCALE
        # self.progress.emit(f"Computed scale: {scale_mm_per_unit:.3f} mm / unit")
        # # Persist scale for downstream steps
        # scale_meta = {
        #     "real_blade_width_mm": self.real_width_mm,
        #     "median_scene_width": width_mean,
        #     "adjust_scale": self.adjust_scale,
        #     "scale_mm_per_unit": scale_mm_per_unit,
        # }
        
        ex_all, in_all = self.vggt.get_camera_params(preds=self.preds, as_numpy=True, squeezed=False)
        # in_all: [1, T, 3, 3]

        if self.mode == "single":
            i = int(self.frame_idx_clicks)
            self.progress.emit(f"Single-frame: computing mask on frame {i}...")

            clicks_labeled = (
                [{'x': int(p['x']), 'y': int(p['y']), 'label': 1} for p in self.saw_pos_1024] +
                [{'x': int(p['x']), 'y': int(p['y']), 'label': 0} for p in self.saw_neg_1024]
            )
            mask_1024 = self.sam.segment_with_clicks(self.imgs_1024_rgb[i], clicks_labeled)

            # 1) measure pixel width at 1024-res, across bottom N rows; convert to model-res
            px_w_model = self._pixel_width_from_mask(mask_1024)
            self.progress.emit(f"Pixel width @model-res (frame {i}): {px_w_model:.2f}px")

            # 2) compute scale from intrinsics
            K = in_all[0, i, :, :]
            scale_i = compute_scale_from_width(
                pixel_width=px_w_model,
                intrinsics=K,
                blade_width_real=self.real_width_mm,
                depth=None  # keep None unless you have Z
            )
            scales = np.array([scale_i], dtype=float)
            scale_mm_per_unit = float(np.mean(scales)) * self.adjust_scale
            widths = np.array([px_w_model], dtype=float)  # keep for UI/debug if you want

        else:
            self.progress.emit("Tracking sawblade across frames...")
            masks_1024 = self.sam.track_video_single_object(
                folder_path=str(self.sam_frames_dir),
                positive_clicks=self.saw_pos_1024,
                negative_clicks=self.saw_neg_1024,
                frame_idx=self.frame_idx_clicks,
            )

            px_widths_model = []
            scales_per_frame = []

            self.progress.emit("Estimating blade pixel width (per frame) and per-frame scale...")
            T = len(masks_1024)
            for i, mask_1024 in enumerate(masks_1024):
                px_w_model = self._pixel_width_from_mask(mask_1024)
                px_widths_model.append(px_w_model)

                K = in_all[0, i, :, :] if i < in_all.shape[1] else in_all[0, -1, :, :]
                scale_i = compute_scale_from_width(
                    pixel_width=px_w_model,
                    intrinsics=K,
                    blade_width_real=self.real_width_mm,
                    depth=None
                )
                scales_per_frame.append(scale_i)

            widths = np.asarray(px_widths_model, dtype=float)
            scales = np.asarray(scales_per_frame, dtype=float)

            self.progress.emit(
                f"Mean pixel width @model-res: {float(np.mean(widths)):.2f}px "
                f"(N={len(widths)})"
            )
            scale_mm_per_unit = float(np.mean(scales)) * self.adjust_scale

        # ---- DO NOT override the computed scale ----
        # scale_mm_per_unit = NEEDED_SCALE   # <-- REMOVE this hard override
        self.progress.emit(f"Computed scale: {scale_mm_per_unit:.6f} mm / unit")
        
        scale_meta = {
            "real_blade_width_mm": self.real_width_mm,
            "avg_pixel_width_model_px": float(np.mean(widths)) if len(widths) else None,
            "adjust_scale": self.adjust_scale,
            "scale_mm_per_unit": float(scale_mm_per_unit),
            "mode": self.mode,
            "n_rows_bottom": self.n_rows_bottom,
        }
        
        with open(TEMP_DATA_DIR / "scale_meta.json", "w", encoding="utf-8") as f:
            json.dump(scale_meta, f, indent=2)

        # Find all *_points.npy clouds produced by segmentation step and scale them
        out_dir = TEMP_DATA_DIR / "pts_scaled"
        out_dir.mkdir(exist_ok=True)

        pts_files = sorted(TEMP_DATA_DIR.glob("*_points.npy"))
        scaled_layers = []  # list of (pts_scaled, cols, name)

        if not pts_files:
            self.progress.emit("No *_points.npy found in temp_data. Nothing to scale.")
        else:
            self.progress.emit("Scaling and saving data...")
            for pts_path in pts_files:
                name = pts_path.stem.replace("_points", "")
                # Skip Sawblade cloud if user doesn't want it scaled; here we include all by default
                pts = np.load(pts_path)
                cols_path = TEMP_DATA_DIR / f"{name}_colors.npy"
                cols = np.load(cols_path) if cols_path.exists() else None

                pts_scaled = pts.astype(np.float32) * scale_mm_per_unit * self.adjust_scale
                np.save(out_dir / f"{name}_points.npy", pts_scaled)
                if cols is not None:
                    np.save(out_dir / f"{name}_colors.npy", cols)

                scaled_layers.append((pts_scaled, cols, name))
                self.progress.emit(f"Saved scaled: {name}_points.npy (+colors if present)")

            # Also save a combined GLB for convenience
            try:
                all_pts = np.vstack([s[0] for s in scaled_layers if s[0] is not None])
                all_cols = np.vstack([s[1] if s[1] is not None else np.full((len(s[0]),3), 200, np.uint8)
                                      for s in scaled_layers])
                glb = out_dir / "combined_scaled.glb"
                self.vggt.save_points_as_glb(all_pts, all_cols, str(glb))
                self.progress.emit("Saved combined_scaled.glb")
            except Exception as e:
                self.progress.emit(f"(warn) Combined GLB not saved: {e}")

        # Return scaled layers for rendering in UI
        self.finished.emit({
            "scaled_layers": scaled_layers,
            "scale_mm_per_unit": scale_mm_per_unit,
            "widths": widths,
        })


# ------------------------
# Main window
# ------------------------
class ScalingWindow(QMainWindow):
    def __init__(self, splash: QSplashScreen | None = None):
        super().__init__()
        self.splash = splash

        def _splash_msg(msg: str):
            if self.splash is not None:
                self.splash.showMessage(
                    msg,
                    Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom,
                    QColor("white"),
                )
                QApplication.processEvents()
        self._splash_msg = _splash_msg

        self.setWindowTitle("Scaling via Sawblade Width")
        self.resize(1200, 700)

        # Load preds + frames (orig and 1024) — identical to segmentation_app
        self._splash_msg("Loading data…")
        npz = np.load(TEMP_DATA_DIR / "preds.npz")
        self.preds_np = {k: npz[k] for k in npz.files}

        # Use CPU (safe + avoids needless GPU hops). Switch to 'cuda' if you wish.
        self.device = torch.device("cpu")
        self.preds_torch = to_torch_preds(self.preds_np, self.device)


        orig_dir = TEMP_DATA_DIR / "orig"
        jpgs = sorted(orig_dir.glob("*.jpg"))
        imgs = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in jpgs]
        self.imgs_np = np.stack(imgs, 0)

        self.sam_dir = TEMP_DATA_DIR / f"s{TARGET_SHORT}"
        self.sam_dir.mkdir(exist_ok=True)
        self._splash_msg("Preparing resized frames (1024 short side)…")

        self.jpgs_1024 = []
        for jpg in jpgs:
            outp = self.sam_dir / jpg.name
            self.jpgs_1024.append(outp)
            if not outp.exists():
                im = cv2.imread(str(jpg))
                h, w = im.shape[:2]
                scale = TARGET_SHORT / min(h, w)
                nh, nw = int(h * scale), int(w * scale)
                imr = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
                cv2.imwrite(str(outp), imr)

        imgs_1024 = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in self.jpgs_1024]
        self.imgs_1024_np = np.stack(imgs_1024, 0)

        self._orig_sizes = [(im.shape[1], im.shape[0]) for im in self.imgs_np]
        self._sam_sizes  = [(im.shape[1], im.shape[0]) for im in self.imgs_1024_np]

        torch.cuda.empty_cache()
        self._splash_msg("Initializing segmentation processes...")
        self.sam = SegmentAnything2()
        self.vggt = VGGTProcessor(skip_model=True)

        if self.splash is not None:
            self.splash.finish(self)
            self.splash = None

        # ------------------------
        # Single object: "Sawblade"
        # ------------------------
        self.ov = [{
            "name": "Sawblade",
            "pos": [],           # list[{"x": int, "y": int}]
            "neg": [],           # list[{"x": int, "y": int}]
            "history": [],       # list[("add"|"del", "pos"|"neg", {"x":..,"y":..})]
            "redo": [],          # same as history
            "frame_idx": 0
        }]


        # ------------------------
        # UI layout (left: clicks/mask; right: 3D)
        # ------------------------
        cen = QWidget(); self.setCentralWidget(cen)
        main = QVBoxLayout(cen)
        self.split = QSplitter(Qt.Orientation.Horizontal)
        main.addWidget(self.split, 1)

        # Left panel
        left = QWidget(); L = QVBoxLayout(left)
        row = QHBoxLayout()
        row.addWidget(QLabel("Real blade width (mm):"))
        self.spn_real_width = QDoubleSpinBox()
        self.spn_real_width.setDecimals(3)
        self.spn_real_width.setRange(0.001, 10000.0)
        self.spn_real_width.setValue(19)  # default
        self.spn_real_width.setSingleStep(0.01)
        row.addWidget(self.spn_real_width)
        L.addLayout(row)
        
        rowN = QHBoxLayout()
        rowN.addWidget(QLabel("Rows from bottom (N):"))
        self.spn_rows_bottom = QSpinBox()
        self.spn_rows_bottom.setRange(1, 1024)
        self.spn_rows_bottom.setValue(100)  # default you used in your snippet
        self.spn_rows_bottom.setSingleStep(10)
        rowN.addWidget(self.spn_rows_bottom)
        L.addLayout(rowN)

        
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Scaling mode:"))
        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems(["Multi-frame (track)", "Single frame"])
        self.cmb_mode.setCurrentIndex(0)
        row2.addWidget(self.cmb_mode)
        L.addLayout(row2)

        self.scale_mode = "multi"
        self.cmb_mode.currentIndexChanged.connect(
            lambda i: setattr(self, "scale_mode", "multi" if i == 0 else "single")
        )

        L.addWidget(QLabel("Object:"))
        self.lst = QListWidget(); self.lst.addItem("Sawblade")
        L.addWidget(self.lst)

        L.addWidget(QLabel("Frame:"))
        self.spin = QSpinBox(); self.spin.setMaximum(len(self.jpgs_1024) - 1)
        self.spin.valueChanged.connect(self.update_preview)
        L.addWidget(self.spin)

        self.lbl_click = ClickableLabel(); L.addWidget(self.lbl_click)
        self.lbl_mask  = ClickableLabel(); L.addWidget(self.lbl_mask)
        
        self.btn_undo = QPushButton("Undo")
        self.btn_redo = QPushButton("Redo")
        self.btn_clear = QPushButton("Clear clicks")
        self.btn_undo.clicked.connect(self.undo_click)
        self.btn_redo.clicked.connect(self.redo_click)
        self.btn_clear.clicked.connect(self.clear_clicks)
        row_actions = QHBoxLayout()
        row_actions.addWidget(self.btn_undo)
        row_actions.addWidget(self.btn_redo)
        L.addLayout(row_actions)
        L.addWidget(self.btn_clear)
        
        QShortcut(QKeySequence.StandardKey.Undo, self, activated=self.undo_click)
        QShortcut(QKeySequence.StandardKey.Redo, self, activated=self.redo_click)
        QShortcut(QKeySequence(Qt.Key.Key_Backspace), self, activated=self.undo_click)

        # Actions
        self.btn_scale = QPushButton("Compute scale And Save scaled data")
        self.btn_next  = QPushButton("Next step")
        self.btn_scale.clicked.connect(self.compute_scale_clicked)
        self.btn_next.clicked.connect(self.close)
        L.addWidget(self.btn_scale)
        L.addWidget(self.btn_next)

        self.split.addWidget(left)

        # Right panel
        right = QWidget(); R = QVBoxLayout(right)
        if USE_PYQTGRAPH:
            self.gl = create_gl_view(); R.addWidget(self.gl)
            self._gl_layers = []
        else:
            self.web = QWebEngineView(); R.addWidget(self.web)

        self.log = QTextEdit(); self.log.setReadOnly(True); self.log.setFixedHeight(150)
        R.addWidget(self.log)

        self.split.addWidget(right)
        self.split.setStretchFactor(0, 1)
        self.split.setStretchFactor(1, 4)

        # Signals
        self.lst.currentRowChanged.connect(self._force_sawblade)
        self.lst.setCurrentRow(0)
        self.lbl_click.clicked.connect(self.on_image_click)

        # Initial state
        self.spin.setValue(0)
        self.update_preview()

        # Try to show already scaled clouds if present
        self._try_load_and_show_scaled()

    # --------------
    # UI helpers
    # --------------
    def _force_sawblade(self, *_):
        # Always one object
        pass

    def update_preview(self):
        idx = self.spin.value()
        sam_im = self.imgs_1024_np[idx]
        preview = cv2.resize(sam_im, (PREVIEW_W, PREVIEW_H), interpolation=cv2.INTER_AREA)
        qimg = QImage(preview.data, PREVIEW_W, PREVIEW_H, 3*PREVIEW_W, QImage.Format.Format_RGB888)
        self.lbl_click.setPixmap(QPixmap.fromImage(qimg))

        self._preview = preview
        self._sam_dims  = self._sam_sizes[idx]
        self._orig_dims = self._orig_sizes[idx]
        self._sam_full  = sam_im

        self.update_mask_preview()

    def on_image_click(self, x, y, btn):
        # Map preview coords -> SAM-1024 coords
        w_sam, h_sam = self._sam_dims
        ox = int(x * w_sam / PREVIEW_W)
        oy = int(y * h_sam / PREVIEW_H)

        o = self.ov[0]
        o['frame_idx'] = self.spin.value()

        # left = positive, right = negative
        channel = "pos" if btn == 1 else "neg"
        pt = {"x": ox, "y": oy}
        o[channel].append(pt)

        # history: record add, clear redo on new action
        o["history"].append(("add", channel, pt))
        o["redo"].clear()

        self.log.append(f"{'+' if channel=='pos' else '-'} Click @({ox},{oy}) on 'Sawblade' [frame {o['frame_idx']}]")
        self.draw_markers()
        self.update_mask_preview()
        
    def undo_click(self):
        o = self.ov[0]
        if not o["history"]:
            return
        action, channel, pt = o["history"].pop()

        if action == "add":
            # remove the last matching point from that channel (fallback: pop)
            try:
                # remove last occurrence equal to pt
                idx = next(i for i in range(len(o[channel])-1, -1, -1) if o[channel][i] == pt)
                o[channel].pop(idx)
            except StopIteration:
                if o[channel]: o[channel].pop()
            # push inverse to redo
            o["redo"].append(("add", channel, pt))
        elif action == "del":
            # re-add the point
            o[channel].append(pt)
            o["redo"].append(("del", channel, pt))

        self.log.append(f"Undo ({channel})")
        self.draw_markers()
        self.update_mask_preview()

    def redo_click(self):
        o = self.ov[0]
        if not o["redo"]:
            return
        action, channel, pt = o["redo"].pop()  # replay same operation

        if action == "add":
            o[channel].append(pt)
            o["history"].append(("add", channel, pt))
        elif action == "del":
            try:
                idx = next(i for i in range(len(o[channel])-1, -1, -1) if o[channel][i] == pt)
                o[channel].pop(idx)
            except StopIteration:
                if o[channel]: o[channel].pop()
            o["history"].append(("del", channel, pt))

        self.log.append(f"Redo ({channel})")
        self.draw_markers()
        self.update_mask_preview()

    def clear_clicks(self):
        o = self.ov[0]
        # push deletes into history so a single Undo restores them in order
        for pt in reversed(o["pos"]):
            o["history"].append(("del", "pos", pt))
        for pt in reversed(o["neg"]):
            o["history"].append(("del", "neg", pt))
        o["pos"].clear()
        o["neg"].clear()
        o["redo"].clear()
        self.log.append("Cleared all clicks")
        self.draw_markers()
        self.update_mask_preview()

    def draw_markers(self):
        pre = self._preview.copy()
        w_sam, h_sam = self._sam_dims

        # positives = green filled circles
        for pt in self.ov[0]['pos']:
            dx = int(pt['x'] * PREVIEW_W / w_sam)
            dy = int(pt['y'] * PREVIEW_H / h_sam)
            cv2.circle(pre, (dx, dy), 6, (0, 255, 0), -1)

        # negatives = red X
        for pt in self.ov[0]['neg']:
            dx = int(pt['x'] * PREVIEW_W / w_sam)
            dy = int(pt['y'] * PREVIEW_H / h_sam)
            s = 6
            cv2.line(pre, (dx - s, dy - s), (dx + s, dy + s), (255, 0, 0), 2)
            cv2.line(pre, (dx - s, dy + s), (dx + s, dy - s), (255, 0, 0), 2)

        qimg = QImage(pre.data, PREVIEW_W, PREVIEW_H, 3*PREVIEW_W, QImage.Format.Format_RGB888)
        self.lbl_click.setPixmap(QPixmap.fromImage(qimg))

    def update_mask_preview(self):
        o = self.ov[0]
        if not (o['pos'] or o['neg']):
            self.lbl_mask.clear()
            return

        clicks = (
            [{'x': p['x'], 'y': p['y'], 'label': 1} for p in o['pos']] +
            [{'x': p['x'], 'y': p['y'], 'label': 0} for p in o['neg']]
        )
        mask_1024 = self.sam.segment_with_clicks(self._sam_full, clicks)
        mu8 = cv2.resize(mask_1024.astype(np.uint8), (PREVIEW_W, PREVIEW_H), interpolation=cv2.INTER_NEAREST) * 255
        qimg = QImage(mu8.data, PREVIEW_W, PREVIEW_H, PREVIEW_W, QImage.Format.Format_Grayscale8)
        self.lbl_mask.setPixmap(QPixmap.fromImage(qimg))

    # --------------
    # Scaling flow
    # --------------
    def compute_scale_clicked(self):
        o = self.ov[0]
        if not (o['pos'] or o['neg']):
            self.log.append("Add a few clicks on the Sawblade first.")
            return

        # Get values from UI field and hardcoded constant
        real_mm = float(self.spn_real_width.value())
        if real_mm <= 0:
            self.log.append("Real blade width must be > 0.")
            return
        adjust = float(ADJUST_SCALE)  # hardcoded (no UI)

        # Build clicks in SAM-1024 space (already stored that way)
        saw_pos = [{'x': int(p['x']), 'y': int(p['y'])} for p in o['pos']]
        saw_neg = [{'x': int(p['x']), 'y': int(p['y'])} for p in o['neg']]

        frame_idx_clicks = int(o['frame_idx'])

        self.log.append(
            f"Computing scale and saving scaled data..."
        )

        # Keep references on self so they don't get GC'd
        self.scale_thread = QThread()
        self.scale_worker = ScaleAndSaveWorker(
            self.sam, self.vggt, self.sam_dir,
            self.preds_torch, self.imgs_np, self.imgs_1024_np,
            saw_pos, saw_neg, frame_idx_clicks,
            real_width_mm=real_mm,
            adjust_scale=adjust,
            mode=self.scale_mode,
            n_rows_bottom=int(self.spn_rows_bottom.value()),  # <-- NEW
        )

        self.scale_worker.moveToThread(self.scale_thread)
        self.scale_thread.started.connect(self.scale_worker.run)
        self.scale_worker.progress.connect(self.log.append)
        self.scale_worker.finished.connect(self.on_scale_done)
        self.scale_worker.finished.connect(self.scale_thread.quit)
        self.scale_thread.finished.connect(self.scale_thread.deleteLater)
        self.scale_worker.finished.connect(self.scale_worker.deleteLater)
        self.scale_thread.start()


    def on_scale_done(self, payload: dict):
        # Render scaled layers (if any) and log the computed scale
        scl = payload.get("scale_mm_per_unit", None)
        if scl is not None:
            self.log.append(f"Final scale: {scl:.3f} mm/unit")
        layers = payload.get("scaled_layers", [])
        if not layers:
            self.log.append("No scaled layers to render.")
            return

        if USE_PYQTGRAPH:
            # Clear layers
            for it in getattr(self, "_gl_layers", []):
                try: self.gl.removeItem(it)
                except Exception: pass
            self._gl_layers = []
            # Add layers
            for i, (pts, cols, name) in enumerate(layers):
                it = plot_pointcloud_pyqtgraph_into(self.gl, pts_flat=pts, cols_flat=cols,
                                                    max_points=250000, point_size=2.2)
                if it: self._gl_layers.append(it)
                if i == 0: set_view_to_points(self.gl, pts, margin=1.6)
            self.log.append("Rendered scaled data (pyqtgraph).")
        else:
            # Plotly fallback
            pts0, cols0, name0 = layers[0]
            fig = plot_pointcloud_plotly(pts_flat=pts0, cols_flat=cols0,
                                         max_points=25000, marker_size=2, showlegend=True, name=name0)
            for pts, cols, name in layers[1:]:
                fig = append_pointcloud(fig, pts, cols, max_points=25000, marker_size=2, name=name)
            fig.update_layout(autosize=True, margin=dict(l=0, r=0, t=20, b=0))
            self.web.setHtml(fig.to_html(full_html=True, include_plotlyjs='cdn', config={'responsive': True}))
            self.log.append("Rendered scaled data (Plotly).")

        self.log.append("Saved scaled data.")

    def _try_load_and_show_scaled(self):
        """If scaled clouds already exist (from a previous run), show them immediately."""
        out_dir = TEMP_DATA_DIR / "pts_scaled"
        if not out_dir.exists():
            return
        pts_files = sorted(out_dir.glob("*_points.npy"))
        if not pts_files:
            return
        layers = []
        for p in pts_files:
            name = p.stem.replace("_points", "")
            pts = np.load(p)
            cols_p = out_dir / f"{name}_colors.npy"
            cols = np.load(cols_p) if cols_p.exists() else None
            layers.append((pts, cols, name))

        if USE_PYQTGRAPH:
            for it in getattr(self, "_gl_layers", []):
                try: self.gl.removeItem(it)
                except Exception: pass
            self._gl_layers = []
            for i, (pts, cols, name) in enumerate(layers):
                it = plot_pointcloud_pyqtgraph_into(self.gl, pts_flat=pts, cols_flat=cols,
                                                    max_points=250000, point_size=2.2)
                if it: self._gl_layers.append(it)
                if i == 0: set_view_to_points(self.gl, pts, margin=1.6)
            self.log.append("Loaded existing scaled data (pyqtgraph).")
        else:
            pts0, cols0, name0 = layers[0]
            fig = plot_pointcloud_plotly(pts_flat=pts0, cols_flat=cols0,
                                         max_points=25000, marker_size=2, showlegend=True, name=name0)
            for pts, cols, name in layers[1:]:
                fig = append_pointcloud(fig, pts, cols, max_points=25000, marker_size=2, name=name)
            fig.update_layout(autosize=True, margin=dict(l=0, r=0, t=20, b=0))
            self.web.setHtml(fig.to_html(full_html=True, include_plotlyjs='cdn', config={'responsive': True}))
            self.log.append("Loaded existing scaled data (Plotly).")


# ------------------------
# Entrypoint (with splash)
# ------------------------
if __name__ == '__main__':
    from PyQt6.QtCore import QCoreApplication
    if USE_PYQTGRAPH:
        os.environ["QT_OPENGL"] = "desktop"
        fmt = QSurfaceFormat()
        fmt.setRenderableType(QSurfaceFormat.RenderableType.OpenGL)
        fmt.setVersion(2, 1)
        fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CompatibilityProfile)
        fmt.setSwapBehavior(QSurfaceFormat.SwapBehavior.DoubleBuffer)
        QSurfaceFormat.setDefaultFormat(fmt)
        QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    else:
        QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_UseSoftwareOpenGL)

    app = QApplication(sys.argv)

    # Splash: show immediately so perceived latency is intentional
    pm = QPixmap(480, 220); pm.fill(Qt.GlobalColor.black)
    splash = QSplashScreen(pm)
    splash.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
    splash.show()
    splash.showMessage("Loading scaling step...",
                       Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom,
                       QColor("white"))
    app.processEvents()

    win = ScalingWindow(splash=splash)
    win.show()
    sys.exit(app.exec())
