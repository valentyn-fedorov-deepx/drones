import os
import sys
import shutil
from pathlib import Path
import numpy as np
import cv2

from PyQt6 import QtGui
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QLabel,
    QSpinBox, QComboBox, QFileDialog,
    QHBoxLayout, QVBoxLayout, QSplitter, QTextEdit,
    QProgressBar, QDoubleSpinBox
)
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject, QCoreApplication
from PyQt6.QtWebEngineWidgets import QWebEngineView

# ——— Set up paths ———
root_dir = Path(__file__).parent.parent.parent.parent.parent

data_dir = root_dir / "data"
data_dir.mkdir(parents=True, exist_ok=True)

USE_PYQTGRAPH = True
VGGT_MODEL_PATH = data_dir / "checkpoints" / "3d_model.pt"
VGGT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

TEMP_DATA_DIR = data_dir / "temp_data"
if TEMP_DATA_DIR.exists():
    shutil.rmtree(TEMP_DATA_DIR)
TEMP_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ——— Add project paths ———
for p in [
    root_dir / "vggt",
    root_dir / "sam2_root",
    root_dir / "dx_vyzai_python",
    root_dir / "dx_vyzai_python" / "aux_codes" / "sawbone_surgical",
    root_dir / "dx_vyzai_python" / "aux_codes" / "sawbone_surgical" / "code"
]:
    sys.path.append(str(p))

from frame_loader import FrameLoader
from vggt_processor import VGGTProcessor
from visualization import plot_pointcloud_plotly, create_gl_view, plot_pointcloud_pyqtgraph_into, set_view_to_points


def apply_brightness_cv2(img: np.ndarray, factor: float) -> np.ndarray:
    """
    Brighten/darken with OpenCV.
    - For uint8 images: use convertScaleAbs (fast, safe).
    - For higher bit depths: multiply + clip to preserve depth.
    """
    if factor == 1.0 or img is None:
        return img

    if img.dtype == np.uint8:
        # y = alpha*x + beta; here beta=0, alpha=factor
        return cv2.convertScaleAbs(img, alpha=float(factor), beta=0)
    else:
        # keep dtype/bit depth (e.g., 12/16-bit pipelines before later conversion)
        out = cv2.multiply(img, np.array([factor], dtype=np.float32))
        info = np.iinfo(img.dtype) if np.issubdtype(img.dtype, np.integer) else None
        if info is not None:
            out = np.clip(out, 0, info.max).astype(img.dtype)
        return out

# ——— Worker for VGGT 3D inference ———
class VGGTWorker(QObject):
    progress = pyqtSignal(str)
    progress_percent = pyqtSignal(int)
    finished = pyqtSignal(object)  # emits (preds, imgs_np, pts_flat, cols_flat)

    def __init__(self, paths, downscale=1, bit_depth=8, color_mode=True, brighten_factor=1.0):
        super().__init__()
        self.paths = paths
        self.downscale = downscale
        self.bit_depth = bit_depth
        self.color_mode = color_mode
        self.brighten_factor = brighten_factor
        self.loader = FrameLoader(bit_depth=self.bit_depth, color_data=self.color_mode)
        self.progress.emit("Preparing model code...")
        self.progress_percent.emit(5)
        self.processor = VGGTProcessor(model_path=VGGT_MODEL_PATH)

    def run(self):
        # clear and recreate temp folder
        shutil.rmtree(TEMP_DATA_DIR)
        TEMP_DATA_DIR.mkdir(parents=True, exist_ok=True)

        self.progress.emit("Loading and downscaling frames...")
        self.progress_percent.emit(10)
        imgs = []
        for p in self.paths:
            img = self.loader.load_bgr_frompxi(p)
            img = apply_brightness_cv2(img, self.brighten_factor)
            imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


        # save as JPG for VGGT
        # original pictures to temp_data_dir/orig, downscaled straight to the dir
        orig_dir = TEMP_DATA_DIR / "orig"
        orig_dir.mkdir(parents=True, exist_ok=True)
        for i, im in enumerate(imgs):
            cv2.imwrite(str(orig_dir / f"{i:05d}.jpg"), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
            if self.downscale > 1:
                h, w = im.shape[:2]
                im = cv2.resize(im, (w//self.downscale, h//self.downscale), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(TEMP_DATA_DIR / f"{i:05d}.jpg"), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
        
        if self.downscale > 1:
            for i, img in enumerate(imgs):
                h, w = img.shape[:2]
                imgs[i] = cv2.resize(img, (w//self.downscale, h//self.downscale), interpolation=cv2.INTER_AREA)
        
        imgs_np = np.stack(imgs, axis=0)
        jpgs = sorted(TEMP_DATA_DIR.glob("*.jpg"))

        self.progress.emit("Starting 3D Reconstruction...")
        self.progress_percent.emit(40)
        preds = self.processor.infer(jpgs)

        self.progress.emit("Saving raw outputs...")
        self.progress_percent.emit(70)
        np_preds = self.processor.preds_to_numpy(preds)
        np.savez(TEMP_DATA_DIR / "preds.npz", **np_preds)

        self.progress.emit("Extracting 3D model...")
        pts = self.processor.get_points(preds, squeezed=True, as_numpy=True)
        cols = self.processor.get_points_colors(points=pts, imgs=imgs_np, flatten=True)
        pts_flat = pts.reshape(-1, 3)
        cols_flat = cols.reshape(-1, 3).astype(np.uint8)
        self.progress_percent.emit(100)

        self.finished.emit((preds, imgs_np, pts_flat, cols_flat))

# ——— Main application window ———
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Model Processing")
        self.resize(900, 600)

        cen = QWidget()
        self.setCentralWidget(cen)
        lay = QVBoxLayout(cen)

        # — Controls bar split into two rows —
        ctrl_layout = QVBoxLayout()
        lay.addLayout(ctrl_layout)

        # First row
        ctrl_row1 = QHBoxLayout()
        ctrl_layout.addLayout(ctrl_row1)

        self.btn_folder = QPushButton("Select .pxi Folder")
        ctrl_row1.addWidget(self.btn_folder)
        self.btn_folder.clicked.connect(self.select_folder)

        self.lbl_folder = QLabel("No folder selected")
        ctrl_row1.addWidget(self.lbl_folder)

        ctrl_row1.addWidget(QLabel("Start:"))
        self.spin_start = QSpinBox()
        ctrl_row1.addWidget(self.spin_start)

        ctrl_row1.addWidget(QLabel("End:"))
        self.spin_end = QSpinBox()
        ctrl_row1.addWidget(self.spin_end)

        ctrl_row1.addWidget(QLabel("Downscale:"))
        self.combo_down = QComboBox()
        self.combo_down.addItems(["1","2","4","8","16"])
        ctrl_row1.addWidget(self.combo_down)

        ctrl_row1.addWidget(QLabel("Count:"))
        self.combo_count = QComboBox()
        self.combo_count.addItems(["5","10","20","30","40","50"])
        ctrl_row1.addWidget(self.combo_count)

        # Second row
        ctrl_row2 = QHBoxLayout()
        ctrl_layout.addLayout(ctrl_row2)

        ctrl_row2.addWidget(QLabel("Bit Depth:"))
        self.combo_bit = QComboBox()
        self.combo_bit.addItems(["8-bit","12-bit"])
        ctrl_row2.addWidget(self.combo_bit)

        ctrl_row2.addWidget(QLabel("Mode:"))
        self.combo_color = QComboBox()
        self.combo_color.addItems(["Color","Grayscale"])
        ctrl_row2.addWidget(self.combo_color)
        
        ctrl_row2.addWidget(QLabel("Brighten:"))
        self.spin_brighten = QDoubleSpinBox()
        self.spin_brighten.setRange(0.10, 3.00)   # 0.10x (darker) … 3.00x (brighter)
        self.spin_brighten.setSingleStep(0.10)
        self.spin_brighten.setDecimals(2)
        self.spin_brighten.setValue(1.00)
        self.spin_brighten.setSuffix("x")

        ctrl_row2.addWidget(self.spin_brighten)


        self.btn_set_images = QPushButton("Set Images")
        ctrl_row2.addWidget(self.btn_set_images)
        self.btn_set_images.clicked.connect(self.load_previews)

        self.btn_process = QPushButton("Process 3D")
        ctrl_row2.addWidget(self.btn_process)
        self.btn_process.clicked.connect(self.process_3d)

        self.btn_next = QPushButton("Next Step")
        self.btn_next.setEnabled(False)
        ctrl_row2.addWidget(self.btn_next)
        self.btn_next.clicked.connect(self.launch_segmentation)

        # — Split view: left for previews, right for Plotly —
        self.split = QSplitter(Qt.Orientation.Horizontal)
        lay.addWidget(self.split, 1)

        self.left = QWidget()
        self.layL = QVBoxLayout(self.left)
        self.split.addWidget(self.left)

        self.lbl_p0 = QLabel("Start Image")
        self.lbl_p1 = QLabel("End Image")
        for lbl in (self.lbl_p0, self.lbl_p1):
            lbl.setFixedSize(200,200)
            lbl.setFrameShape(QLabel.Shape.Box)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.layL.addWidget(lbl)
        
        if USE_PYQTGRAPH:
            self.gl = create_gl_view()
            self.split.addWidget(self.gl)
        else:
            self.web = QWebEngineView()
            self.split.addWidget(self.web)
        self.split.setSizes([250, 900])

        # — Progress bar & Log output —
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        lay.addWidget(self.progress_bar)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFixedHeight(150)
        lay.addWidget(self.log)

        # — State —
        self.files = []
        self.sel_paths = []
        self.down = 1

    def select_folder(self):
        fld = QFileDialog.getExistingDirectory(self, "Select .pxi Folder")
        if not fld:
            return
        self.folder = Path(fld)
        self.lbl_folder.setText(fld)
        self.files = sorted(self.folder.glob("*.pxi"))
        m = max(0, len(self.files)-1)
        for sb in (self.spin_start, self.spin_end):
            sb.setMaximum(m)
        self.spin_end.setValue(m)
        self.log.append(f"Loaded {len(self.files)} files")

    def load_previews(self):
        if not self.files:
            self.log.append("No folder selected.")
            return
        s, e = self.spin_start.value(), self.spin_end.value()
        factor = float(self.spin_brighten.value())
        bit_depth = 8 if self.combo_bit.currentText() == "8-bit" else 12
        color_mode = self.combo_color.currentText() == "Color"
        loader = FrameLoader(bit_depth=bit_depth, color_data=color_mode)
        img_s = loader.load_bgr_frompxi(self.files[s])
        img_e = loader.load_bgr_frompxi(self.files[e])
        
        img_s = apply_brightness_cv2(img_s, factor)
        img_e = apply_brightness_cv2(img_e, factor)

        self.lbl_p0.setPixmap(self.to_pixmap(img_s, bgr=True))
        self.lbl_p1.setPixmap(self.to_pixmap(img_e, bgr=True))
        self.log.append(f"Previews loaded: frames {s} and {e}")

    def to_pixmap(self, img, bgr=False):
        if bgr:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img.shape
        bpl = ch * w
        qimg = QtGui.QImage(img.data, w, h, bpl, QtGui.QImage.Format.Format_RGB888)
        return QtGui.QPixmap.fromImage(qimg).scaled(
            200, 200,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

    def process_3d(self):
        if not self.files:
            self.log.append("No folder selected.")
            return
        s, e = self.spin_start.value(), self.spin_end.value()
        if s > e:
            self.log.append("Start index must be ≤ End index.")
            return
        count = int(self.combo_count.currentText())
        down = int(self.combo_down.currentText())
        bit_depth = 8 if self.combo_bit.currentText() == "8-bit" else 12
        color_mode = self.combo_color.currentText() == "Color"

        self.sel_paths = FrameLoader(bit_depth=bit_depth, color_data=color_mode).select_frames(
            self.files, s, e, count
        )
        self.down = down
        self.log.append(f"Processing {len(self.sel_paths)} frames...")
        self.progress_bar.setValue(0)

        self.btn_process.setEnabled(False)
        
        factor = float(self.spin_brighten.value())

        self.vggt_thread = QThread()
        self.vggt_worker = VGGTWorker(self.sel_paths, self.down, bit_depth, color_mode, factor)
        self.vggt_worker.moveToThread(self.vggt_thread)

        # hook up signals
        self.vggt_worker.progress.connect(self.log.append)
        self.vggt_worker.progress_percent.connect(self.progress_bar.setValue)
        self.vggt_worker.finished.connect(self.on_3d_done)
        self.vggt_thread.started.connect(self.vggt_worker.run)
        self.vggt_worker.finished.connect(self.vggt_thread.quit)
        self.vggt_thread.finished.connect(self.vggt_thread.deleteLater)

        self.vggt_thread.start()

    def on_3d_done(self, data):
        preds, imgs_np, pts, cols = data
        self.preds = preds
        self.imgs_np = imgs_np

        self.log.append("Rendering 3D scene...")
        if USE_PYQTGRAPH:
            # clear old layers if rerun
            if hasattr(self, "_gl_layers"):
                for it in self._gl_layers:
                    try: self.gl.removeItem(it)
                    except Exception: pass
            self._gl_layers = []
            it = plot_pointcloud_pyqtgraph_into(
                self.gl, pts_flat=pts, cols_flat=cols,
                max_points=250000, point_size=2.2
            )
            if it: self._gl_layers.append(it)
            set_view_to_points(self.gl, pts, margin=1.6)
        else:
            fig = plot_pointcloud_plotly(
                pts_flat=pts, cols_flat=cols,
                max_points=25000, marker_size=2.2,
            )
            fig.update_layout(autosize=True, margin=dict(l=0, r=0, t=0, b=0))
            html = fig.to_html(
                full_html=True,
                include_plotlyjs='cdn',
                config={'responsive': True}
            )
            self.web.setHtml(html)

        self.btn_process.setEnabled(True)
        self.btn_next.setEnabled(True)
        self.log.append("3D processing complete. Ready for next step.")

    def launch_segmentation(self):
        if hasattr(self, 'seg_thread') and self.seg_thread.isRunning():
            self.seg_thread.quit()
            self.seg_thread.wait()

        self.close()
        QApplication.instance().quit()

if __name__ == "__main__":
    from PyQt6.QtCore import Qt, QCoreApplication
    from PyQt6.QtGui import QSurfaceFormat

    if USE_PYQTGRAPH:
        os.environ["QT_OPENGL"] = "desktop"

        # 2) Request a COMPATIBILITY profile with fixed-function support
        fmt = QSurfaceFormat()
        fmt.setRenderableType(QSurfaceFormat.RenderableType.OpenGL)
        # 2.1 is enough; or use 3.3 with CompatibilityProfile
        fmt.setVersion(2, 1)
        fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CompatibilityProfile)
        fmt.setSwapBehavior(QSurfaceFormat.SwapBehavior.DoubleBuffer)
        QSurfaceFormat.setDefaultFormat(fmt)

        # 3) Allow context sharing (nice when mixing views)
        QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)

    else:
        # Plotly path can keep software GL to appease QWebEngine if you want
        QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_UseSoftwareOpenGL)

    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
