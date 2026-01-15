import os
import sys
import random
from pathlib import Path

import numpy as np
import cv2
import torch

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QLabel,
    QSpinBox, QTextEdit, QVBoxLayout, QSplitter,
    QListWidget, QInputDialog, QSplashScreen
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt6.QtGui import QImage, QPixmap, QSurfaceFormat, QColor
from PyQt6.QtWebEngineWidgets import QWebEngineView

# Fixed preview dimensions
PREVIEW_W = 300
PREVIEW_H = 250
TARGET_SHORT = 1024
USE_PYQTGRAPH = True

# ——— Paths ———
root_dir = Path(__file__).parents[4]
data_dir = root_dir / "data"; data_dir.mkdir(exist_ok=True, parents=True)
TEMP_DATA_DIR = data_dir / "temp_data"
if not TEMP_DATA_DIR.exists():
    raise FileNotFoundError(f"Missing {TEMP_DATA_DIR}; run preprocessing first")

# Add project modules to PYTHONPATH
for sub in [
    "vggt", "sam2_root",
    "dx_vyzai_python",
    "dx_vyzai_python/aux_codes/sawbone_surgical",
    "dx_vyzai_python/aux_codes/sawbone_surgical/code"
]:
    sys.path.append(str(root_dir / sub))

torch.cuda.empty_cache()

from frame_loader    import FrameLoader
from vggt_processor  import VGGTProcessor
from sam2_model      import SegmentAnything2
from visualization   import plot_pointcloud_plotly, append_pointcloud, create_gl_view, plot_pointcloud_pyqtgraph_into, set_view_to_points

# ——— Clickable preview label ———
class ClickableLabel(QLabel):
    clicked = pyqtSignal(int, int)
    def __init__(self):
        super().__init__()
        self.setFixedSize(PREVIEW_W, PREVIEW_H)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(int(ev.position().x()), int(ev.position().y()))

# ——— 3D segmentation worker ———
class SegmentationWorker(QObject):
    progress = pyqtSignal(str)
    finished = pyqtSignal(object)

    def __init__(self, sam, vggt, preds, imgs_np, jpgs, objects, sam_frames_dir):
        super().__init__()
        self.sam, self.vggt = sam, vggt
        self.preds, self.imgs, self.jpgs, self.ov = preds, imgs_np, jpgs, objects
        self.sam_frames_dir = sam_frames_dir

    def run(self):
        segs = []
        for obj in self.ov:
            name = obj['name']
            self.progress.emit(f"Segmenting '{name}'...")
            masks = self.sam.track_video_single_object(
                folder_path=str(self.sam_frames_dir),
                positive_clicks=obj['points'],
                negative_clicks=[
                    p for o in self.ov if o is not obj for p in o['points']
                ],
                frame_idx=(obj.get('frame_idx', 0))
            )
            H0, W0 = self.imgs.shape[1], self.imgs.shape[2]
            masks_up = [
                cv2.resize(m.astype(np.uint8), (W0, H0), interpolation=cv2.INTER_NEAREST).astype(bool)
                for m in masks
            ]
            pts, cols = self.vggt.segment_pointcloud(self.preds, self.imgs, masks_up)
            segs.append((pts, cols, name))

        # save individual clouds
        for pts, cols, name in segs:
            np.save(TEMP_DATA_DIR/f"{name}_points.npy", pts)
            np.save(TEMP_DATA_DIR/f"{name}_colors.npy", cols)
            glb = TEMP_DATA_DIR/f"{name}.glb"
            self.vggt.save_points_as_glb(pts, cols, str(glb))
            self.progress.emit(f"Saved {name}.glb + .npy")

        # combined
        all_pts  = np.vstack([s[0] for s in segs])
        all_cols = np.vstack([s[1] for s in segs])
        combined = TEMP_DATA_DIR/"combined.glb"
        self.vggt.save_points_as_glb(all_pts, all_cols, str(combined))
        self.progress.emit("Saved combined.glb")

        # # Plotly view
        # self.progress.emit("Building 3D view…")
        # pts0, cols0, name0 = segs[0]
        # fig = plot_pointcloud_plotly(
        #     pts_flat=pts0, cols_flat=cols0,
        #     max_points=25000, marker_size=2,
        #     showlegend=True, name=name0
        # )
        # for pts, cols, name in segs[1:]:
        #     fig = append_pointcloud(fig, pts, cols,
        #                             max_points=25000, marker_size=2, name=name)
        # fig.update_layout(autosize=True, margin=dict(l=0,r=0,t=20,b=0))
        # html = fig.to_html(
        #     full_html=True, include_plotlyjs='cdn',
        #     config={'responsive': True}
        # )
        # self.finished.emit(html)
        self.progress.emit("Preparing 3D view in UI...")
        self.finished.emit(segs)


# ——— Main window ———
class SegmentationWindow(QMainWindow):
    def __init__(self, splash: QSplashScreen | None = None):
        super().__init__()
        self.splash = splash
        
        def _splash_msg(msg: str):
            if self.splash is not None:
                 # bottom-center message in white so it’s visible on dark bg
                self.splash.showMessage(
                    msg,
                    Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom,
                    QColor("white")
                )
                # let the splash repaint even while we’re doing sync work
                QApplication.processEvents()
        self._splash_msg = _splash_msg
        
        self.setWindowTitle("3D Segmentation")
        self.resize(1200, 700)

        # load preds + frames
        self._splash_msg("Loading data...")
        npz = np.load(TEMP_DATA_DIR/"preds.npz")
        self.preds = {k: npz[k] for k in npz.files}
        orig_dir = TEMP_DATA_DIR / "orig"
        jpgs = sorted(orig_dir.glob("*.jpg"))
        imgs = [
            cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
            for p in jpgs
        ]
        self.imgs_np, self.jpgs = np.stack(imgs, 0), jpgs
        
        self.sam_dir = TEMP_DATA_DIR / f"s{TARGET_SHORT}"
        self.sam_dir.mkdir(exist_ok=True)
        self._splash_msg("Preparing resized frames...")
        
        self.jpgs_1024 = []
        for jpg in self.jpgs:
            new_jpg_path = self.sam_dir / jpg.name
            self.jpgs_1024.append(new_jpg_path)
            
            if not new_jpg_path.exists():
                img = cv2.imread(str(jpg))
                if img is None:
                    raise FileNotFoundError(f"Failed to read {jpg}")
                # resize to TARGET_SHORT while keeping aspect ratio
                h, w = img.shape[:2]
                scale = TARGET_SHORT / min(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                imr = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
                cv2.imwrite(str(new_jpg_path), imr)
        
        #rgb stack
        imgs_1024 = [
            cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
            for path in self.jpgs_1024
        ]
        self.imgs_1024_np = np.stack(imgs_1024, 0)
        self._orig_sizes = [(im.shape[1], im.shape[0]) for im in self.imgs_np]
        self._sam_sizes  = [(im.shape[1], im.shape[0]) for im in self.imgs_1024_np]

        torch.cuda.empty_cache()
        self._splash_msg("Initializing segmentation processes...")
        self.sam = SegmentAnything2()
        self.vggt = VGGTProcessor(skip_model=True)

        # Heavy init done — remove splash right before we ask for inputs
        if self.splash is not None:
            self.splash.finish(self)
            self.splash = None

        # ask object count
        cnt, ok = QInputDialog.getInt(
            self, "Objects Count", "How many objects to segment?",
            min=1, value=2
        )
        if not ok:
            self.close(); return

        self.ov = []
        for i in range(cnt):
            nm, ok2 = QInputDialog.getText(
                self, "Object Name", f"Name for object {i+1}:"
            )
            if not ok2 or not nm:
                nm = f"Object {i+1}"
            self.ov.append({'name': nm, 'points': [], 'undo': [], "frame_idx": 0})

        # UI layout
        cen = QWidget(); self.setCentralWidget(cen)
        main_lay = QVBoxLayout(cen)
        self.split = QSplitter(Qt.Orientation.Horizontal)
        main_lay.addWidget(self.split, 1)

        # left pane
        left = QWidget(); v = QVBoxLayout(left)
        v.addWidget(QLabel("Objects:"))
        self.lst = QListWidget()
        for o in self.ov:
            self.lst.addItem(o['name'])
        v.addWidget(self.lst)

        v.addWidget(QLabel("Frame:"))
        self.spin = QSpinBox()
        self.spin.setMaximum(len(self.jpgs) - 1)
        v.addWidget(self.spin)
        self.spin.valueChanged.connect(self.update_preview)

        self.lbl_click = ClickableLabel()
        v.addWidget(self.lbl_click)
        self.lbl_mask = ClickableLabel()
        v.addWidget(self.lbl_mask)

        for text, fn in [("Undo", self.undo_pt),
                         ("Redo", self.redo_pt),
                         ("Segment 3D", self.segment)]:
            btn = QPushButton(text)
            btn.clicked.connect(fn)
            v.addWidget(btn)
            
        self.btn_next  = QPushButton("Next step")
        self.btn_next.clicked.connect(self.close)
        v.addWidget(self.btn_next)
        self.split.addWidget(left)

        # right pane
        right = QWidget(); w = QVBoxLayout(right)
        if USE_PYQTGRAPH:
            self.gl = create_gl_view()
            w.addWidget(self.gl)
            self._gl_layers = []
        else:
            self.web = QWebEngineView()
            w.addWidget(self.web)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFixedHeight(150)
        w.addWidget(self.log)
        self.split.addWidget(right)

        # make 3D view extra wide
        self.split.setStretchFactor(0, 1)
        self.split.setStretchFactor(1, 4)

        # wire signals
        self.lst.currentRowChanged.connect(self.on_object_changed)
        self.lst.setCurrentRow(0)
        self.lbl_click.clicked.connect(self.on_image_click)
        self.spin.setValue(0)

        # initial draw
        self.update_preview()

    def update_preview(self):
        idx = self.spin.value()
        sam_im = self.imgs_1024_np[idx]  # work in 1024 space
        preview = cv2.resize(sam_im, (PREVIEW_W, PREVIEW_H), interpolation=cv2.INTER_AREA)

        qimg = QImage(
            preview.data, PREVIEW_W, PREVIEW_H,
            3*PREVIEW_W, QImage.Format.Format_RGB888
        )
        pix = QPixmap.fromImage(qimg)
        # set both click & mask previews
        self.lbl_click.setPixmap(pix)
        self._preview = preview
        self._sam_dims  = self._sam_sizes[idx]
        self._orig_dims = self._orig_sizes[idx]  # w0, h0
        self._sam_full  = sam_im
        self.update_mask_preview()

    def on_object_changed(self, idx):
        self.act = idx
        self.update_preview()

    def on_image_click(self, x, y):
        w_sam, h_sam = self._sam_dims
        ox = int(x * w_sam / PREVIEW_W)
        oy = int(y * h_sam / PREVIEW_H)
        obj = self.ov[self.act]
        cur_idx = self.spin.value()
        obj['frame_idx'] = cur_idx
        obj['points'].append({'x': ox, 'y': oy})
        obj['undo'].clear()
        self.log.append(f"Click @({ox},{oy}) on '{obj['name']}'")
        self.draw_markers()
        self.update_mask_preview()

    def draw_markers(self):
        pre = self._preview.copy()
        w_sam, h_sam = self._sam_dims
        for pt in self.ov[self.act]['points']:
            dx = int(pt['x'] * PREVIEW_W / w_sam)
            dy = int(pt['y'] * PREVIEW_H / h_sam)
            cv2.circle(pre, (dx, dy), 5, (0, 255, 0), -1)
        qimg = QImage(pre.data, PREVIEW_W, PREVIEW_H,
                      3*PREVIEW_W, QImage.Format.Format_RGB888)
        self.lbl_click.setPixmap(QPixmap.fromImage(qimg))

    def update_mask_preview(self):
        obj = self.ov[self.act]
        if not obj['points']:
            self.lbl_mask.clear()
            return

        clicks = (
            [{'x':p['x'], 'y':p['y'], 'label':1}
             for p in obj['points']] +
            [{'x':p['x'], 'y':p['y'], 'label':0}
             for o in self.ov if o is not obj for p in o['points']]
        )
        mask_1024 = self.sam.segment_with_clicks(self._sam_full, clicks)  # Hs×Ws, bool/0-1
        mu8 = cv2.resize(mask_1024.astype(np.uint8), (PREVIEW_W, PREVIEW_H), interpolation=cv2.INTER_NEAREST) * 255
        qimg = QImage(mu8.data, PREVIEW_W, PREVIEW_H, PREVIEW_W, QImage.Format.Format_Grayscale8)

        self.lbl_mask.setPixmap(QPixmap.fromImage(qimg))

    def undo_pt(self):
        o = self.ov[self.act]
        if o['points']:
            o['undo'].append(o['points'].pop())
            self.log.append(f"Undo '{o['name']}'")
            self.draw_markers()
            self.update_mask_preview()

    def redo_pt(self):
        o = self.ov[self.act]
        if o['undo']:
            o['points'].append(o['undo'].pop())
            self.log.append(f"Redo '{o['name']}'")
            self.draw_markers()
            self.update_mask_preview()

    def segment(self):
        self.log.append("Running full segmentation...")

        # keep these on self so they don't get GC'd
        self.seg_thread  = QThread()
        self.seg_worker  = SegmentationWorker(
            self.sam, self.vggt,
            self.preds, self.imgs_np,
            self.jpgs, self.ov,
            self.sam_dir
        )

        # move worker into the thread
        self.seg_worker.moveToThread(self.seg_thread)

        # wire up signals
        self.seg_thread.started.connect(self.seg_worker.run)
        self.seg_worker.progress.connect(self.log.append)
        self.seg_worker.finished.connect(self.on_seg_done)

        # when the worker says it’s finished, stop the thread
        self.seg_worker.finished.connect(self.seg_thread.quit)

        # clean up both thread and worker when done
        self.seg_thread.finished.connect(self.seg_thread.deleteLater)
        self.seg_worker.finished.connect(self.seg_worker.deleteLater)

        # start it
        self.seg_thread.start()


    def on_seg_done(self, payload):
        if USE_PYQTGRAPH:
            segs = payload  # list of (pts, cols, name)
            # clear old layers
            for it in getattr(self, "_gl_layers", []):
                try:
                    self.gl.removeItem(it)
                except Exception:
                    pass
            self._gl_layers = []
            # add layers
            for i, (pts, cols, name) in enumerate(segs):
                it = plot_pointcloud_pyqtgraph_into(
                    self.gl, pts_flat=pts, cols_flat=cols,
                    max_points=250000, point_size=2.2
                )
                if it: self._gl_layers.append(it)
                if i == 0:
                    set_view_to_points(self.gl, pts, margin=1.6)
            self.log.append("Rendered 3D (pyqtgraph).")
        else:
            # Either got html already, or build Plotly here
            if isinstance(payload, str):
                self.web.setHtml(payload)
            else:
                segs = payload
                pts0, cols0, name0 = segs[0]
                fig = plot_pointcloud_plotly(
                    pts_flat=pts0, cols_flat=cols0,
                    max_points=25000, marker_size=2, showlegend=True, name=name0
                )
                for pts, cols, name in segs[1:]:
                    fig = append_pointcloud(fig, pts, cols, max_points=25000, marker_size=2, name=name)
                fig.update_layout(autosize=True, margin=dict(l=0,r=0,t=20,b=0))
                self.web.setHtml(fig.to_html(full_html=True, include_plotlyjs='cdn', config={'responsive': True}))
        self.log.append("Segmentation complete.")

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
    # Show a splash IMMEDIATELY so users see “it’s working”
    pm = QPixmap(480, 220)
    pm.fill(Qt.GlobalColor.black)  # dark bg looks clean in dark mode
    splash = QSplashScreen(pm)
    splash.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
    splash.show()
    splash.showMessage(
        "Loading segmentation step...",
        Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom,
        QColor("white")
    )
    # Ensure the splash is actually painted before heavy work starts
    app.processEvents()

    win = SegmentationWindow(splash=splash)
    win.show()
    sys.exit(app.exec())
