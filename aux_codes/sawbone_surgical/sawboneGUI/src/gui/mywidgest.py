import os
import sys
import logging

import numpy as np
from collections import deque

from PyQt6.QtCore import Qt, pyqtSignal, QCoreApplication
from PyQt6.QtGui import QImage, QPixmap, QMouseEvent, QSurfaceFormat
from PyQt6.QtWidgets import (QFrame, QLabel, QGridLayout, QPushButton,
    QHBoxLayout, QVBoxLayout, QDoubleSpinBox, QWidget
)

try:
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl
    HAVE_PG = True
except Exception:
    pg = None
    gl = None
    HAVE_PG = False


class ImagePane(QFrame):
    """An image pane with 3 buttons: start, start (no recording), stop"""
    start_signal = pyqtSignal(bool)
    stop_signal = pyqtSignal()
    points_placed = pyqtSignal(list)
    stop_reconstruction_signal = pyqtSignal()

    def __init__(self, name, parent):
        super().__init__(parent)
        self.name = name

        # GUI elements
        self.video_label = QLabel(self)
        self.video_label.setFrameShape(QFrame.Shape.Box)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; color: white;")
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setMouseTracking(True)

        self.btn_start = QPushButton(f'Start {name}', self)
        self.btn_start.clicked.connect(lambda: self.start_signal.emit(True))
        self.btn_pause = QPushButton(f'Pause {name}', self)
        self.btn_pause.clicked.connect(lambda: self.stop_signal.emit())
        self.btn_stop = QPushButton(f'Stop Reconstraction', self)
        self.btn_stop.clicked.connect(lambda: self.stop_reconstruction_signal.emit())
        
        self.my_layout = QGridLayout()
        self.my_layout.addWidget(self.video_label, 0, 0, 1, 3)
        self.my_layout.addWidget(self.btn_start, 1, 0, 1, 1)
        self.my_layout.addWidget(self.btn_pause, 1, 1, 1, 1)
        self.my_layout.addWidget(self.btn_stop, 1, 2, 1, 1)
        
        self.my_layout.setRowStretch(0, 4)
        self.my_layout.setRowStretch(1, 1)

        self.setLayout(self.my_layout)
        self.setFrameShape(QFrame.Shape.Box)
        self.setLineWidth(3)
        
        self._paused = False
        self._display_img = None
        self._points = deque(maxlen=5)
        
    def convert_cv_qt(self, rgb_image: np.ndarray) -> QPixmap:
        """Convert an OpenCV RGB image to QPixmap"""
        height, width, channel = rgb_image.shape
        bytes_per_line = channel * width
        qt_image = QImage(
            rgb_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888
        )
        scaled = qt_image.scaled(
            self.video_label.width(), self.video_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        return QPixmap.fromImage(scaled)

    def set_frame(self, frame: np.ndarray):
        import cv2
        """Update with an OpenCV BGR frame"""
        logging.debug(f"Updating image for {self.name}")
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.video_label.setPixmap(self.convert_cv_qt(rgb_image))
        logging.debug(f"Updated image for {self.name}")
        
        self._display_img = frame
        
    def mousePressEvent(self, event: QMouseEvent):
        import cv2
        if self._display_img is None:
            return
        if not self.video_label.pixmap():
            return

        pm = self.video_label.pixmap()
        label_size = self.video_label.size()
        pm_size = pm.size()

        x_off = (label_size.width() - pm_size.width()) // 2
        y_off = (label_size.height() - pm_size.height()) // 2
        x = event.position().x() - x_off
        y = event.position().y() - y_off
        if x < 0 or y < 0 or x >= pm_size.width() or y >= pm_size.height():
            return

        h, w, _ = self._display_img.shape
        img_x = int(x * w / pm_size.width())
        img_y = int(y * h / pm_size.height())
        
        if event.button() == Qt.MouseButton.LeftButton:
            self._points.append([img_x, img_y, (h, w)])
            self.points_placed.emit(self._points)
        
        if self._points:
            pre_display_img = self._display_img.copy()
            
            for x, y, (w, h) in self._points:
                cv2.circle(pre_display_img, (x,y), 3, (0,255,0), -1)
            
            self.set_frame(pre_display_img)


# ---- Optional: one-time GL configuration (call BEFORE creating QApplication) ----
def configure_qt_opengl_desktop_21():
    fmt = QSurfaceFormat()
    fmt.setRenderableType(QSurfaceFormat.RenderableType.OpenGL)  # Desktop GL
    fmt.setVersion(2, 1)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CompatibilityProfile)
    fmt.setDepthBufferSize(24)
    fmt.setStencilBufferSize(8)
    fmt.setSwapBehavior(QSurfaceFormat.SwapBehavior.DoubleBuffer)
    QSurfaceFormat.setDefaultFormat(fmt)

    # ДО QApplication:
    QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)


if HAVE_PG:
    class OrbitGLView(gl.GLViewWidget):
        """
        Trackball-style orbit around a target 'center'.
        LMB: orbit, RMB: pan, Wheel: dolly.
        """
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setBackgroundColor((10, 10, 14))
            self.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
            self.opts['elevation'] = 20
            self.opts['azimuth']   = 45
            self.opts['distance']  = 600
            self.opts['azimuth']  %= 360
            self._last_pos = None
            self._btn = None

        # --- mouse handling ---
        def mousePressEvent(self, ev):
            self._last_pos = ev.position()
            self._btn = ev.button()
            super().mousePressEvent(ev)

        def mouseReleaseEvent(self, ev):
            self._last_pos = None
            self._btn = None
            super().mouseReleaseEvent(ev)

        def _pan_scale(self):
            return max(1e-6, self.opts['distance'] * 0.5)

        def mouseMoveEvent(self, ev):
            if self._last_pos is None:
                return super().mouseMoveEvent(ev)

            d = ev.position() - self._last_pos
            if self._btn == Qt.MouseButton.LeftButton:
                self.orbit(-d.x() * 0.5, d.y() * 0.5)
            elif self._btn == Qt.MouseButton.RightButton:
                s = self._pan_scale()
                self.pan(-d.x() * s, d.y() * s, 0.0, relative="view-upright")
            self._last_pos = ev.position()
            super().mouseMoveEvent(ev)

        def wheelEvent(self, ev):
            factor = 1.0 - (ev.angleDelta().y() / 1200.0)
            factor = float(np.clip(factor, 0.8, 1.25))
            self.opts['distance'] *= factor
            self.update()
            super().wheelEvent(ev)

class ReconstructionViewer(QFrame):
    """
    Drop-in widget to visualize 3D point clouds (points + colors) using pyqtgraph.opengl.

    Public API:
      - set_pointcloud(points, colors, **kwargs): clears old layers and shows a single cloud
      - add_pointcloud(points, colors, **kwargs): adds another layer (overlay)
      - clear(): remove all layers
      - fit_to_points(margin=1.6): center & set camera distance based on current layers
      - set_point_size(size): update size for all layers
    """
    # Optional signals if you want to react from outside:
    layersChanged = pyqtSignal(int)

    def __init__(self, parent: QWidget | None = None, title: str = "3D Reconstruction"):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.Box)
        self.setLineWidth(3)

        self._layers = []  # type: list
        self._point_size = 2.2
        self._max_points_default = 250_000
        self._px_mode = True
        self._gl_options = "opaque"

        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)

        header = QHBoxLayout()
        self.lbl_title = QLabel(title)
        self.lbl_title.setStyleSheet("color: white; background:#333; padding:4px;")
        self.btn_fit   = QPushButton("Fit")
        self.btn_clear = QPushButton("Clear")
        self.spin_size = QDoubleSpinBox()
        self.spin_size.setRange(0.1, 20.0)
        self.spin_size.setSingleStep(0.1)
        self.spin_size.setValue(self._point_size)
        self.spin_size.setPrefix("• ")
        self.spin_size.setToolTip("Point size")

        header.addWidget(self.lbl_title, 1)
        header.addWidget(self.btn_fit, 0)
        header.addWidget(self.btn_clear, 0)
        header.addWidget(self.spin_size, 0)

        outer.addLayout(header)

        self._fallback_label = None

        if HAVE_PG:
            self.view = OrbitGLView(self)
            outer.addWidget(self.view, 1)
            self.btn_fit.clicked.connect(self.fit_to_points)
            self.btn_clear.clicked.connect(self.clear)
            self.spin_size.valueChanged.connect(self.set_point_size)
        else:
            # Graceful fallback if pyqtgraph isn't available
            self.view = None
            self._fallback_label = QLabel(
                "pyqtgraph / pyqtgraph.opengl not available.\n"
                "Install pyqtgraph and ensure OpenGL is working."
            )
            self._fallback_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._fallback_label.setStyleSheet("background:#111; color:#bbb; padding:18px;")
            outer.addWidget(self._fallback_label, 1)

        self.setLayout(outer)

    # ---------- Public API ----------
    def clear(self):
        if not HAVE_PG or self.view is None:
            return
        for it in self._layers:
            try:
                self.view.removeItem(it)
            except Exception:
                pass
        self._layers.clear()
        self.layersChanged.emit(0)

    def set_point_size(self, size: float):
        """Update point size for all layers."""
        self._point_size = float(size)
        if not HAVE_PG or self.view is None:
            return
        for it in self._layers:
            try:
                it.setData(size=self._point_size)
            except Exception:
                pass

    def fit_to_points(self, margin: float = 1.6):
        """Center camera on the union of all layer points."""
        if not HAVE_PG or self.view is None or not self._layers:
            return

        # Collect all points from layers (as stored inside GLScatterPlotItem)
        pts_list = []
        for it in self._layers:
            try:
                pts = it.pos  # ndarray (N,3)
                if pts is not None and len(pts) > 0:
                    pts_list.append(pts)
            except Exception:
                pass
        if not pts_list:
            return

        pts = np.vstack(pts_list)
        self._set_view_to_points(self.view, pts, margin=margin)

    def set_pointcloud(
        self,
        points: np.ndarray,
        colors: np.ndarray | None = None,
        *,
        max_points: int | None = None,
        point_size: float | None = None,
        px_mode: bool | None = None,
        gl_options: str | None = None,
        fit: bool = True,
        margin: float = 1.6,
    ):
        """
        Clear old layers and draw a single cloud.
        """
        self.clear()
        item = self.add_pointcloud(
            points, colors,
            max_points=max_points, point_size=point_size,
            px_mode=px_mode, gl_options=gl_options
        )
        if fit and item is not None:
            self.fit_to_points(margin=margin)
        return item

    def add_pointcloud(
        self,
        points: np.ndarray,
        colors: np.ndarray | None = None,
        *,
        max_points: int | None = None,
        point_size: float | None = None,
        px_mode: bool | None = None,
        gl_options: str | None = None,
    ):
        """
        Add another cloud layer (returns the GLScatterPlotItem).
        """
        if not HAVE_PG or self.view is None:
            return None

        if points is None or points.size == 0:
            return None

        pts = np.asarray(points, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("points must be (N, 3)")

        # Downsample if needed
        maxp = self._max_points_default if max_points is None else int(max_points)
        if len(pts) > maxp:
            idx = np.random.choice(len(pts), maxp, replace=False)
            pts = pts[idx]
            if colors is not None:
                colors = np.asarray(colors)[idx]

        # Colors -> RGBA float32 in [0,1]
        rgba = self._coerce_colors(colors, len(pts))

        size = self._point_size if point_size is None else float(point_size)
        px   = self._px_mode if px_mode is None else bool(px_mode)
        glo  = self._gl_options if gl_options is None else str(gl_options)

        item = gl.GLScatterPlotItem(
            pos=pts,
            size=size,
            color=rgba,
            pxMode=px,
            glOptions=glo,
        )
        self.view.addItem(item)
        self._layers.append(item)
        self.layersChanged.emit(len(self._layers))
        return item

    # ---------- Internals ----------
    @staticmethod
    def _set_view_to_points(view, pts: np.ndarray, margin: float = 1.5):
        lo = pts.min(axis=0)
        hi = pts.max(axis=0)
        center = (lo + hi) * 0.5
        span = float(np.linalg.norm(hi - lo))
        span = max(span, 1e-4)

        # Update orbit center & distance
        view.opts['center'] = pg.Vector(center[0], center[1], center[2])
        view.opts['distance'] = span * margin

    @staticmethod
    def _coerce_colors(colors: np.ndarray | None, n: int) -> np.ndarray:
        """
        Accepts:
          - None -> white
          - (N,3) or (N,4) RGB/RGBA, either 0..255 or 0..1
          - (3,) or (4,) broadcast to all points
        Returns float32 RGBA in [0,1].
        """
        if colors is None:
            rgba = np.ones((n, 4), dtype=np.float32)
            rgba[:, :3] = 1.0
            return rgba

        cols = np.asarray(colors)
        if cols.ndim == 1:
            cols = np.repeat(cols[None, :], n, axis=0)

        if cols.shape[0] != n:
            raise ValueError("colors length must match points length")

        if cols.shape[1] == 3:
            # Add alpha
            cols = np.c_[cols, np.ones((n, 1), dtype=cols.dtype)]
        elif cols.shape[1] != 4:
            raise ValueError("colors must be (..., 3) or (..., 4)")

        cols = cols.astype(np.float32, copy=False)
        # If likely 0..255, normalize
        if cols.max() > 1.5:
            cols = cols / 255.0
        cols = np.clip(cols, 0.0, 1.0)
        return cols