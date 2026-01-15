from plotly import graph_objects as go
import numpy as np
from PyQt6.QtCore import Qt

# visualization.py  (add near the top)
try:
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl
    _HAVE_PG = True
except Exception:
    _HAVE_PG = False
    
class OrbitGLView(gl.GLViewWidget):
    """
    Trackball-style orbit around a target 'center', with RMB pan and wheel zoom.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackgroundColor((10, 10, 14))
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        # Sensible defaults; you can tweak per scene after adding data
        self.opts['elevation'] = 20
        self.opts['azimuth']   = 45
        self.opts['distance']  = 600
        # Keep a stable 'up' so right-drag pans in a natural screen space
        self.opts['azimuth'] %= 360
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
        # scale pan speed with distance; bump up/down to taste
        return max(1e-6, self.opts['distance'] * 0.5)

    def mouseMoveEvent(self, ev):
        if self._last_pos is None:
            return super().mouseMoveEvent(ev)

        d = ev.position() - self._last_pos
        if self._btn == Qt.MouseButton.LeftButton:
            # Orbit around current 'center' (plotly-like)
            self.orbit(-d.x() * 0.5, d.y() * 0.5)
        elif self._btn == Qt.MouseButton.RightButton:
            # Pan in view-upright coordinates (keeps horizon sensible)
            s = self._pan_scale()
            self.pan(-d.x() * s, d.y() * s, 0.0, relative="view-upright")
        self._last_pos = ev.position()
        super().mouseMoveEvent(ev)

    def wheelEvent(self, ev):
        # Smooth dolly (zoom)
        factor = 1.0 - (ev.angleDelta().y() / 1200.0)
        factor = float(np.clip(factor, 0.8, 1.25))
        self.opts['distance'] *= factor
        self.update()
        super().wheelEvent(ev)


def create_gl_view(parent=None):
    """Return a GLViewWidget ready to accept scatter items."""
    # w = gl.GLViewWidget(parent=parent)
    # w.opts['distance'] = 600  # sensible default; you can tweak after adding data
    
    w = OrbitGLView(parent=parent)
    return w

def set_view_to_points(view: OrbitGLView, pts: np.ndarray, margin: float = 1.5):
    """
    Center the camera on the given point cloud and set a good distance.
    Call this after you add the first cloud (or whenever the scene changes a lot).
    """
    if pts.size == 0:
        return
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    center = (lo + hi) * 0.5
    span = float(np.linalg.norm(hi - lo))
    span = max(span, 1e-4)

    # update camera "target" and distance (orbit pivots around this center)
    view.opts['center'] = pg.Vector(center[0], center[1], center[2])
    view.opts['distance'] = span * margin    # margin ~ how much of the object fills the view

def plot_pointcloud_pyqtgraph_into(
    view, pts_flat, cols_flat,
    max_points=60000, point_size=2.5,
    px_mode=True,
    gl_options="opaque",
):
    """
    Add a downsampled scatter layer into an existing GLViewWidget.
    Returns the created GLScatterPlotItem so you can manage/remove later.
    """
    if pts_flat.size == 0:
        return None
    pts = pts_flat
    cols = cols_flat
    if len(pts) > max_points:
        idx = np.random.choice(len(pts), max_points, replace=False)
        pts, cols = pts[idx], cols[idx]

    # RGBA in [0,1]
    if cols.shape[1] == 3:
        rgba = np.c_[cols.astype(np.float32)/255.0, np.ones((len(cols), 1), np.float32)]
    else:
        rgba = cols.astype(np.float32)

    item = gl.GLScatterPlotItem(
        pos=pts.astype(np.float32),
        size=point_size,
        color=rgba,
        pxMode=px_mode,
        glOptions=gl_options,
    )
    view.addItem(item)

    # set a reasonable camera distance once
    try:
        span = np.linalg.norm(pts.max(0) - pts.min(0))
        if span > 0:
            view.opts['distance'] = span * 1.5
    except Exception:
        pass

    return item

def append_pointcloud_pyqtgraph(view, pts, cols, point_size=2.0, max_points=60000):
    """Convenience wrapper for adding another layer."""
    return plot_pointcloud_pyqtgraph_into(view, pts, cols, max_points=max_points, point_size=point_size)

def filter_by_confidence(pts, conf, cols, thresh):
    """Keep only points with confidence > thresh."""
    thresh_pc = np.percentile(conf, thresh*100)
    mask = conf > thresh_pc
    return pts[mask], cols[mask]

def filter_by_color_background(pts, cols,
                               black_bg=True, white_bg=True,
                               black_thresh=16, white_thresh=240):
    """Remove pure‑black or pure‑white background pixels."""
    mask = np.ones(len(cols), dtype=bool)
    if black_bg:
        mask &= cols.sum(axis=1) >= black_thresh
    if white_bg:
        mask &= ~((cols[:,0] > white_thresh) &
                  (cols[:,1] > white_thresh) &
                  (cols[:,2] > white_thresh))
    return pts[mask], cols[mask]

def filter_by_sky_mask(pts, cols, sky_mask):
    """Keep only points where sky_mask is True."""
    mask = np.ravel(sky_mask).astype(bool)
    return pts[mask], cols[mask]

def plot_pointcloud_plotly(
    pts_flat: np.ndarray,
    cols_flat: np.ndarray,
    width: int = None,
    height: int = None,
    conf_flat: np.ndarray = None,
    conf_thresh: float = 0.5,
    max_points: int = 60000,
    marker_size: float = 1.0,
    opacity: float = 1.0,
    conf_based_filter: bool = True,
    color_bg_removal: bool = False,
    black_bg: bool = True,
    white_bg: bool = True,
    sky_seg: bool = False,
    sky_mask: np.ndarray | None = None,
    downscale: float = 2.0,
    title="",
    name="",
    showlegend: bool = False,
) -> go.Figure:
    """
    Builds a Plotly 3D scatter of provided points, colors, and confidence values with optional filters.
    """
    # 1) Determine plot dimensions
    plot_width = int(width / downscale) if width is not None else None
    plot_height = int(height / downscale) if height is not None else None

    # 2) Apply filters
    if conf_based_filter and conf_flat is not None:
        pts_flat, cols_flat = filter_by_confidence(
            pts_flat, conf_flat, cols_flat, conf_thresh
        )
    if color_bg_removal:
        pts_flat, cols_flat = filter_by_color_background(
            pts_flat, cols_flat, black_bg=black_bg, white_bg=white_bg
        )
    if sky_seg:
        if sky_mask is None:
            raise KeyError("sky_mask is required for sky_seg=True")
        pts_flat, cols_flat = filter_by_sky_mask(
            pts_flat, cols_flat, sky_mask
        )

    # 3) Randomly downsample to max_points
    N = len(pts_flat)
    if N > max_points:
        idx = np.random.choice(N, max_points, replace=False)
        pts_flat, cols_flat = pts_flat[idx], cols_flat[idx]

    # 4) Build Plotly scatter
    hex_colors = [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in cols_flat]
    fig = go.Figure(data=go.Scatter3d(
        x=pts_flat[:, 0], y=pts_flat[:, 1], z=pts_flat[:, 2],
        mode='markers',
        marker=dict(size=marker_size, color=hex_colors, opacity=opacity),
        name=name,
    ))
    fig.update_layout(
        scene=dict(
            xaxis_showgrid=False, xaxis_showticklabels=False, xaxis_title='',
            yaxis_showgrid=False, yaxis_showticklabels=False, yaxis_title='',
            zaxis_showgrid=False, zaxis_showticklabels=False, zaxis_title='',
            aspectmode='data',
            dragmode='orbit',
        ),
        margin=dict(l=0, r=0, b=0, t=20),
        template='plotly_white',
        showlegend=showlegend,
        title="<br>" + title,
    )
    if plot_width is not None and plot_height is not None:
        fig.update_layout(width=plot_width, height=plot_height)
    return fig

def append_pointcloud(
    fig: go.Figure,
    pts: np.ndarray,
    cols: np.ndarray,
    max_points: int = 60000,
    name: str = "",
    marker_size: float = 1.0,
    opacity: float = 1.0,
):
    """
    Add a new 3D scatter trace to `fig` from pts (N,3) and cols (N,3 uint8).
    """
    if pts.size == 0:
        return
    if len(pts) > max_points:
        idx = np.random.choice(len(pts), max_points, replace=False)
        pts = pts[idx]
        cols = cols[idx]
    hex_colors = [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in cols]
    fig.add_trace(go.Scatter3d(
        x=pts[:,0], y=pts[:,1], z=pts[:,2],
        mode='markers',
        marker=dict(size=marker_size, color=hex_colors, opacity=opacity),
        name=name,
    ))
    
    return fig