import os
import sys
from pathlib import Path
import json
import numpy as np
import cv2
from typing import Dict, List, Tuple

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QLabel,
    QTextEdit, QVBoxLayout, QSplitter, QHBoxLayout, QDoubleSpinBox,
    QCheckBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QSurfaceFormat, QColor

# -------------------------------------
# Paths & repo imports (same layout as scaling_app)
# -------------------------------------
root_dir = Path(__file__).parents[4]
data_dir = root_dir / "data"
data_dir.mkdir(parents=True, exist_ok=True)
TEMP_DATA_DIR = data_dir / "temp_data"
PTS_SCALED_DIR = TEMP_DATA_DIR / "pts_scaled"

# Make project modules importable (same set used before)
for sub in [
    "vggt", "sam2_root",
    "dx_vyzai_python",
    "dx_vyzai_python/aux_codes/sawbone_surgical",
    "dx_vyzai_python/aux_codes/sawbone_surgical/code"
]:
    sys.path.append(str(root_dir / sub))

# Core helpers from user's modules
from visualization import (
    create_gl_view, plot_pointcloud_pyqtgraph_into, set_view_to_points
)
from geometry_processor import (
    PCAAnalyzer, pick_extremes_by_half, most_collinear, split_femur_knee_shaft
)
from vggt_processor import VGGTProcessor

# External deps
from sklearn.neighbors import NearestNeighbors

# -------------------------------------
# Utility: convert npz preds to torch tensors
# (kept lightweight to avoid importing torch here unless needed by VGGT)
# -------------------------------------
def to_torch_preds(preds_np: dict, device: str = "cpu") -> dict:
    try:
        import torch
    except Exception:
        raise RuntimeError("PyTorch is required to load camera poses from preds.npz")
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

# -------------------------------------
# Point-cloud filters & sampling
# -------------------------------------
def filter_low_density(pts: np.ndarray,
                       cols: np.ndarray,
                       k: int = 10,
                       percentile: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove points whose local density (1 / distance_to_kth_neighbor)
    falls below the given percentile.
    """
    if len(pts) == 0:
        return pts, cols
    nbrs = NearestNeighbors(n_neighbors=min(k+1, len(pts)), algorithm='auto').fit(pts)
    dists, _ = nbrs.kneighbors(pts)
    d_k = dists[:, -1]
    density = 1.0 / (d_k + 1e-8)
    thresh = np.percentile(density, percentile)
    mask = density >= thresh
    return pts[mask], cols[mask]


def limit_cloud_points(pts: np.ndarray, cols: np.ndarray, max_points: int = 250_000):
    if len(pts) > max_points:
        idx = np.random.choice(len(pts), max_points, replace=False)
        return pts[idx], cols[idx]
    return pts, cols

# -------------------------------------
# Geometry primitives (planes) and refinements (kept compact)
# -------------------------------------
class Plane:
    def __init__(self, A: float, B: float, C: float, D: float):
        n = np.linalg.norm([A, B, C])
        if n == 0:
            raise ValueError("Plane normal cannot be zero")
        self.A, self.B, self.C, self.D = A/n, B/n, C/n, D/n
        self.normal = np.array([self.A, self.B, self.C], dtype=float)

    @staticmethod
    def from_normal_point(normal: np.ndarray, point: np.ndarray) -> "Plane":
        n = np.asarray(normal, float)
        n /= (np.linalg.norm(n) + 1e-12)
        D = -float(n @ np.asarray(point, float))
        return Plane(n[0], n[1], n[2], D)

    @staticmethod
    def from_two_vectors(v1: np.ndarray, v2: np.ndarray, point: np.ndarray | None = None) -> "Plane":
        n = np.cross(np.asarray(v1, float), np.asarray(v2, float))
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            raise ValueError("v1 and v2 are parallel; cannot make a plane")
        n /= nn
        D = 0.0 if point is None else -float(n @ np.asarray(point, float))
        return Plane(n[0], n[1], n[2], D)

    def orient_away_from(self, centroid: np.ndarray) -> "Plane":
        if float(self.normal @ centroid + self.D) > 0:
            self.A, self.B, self.C, self.D = -self.A, -self.B, -self.C, -self.D
            self.normal = -self.normal
        return self

    def signed_distance(self, pts: np.ndarray) -> np.ndarray:
        return self.A*pts[:,0] + self.B*pts[:,1] + self.C*pts[:,2] + self.D

    def project_point(self, p: np.ndarray, distance: float) -> np.ndarray:
        return np.asarray(p, float) + self.normal * distance
    
    def points_above_plane(self, pts: np.ndarray, eps: float = 0.0) -> np.ndarray:
        """
        Points with signed distance > eps are considered above (in the direction of the plane normal).
        With your orient_away_from(centroid), 'above' means 'outside bone' consistently.
        """
        return self.signed_distance(pts) > eps



def refine_center(pts: np.ndarray, res: dict,
                  division_axis: str, projection_axis: str,
                  positive_dir: bool, tol: float = 1e-4, max_iter: int = 10) -> np.ndarray:
    div_n = res['axes'][division_axis]['dir']
    div_n = div_n / (np.linalg.norm(div_n) + 1e-12)
    c_cur = res['centroid'].copy()
    for _ in range(max_iter):
        res['centroid'] = c_cur
        pa, pb = pick_extremes_by_half(
            pts, res,
            division_axis=division_axis,
            projection_axis=projection_axis,
            positive_dir=positive_dir
        )
        sd_a = (pa - c_cur) @ div_n
        sd_b = (pb - c_cur) @ div_n
        shift = 0.5 * (sd_a + sd_b)
        if abs(shift) < tol:
            break
        c_cur = c_cur + div_n * shift
    return c_cur


def refine_axes_in_plane(res: dict, pts: np.ndarray,
                         axis_primary: str, axis_secondary: str,
                         pt1: np.ndarray, pt2: np.ndarray) -> dict:
    all_axes = list(res['axes'].keys())
    axis_other = next(a for a in all_axes if a not in (axis_primary, axis_secondary))
    v1_old = res['axes'][axis_primary]['dir']
    v2_old = res['axes'][axis_secondary]['dir']
    v3     = res['axes'][axis_other]['dir']
    k = v3 / (np.linalg.norm(v3) + 1e-12)
    t = pt2 - pt1
    t_proj = t - (t @ k) * k
    if np.linalg.norm(t_proj) < 1e-8:
        return res
    v1_new = t_proj / np.linalg.norm(t_proj)
    if v1_old @ v1_new < 0:
        v1_old = -v1_old
    cosang = float(v1_old @ v1_new)
    sinang = float(np.cross(v1_old, v1_new) @ k)
    K = np.array([[    0, -k[2],  k[1]],
                  [ k[2],     0, -k[0]],
                  [-k[1],  k[0],     0]])
    R = np.eye(3) + K * sinang + (K @ K) * (1 - cosang)
    v1r = R @ v1_old; v2r = R @ v2_old
    v1r /= (np.linalg.norm(v1r) + 1e-12)
    v2r /= (np.linalg.norm(v2r) + 1e-12)
    res['axes'][axis_primary]['dir']   = v1r
    res['axes'][axis_secondary]['dir'] = v2r
    res['axes'][axis_other]['dir']     = k
    c = res['centroid']
    for info in res['axes'].values():
        d = info['dir']
        proj = (pts - c) @ d
        ext = proj.max() - proj.min()
        info['extent'] = float(ext)
        info['start'] = c - d*(ext/2)
        info['end']   = c + d*(ext/2)
    return res


def refine_axes_by_midpoints(res: dict, pts: np.ndarray,
                             axis_primary: str, axis_secondary: str,
                             p1: np.ndarray, p2: np.ndarray) -> dict:
    return refine_axes_in_plane(res, pts, axis_primary, axis_secondary, p1, p2)


def refine_centroid_along_axis(res: dict, axis: str, targets: List[np.ndarray]) -> dict:
    c = res['centroid']
    d = res['axes'][axis]['dir']
    d = d / (np.linalg.norm(d) + 1e-12)
    mean_pt = np.mean(np.stack(targets, 0), axis=0)
    shift = float((mean_pt - c) @ d)
    c_new = c + d * shift
    res['centroid'] = c_new
    for info in res['axes'].values():
        dir_vec = info['dir']; ext = info['extent']
        info['start'] = c_new - dir_vec * (ext/2)
        info['end']   = c_new + dir_vec * (ext/2)
    return res

def move_cut_regions_away(
    pts: np.ndarray,
    planes: list[Plane],
    distance_mm: float = 15.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each point, detect which planes it is 'above' (positive side),
    average those plane normals, and move the point outward by distance_mm.
    """
    if len(pts) == 0 or len(planes) == 0:
        return pts.copy(), np.zeros_like(pts)

    pts_moved = pts.copy()
    to_move = np.zeros((len(pts), len(planes)), dtype=bool)

    # mark “above” points for every plane (strictly >0; our cut patches will get a tiny +0.01 normal offset)
    for idx, plane in enumerate(planes):
        to_move[:, idx] = plane.points_above_plane(pts)

    normals = np.array([p.normal for p in planes], dtype=float)  # (P,3)
    move_direction = to_move @ normals  # (N,3) sum of normals for each point

    # normalize non-zero directions
    norms = np.linalg.norm(move_direction, axis=1, keepdims=True)
    non_null = norms[:, 0] > 1e-12
    move_direction[non_null] /= norms[non_null]

    pts_moved[non_null] += move_direction[non_null] * float(distance_mm)
    return pts_moved, move_direction


def generate_bone_cut_patches(
    pts: np.ndarray,
    planes: List[Plane],
    patch_colors: List = None,
    patch_names: List[str] = None,
    fill_density: float = 0.2,
    tol: float = 0.5
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Build dense, colored fill points for each cut by projecting near-plane points,
    building a 2D convex hull, and sampling a grid inside the hull.
    Returns {name: {"pts": (M,3) float, "col": (M,3) uint8}}
    """
    from scipy.spatial import ConvexHull
    from matplotlib.path import Path

    if patch_names is None:
        patch_names = [f"Cut Surface {i+1}" for i in range(len(planes))]

    # default: red for all
    if patch_colors is None:
        patch_colors = [(255, 0, 0)] * len(planes)

    # normalize colors to (r,g,b) tuples
    def _as_rgb(c):
        if isinstance(c, (tuple, list)) and len(c) == 3:
            return tuple(int(v) for v in c)
        if isinstance(c, str):
            c = c.strip()
            if c.startswith("#") and len(c) >= 7:
                return (int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16))
            if c.startswith("rgb"):
                nums = c.strip("rgb()").split(",")
                return tuple(int(float(v)) for v in nums[:3])
        return (255, 0, 0)

    rgb_colors = [_as_rgb(c) for c in patch_colors]

    out: Dict[str, Dict[str, np.ndarray]] = {}
    if len(pts) == 0:
        return out

    for plane, color, name in zip(planes, rgb_colors, patch_names):
        d = plane.signed_distance(pts)
        near_mask = np.abs(d) <= float(tol)
        if np.count_nonzero(near_mask) < 3:
            continue

        cut_pts = pts[near_mask]  # points near the plane
        d_exact = plane.signed_distance(cut_pts)[:, None]
        proj = cut_pts - d_exact * plane.normal[None, :]

        # basis on plane
        arb = np.array([1.0, 0.0, 0.0])
        if abs(plane.normal @ arb) > 0.9:
            arb = np.array([0.0, 1.0, 0.0])
        u = np.cross(plane.normal, arb); u /= (np.linalg.norm(u) + 1e-12)
        v = np.cross(plane.normal, u);   v /= (np.linalg.norm(v) + 1e-12)

        center = proj.mean(axis=0)
        pts2 = np.c_[ (proj - center) @ u, (proj - center) @ v ]

        try:
            hull = ConvexHull(pts2)
            poly = pts2[hull.vertices]
            x_min, y_min = poly.min(axis=0)
            x_max, y_max = poly.max(axis=0)
            area = max((x_max - x_min) * (y_max - y_min), 1e-6)

            # smaller fill_density -> denser grid
            target_points = int(max(200, area / max(fill_density, 1e-6)))
            grid_n = int(np.sqrt(target_points))
            xs = np.linspace(x_min, x_max, grid_n)
            ys = np.linspace(y_min, y_max, grid_n)
            XX, YY = np.meshgrid(xs, ys)
            grid2 = np.c_[XX.ravel(), YY.ravel()]

            inside = Path(poly).contains_points(grid2)
            in2 = grid2[inside]

            fill_pts = center + in2[:, 0, None] * u + in2[:, 1, None] * v
            # tiny push so they are definitely 'above' for moving step
            fill_pts = fill_pts + plane.normal[None, :] * 0.01

            fill_cols = np.tile(np.array(color, dtype=np.uint8), (len(fill_pts), 1))
            out[name] = {"pts": fill_pts.astype(np.float32), "col": fill_cols}
        except Exception:
            # fallback: use the projected near points (still offset)
            fill_pts = proj + plane.normal[None, :] * 0.01
            fill_cols = np.tile(np.array(color, dtype=np.uint8), (len(fill_pts), 1))
            out[name] = {"pts": fill_pts.astype(np.float32), "col": fill_cols}

    return out


# -------------------------------------
# Cut set builder (distal/proximal + chamfers + anterior)
# -------------------------------------
class ImplantCuts:
    def __init__(self, centroid: np.ndarray,
                 AP_wo_CondThick: float = 37.37,
                 DistCut_length: float = 19.05,
                 AntChamfCut_length: float = 18.6,
                 post_to_post_chamfer: float = 50.0,
                 dist_to_ant_chamfer: float = 44.0,
                 ant_to_ant_chamfer: float = 41.0):
        self.centroid = np.asarray(centroid, float)
        self.AP_wo_CondThick = float(AP_wo_CondThick)
        self.DistCut_length = float(DistCut_length)
        self.AntChamfCut_length = float(AntChamfCut_length)
        self.post_to_post_chamfer = float(post_to_post_chamfer)
        self.dist_to_ant_chamfer = float(dist_to_ant_chamfer)
        self.ant_to_ant_chamfer = float(ant_to_ant_chamfer)

        r1 = np.deg2rad(self.post_to_post_chamfer)
        r2 = np.deg2rad(self.dist_to_ant_chamfer)
        self._PostChamfCut_projected_height = (
            self.AP_wo_CondThick
            - self.AntChamfCut_length * np.cos(r2)
            - self.DistCut_length
        )
        s = np.sin(r1)
        if abs(s) < 1e-8:
            raise ValueError("post_to_post_chamfer leads to sin≈0")
        self.PostChamfCut_length = self._PostChamfCut_projected_height / s

        self.normals = None
        self.k = None  # rotation axis (posterior → distal)

    @staticmethod
    def _n(v):
        v = np.asarray(v, float)
        n = np.linalg.norm(v) + 1e-12
        return v / n

    @staticmethod
    def _to_radians(deg: float) -> float:
        return np.deg2rad(deg)

    def compute_cut_normals(self, distal_normal: np.ndarray, proximal_normal: np.ndarray) -> dict[str, np.ndarray]:
        # Unitize inputs
        n_dist = self._n(distal_normal)
        n_prox = self._n(proximal_normal)

        # (Optional) enforce orthogonality if slightly skewed
        dp = float(n_dist @ n_prox)
        if abs(dp) > 1e-3:
            n_dist = self._n(n_dist - dp * n_prox)

        # Right-handed axis so rotating n_prox → n_dist is +CCW
        self.k = self._n(np.cross(n_prox, n_dist))

        def rotate_k(v: np.ndarray, angle_deg: float) -> np.ndarray:
            th = self._to_radians(angle_deg)
            k = self.k
            return self._n(
                v*np.cos(th) + np.cross(k, v)*np.sin(th) + k*(k@v)*(1-np.cos(th))
            )

        normals = {
            "posterior"    : n_prox,
            "post_chamfer" : rotate_k(n_prox, self.post_to_post_chamfer),
            "distal"       : n_dist,
            "ant_chamfer"  : rotate_k(n_dist, self.dist_to_ant_chamfer),
        }
        normals["anterior"] = rotate_k(normals["ant_chamfer"], self.ant_to_ant_chamfer)
        self.normals = normals
        return normals

    @staticmethod
    def _three_plane_intersection(p1: Plane, p2: Plane, p3: Plane) -> np.ndarray:
        A = np.vstack([p1.normal, p2.normal, p3.normal])
        b = -np.array([p1.D, p2.D, p3.D], float)
        if np.linalg.matrix_rank(A) < 3:
            raise ValueError("Planes do not intersect at a single point")
        return np.linalg.solve(A, b)

    def _normals_plane(self) -> Plane:
        # Keep the same ordering as the rotation axis definition (posterior, distal)
        return Plane.from_two_vectors(self.normals["posterior"], self.normals["distal"], point=self.centroid)

    def _compute_post_chamf_cut(self, distal_plane: Plane, proximal_plane: Plane) -> Plane:
        if self.normals is None:
            raise ValueError("compute_cut_normals() first")
        normals_plane = self._normals_plane()
        start = self._three_plane_intersection(normals_plane, distal_plane, proximal_plane)
        move = (
            self.PostChamfCut_length
            * np.cos(self._to_radians(self.post_to_post_chamfer))
            * np.sin(self._to_radians(self.dist_to_ant_chamfer))
        )
        cut_pt = start - move * self.normals["post_chamfer"]
        return Plane.from_normal_point(self.normals["post_chamfer"], cut_pt).orient_away_from(self.centroid)

    def _compute_ant_chamf_cut(self, distal_plane: Plane, proximal_plane: Plane, post_chamf_plane: Plane) -> Plane:
        if self.normals is None:
            raise ValueError("compute_cut_normals() first")
        normals_plane = self._normals_plane()
        start = self._three_plane_intersection(normals_plane, distal_plane, post_chamf_plane)
        # Treat AntChamfCut_length as AP/normal offset along the posterior/proximal direction
        cut_pt = start - self.AntChamfCut_length * self.normals["posterior"]
        return Plane.from_normal_point(self.normals["ant_chamfer"], cut_pt).orient_away_from(self.centroid)

    def _compute_ant_cut(self, distal_plane: Plane, proximal_plane: Plane, ant_chamf_plane: Plane) -> Plane:
        if self.normals is None:
            raise ValueError("compute_cut_normals() first")
        normals_plane = self._normals_plane()
        start = self._three_plane_intersection(normals_plane, ant_chamf_plane, distal_plane)

        # Direction along the chamfer edge (90° inside normals-plane, consistent axis)
        dist_to_ant_dir = self._n(self.normals["ant_chamfer"] - self.normals["distal"])
        k = self.k
        th = self._to_radians(90.0)
        K = np.array([[0, -k[2],  k[1]],
                      [k[2],  0, -k[0]],
                      [-k[1], k[0],  0]])
        v_along = self._n(dist_to_ant_dir*np.cos(th) + (K @ dist_to_ant_dir)*np.sin(th) + k*(k@dist_to_ant_dir)*(1-np.cos(th)))

        cut_pt = start + v_along * self.AntChamfCut_length
        return Plane.from_normal_point(self.normals["anterior"], cut_pt).orient_away_from(self.centroid)

    def compute_all_planes(self, distal_plane: Plane, proximal_plane: Plane) -> dict[str, Plane]:
        """
        Convenience: build the three derived planes. The caller already has distal/proximal.
        Returns: {'proximal','post_chamfer','distal','ant_chamfer','anterior'}
        """
        if self.normals is None:
            raise ValueError("compute_cut_normals() first")
        post_chamf = self._compute_post_chamf_cut(distal_plane, proximal_plane)
        ant_chamf  = self._compute_ant_chamf_cut(distal_plane, proximal_plane, post_chamf)
        ant        = self._compute_ant_cut(distal_plane, proximal_plane, ant_chamf)
        return {
            "proximal"     : proximal_plane,
            "post_chamfer" : post_chamf,
            "distal"       : distal_plane,
            "ant_chamfer"  : ant_chamf,
            "anterior"     : ant,
        }



    def _three_plane_intersection(self, p1: Plane, p2: Plane, p3: Plane) -> np.ndarray:
        A = np.vstack([p1.normal, p2.normal, p3.normal])
        b = -np.array([p1.D, p2.D, p3.D], float)
        if np.linalg.matrix_rank(A) < 3:
            raise ValueError("Planes do not intersect at a single point")
        return np.linalg.solve(A, b)

    def posterior_chamfer_plane(self, distal: Plane, proximal: Plane) -> Plane:
        if self.normals is None:
            raise ValueError("call compute_normals first")
        normals_plane = Plane.from_two_vectors(distal.normal, proximal.normal, point=self.centroid)
        start = self._three_plane_intersection(normals_plane, distal, proximal)
        move = self.PostChamfCut_length * np.cos(np.deg2rad(self.post_to_post_chamfer)) * np.sin(np.deg2rad(self.dist_to_ant_chamfer))
        cut_point = start - move * self.normals['post_chamfer']
        return Plane.from_normal_point(self.normals['post_chamfer'], cut_point).orient_away_from(self.centroid)

    def anterior_chamfer_plane(self, distal: Plane, proximal: Plane, post_chamf: Plane) -> Plane:
        normals_plane = Plane.from_two_vectors(distal.normal, proximal.normal, point=self.centroid)
        start = self._three_plane_intersection(normals_plane, distal, post_chamf)
        cut_point = start - self.AntChamfCut_length * proximal.normal
        return Plane.from_normal_point(self.normals['ant_chamfer'], cut_point).orient_away_from(self.centroid)

    def anterior_plane(self, distal: Plane, ant_chamf: Plane) -> Plane:
        normals_plane = Plane.from_two_vectors(distal.normal, self.normals['posterior'], point=self.centroid)
        # Recreate vector along chamfer direction (90° from dist->ant_chamf in normals_plane)
        dist_to_ant_chamfer_dir = self.normals['ant_chamfer'] - self.normals['distal']
        dist_to_ant_chamfer_dir /= (np.linalg.norm(dist_to_ant_chamfer_dir) + 1e-12)
        k = normals_plane.normal
        th = np.deg2rad(90.0)
        K = np.array([[0, -k[2], k[1]],[k[2], 0, -k[0]],[-k[1], k[0], 0]])
        R = np.eye(3) + K * np.sin(th) + (K @ K) * (1 - np.cos(th))
        v_along = R @ dist_to_ant_chamfer_dir
        v_along /= (np.linalg.norm(v_along) + 1e-12)
        # intersection of normals_plane, ant_chamf, distal
        start = self._three_plane_intersection(normals_plane, ant_chamf, distal)
        cut_point = start + v_along * self.AntChamfCut_length
        return Plane.from_normal_point(self.normals['anterior'], cut_point).orient_away_from(self.centroid)

# -------------------------------------
# Main Window
# -------------------------------------
class CutPlanesWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Femur Cut Planes — Visualization")
        self.resize(1200, 700)

        # UI: left controls, right GL + log
        cen = QWidget(); self.setCentralWidget(cen)
        main = QVBoxLayout(cen)
        split = QSplitter(Qt.Orientation.Horizontal)
        main.addWidget(split, 1)

        left = QWidget(); L = QVBoxLayout(left)
        row = QHBoxLayout()
        row.addWidget(QLabel("Fill density (lower=denser):"))
        self.spn_density = QDoubleSpinBox(); self.spn_density.setRange(0.01, 5.0); self.spn_density.setDecimals(3)
        self.spn_density.setSingleStep(0.01); self.spn_density.setValue(0.15)
        row.addWidget(self.spn_density)
        L.addLayout(row)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Plane tol (mm):"))
        self.spn_tol = QDoubleSpinBox(); self.spn_tol.setRange(0.01, 10.0); self.spn_tol.setDecimals(2)
        self.spn_tol.setSingleStep(0.05); self.spn_tol.setValue(0.50)
        row2.addWidget(self.spn_tol)
        L.addLayout(row2)
        
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Move distance (mm):"))
        self.spn_move = QDoubleSpinBox()
        self.spn_move.setRange(0.0, 100.0)
        self.spn_move.setDecimals(1)
        self.spn_move.setSingleStep(1.0)
        self.spn_move.setValue(20.0)
        row3.addWidget(self.spn_move)
        L.addLayout(row3)
        self.chk_shaft_present = QCheckBox("Shaft present (remove shaft)")
        
        self.chk_shaft_present.setToolTip(
            "If checked, the femur cloud will be auto-split and the shaft removed before computing cut planes."
        )
        self.chk_shaft_present.setChecked(False)
        L.addWidget(self.chk_shaft_present)


        self.btn_render = QPushButton("Compute & Render")
        self.btn_render.clicked.connect(self.compute_and_render)
        L.addWidget(self.btn_render)

        self.btn_close = QPushButton("Quit")
        self.btn_close.clicked.connect(self.close)
        L.addWidget(self.btn_close)

        L.addStretch(1)
        split.addWidget(left)

        right = QWidget(); R = QVBoxLayout(right)
        self.gl = create_gl_view(); R.addWidget(self.gl, 1)
        self.log = QTextEdit(); self.log.setReadOnly(True); self.log.setFixedHeight(150)
        R.addWidget(self.log)
        split.addWidget(right)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 5)

        # state placeholders
        self.f_pts = None
        self.f_cols = None
        self.preds_torch = None
        self.vggt = None
        self._gl_layers = []

        # Try immediate render
        # self.compute_and_render()
        self.log.append("Ready. Click 'Compute & Render' to start.")

    # ---------------------------------
    # Data loading
    # ---------------------------------
    def _load_femur_cloud(self) -> tuple[np.ndarray, np.ndarray]:
        # Prefer scaled femur
        cand = []
        if PTS_SCALED_DIR.exists():
            cand.extend(sorted(PTS_SCALED_DIR.glob("*femur*_points.npy")))
            cand.extend(sorted(PTS_SCALED_DIR.glob("*Femur*_points.npy")))
        for p in cand:
            cols = None
            cpath = p.with_name(p.stem.replace("_points", "_colors") + ".npy")
            if cpath.exists():
                cols = np.load(cpath)
            else:
                cols = np.full((len(np.load(p)), 3), 200, np.uint8)
            return np.load(p), cols
        # Fallback to checkpoints
        ckpt = data_dir / "reconstr_checkpoint_cadaver"
        fpts = ckpt / "femur_points.npy"
        fcols = ckpt / "femur_colors.npy"
        if fpts.exists() and fcols.exists():
            return np.load(fpts), np.load(fcols)
        raise FileNotFoundError("Femur cloud not found in pts_scaled/ or reconstr_checkpoint_cadaver/")

    def _load_camera_axes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Load preds and query VGGT for camera axes
        npz_path = TEMP_DATA_DIR / "preds.npz"
        if not npz_path.exists():
            raise FileNotFoundError("preds.npz not found in temp_data; needed for camera axes")
        npz = np.load(npz_path)
        preds_np = {k: npz[k] for k in npz.files}
        self.preds_torch = to_torch_preds(preds_np, device="cpu")
        self.vggt = VGGTProcessor(skip_model=True)
        cam_positions, x_axes, y_axes, z_axes = self.vggt.get_camera_poses(self.preds_torch)
        # handle potential batch dims
        xa = np.asarray(x_axes); ya = np.asarray(y_axes); za = np.asarray(z_axes)
        if xa.ndim == 3: xa = xa[0]
        if ya.ndim == 3: ya = ya[0]
        if za.ndim == 3: za = za[0]
        return xa, ya, za

    # ---------------------------------
    # Pipeline & rendering
    # ---------------------------------
    def compute_and_render(self):
        # Clear old GL layers
        for it in getattr(self, "_gl_layers", []):
            try: self.gl.removeItem(it)
            except Exception: pass
        self._gl_layers = []

        # -----------------------
        # Load data
        # -----------------------
        try:
            f_pts_full, f_cols_full = self._load_femur_cloud()
            self.log.append(f"Loaded femur: {len(f_pts_full)} vts")
        except Exception as e:
            self.log.append(f"(error) {e}")
            return

        # Light density filter for robustness (apply on full before optional splitting)
        f_pts_full, f_cols_full = filter_low_density(
            f_pts_full, f_cols_full, k=10, percentile=21.0
        )
        self.log.append(f"After density filter: {len(f_pts_full)} vts")

        # -----------------------
        # Camera axes (for div/proj/dist selection)
        # -----------------------
        try:
            x_axes, y_axes, z_axes = self._load_camera_axes()
            avg_x = x_axes.mean(axis=0)
            avg_y = y_axes.mean(axis=0)
            avg_z = z_axes.mean(axis=0)
            cam_ok = True
        except Exception as e:
            self.log.append(f"(warn) Camera axes not available, fallback to PCA defaults: {e}")
            avg_x = avg_y = avg_z = None
            cam_ok = False

        # -----------------------
        # PCA on full cloud (for splitting)
        # -----------------------
        pca_res_full = PCAAnalyzer.analyze(f_pts_full)
        axes_full = pca_res_full['axes']
        axes_array_full = [axes_full[a]['dir'] for a in ('length','width','height')]

        if cam_ok:
            div_vec, div_sign, _  = most_collinear(axes_array_full, avg_x)
            proj_vec, proj_sign, _ = most_collinear(axes_array_full, avg_y)
            dist_vec, dist_sign, _ = most_collinear(axes_array_full, avg_z)
            div_axis_name = next(a for a in ('length','width','height') if np.allclose(axes_full[a]['dir'], div_vec))
            proj_axis_name_guess = next(a for a in ('length','width','height') if np.allclose(axes_full[a]['dir'], proj_vec))
            dist_axis_name_guess = next(a for a in ('length','width','height') if np.allclose(axes_full[a]['dir'], dist_vec))
            prox_positive_dir_guess = (proj_sign == 1)
            dist_positive_dir_guess = (dist_sign == -1)
        else:
            # Fallback guesses (same as your current behavior)
            div_axis_name = 'length'
            proj_axis_name_guess = 'height'
            dist_axis_name_guess = 'height'
            prox_positive_dir_guess = True
            dist_positive_dir_guess = False

        # -----------------------
        # Optional: split & drop shaft
        # -----------------------
        if self.chk_shaft_present.isChecked():
            try:
                knee_pts, knee_cols, shaft_pts, shaft_cols, dbg = split_femur_knee_shaft(
                    f_pts=f_pts_full,
                    f_cols=f_cols_full,
                    pca_res=pca_res_full,
                    div_axis_name=div_axis_name,   # from camera mapping/fallback
                    n_bins=200,
                    knee_head_frac=0.15,
                    min_drop=0.25,
                    persist=6,
                    smooth_win=7,
                    p_low=5.0,
                    p_high=95.0,
                    enforce_knee_side_by_width=True
                )
                # Choose knee/condyles and drop shaft
                if len(knee_pts) >= 500:   # simple sanity floor to avoid degenerate splits
                    f_pts, f_cols = knee_pts, knee_cols
                    self.log.append(
                        f"Shaft checkbox ON ⇒ using knee-only region: {len(knee_pts)} pts (shaft {len(shaft_pts)} pts dropped)"
                    )
                else:
                    f_pts, f_cols = f_pts_full, f_cols_full
                    self.log.append(
                        "(warn) Split produced too few knee points; falling back to full femur."
                    )
            except Exception as e:
                f_pts, f_cols = f_pts_full, f_cols_full
                self.log.append(f"(warn) Shaft split failed ({e}); using full femur.")
        else:
            f_pts, f_cols = f_pts_full, f_cols_full
            self.log.append("Shaft checkbox OFF ⇒ using full femur point cloud.")

        # -----------------------
        # Recompute PCA on the final working cloud
        # -----------------------
        pca_res = PCAAnalyzer.analyze(f_pts)
        axes = pca_res['axes']

        # Re-derive camera axis alignment on the *current* cloud (safer than reusing guesses)
        if cam_ok:
            axes_array = [axes[a]['dir'] for a in ('length','width','height')]
            try:
                div_vec, div_sign, _  = most_collinear(axes_array, avg_x)
                proj_vec, proj_sign, _ = most_collinear(axes_array, avg_y)
                dist_vec, dist_sign, _ = most_collinear(axes_array, avg_z)
                div_axis_name = next(a for a in ('length','width','height') if np.allclose(axes[a]['dir'], div_vec))
                proj_axis_name = next(a for a in ('length','width','height') if np.allclose(axes[a]['dir'], proj_vec))
                dist_axis_name = next(a for a in ('length','width','height') if np.allclose(axes[a]['dir'], dist_vec))
                prox_positive_dir = (proj_sign == 1)
                dist_positive_dir = (dist_sign == -1)
            except Exception as e:
                self.log.append(f"(warn) Camera-based axis mapping failed on current cloud: {e}")
                div_axis_name   = div_axis_name            # keep previous
                proj_axis_name  = proj_axis_name_guess
                dist_axis_name  = dist_axis_name_guess
                prox_positive_dir = prox_positive_dir_guess
                dist_positive_dir = dist_positive_dir_guess
        else:
            div_axis_name   = div_axis_name
            proj_axis_name  = proj_axis_name_guess
            dist_axis_name  = dist_axis_name_guess
            prox_positive_dir = prox_positive_dir_guess
            dist_positive_dir = dist_positive_dir_guess

        # -----------------------
        # (everything below remains your original logic)
        # Proximal/distal extreme picks, plane builds, visualization, etc…
        # -----------------------
        # Proximal extremes (initial)
        prox1, prox2 = pick_extremes_by_half(
            pts=f_pts, pca_res=pca_res,
            division_axis=div_axis_name,
            projection_axis=proj_axis_name,
            positive_dir=prox_positive_dir,
        )

        # Center refine
        c_ref = refine_center(f_pts, pca_res, div_axis_name, proj_axis_name, prox_positive_dir)
        pca_res['centroid'] = c_ref
        pca_res = refine_axes_in_plane(pca_res, f_pts, div_axis_name, proj_axis_name, prox1, prox2)

        # Re-pick extremes with refined axes
        prox1, prox2 = pick_extremes_by_half(f_pts, pca_res, div_axis_name, proj_axis_name, prox_positive_dir)
        dist1, dist2 = pick_extremes_by_half(f_pts, pca_res, div_axis_name, dist_axis_name, dist_positive_dir)

        # Midpoints-guided tiny refinement
        mid1 = (prox1 + dist1) / 2
        mid2 = (prox2 + dist2) / 2
        pca_res = refine_axes_by_midpoints(pca_res, f_pts, div_axis_name, proj_axis_name, mid1, mid2)
        pca_res = refine_centroid_along_axis(pca_res, div_axis_name, [mid1, mid2])
        
        prox1, prox2 = pick_extremes_by_half(f_pts, pca_res, div_axis_name, proj_axis_name, prox_positive_dir)
        dist1, dist2 = pick_extremes_by_half(f_pts, pca_res, div_axis_name, dist_axis_name, dist_positive_dir)
        
        axes = pca_res['axes']
        centroid = pca_res['centroid']

        w_dir = -axes[proj_axis_name]["dir"] if prox_positive_dir else axes[proj_axis_name]["dir"]
        h_dir = -axes[dist_axis_name]["dir"] if dist_positive_dir else axes[dist_axis_name]["dir"]
        
        # Pick proximal-most between (prox1, prox2): argmax along w_dir
        proj_w = [ (p - centroid) @ w_dir for p in (prox1, prox2) ]
        most_prox = prox1 if proj_w[0] >= proj_w[1] else prox2

        # Pick distal-most between (dist1, dist2): argmin along h_dir
        proj_h = [ (p - centroid) @ h_dir for p in (dist1, dist2) ]
        most_dist = dist1 if proj_h[0] <= proj_h[1] else dist2

        # Place planes slightly away from the cortex to avoid z-fighting; then orient consistently
        proximal_plane = Plane.from_normal_point(w_dir, most_prox + w_dir * 5.0).orient_away_from(centroid)
        
        unmoved_distal_plane   = Plane.from_normal_point(h_dir, most_dist).orient_away_from(centroid)
        unmoved_sign = np.sign(unmoved_distal_plane.signed_distance(
            centroid.reshape(1, 3)
        ))[0]
        distal_plane = Plane.from_normal_point(h_dir, most_dist + h_dir * 6.0).orient_away_from(centroid)
        shifted_sign = np.sign(distal_plane.signed_distance(
            centroid.reshape(1, 3)
        ))[0]

        if not np.allclose(distal_plane.normal, unmoved_distal_plane.normal):
            # rotate distal plane normal to match unmoved
            distal_plane.A, distal_plane.B, distal_plane.C = unmoved_distal_plane.normal
            distal_plane.normal = unmoved_distal_plane.normal

        # Build implant cut family
        cuts = ImplantCuts(centroid=centroid)
        normals = cuts.compute_cut_normals(distal_plane.normal, proximal_plane.normal)
        post_chamf = cuts.posterior_chamfer_plane(distal_plane, proximal_plane)
        ant_chamf  = cuts.anterior_chamfer_plane(distal_plane, proximal_plane, post_chamf)
        ant        = cuts.anterior_plane(distal_plane, ant_chamf)

        cut_planes = [
            ("Posterior Cut", proximal_plane.normal),                # as a normal only (ref plane below)
            ("Posterior Chamfer Cut", post_chamf.normal),          # normals for legend convenience
            ("Distal Cut", distal_plane.normal),
            ("Anterior Chamfer Cut", ant_chamf.normal),
            ("Anterior Cut", ant.normal)
        ]
        # For fills we need the actual plane objects
        plane_objs = [
            proximal_plane,
            post_chamf,
            distal_plane,
            ant_chamf,
            ant,
        ]

        # ---- Visualize (pyqtgraph): femur + colored fills ----


        patch_names = [
            "Posterior Cut",
            "Posterior Chamfer Cut",
            "Distal Cut",
            "Anterior Chamfer Cut",
            "Anterior Cut",
        ]
        patch_colors = [
            (255,   0,   0),  # posterior
            (  0, 255,   0),  # posterior chamfer
            (  0,   0, 255),  # distal
            (255,   0, 255),  # anterior chamfer
            (255, 255,   0),  # anterior
        ]

        move_dist = float(self.spn_move.value())
        dens = float(self.spn_density.value())
        tol  = float(self.spn_tol.value())

        # 1) Move femur points away from the cut planes
        f_pts_moved, f_move_dir = move_cut_regions_away(f_pts, plane_objs, distance_mm=move_dist)
        femur_pts_vis, femur_cols_vis = limit_cloud_points(f_pts_moved, f_cols, max_points=300_000)
        it0 = plot_pointcloud_pyqtgraph_into(self.gl, femur_pts_vis.astype(np.float32), femur_cols_vis,
                                             max_points=len(femur_pts_vis), point_size=2.0)
        if it0: self._gl_layers.append(it0)
        set_view_to_points(self.gl, f_pts, margin=1.6)

        # 2) Build cut patches on the ORIGINAL cloud (so we can find plane intersections),
        # then move the patch fill points by the same rule.
        patches = generate_bone_cut_patches(
            pts=f_pts,
            planes=plane_objs,
            patch_colors=patch_colors,
            patch_names=patch_names,
            fill_density=dens,
            tol=tol
        )

        # 3) Move each patch by the same rule and render
        for nm in patch_names:
            if nm not in patches:
                self.log.append(f"(warn) No patch generated for '{nm}'")
                continue
            pts_patch = patches[nm]["pts"]
            cols_patch = patches[nm]["col"]
            # pts_patch_moved, _ = move_cut_regions_away(pts_patch, plane_objs, distance_mm=move_dist)

            itp = plot_pointcloud_pyqtgraph_into(self.gl,
                                                 pts_flat=pts_patch.astype(np.float32),
                                                 cols_flat=cols_patch,
                                                 max_points=len(pts_patch),
                                                 point_size=2.2)
            if itp: self._gl_layers.append(itp)

        
        def _add_marker(pt: np.ndarray, color: tuple[int,int,int], size: float = 7.0, name: str | None = None):
            arr = np.asarray(pt, dtype=np.float32).reshape(1, 3)
            cols = np.array([color], dtype=np.uint8)
            it = plot_pointcloud_pyqtgraph_into(self.gl, pts_flat=arr, cols_flat=cols,
                                                max_points=1, point_size=size)
            if it: self._gl_layers.append(it)

        def _add_vector(origin: np.ndarray, direction: np.ndarray,
                        color: tuple[int,int,int], length: float,
                        name: str | None = None, npts: int = 24, size: float = 3.0):
            d = np.asarray(direction, float)
            n = np.linalg.norm(d) + 1e-12
            d = d / n
            t = np.linspace(0.0, float(length), int(npts))[:, None]  # (n,1)
            pts = origin[None, :] + t * d[None, :]
            cols = np.tile(np.array(color, np.uint8), (len(pts), 1))
            it = plot_pointcloud_pyqtgraph_into(self.gl, pts_flat=pts.astype(np.float32), cols_flat=cols,
                                                max_points=len(pts), point_size=size)
            if it: self._gl_layers.append(it)

        # Choose a reasonable vector length from PCA extents
        extent_max = max(info['extent'] for info in axes.values())
        vec_len = max(10.0, 0.20 * float(extent_max))  # ~20% of longest PCA extent, at least 10 mm

        # Centroid
        _add_marker(centroid, (255, 255, 255), size=8.0, name="centroid")

        # # Candidate & selected proximal points
        _add_marker(prox1, (255,   0, 255), size=7.0, name="prox1")
        _add_marker(prox2, (  0, 255, 255), size=7.0, name="prox2")
        _add_marker(most_prox, (255, 128,   0), size=8.5, name="most_prox")

        # # Candidate & selected distal points
        _add_marker(dist1, (  0, 255,   0), size=7.0, name="dist1")
        _add_marker(dist2, (  0, 128, 255), size=7.0, name="dist2")
        _add_marker(most_dist, (255, 200,   0), size=8.5, name="most_dist")

        # Implant normals (from ImplantCuts)
        _add_vector(centroid, normals['posterior']   , (255,   0,   0), vec_len, name="n_posterior")
        _add_vector(centroid, normals['post_chamfer'], (  0, 255,   0), vec_len, name="n_post_chamfer")
        _add_vector(centroid, normals['distal']      , (  0,   0, 255), vec_len, name="n_distal")
        _add_vector(centroid, normals['ant_chamfer'] , (255,   0, 255), vec_len, name="n_ant_chamfer")
        _add_vector(centroid, normals['anterior']    , (255, 255,   0), vec_len, name="n_anterior")

        # # Plane normals actually used for the prox/dist planes (white/gray)
        # _add_vector(centroid, prox_normal, (240, 240, 240), vec_len * 0.9, name="prox_plane_normal")
        # _add_vector(centroid, dist_normal, (160, 160, 160), vec_len * 0.9, name="dist_plane_normal")

        # # Rotation axis k (if available)
        # if hasattr(cuts, 'k') and cuts.k is not None:
        #     _add_vector(centroid, cuts.k, (  0, 200, 200), vec_len * 0.8, name="rotation_axis_k")
        self.log.append(f"Rendered CUT femur (+{move_dist:.1f} mm displacement) + cut patches")


# -------------------------------------
# Entrypoint (OpenGL context like scaling_app)
# -------------------------------------
if __name__ == '__main__':
    from PyQt6.QtCore import QCoreApplication
    os.environ["QT_OPENGL"] = "desktop"
    fmt = QSurfaceFormat()
    fmt.setRenderableType(QSurfaceFormat.RenderableType.OpenGL)
    fmt.setVersion(2, 1)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CompatibilityProfile)
    fmt.setSwapBehavior(QSurfaceFormat.SwapBehavior.DoubleBuffer)
    QSurfaceFormat.setDefaultFormat(fmt)
    QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)

    app = QApplication(sys.argv)
    # Quick splash so early latency is intentional
    pm = QPixmap(480, 200); pm.fill(Qt.GlobalColor.black)
    from PyQt6.QtWidgets import QSplashScreen
    splash = QSplashScreen(pm)
    splash.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
    splash.show()
    splash.showMessage("Loading cut planes…", Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom, QColor("white"))
    app.processEvents()

    win = CutPlanesWindow()
    splash.finish(win)
    win.show()
    sys.exit(app.exec())
