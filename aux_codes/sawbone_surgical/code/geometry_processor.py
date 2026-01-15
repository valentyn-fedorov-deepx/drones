from __future__ import annotations
import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.neighbors import NearestNeighbors

class PCAAnalyzer:
    """Computes principal axes and their extents."""
    @staticmethod
    def analyze(pts: np.ndarray) -> dict:
        c = pts.mean(0)
        pc = pts - c
        w,v = np.linalg.eigh(np.cov(pc.T))
        idx = np.argsort(w)[::-1]
        v = v[:,idx]
        axes = {}
        for n,vec in zip(['length','width','height'],v.T):
            proj = pc@vec
            extent = proj.max() - proj.min()
            axes[n] = {
                'dir':vec,
                'extent': extent,
                'start': c - vec*(extent/2),
                'end'  : c + vec*(extent/2)
                }
        return {'centroid':c,'axes':axes}
    
def filter_low_density(
    pts: np.ndarray,
    cols: np.ndarray | None = None,
    k: int = 10,
    percentile: float = 5.0,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """
    Remove the sparsest points by KNN density (1 / d_k).
    Returns (filtered_pts, filtered_cols_or_None, mask).
    """
    # k+1 because the first neighbor is the point itself
    dists, _ = NearestNeighbors(n_neighbors=k + 1).fit(pts).kneighbors(pts)
    d_k = dists[:, -1]
    density = 1.0 / (d_k + 1e-8)

    thresh = np.percentile(density, percentile)
    mask = density >= thresh

    return pts[mask], (None if cols is None else cols[mask]), mask

def pick_extremes_by_half(
    pts: np.ndarray,
    pca_res: dict,
    division_axis: str = 'length',
    projection_axis: str = 'height',
    positive_dir: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    1) Split pts by the plane through pca_res['centroid'] whose normal is
       pca_res['axes'][division_axis]['dir'].
    2) In each half (above vs. below that plane), project onto
       pca_res['axes'][projection_axis]['dir'], and pick max or min.
    Returns (pt_above_plane, pt_below_plane).
    """
    c = pca_res['centroid']
    div_v = pca_res['axes'][division_axis]['dir']
    div_n = div_v / np.linalg.norm(div_v)
    proj_v = pca_res['axes'][projection_axis]['dir']
    proj_n = proj_v / np.linalg.norm(proj_v)

    signed = (pts - c) @ div_n
    above = pts[signed >= 0]
    below = pts[signed <  0]

    pa = (above - c) @ proj_n
    pb = (below - c) @ proj_n

    if positive_dir:
        i_above = np.argmax(pa)
        i_below = np.argmax(pb)
    else:
        i_above = np.argmin(pa)
        i_below = np.argmin(pb)

    return above[i_above], below[i_below]

def most_collinear(
    vectors: list[np.ndarray],
    base: np.ndarray
) -> tuple[np.ndarray, int, float]:
    """
    Pick the vector from `vectors` that is most collinear with `base`
    (by maximum absolute cosine similarity).

    Returns (best_vector, sign, cos_sim_abs) where:
      - best_vector: the selected candidate (original object from `vectors`)
      - sign: +1 if it points the same way as `base`, -1 if opposite
      - cos_sim_abs: |cosine similarity| in [0, 1]

    Zero-norm candidates are ignored. Raises if `base` is zero or no valid candidates.
    """
    
    b = np.asarray(base, dtype=float)
    b_norm = np.linalg.norm(b)
    if b_norm == 0:
        raise ValueError("Base vector has zero norm; collinearity is undefined.")

    # Stack candidates
    A = np.asarray([np.asarray(v, dtype=float) for v in vectors])
    if A.ndim != 2 or A.shape[1] != b.shape[0]:
        raise ValueError("All candidate vectors must have the same 1D shape as `base`.")

    norms = np.linalg.norm(A, axis=1)
    valid = norms > 0
    if not np.any(valid):
        raise ValueError("All candidate vectors have zero norm; nothing to compare.")

    dots = A @ b
    cos = np.full(len(vectors), np.nan, dtype=float)
    cos[valid] = dots[valid] / (norms[valid] * b_norm)

    idx = int(np.nanargmax(np.abs(cos)))
    sign = 1 if cos[idx] >= 0 else -1
    return vectors[idx], sign, float(abs(cos[idx]))

def _moving_average(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1 or win > len(x):
        return x.copy()
    c = np.cumsum(np.insert(x, 0, 0.0))
    y = (c[win:] - c[:-win]) / float(win)
    # center the window
    pad_left = win // 2
    pad_right = len(x) - len(y) - pad_left
    return np.pad(y, (pad_left, pad_right), mode='edge')

def _percentile_range(a: np.ndarray, p_low: float, p_high: float) -> float:
    if a.size == 0:
        return 0.0
    lo = np.percentile(a, p_low)
    hi = np.percentile(a, p_high)
    return float(hi - lo)

def split_femur_knee_shaft(
    f_pts: np.ndarray,                    # (N, 3)
    f_cols: np.ndarray,                   # (N, C) C=3 or 4
    pca_res: Dict,                        # {'centroid': (3,), 'axes': {'length': {'dir': (3,)}, ...}}
    div_axis_name: Optional[str] = None,  # PCA axis name used to measure "width" (e.g., 'width' or your previously found div_axis_name)
    n_bins: int = 200,
    knee_head_frac: float = 0.15,         # first fraction of bins to estimate knee baseline width
    min_drop: float = 0.25,               # required relative drop from baseline (e.g., 0.25 -> 25%)
    persist: int = 6,                     # bins the drop must persist (stay low) to be accepted
    smooth_win: int = 7,                  # moving-average window for width smoothing
    p_low: float = 5.0,                   # robust width: use percentile range [p_low, p_high]
    p_high: float = 95.0,
    enforce_knee_side_by_width: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Returns:
        knee_pts, knee_cols, shaft_pts, shaft_cols, debug
    Steps:
      1) Auto-pick progression axis (largest extent among PCA axes), and orient it so that width decreases
         from start->end (i.e., start near knee, end toward shaft). If div_axis_name is None, pick the
         axis orthogonal to progression with larger extent as "width axis".
      2) Bin along progression; per bin compute robust width along 'div_axis' (percentile range).
      3) Smooth width curve; detect first sharp drop vs. baseline that then persists; cut there.
    """

    assert f_pts.ndim == 2 and f_pts.shape[1] == 3, "f_pts must be (N,3)"
    assert f_cols.ndim == 2 and f_cols.shape[0] == f_pts.shape[0], "f_cols must align with f_pts"

    # --- Unpack PCA
    c = np.asarray(pca_res['centroid']).reshape(3)
    axes = pca_res['axes']  # dict with keys among ['length','width','height'], each has {'dir': (3,)}

    # Build a clean axis dict -> unit vectors
    axis_names = ['length', 'width', 'height']
    A = {}
    for name in axis_names:
        if name in axes:
            v = np.asarray(axes[name]['dir']).astype(float)
            nv = np.linalg.norm(v) + 1e-12
            A[name] = v / nv
    # Fall back if some names missing
    if len(A) < 2:
        raise ValueError("pca_res['axes'] must contain at least two axes with 'dir'.")

    # --- Compute extents along each PCA axis
    pts_c = f_pts - c
    extents = {}
    proj_cache = {}
    for name, v in A.items():
        s = pts_c @ v
        proj_cache[name] = s
        extents[name] = float(s.max() - s.min())

    # 1) Pick progression axis = axis with maximum extent (knee -> shaft runs along the longest bone axis)
    prog_axis_name = max(extents.items(), key=lambda kv: kv[1])[0]
    prog_v = A[prog_axis_name]
    s_prog = proj_cache[prog_axis_name]  # scalar parameter along progression

    # 2) Pick division axis:
    #    If not provided, choose the orthogonal axis with the larger extent (between the remaining axes).
    if div_axis_name is None:
        candidates = [n for n in A.keys() if n != prog_axis_name]
        if not candidates:
            raise ValueError("Cannot auto-select div_axis_name (need at least two PCA axes).")
        # choose the one with larger extent
        div_axis_name = max(candidates, key=lambda nm: extents[nm])
    div_v = A[div_axis_name]
    s_div = proj_cache[div_axis_name]

    # 3) Bin along progression
    s_min, s_max = s_prog.min(), s_prog.max()
    if n_bins < 10:
        n_bins = 10
    edges = np.linspace(s_min, s_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    widths = np.zeros(n_bins, dtype=float)
    for i in range(n_bins):
        m = (s_prog >= edges[i]) & (s_prog < edges[i + 1])
        if not np.any(m):
            widths[i] = 0.0
        else:
            # robust width along div axis in this bin
            widths[i] = _percentile_range(s_div[m], p_low, p_high)

    # 4) Smooth the width curve
    w_smooth = _moving_average(widths, smooth_win)

    # 5) Determine direction so that we move from KNEE (wide) -> SHAFT (narrow)
    head_bins = max(3, int(n_bins * knee_head_frac))
    tail_bins = max(3, int(n_bins * knee_head_frac))
    mean_head = np.mean(w_smooth[:head_bins]) if head_bins < len(w_smooth) else w_smooth[0]
    mean_tail = np.mean(w_smooth[-tail_bins:]) if tail_bins < len(w_smooth) else w_smooth[-1]
    forward_is_knee_to_shaft = True
    if mean_head < mean_tail:
        # flip direction convention: reverse arrays so index increases knee->shaft
        forward_is_knee_to_shaft = False
        w_smooth = w_smooth[::-1]
        widths = widths[::-1]
        centers = centers[::-1]
        edges = edges[::-1]

    # Recompute baseline after ensuring direction
    baseline = np.mean(w_smooth[:head_bins])

    # 6) Detect first significant drop that persists
    #    Condition: width <= baseline*(1 - min_drop) AND remains low for `persist` bins.
    thresh = baseline * (1.0 - min_drop)
    drop_idx = None
    for i in range(head_bins, n_bins - persist):
        if w_smooth[i] <= thresh:
            if np.all(w_smooth[i:i + persist] <= (baseline * (1.0 - 0.5 * min_drop))):
                drop_idx = i
                break

    # Fallback: if no persistent drop, use the steepest negative gradient point past the head
    if drop_idx is None:
        grad = np.diff(w_smooth, prepend=w_smooth[0])
        # ignore the first head_bins (knee region)
        j = np.argmin(grad[head_bins:]) + head_bins
        drop_idx = int(j)

    # 7) Define the cut plane at the drop bin center
    s_cut = centers[drop_idx]

    # If we flipped arrays earlier, we must adjust the criterion accordingly.
    # We always want: increasing index means knee->shaft.
    # Assign knee = s_prog <= s_cut (in the "forward" convention), shaft = s_prog > s_cut.
    if forward_is_knee_to_shaft:
        knee_mask = s_prog <= s_cut
        shaft_mask = ~knee_mask
    else:
        # original s_prog direction was opposite → invert the assignment
        knee_mask = s_prog >= s_cut
        shaft_mask = ~knee_mask

    # Optional sanity: enforce that knee side is the wider side near the boundary
    if enforce_knee_side_by_width:
        # take a small neighborhood around the cut on each side and compare widths
        nb = max(3, int(0.03 * n_bins))
        # build bin masks
        bin_idx = np.searchsorted(edges, s_prog, side='right') - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)

        near_cut = np.abs(bin_idx - drop_idx) <= nb
        side_knee = knee_mask & near_cut
        side_shaft = shaft_mask & near_cut

        knee_w = _percentile_range(s_div[side_knee], p_low, p_high) if np.any(side_knee) else 0.0
        shaft_w = _percentile_range(s_div[side_shaft], p_low, p_high) if np.any(side_shaft) else 0.0

        if knee_w < shaft_w:
            # swap sides if mis-assigned
            knee_mask, shaft_mask = shaft_mask, knee_mask

    knee_pts = f_pts[knee_mask]
    knee_cols = f_cols[knee_mask]
    shaft_pts = f_pts[shaft_mask]
    shaft_cols = f_cols[shaft_mask]

    debug = {
        'prog_axis_name': prog_axis_name,
        'div_axis_name': div_axis_name,
        'prog_dir': A[prog_axis_name],
        'div_dir': A[div_axis_name],
        'centers': centers,
        'widths_raw': widths,
        'widths_smooth': w_smooth,
        'baseline': baseline,
        'drop_idx': drop_idx,
        's_cut': s_cut,
        'forward_is_knee_to_shaft': forward_is_knee_to_shaft,
        'edges': edges,
    }

    return knee_pts, knee_cols, shaft_pts, shaft_cols, debug

def refine_center(pts: np.ndarray,
                  res: dict,
                  division_axis: str = 'length',
                  projection_axis: str = 'height',
                  positive_dir: bool = True,
                  tol: float = 1e-4,
                  max_iter: int = 10
                 ) -> np.ndarray:
    """
    Iteratively shift res['centroid'] so the division plane
    bisects the two extremes along projection_axis.
    """
    # normalize once
    div_n = res['axes'][division_axis]['dir']
    div_n = div_n / np.linalg.norm(div_n)
    c_cur = res['centroid'].copy()

    for _ in range(max_iter):
        res['centroid'] = c_cur
        pa, pb = pick_extremes_by_half(
            pts, res,
            division_axis=division_axis,
            projection_axis=projection_axis,
            positive_dir=positive_dir
        )
        sd_a = np.dot(pa - c_cur, div_n)
        sd_b = np.dot(pb - c_cur, div_n)
        shift = 0.5 * (sd_a + sd_b)
        if abs(shift) < tol:
            break
        c_cur = c_cur + div_n * shift

    return c_cur


def refine_axes_in_plane(res: dict,
                         pts: np.ndarray,
                         axis_primary: str,
                         axis_secondary: str,
                         pt1: np.ndarray,
                         pt2: np.ndarray) -> dict:
    """
    Rotate the PCA axes in the plane spanned by axis_primary and axis_secondary
    so that axis_primary becomes parallel to (pt2 - pt1). The rotation is taken
    to be the minimal‑angle (< 90°) around the normal to that plane, and both
    axes in that plane (primary & secondary) rotate together; the third axis
    (plane normal) stays fixed.
    """
    # 1) identify the fixed axis (the plane normal)
    all_axes = list(res['axes'].keys())
    axis_other = next(a for a in all_axes if a not in (axis_primary, axis_secondary))

    # 2) old directions
    v1_old = res['axes'][axis_primary]['dir']
    v2_old = res['axes'][axis_secondary]['dir']
    v3     = res['axes'][axis_other]['dir']
    k = v3 / np.linalg.norm(v3)     # plane normal

    # 3) compute target direction in that plane
    t = pt2 - pt1
    t_proj = t - (t.dot(k)) * k
    if np.linalg.norm(t_proj) < 1e-8:
        return res  # no meaningful rotation
    v1_new = t_proj / np.linalg.norm(t_proj)

    # 4) ensure minimal rotation: flip v1_old if angle > 90°
    if np.dot(v1_old, v1_new) < 0:
        v1_old = -v1_old
        # flipping primary only — secondary will follow via R

    # 5) compute rotation angle & build R about k
    cosang = np.dot(v1_old, v1_new)
    sinang = np.dot(np.cross(v1_old, v1_new), k)
    K = np.array([[    0, -k[2],  k[1]],
                  [ k[2],     0, -k[0]],
                  [-k[1],  k[0],     0]])
    R = np.eye(3) + K * sinang + (K @ K) * (1 - cosang)

    # 6) rotate both primary & secondary
    v1r = R.dot(v1_old)
    v2r = R.dot(v2_old)
    v1r /= np.linalg.norm(v1r)
    v2r /= np.linalg.norm(v2r)

    # 7) write back
    res['axes'][axis_primary]['dir']   = v1r
    res['axes'][axis_secondary]['dir'] = v2r
    # axis_other remains k
    res['axes'][axis_other]['dir']     = k

    # 8) recompute extents
    c = res['centroid']
    for name, info in res['axes'].items():
        d = info['dir']
        proj = (pts - c) @ d
        ext = proj.max() - proj.min()
        info['extent'] = ext
        info['start'] = c - d*(ext / 2)
        info['end'] = c + d*(ext / 2)

    return res

def refine_axes_by_midpoints(res: dict,
                             pts: np.ndarray,
                             axis_primary: str,
                             axis_secondary: str,
                             p1: np.ndarray,
                             p2: np.ndarray) -> dict:
    """
    Wrap your existing refine_axes_in_plane to take mid‑points directly.
    """
    return refine_axes_in_plane(
        res,
        pts=pts,
        axis_primary=axis_primary,
        axis_secondary=axis_secondary,
        pt1=p1,
        pt2=p2
    )

def refine_centroid_along_axis(res: dict,
                               axis: str,
                               targets: list[np.ndarray]) -> dict:
    """
    Slide res['centroid'] along the given axis so that it
    lands at the midpoint of `targets`.
    """
    c = res['centroid']
    d = res['axes'][axis]['dir']
    d = d / np.linalg.norm(d)

    # compute the average of your target points
    mean_pt = np.stack(targets, axis=0).mean(axis=0)

    # how far along d to move?
    shift = np.dot(mean_pt - c, d)
    c_new = c + d * shift
    res['centroid'] = c_new

    # update all start/end for each axis
    for info in res['axes'].values():
        dir_vec = info['dir']
        ext     = info['extent']
        info['start'] = c_new - dir_vec * (ext/2)
        info['end']   = c_new + dir_vec * (ext/2)

    return res

def rotate_pca_around_axis(res: dict,
                      pts: np.ndarray,
                      rotation_axis: str,
                      angle_deg: float,
                      clockwise: bool = True,
                      positive_dir: bool = True) -> dict:
    """
    Rotate all axes around the specified axis by the given angle.
    
    Parameters:
        res: The PCA results dictionary
        pts: The point cloud array
        rotation_axis: Which axis to rotate around ('length', 'width', or 'height')
        angle_deg: Rotation angle in degrees
        clockwise: If True, rotate clockwise; if False, rotate counterclockwise
        positive_dir: If True, use the positive direction of the axis; if False, use negative
    
    Returns:
        Updated PCA results dictionary
    """
    # Determine which axes to rotate
    all_axes = list(res['axes'].keys())
    axes_to_rotate = [axis for axis in all_axes if axis != rotation_axis]
    
    # Get the rotation axis direction (normalized)
    k = res['axes'][rotation_axis]['dir']
    k = k / np.linalg.norm(k)
    
    # Adjust direction if needed
    if not positive_dir:
        k = -k
    
    # Convert angle to radians and adjust sign for clockwise/counterclockwise
    angle_rad = np.deg2rad(angle_deg)
    if clockwise:
        angle_rad = -angle_rad  # Negate for clockwise rotation
    
    # Rodrigues' rotation formula
    cosθ = np.cos(angle_rad)
    sinθ = np.sin(angle_rad)
    
    # Skew-symmetric matrix from rotation axis
    K = np.array([[    0, -k[2],  k[1]],
                  [ k[2],     0, -k[0]],
                  [-k[1],  k[0],     0]])
    
    # Rotation matrix
    R = np.eye(3) + K * sinθ + (K @ K) * (1 - cosθ)
    
    # Apply rotation to the specified axes
    for axis in axes_to_rotate:
        v_old = res['axes'][axis]['dir']
        v_new = R @ v_old
        v_new = v_new / np.linalg.norm(v_new)
        res['axes'][axis]['dir'] = v_new
    
    # Recompute extents
    c = res['centroid']
    for name, info in res['axes'].items():
        d = info['dir']
        proj = (pts - c) @ d
        ext = proj.max() - proj.min()
        info['extent'] = ext
        info['start'] = c - d*(ext / 2)
        info['end'] = c + d*(ext / 2)
    
    return res

def first_planes_from_dirs(
    centroid: np.ndarray,
    w_dir: np.ndarray, h_dir: np.ndarray,
    new_prox1: np.ndarray, new_prox2: np.ndarray,
    new_dist1: np.ndarray, new_dist2: np.ndarray,
    shift_posterior: float = 5.0,
    shift_distal: float = 6.0,
) -> Tuple[Plane, Plane]:
    """
    Build (distal_plane, posterior_plane) given posterior (w_dir) and distal (h_dir) directions.

    - Plane normals point OUTWARD (from centroid to the chosen cortex point).
    - Anchor points are moved TOWARD the centroid by the given shift (i.e., always 'inside').
    - Uses only Plane.from_normal_and_point(...).
    """

    c = np.asarray(centroid, float)

    # unitize directions
    w_dir = np.asarray(w_dir, float); w_dir /= (np.linalg.norm(w_dir) + 1e-12)
    h_dir = np.asarray(h_dir, float); h_dir /= (np.linalg.norm(h_dir) + 1e-12)

    new_prox1 = np.asarray(new_prox1, float); new_prox2 = np.asarray(new_prox2, float)
    new_dist1 = np.asarray(new_dist1, float); new_dist2 = np.asarray(new_dist2, float)

    # choose extremes: posterior=max along w_dir, distal=min along h_dir
    most_posterior = new_prox1 if (new_prox1 - c) @ w_dir >= (new_prox2 - c) @ w_dir else new_prox2
    most_distal    = new_dist1 if (new_dist1 - c) @ h_dir <= (new_dist2 - c) @ h_dir else new_dist2

    # outward normals = direction from centroid to the selected extreme
    out_w = (1.0 if (most_posterior - c) @ w_dir >= 0.0 else -1.0) * w_dir
    out_h = (1.0 if (most_distal    - c) @ h_dir >= 0.0 else -1.0) * h_dir

    sp = abs(float(shift_posterior))
    sd = abs(float(shift_distal))

    # anchors moved INSIDE (toward centroid)
    p_post_inside = most_posterior - out_w * sp
    p_dist_inside = most_distal    - out_h * sd

    # construct planes (no flipping; normals stay outward)
    posterior_plane = Plane.from_normal_and_point(out_w, p_post_inside)
    distal_plane    = Plane.from_normal_and_point(out_h, p_dist_inside)

    # if, for any reason, centroid ended up on the positive side, push deeper inside (rare)
    if posterior_plane.signed_distance(c) > 0.0:
        depth = max(0.0, (most_posterior - c) @ out_w)
        posterior_plane = Plane.from_normal_and_point(out_w, most_posterior - out_w * (depth + sp + 1e-6))
    if distal_plane.signed_distance(c) > 0.0:
        depth = max(0.0, (most_distal - c) @ out_h)
        distal_plane = Plane.from_normal_and_point(out_h, most_distal - out_h * (depth + sd + 1e-6))

    return distal_plane, posterior_plane

class Plane:
    """
    Ax + By + Cz + D = 0, with unit-length normal (A,B,C).
    """
    def __init__(self, A: float, B: float, C: float, D: float):
        n = float(np.linalg.norm([A, B, C]))
        if n == 0:
            raise ValueError("Plane normal cannot be zero.")
        self.A, self.B, self.C, self.D = A / n, B / n, C / n, D / n
        self.normal = np.array([self.A, self.B, self.C], dtype=float)

    @staticmethod
    def from_normal_and_point(normal: np.ndarray, point: np.ndarray) -> "Plane":
        n = np.asarray(normal, float)
        ln = float(np.linalg.norm(n))
        if ln == 0:
            raise ValueError("Normal vector must be nonzero.")
        n /= ln
        D = -float(n @ np.asarray(point, float))
        return Plane(n[0], n[1], n[2], D)

    @staticmethod
    def from_two_vectors(v1: np.ndarray, v2: np.ndarray, point: np.ndarray | None = None) -> "Plane":
        v1 = np.asarray(v1, float)
        v2 = np.asarray(v2, float)
        n = np.cross(v1, v2)
        nn = float(np.linalg.norm(n))
        if nn < 1e-12:
            raise ValueError("v1 and v2 are parallel/colinear; cannot define a plane.")
        n /= nn
        D = 0.0 if point is None else -float(n @ np.asarray(point, float))
        return Plane(n[0], n[1], n[2], D)

    def coefficients(self) -> tuple[float, float, float, float]:
        return self.A, self.B, self.C, self.D

    def signed_distance(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, float)
        return pts @ self.normal + self.D

    def project_point(self, pt: np.ndarray, distance: float) -> np.ndarray:
        return np.asarray(pt, float) + self.normal * float(distance)

    def orient_away_from_point(self, point: np.ndarray) -> "Plane":
        """Flip normal if point lies on the normal-positive side (keeps 'outside' consistent)."""
        if float(self.normal @ np.asarray(point, float) + self.D) > 0.0:
            self.A, self.B, self.C, self.D = -self.A, -self.B, -self.C, -self.D
            self.normal = -self.normal
        return self

    def points_above_plane(self, points: np.ndarray) -> np.ndarray:
        """Boolean mask: signed_distance > 0 (above, in direction of plane normal)."""
        return self.signed_distance(points) > 0.0


# three-plane-intersection
def intersect_three_planes(p1: Plane, p2: Plane, p3: Plane, atol: float = 1e-12) -> np.ndarray:
    A = np.vstack([p1.normal, p2.normal, p3.normal]).astype(float)
    b = -np.array([p1.D, p2.D, p3.D], dtype=float)
    if np.linalg.matrix_rank(A, tol=atol) < 3:
        raise ValueError("Planes do not intersect at a single point (normals are coplanar or nearly so).")
    return np.linalg.solve(A, b)


class FemurImplant:
    """
    Computes unit cut normals and three derived planes
    (posterior chamfer, anterior chamfer, anterior)
    from distal/posterior reference planes and design angles.

    Angle convention:
      - Angles are in degrees and applied CCW around the axis k = cross(posterior, distal),
        so that rotating n_prox → n_dist is +CCW.
    """
    def __init__(
        self,
        AP_wo_CondThick: float, # Anterior-Posterior length without condyles thickness
        DistCut_length: float, # Distal Cut Lengt
        AntChamfCut_length: float, # Anterior Chamfer Cut Length
        centroid: np.ndarray | None = None, # centroid of the Implant
        
        # angles in degrees between consecutive normals, counter-clockwise from right side of the knee
        post_to_post_chamfer: float = 50.0,
        dist_to_ant_chamfer: float = 44.0,
        ant_to_ant_chamfer: float = 41.0,
    ):
        self.AP_wo_CondThick = float(AP_wo_CondThick)
        self.DistCut_length = float(DistCut_length)
        self.AntChamfCut_length = float(AntChamfCut_length)

        self.post_to_post_chamfer = float(post_to_post_chamfer)
        self.dist_to_ant_chamfer = float(dist_to_ant_chamfer)
        self.ant_to_ant_chamfer = float(ant_to_ant_chamfer)

        self.centroid = np.asarray(centroid, dtype=float) if centroid is not None else None
        self.normals: Dict[str, np.ndarray] = {}
        self.k: np.ndarray | None = None  # rotation axis
        self._last_planes: dict[str, Plane] | None = None


        # geometry derived length
        r1 = np.deg2rad(self.post_to_post_chamfer)
        r2 = np.deg2rad(self.dist_to_ant_chamfer)
        sin_r1 = np.sin(r1)
        if abs(sin_r1) < 1e-8:
            raise ValueError("post_to_post_chamfer leads to sin≈0; choose a non-degenerate angle.")

        self._PostChamfCut_projected_height = (
            self.AP_wo_CondThick
            - self.AntChamfCut_length * np.cos(r2)
            - self.DistCut_length
        )
        self.PostChamfCut_length = self._PostChamfCut_projected_height / sin_r1
        
    @staticmethod
    def _unit(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, float)
        n = float(np.linalg.norm(v))
        if n == 0:
            raise ValueError("Zero-length vector.")
        return v / n

    @staticmethod
    def _to_radians(deg: float) -> float:
        return float(np.deg2rad(deg))

    def _rotate_about_k(self, v: np.ndarray, angle_deg: float) -> np.ndarray:
        """Rodrigues rotation of vector v around self.k by angle_deg (degrees)."""
        if self.k is None:
            raise RuntimeError("Rotation axis k is not set.")
        k = self.k
        th = self._to_radians(angle_deg)
        return self._unit(
            v * np.cos(th) + np.cross(k, v) * np.sin(th) + k * (k @ v) * (1 - np.cos(th))
        )

    def compute_cut_normals(self, distal_normal: np.ndarray, posterior_normal: np.ndarray) -> Dict[str, np.ndarray]:
        """Build the five unit normals with consistent CCW convention."""
        n_dist = self._unit(distal_normal)
        n_prox = self._unit(posterior_normal)

        # if slightly non-orthogonal, orthogonalize distal to posterior
        dp = float(n_dist @ n_prox)
        if abs(dp) > 1e-3:
            n_dist = self._unit(n_dist - dp * n_prox)

        # right-handed axis: rotating posterior -> distal is +CCW
        self.k = self._unit(np.cross(n_prox, n_dist))

        normals = {
            "posterior": n_prox,
            "post_chamfer": self._rotate_about_k(n_prox, self.post_to_post_chamfer),
            "distal": n_dist,
            "ant_chamfer": self._rotate_about_k(n_dist, self.dist_to_ant_chamfer),
        }
        normals["anterior"] = self._rotate_about_k(normals["ant_chamfer"], self.ant_to_ant_chamfer)
        self.normals = normals
        return normals

    #  helpers for the normals-plane used in intersections
    def _normals_plane_via_ref(self, distal_plane: Plane, posterior_plane: Plane) -> Plane:
        """
        The 'normals plane' is the plane spanned by the distal & posterior normals, through centroid.
        This matches your prior construction and keeps the 90° turn well-defined.
        """
        if self.centroid is None:
            raise ValueError("centroid must be provided to Implant for plane construction.")
        return Plane.from_two_vectors(distal_plane.normal, posterior_plane.normal, point=self.centroid)

    # individual cuts
    def _compute_post_chamf_cut(self, distal_plane: Plane, posterior_plane: Plane) -> Plane:
        if not self.normals:
            raise ValueError("Compute normals first.")
        normals_plane = self._normals_plane_via_ref(distal_plane, posterior_plane)
        start = intersect_three_planes(normals_plane, distal_plane, posterior_plane)

        move = (
            self.PostChamfCut_length
            * np.cos(self._to_radians(self.post_to_post_chamfer))
            * np.sin(self._to_radians(self.dist_to_ant_chamfer))
        )
        n_post = self.normals["post_chamfer"]
        cut_point = start - move * n_post
        return Plane.from_normal_and_point(n_post, cut_point).orient_away_from_point(self.centroid)

    def _compute_ant_chamf_cut(self, distal_plane: Plane, posterior_plane: Plane, post_chamf_plane: Plane) -> Plane:
        if not self.normals:
            raise ValueError("Compute normals first.")
        normals_plane = self._normals_plane_via_ref(distal_plane, posterior_plane)
        start = intersect_three_planes(normals_plane, distal_plane, post_chamf_plane)
        # interpret AntChamfCut_length as offset along posterior (posterior) direction
        cut_point = start - self.AntChamfCut_length * self.normals["posterior"]
        return Plane.from_normal_and_point(self.normals["ant_chamfer"], cut_point).orient_away_from_point(self.centroid)

    def _compute_ant_cut(self, distal_plane: Plane, posterior_plane: Plane, ant_chamf_plane: Plane) -> Plane:
        if not self.normals:
            raise ValueError("Compute normals first.")
        normals_plane = self._normals_plane_via_ref(distal_plane, posterior_plane)
        start = intersect_three_planes(normals_plane, ant_chamf_plane, distal_plane)

        # vector along chamfer edge: 90 deg turn inside the normals plane
        dist_to_ant_dir = self._unit(self.normals["ant_chamfer"] - self.normals["distal"])
        k = normals_plane.normal
        th = np.deg2rad(90.0)
        K = np.array([[0, -k[2],  k[1]],
                      [k[2],  0, -k[0]],
                      [-k[1], k[0],  0]])
        R = np.eye(3) + K * np.sin(th) + (K @ K) * (1 - np.cos(th))
        v_along = self._unit(R @ dist_to_ant_dir)

        cut_point = start + v_along * self.AntChamfCut_length
        return Plane.from_normal_and_point(self.normals["anterior"], cut_point).orient_away_from_point(self.centroid)

    def compute_all_planes(self, distal_plane: Plane, posterior_plane: Plane) -> Dict[str, Plane]:
        """
        Returns a complete family of planes (including the input two).
          keys: 'posterior','post_chamfer','distal','ant_chamfer','anterior'
        """
        post_chamf = self._compute_post_chamf_cut(distal_plane, posterior_plane)
        ant_chamf  = self._compute_ant_chamf_cut(distal_plane, posterior_plane, post_chamf)
        anterior   = self._compute_ant_cut(distal_plane, posterior_plane, ant_chamf)
        self._last_planes = {
            "posterior":     posterior_plane,
            "post_chamfer": post_chamf,
            "distal":       distal_plane,
            "ant_chamfer":  ant_chamf,
            "anterior":     anterior,
        }
        return self._last_planes
    
    def move_cut_regions_away(
        self,
        pts: np.ndarray,
        distance_mm: float = 15.0,
        planes: list[Plane] | dict[str, Plane] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Move points that are 'above' any of the given planes in the average outward direction.

        - If `planes` is None, uses the last planes computed by `compute_all_planes(...)`.
        - Plane orientation should be such that 'above' (signed_distance > 0) is outward
        (our pipeline orients planes away from the centroid).

        Returns:
            pts_moved : (N,3) moved points
            move_dir  : (N,3) unit vectors used for the move (zeros where no move)
        """
        # choose planes: cached by default
        if planes is None:
            if self._last_planes is None:
                raise ValueError("No planes provided and no cached planes. Call compute_all_planes(...) first or pass planes.")
            use_planes = list(self._last_planes.values())
        elif isinstance(planes, dict):
            use_planes = list(planes.values())
        else:
            use_planes = list(planes)

        if len(use_planes) == 0:
            return pts.copy(), np.zeros_like(pts, dtype=float)

        # boolean matrix: N x M (point is above plane i ?)
        to_move = np.column_stack([pl.points_above_plane(pts) for pl in use_planes])  # (N,M)

        # normals matrix: M x 3
        normals = np.array([pl.normal for pl in use_planes], dtype=float)             # (M,3)

        # average outward direction for each point
        move_dir = to_move @ normals                                                  # (N,3)

        # normalize non-zero rows
        norms = np.linalg.norm(move_dir, axis=1, keepdims=True)
        nz = norms.squeeze(-1) > 0
        move_dir[nz] /= norms[nz]

        pts_moved = pts + move_dir * float(distance_mm)
        return pts_moved, move_dir
    
    def shift_cut_planes(
        self,
        direction: np.ndarray,
        distance_mm: float,
        planes: dict[str, Plane] | list[Plane] | None = None,
        *,
        update_cache: bool = True,
        update_centroid: bool = False,
    ) -> dict[str, Plane]:
        """
        Translate all cut planes by (unit(direction) * distance_mm). Normals stay the same.

        Parameters
        ----------
        direction : (3,) array-like
            Shift direction. Will be normalized internally.
        distance_mm : float
            Translation magnitude in the same units as your geometry (mm).
        planes : dict[str, Plane] | list[Plane] | None
            Which planes to shift. If None, uses the last-computed planes (self._last_planes).
            If a list is passed, keys 'p0','p1',... are used in the returned dict.
        update_cache : bool, default=True
            If True and planes is None, overwrite self._last_planes with shifted planes.
        update_centroid : bool, default=False
            If True and self.centroid is set, translate the centroid by the same vector.

        Returns
        -------
        shifted : dict[str, Plane]
            A dict of shifted planes with the same normals and translated offsets.
        """
        if planes is None:
            if self._last_planes is None:
                raise ValueError("No planes to shift. Call compute_all_planes(...) first or pass `planes`.")
            use = self._last_planes
        elif isinstance(planes, dict):
            use = planes
        else:
            # list -> named dict for consistent return type
            use = {f"p{i}": pl for i, pl in enumerate(planes)}

        d = self._unit(direction)
        t = d * float(distance_mm)

        shifted: dict[str, Plane] = {}
        for key, pl in use.items():
            # any point on the plane: x0 = -D * n  (since n·x0 + D = 0 and ||n||=1)
            x0 = -pl.D * pl.normal
            x0_shifted = x0 + t
            # rebuild plane from same normal and translated point
            pl_new = Plane.from_normal_and_point(pl.normal, x0_shifted)
            shifted[key] = pl_new

        if update_cache and planes is None:
            self._last_planes = shifted

        if update_centroid and (self.centroid is not None):
            self.centroid = self.centroid + t

        return shifted



#! DEPRECATION NOTICE
class DirectedAxisFitter:
    """
    Fit the 3D point cloud along a supplied direction vector.
    Direction is fixed (like PCA but constrained), the best-fit line
    passes through the centroid; start/end are the min/max projections
    along that line, yielding a fulcrum (start) and a collinear vector.
    """
    @staticmethod
    def fit(pts: np.ndarray, vec: np.ndarray, trim: float = 0.0) -> dict:
        """
        Parameters
        ----------
        pts : (N,3) array
            3D points.
        vec : (3,) array
            Direction to fit along. Need not be unit; must be nonzero.
        trim : float in [0, 0.5]
            Optional symmetric trimming of extremes (robust to outliers).
            E.g., 0.05 uses the 5th and 95th percentiles instead of min/max.

        Returns
        -------
        dict with:
            - 'dir'    : unit vector collinear with `vec`
            - 'extent' : scalar length along 'dir' (float)
            - 'start'  : 3D point (fulcrum) at lower extreme
            - 'end'    : 3D point at upper extreme
        """
        pts = np.asarray(pts, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("pts must be an (N,3) array.")
        v = np.asarray(vec, dtype=float)
        nv = np.linalg.norm(v)
        if nv == 0:
            raise ValueError("vec has zero norm.")
        if not (0.0 <= trim <= 0.5):
            raise ValueError("trim must be in [0, 0.5].")

        # Unit direction and centroid
        u = v / nv
        c = pts.mean(axis=0)

        # Project points onto the best-fit line parallel to u through the centroid
        t = (pts - c) @ u  # signed offsets along u

        # Optionally trim outliers
        if trim > 0:
            lo = np.quantile(t, trim)
            hi = np.quantile(t, 1.0 - trim)
        else:
            lo = t.min()
            hi = t.max()

        start = c + lo * u            # fulcrum (lower extreme along u)
        end   = c + hi * u            # upper extreme along u
        extent = float(hi - lo)       # length along u (>= 0)

        return {
            'dir'   : u,
            'extent': extent,
            'start' : start,
            'end'   : end,
        }