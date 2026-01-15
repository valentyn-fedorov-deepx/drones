import numpy as np

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
        
        
import numpy as np
from typing import Dict, Tuple, Optional

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
