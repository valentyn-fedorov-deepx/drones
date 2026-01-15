import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub
import cv2
import re

from streamvggt.models.aggregator import Aggregator
from streamvggt.heads.camera_head import CameraHead
from streamvggt.heads.dpt_head import DPTHead
from streamvggt.heads.track_head import TrackHead
from transformers.file_utils import ModelOutput
from typing import Optional, Tuple, List, Any, Dict, Union
from contextlib import contextmanager
from dataclasses import dataclass, field
import numpy as np


################
# STREAM STATE #
################
@dataclass
class StreamState:
    # caches
    past_kv_agg: List[Any]
    past_kv_cam: List[Any]
    frame_idx: int = 0

    # rolling outputs (CPU to limit VRAM; keep tensors if you need speed)
    camera_poses: List[torch.Tensor] = field(default_factory=list)
    pts3d_list:   List[torch.Tensor] = field(default_factory=list)
    depth_list:   List[torch.Tensor] = field(default_factory=list)
    depth_conf_list: List[torch.Tensor] = field(default_factory=list)

    # optional trackers
    last_query_points: Optional[torch.Tensor] = None

    # memory control
    max_keep: Optional[int] = None  # e.g., keep only last N frames (sliding window)
    
@contextmanager
def infer_mode():
    # fast + memory-safe inference by default
    with torch.inference_mode():
        # if you want AMP, flip enabled=True and pick dtype:
        with torch.cuda.amp.autocast(enabled=False):
            yield

def _append_and_prune(lst: List[torch.Tensor], item: torch.Tensor, max_keep: Optional[int]):
    lst.append(item)
    if max_keep is not None and len(lst) > max_keep:
        # drop oldest
        del lst[0]


#################
# Output Buffer #
#################
@dataclass
class OutsBuffer:
    """Holds everything you ever produced in streaming."""
    ress:  List[Dict[str, Any]] = field(default_factory=list)  # per-frame dicts
    views: List[Dict[str, Any]] = field(default_factory=list)  # optional original frames
    # optional: keep a running count for easy indexing
    n_frames: int = 0

def _move_detach(obj, device: Optional[torch.device]):
    """Detach tensors & move to device; leave non-tensors as-is."""
    if torch.is_tensor(obj):
        x = obj.detach()
        if device is not None:
            x = x.to(device, non_blocking=True)
        return x
    if isinstance(obj, dict):
        return {k: _move_detach(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        t = [ _move_detach(v, device) for v in obj ]
        return type(obj)(t) if isinstance(obj, tuple) else t
    return obj

def new_outs_buffer() -> OutsBuffer:
    return OutsBuffer()

def append_outs_inplace(
    buf: OutsBuffer,
    outs: List[Dict[str, Any]],
    frames: Optional[List[Dict[str, Any]]] = None,
    *,
    store_device: Optional[torch.device] = torch.device("cpu"),
    keep_images: bool = False
) -> OutsBuffer:
    """
    Append a batch of per-frame results to the buffer.

    - outs: what your stream_step/inference returned (list of dicts; one per frame).
    - frames: matching list of input frames (for metadata; can be None).
    - store_device: where to store tensors (CPU by default to free VRAM).
    - keep_images: if True, stores full frame['img']; otherwise drops image to save RAM.
    """
    # sanity
    if frames is not None:
        assert len(outs) == len(frames), "outs and frames length mismatch"

    for i, res in enumerate(outs):
        # 1) detach/move all tensors inside the per-frame result
        res_stored = _move_detach(res, store_device)
        buf.ress.append(res_stored)

        # 2) store view metadata if provided
        if frames is not None:
            vw = frames[i]
            if not keep_images and isinstance(vw, dict) and "img" in vw:
                # copy without the heavy image tensor
                vw = {k: v for k, v in vw.items() if k != "img"}
            else:
                vw = _move_detach(vw, store_device)
            buf.views.append(vw)

        buf.n_frames += 1

    return buf

###############
# Save to NPZ #
###############
def _to_numpy(x) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    # allow scalars:
    if isinstance(x, (int, float, bool, np.number, str)):
        return np.array(x)
    raise TypeError(f"Unsupported type for numpy conversion: {type(x)}")


def _maybe_squeeze_B(arr: np.ndarray, squeeze_batch: bool) -> np.ndarray:
    if squeeze_batch and arr.ndim >= 1 and arr.shape[0] == 1:
        return np.squeeze(arr, axis=0)
    return arr


def _gather_sources(
    *,
    ress: Optional[List[Dict[str, Any]]] = None,
    views: Optional[List[Dict[str, Any]]] = None,
    output: Optional[Any] = None,     # StreamVGGTOutput-like with .ress/.views
    buf: Optional[OutsBuffer] = None  # OutsBuffer
) -> Tuple[List[Dict[str, Any]], Optional[List[Dict[str, Any]]]]:
    if ress is not None:
        return ress, views
    if buf is not None:
        return buf.ress, (buf.views if len(buf.views) == len(buf.ress) and len(buf.views) > 0 else None)
    if output is not None:
        return getattr(output, "ress", None), getattr(output, "views", None)
    raise ValueError("Provide one of: (ress[,views]) OR output OR buf.")


def _collect_all_keys(ress: List[Dict[str, Any]]) -> List[str]:
    keys = set()
    for r in ress:
        keys.update(r.keys())
    return sorted(keys)


def _try_stack_key(
    key: str,
    ress: List[Dict[str, Any]],
    squeeze_batch: bool
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[List[int]]]:
    """
    Returns:
      stacked: [T, ...] if shapes consistent (missing frames filled with zeros) else None
      mask:    [T] bool mask where key was present (only if stacked is not None)
      frames_present: list of indices where key exists (only if stacked is None => ragged)
    """
    T = len(ress)
    ref = None
    ref_t = None
    for t, r in enumerate(ress):
        if key in r:
            ref = _maybe_squeeze_B(_to_numpy(r[key]), squeeze_batch)
            ref_t = t
            break
    if ref is None:
        return None, None, []  # key never present

    same_shape = True
    arrs = [None] * T
    present = np.zeros((T,), dtype=bool)
    ref_shape = ref.shape
    arrs[ref_t] = ref
    present[ref_t] = True

    for t, r in enumerate(ress):
        if t == ref_t:
            continue
        if key in r:
            arr = _maybe_squeeze_B(_to_numpy(r[key]), squeeze_batch)
            if arr.shape != ref_shape:
                same_shape = False
                break
            arrs[t] = arr
            present[t] = True

    if not same_shape:
        frames_present = [t for t, r in enumerate(ress) if key in r]
        return None, None, frames_present

    for t in range(T):
        if arrs[t] is None:
            arrs[t] = np.zeros(ref_shape, dtype=ref.dtype)

    stacked = np.stack(arrs, axis=0)  # [T, ...]
    return stacked, present, None


def save_streamvggt_npz(
    path: str,
    *,
    ress: Optional[List[Dict[str, Any]]] = None,
    views: Optional[List[Dict[str, Any]]] = None,
    output: Optional[Any] = None,  # StreamVGGTOutput-like
    buf: Optional[OutsBuffer] = None,
    squeeze_batch: bool = True,
    compress: bool = True,
    include_view_fields: Optional[List[str]] = None  # e.g., ["valid_mask", "timestamp"]
) -> str:
    """
    Save results to a single .npz.

    Stored keys:
      - 'meta_T' (int), 'meta_keys' (str array), 'meta_version'
      - For each prediction key K:
          * If shapes consistent: 'K' = [T,...], 'K_present' = [T] bool
          * If ragged: 'K_frames' (indices) + one array per frame: 'K_000000', ...
      - For selected view fields (prefixed 'view_'), same logic as above.
    """
    ress, views = _gather_sources(ress=ress, views=views, output=output, buf=buf)
    if not isinstance(ress, list) or len(ress) == 0:
        raise ValueError("No results to save.")
    T = len(ress)

    npz_dict: Dict[str, Any] = {}
    all_keys = _collect_all_keys(ress)
    npz_dict["meta_T"] = np.array(T, dtype=np.int64)
    npz_dict["meta_keys"] = np.array(all_keys, dtype="U64")
    npz_dict["meta_version"] = np.array("v1", dtype="U8")

    # Save prediction fields
    for key in all_keys:
        stacked, mask, frames_present = _try_stack_key(key, ress, squeeze_batch)
        if stacked is not None:
            npz_dict[key] = stacked
            npz_dict[f"{key}_present"] = mask
        else:
            npz_dict[f"{key}_frames"] = np.array(frames_present, dtype=np.int64)
            for t in frames_present:
                arr = _maybe_squeeze_B(_to_numpy(ress[t][key]), squeeze_batch)
                npz_dict[f"{key}_{t:06d}"] = arr

    # Optionally save selected fields from views (without raw images by default)
    if include_view_fields and views is not None:
        for vf in include_view_fields:
            # synthesize per-frame containers for this single field
            vf_ress = []
            for vw in views:
                if vw is None or vf not in vw:
                    vf_ress.append({})
                else:
                    vf_val = vw[vf]
                    if torch.is_tensor(vf_val) or isinstance(vf_val, np.ndarray):
                        vf_ress.append({vf: vf_val})
                    else:
                        vf_ress.append({vf: np.array(vf_val)})
            stacked, mask, frames_present = _try_stack_key(vf, vf_ress, squeeze_batch)
            base = f"view_{vf}"
            if stacked is not None:
                npz_dict[base] = stacked
                npz_dict[f"{base}_present"] = mask
            else:
                npz_dict[f"{base}_frames"] = np.array(frames_present, dtype=np.int64)
                for t in frames_present:
                    val = vf_ress[t][vf]
                    arr = _maybe_squeeze_B(_to_numpy(val), squeeze_batch) \
                          if torch.is_tensor(val) or isinstance(val, np.ndarray) \
                          else np.array(val)
                    npz_dict[f"{base}_{t:06d}"] = arr

    if compress:
        np.savez_compressed(path, **npz_dict)
    else:
        np.savez(path, **npz_dict)
    return path

#################
# Load From NPZ #
#################

def _np_to_torch_if_numeric(x: np.ndarray, to_torch: bool, device: Union[str, torch.device]):
    if not to_torch:
        return x
    # treat numeric + boolean arrays as convertible; leave strings/objects alone
    if x.dtype.kind in ("b", "i", "u", "f", "c"):  # bool/int/uint/float/complex
        return torch.from_numpy(x).to(device)
    return x  # leave non-numeric (e.g., strings) as numpy


def _assign_stacked_field(ress: List[Dict[str, Any]], key: str, arr: np.ndarray, mask: Optional[np.ndarray],
                          to_torch: bool, device: Union[str, torch.device]):
    T = len(ress)
    if mask is None:
        mask = np.ones((T,), dtype=bool)
    for t in range(T):
        if mask[t]:
            val = arr[t]
            ress[t][key] = _np_to_torch_if_numeric(val, to_torch, device)


def _assign_ragged_field(ress: List[Dict[str, Any]], key: str, frames_present: np.ndarray, npz: np.lib.npyio.NpzFile,
                         to_torch: bool, device: Union[str, torch.device]):
    for t in frames_present.tolist():
        val = npz[f"{key}_{t:06d}"]
        ress[t][key] = _np_to_torch_if_numeric(val, to_torch, device)


def _collect_view_bases(npz: np.lib.npyio.NpzFile) -> List[str]:
    """Find base names of view fields saved with 'view_' prefix."""
    bases = set()
    for k in npz.files:
        if not k.startswith("view_"):
            continue
        field = k[5:]
        # strip suffixes
        if field.endswith("_present"):
            field = field[:-9]
        elif field.endswith("_frames"):
            field = field[:-7]
        else:
            m = re.match(r"(.+)_\d{6}$", field)
            if m:
                field = m.group(1)
        bases.add(field)
    return sorted(bases)


def load_streamvggt_npz(
    path: str,
    *,
    to_torch: bool = False,
    device: Union[str, torch.device] = "cpu"
) -> Dict[str, Any]:
    """
    Load results saved by save_streamvggt_npz.

    Returns dict with:
      - 'ress' : list of per-frame dicts
      - 'views': list of per-frame dicts OR None (if no view fields saved)
      - 'meta' : {'T': int, 'keys': List[str], 'version': str}
    """
    npz = np.load(path, allow_pickle=False)
    T = int(npz["meta_T"])
    keys = [str(x) for x in npz["meta_keys"]]
    version = str(npz["meta_version"]) if "meta_version" in npz else "v0"
    ress: List[Dict[str, Any]] = [dict() for _ in range(T)]

    # restore prediction fields
    for key in keys:
        if key in npz.files:
            arr = npz[key]
            mask = npz[f"{key}_present"] if f"{key}_present" in npz.files else None
            _assign_stacked_field(ress, key, arr, mask, to_torch, device)
        elif f"{key}_frames" in npz.files:
            frames_present = npz[f"{key}_frames"]
            _assign_ragged_field(ress, key, frames_present, npz, to_torch, device)
        else:
            # key missing entirely (shouldn't happen if saved by our saver)
            pass

    # restore views if any 'view_' keys exist
    view_bases = _collect_view_bases(npz)
    views: Optional[List[Dict[str, Any]]] = None
    if len(view_bases) > 0:
        views = [dict() for _ in range(T)]
        for base in view_bases:
            full = f"view_{base}"
            if full in npz.files:
                arr = npz[full]
                mask = npz[f"{full}_present"] if f"{full}_present" in npz.files else None
                # assign into views[t][base]
                if mask is None:
                    mask = np.ones((T,), dtype=bool)
                for t in range(T):
                    if mask[t]:
                        val = arr[t]
                        views[t][base] = _np_to_torch_if_numeric(val, to_torch, device)
            elif f"{full}_frames" in npz.files:
                frames_present = npz[f"{full}_frames"]
                for t in frames_present.tolist():
                    val = npz[f"{full}_{t:06d}"]
                    views[t][base] = _np_to_torch_if_numeric(val, to_torch, device)

    return {
        "ress": ress,
        "views": views,
        "meta": {"T": T, "keys": keys, "version": version}
    }


def load_npz_into_outsbuffer(
    path: str,
    *,
    to_torch: bool = False,
    device: Union[str, torch.device] = "cpu"
) -> OutsBuffer:
    """
    Load a saved .npz back into an OutsBuffer.
    """
    data = load_streamvggt_npz(path, to_torch=to_torch, device=device)
    buf = OutsBuffer()
    buf.ress = data["ress"]
    buf.views = data["views"] if data["views"] is not None else []
    buf.n_frames = len(buf.ress)
    return buf

########################
# Conversion to Points #
########################
def extract_pts_cols_from_frame_result(
    res: dict,
    img_path,
    *,
    conf_key: str = "conf",
    pts_key: str  = "pts3d_in_other_view",
    conf_thresh: float = 0.50
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (pts Nx3 float32, cols Nx3 uint8) for ONE frame result.
    Uses per-pixel world points + confidence; reads the original RGB for colors.
    """
    if (pts_key not in res) or (conf_key not in res):
        return np.zeros((0,3), np.float32), np.zeros((0,3), np.uint8)

    pts_hw3  = res[pts_key][0].detach().cpu().numpy()   # [H,W,3]
    conf_hw  = res[conf_key][0].detach().cpu().numpy()  # [H,W]

    H, W, _  = pts_hw3.shape
    conf_thresh_pc = np.percentile(conf_hw, conf_thresh * 100)
    valid    = np.isfinite(pts_hw3).all(axis=-1) & (conf_hw >= float(conf_thresh_pc))

    if not np.any(valid):
        return np.zeros((0,3), np.float32), np.zeros((0,3), np.uint8)

    # Colors
    if img_path is not None:
        im_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if im_bgr is not None:
            im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
            if im_rgb.shape[:2] != (H, W):
                im_rgb = cv2.resize(im_rgb, (W, H), interpolation=cv2.INTER_AREA)
        else:
            im_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    else:
        im_rgb = np.zeros((H, W, 3), dtype=np.uint8)

    pts  = pts_hw3.reshape(-1, 3)
    cols = im_rgb.reshape(-1, 3)
    m    = valid.reshape(-1)

    pts_sel  = pts[m].astype(np.float32, copy=False)
    cols_sel = cols[m].astype(np.uint8,  copy=False)
    return pts_sel, cols_sel

class ReservoirPointSampler:
    """
    Classic streaming reservoir sampler (Algorithm R) for points + colors.
    Keeps a uniform random sample of size <= capacity from an arbitrarily long stream.
    """
    def __init__(self, capacity: int, seed = None):
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(seed)
        self.points = np.zeros((capacity, 3), dtype=np.float32)
        self.colors = np.zeros((capacity, 3), dtype=np.uint8)
        self.filled = 0          # how many slots actually filled (<= capacity)
        self.n_total = 0         # total items seen (for correct probabilities)

    def add_batch(self, pts: np.ndarray, cols: np.ndarray):
        assert pts.shape == cols.shape and pts.shape[1] == 3
        n = pts.shape[0]
        for i in range(n):
            if self.n_total < self.capacity:
                # still filling the reservoir
                self.points[self.n_total] = pts[i]
                self.colors[self.n_total] = cols[i]
                self.n_total += 1
                self.filled = self.n_total
            else:
                # replace with probability capacity / (n_total+1)
                j = self.rng.integers(0, self.n_total + 1)
                if j < self.capacity:
                    self.points[j] = pts[i]
                    self.colors[j] = cols[i]
                self.n_total += 1

    def get(self) -> tuple[np.ndarray, np.ndarray]:
        return self.points[:self.filled].copy(), self.colors[:self.filled].copy()
    
def buffer_to_points_and_colors_fixed_n(
    buffer,                          # OutsBuffer
    *,
    target_n: int = 60000,
    conf_thresh: float = 0.50
) -> tuple[np.ndarray, np.ndarray]:
    if not hasattr(buffer, "ress") or buffer.n_frames == 0:
        return np.zeros((0,3), np.float32), np.zeros((0,3), np.uint8)

    sampler = ReservoirPointSampler(capacity=int(target_n), seed=42)

    has_views = len(buffer.views) == buffer.n_frames
    for t in range(buffer.n_frames):
        res = buffer.ress[t]
        img_path = None
        if has_views and isinstance(buffer.views[t], dict):
            img_path = buffer.views[t].get("path", None)

        pts, cols = extract_pts_cols_from_frame_result(
            res, img_path, conf_thresh=conf_thresh
        )
        if pts.shape[0]:
            sampler.add_batch(pts, cols)

    return sampler.get()



#################
# Other helpers #
#################

def stack_over_time(buf: OutsBuffer, key: str):
    """
    Stack a tensor field over time: returns [T, ...].
    Works when each frame has the same shape for that key (common case).
    """
    vals = [r[key] for r in buf.ress if (r is not None and key in r)]
    if len(vals) == 0:
        return None
    # Choose stack or cat depending on your desired leading dim
    try:
        return torch.stack(vals, dim=0)
    except Exception:
        # fallback to cat for already batched [B,...] tensors
        return torch.cat(vals, dim=0)

def last_n(buf: OutsBuffer, n: int) -> List[Dict[str, Any]]:
    """Grab the last n per-frame outs (useful for recent previews)."""
    return buf.ress[-n:]

def _resize_mask_to_hw(mask_any, H: int, W: int) -> np.ndarray:
    """Return HxW bool mask, resizing with nearest-neighbor if needed."""
    if torch.is_tensor(mask_any):
        m = mask_any.detach().cpu().numpy()
    else:
        m = np.asarray(mask_any)
    m = m.astype(np.uint8)
    if m.shape[:2] != (H, W):
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
    return (m > 0)

def _apply_mask_to_preds_inplace(
    pts3d: torch.Tensor,      # [B, H, W, 3]
    conf_pts: torch.Tensor,   # [B, H, W]
    depth: torch.Tensor,      # [B, H, W, 1]
    conf_depth: torch.Tensor, # [B, H, W]
    mask_any                   # HxW (np/torch), any dtype
):
    """Invalidate points (NaN) and zero-out conf/depth outside mask."""
    B, H, W, _ = pts3d.shape
    vm = _resize_mask_to_hw(mask_any, H, W)
    vm_t   = torch.from_numpy(vm).to(pts3d.device, dtype=torch.bool)   # [H,W]
    vm_bhw = vm_t.unsqueeze(0)                                         # [1,H,W]
    vm_bhw1= vm_bhw.unsqueeze(-1)                                      # [1,H,W,1]
    inv1   = ~vm_bhw1

    # Points outside mask → NaN; confidences → 0; depth → 0
    pts3d[inv1.expand_as(pts3d)] = torch.nan
    conf_pts  *= vm_bhw.float()
    conf_depth*= vm_bhw.float()
    depth[inv1.expand_as(depth)] = 0

def _apply_mask_to_preds_copy(
    pts3d: torch.Tensor,      # [B, H, W, 3]
    conf_pts: torch.Tensor,   # [B, H, W]
    depth: torch.Tensor,      # [B, H, W, 1]
    conf_depth: torch.Tensor, # [B, H, W]
    mask_any
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return masked clones (originals untouched)."""
    pts3d_c      = pts3d.clone()
    conf_pts_c   = conf_pts.clone()
    depth_c      = depth.clone()
    conf_depth_c = conf_depth.clone()
    _apply_mask_to_preds_inplace(
        pts3d=pts3d_c, conf_pts=conf_pts_c,
        depth=depth_c, conf_depth=conf_depth_c,
        mask_any=mask_any
    )
    return pts3d_c, conf_pts_c, depth_c, conf_depth_c

###################
# StreamVGGT code #
###################

@dataclass
class StreamVGGTOutput(ModelOutput):
    ress: Optional[List[dict]] = None
    views: Optional[torch.Tensor] = None

class StreamVGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024):
        super().__init__()

        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.camera_head = CameraHead(dim_in=2 * embed_dim)
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1")
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1")
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size)
    


    def forward(
        self,
        views,
        query_points: torch.Tensor = None,
        history_info: Optional[dict] = None,
        past_key_values=None,
        use_cache=False,
        past_frame_idx=0
    ):
        images = torch.stack(
            [view["img"] for view in views], dim=0
        ).permute(1, 0, 2, 3, 4)    # B S C H W

        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        if history_info is None:
            history_info = {"token": None}

        aggregated_tokens_list, patch_start_idx = self.aggregator(images)
        predictions = {}

        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration

            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

            if self.track_head is not None and query_points is not None:
                track_list, vis, conf = self.track_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
                )
                predictions["track"] = track_list[-1]  # track of the last iteration
                predictions["vis"] = vis
                predictions["conf"] = conf
            predictions["images"] = images

            B, S = images.shape[:2]
            ress = []
            for s in range(S):
                res = {
                    'pts3d_in_other_view': predictions['world_points'][:, s],  # [B, H, W, 3]
                    'conf': predictions['world_points_conf'][:, s],  # [B, H, W]

                    'depth': predictions['depth'][:, s],  # [B, H, W, 1]
                    'depth_conf': predictions['depth_conf'][:, s],  # [B, H, W]
                    'camera_pose': predictions['pose_enc'][:, s, :],  # [B, 9]

                    **({'valid_mask': views[s]["valid_mask"]}
                    if 'valid_mask' in views[s] else {}),  # [B, H, W]

                    **({'track': predictions['track'][:, s],  # [B, N, 2]
                        'vis': predictions['vis'][:, s],  # [B, N]
                        'track_conf': predictions['conf'][:, s]}
                    if 'track' in predictions else {})
                }
                ress.append(res)
            return StreamVGGTOutput(ress=ress, views=views)  # [S] [B, C, H, W]
        
    def inference(self, frames, query_points: torch.Tensor = None, past_key_values=None):        
        past_key_values = [None] * self.aggregator.depth
        past_key_values_camera = [None] * self.camera_head.trunk_depth
        
        all_ress = []
        processed_frames = []

        for i, frame in enumerate(frames):
            images = frame["img"].unsqueeze(0) 
            aggregator_output = self.aggregator(
                images, 
                past_key_values=past_key_values,
                use_cache=True, 
                past_frame_idx=i
            )
            
            if isinstance(aggregator_output, tuple) and len(aggregator_output) == 3:
                aggregated_tokens, patch_start_idx, past_key_values = aggregator_output
            else:
                aggregated_tokens, patch_start_idx = aggregator_output
            
            with torch.cuda.amp.autocast(enabled=False):
                if self.camera_head is not None:
                    pose_enc, past_key_values_camera = self.camera_head(aggregated_tokens, past_key_values_camera=past_key_values_camera, use_cache=True)
                    pose_enc = pose_enc[-1]
                    camera_pose = pose_enc[:, 0, :]

                if self.depth_head is not None:
                    depth, depth_conf = self.depth_head(
                        aggregated_tokens, images=images, patch_start_idx=patch_start_idx
                    )
                    depth = depth[:, 0] 
                    depth_conf = depth_conf[:, 0]
                
                if self.point_head is not None:
                    pts3d, pts3d_conf = self.point_head(
                        aggregated_tokens, images=images, patch_start_idx=patch_start_idx
                    )
                    pts3d = pts3d[:, 0] 
                    pts3d_conf = pts3d_conf[:, 0]

                if self.track_head is not None and query_points is not None:
                    track_list, vis, conf = self.track_head(
                        aggregated_tokens, images=images, patch_start_idx=patch_start_idx, query_points=query_points
                )
                    track = track_list[-1][:, 0]  
                    query_points = track
                    vis = vis[:, 0]
                    track_conf = conf[:, 0]

            all_ress.append({
                'pts3d_in_other_view': pts3d,
                'conf': pts3d_conf,
                'depth': depth,
                'depth_conf': depth_conf,
                'camera_pose': camera_pose,
                **({'valid_mask': frame["valid_mask"]}
                    if 'valid_mask' in frame else {}),  

                **({'track': track, 
                    'vis': vis,  
                    'track_conf': track_conf}
                if query_points is not None else {})
            })
            processed_frames.append(frame)
        
        output = StreamVGGTOutput(ress=all_ress, views=processed_frames)
        return output
    
    def new_state(self, *, max_keep: Optional[int] = None) -> StreamState:
        return StreamState(
            past_kv_agg=[None] * self.aggregator.depth,
            past_kv_cam=[None] * self.camera_head.trunk_depth,
            frame_idx=0,
            max_keep=max_keep
        )
        
    def stream_step(
        self,
        frames: List[dict],                 # each with "img", optional "valid_mask"
        state: StreamState,
        query_points: Optional[torch.Tensor] = None,  # (B, N, 2) in normalized coords
        return_per_frame: bool = True,
    ):
        """Process 1..K frames sequentially, updating caches inside `state`."""
        self.eval()
        outs = []
        if query_points is None:
            query_points = state.last_query_points

        for f in frames:
            images = f["img"].unsqueeze(0)  # [B=1, C, H, W]
            images = images.unsqueeze(1)    # [B=1, S=1, C, H, W]

            with infer_mode():
                # 1) aggregator with cache
                agg_out = self.aggregator(
                    images,
                    past_key_values=state.past_kv_agg,
                    use_cache=True,
                    past_frame_idx=state.frame_idx
                )
                if isinstance(agg_out, tuple) and len(agg_out) == 3:
                    aggregated_tokens, patch_start_idx, state.past_kv_agg = agg_out
                else:
                    aggregated_tokens, patch_start_idx = agg_out  # if aggregator doesn’t return caches

                # 2) camera head with cache
                pose_enc, state.past_kv_cam = self.camera_head(
                    aggregated_tokens,
                    past_key_values_camera=state.past_kv_cam,
                    use_cache=True
                )
                pose_enc = pose_enc[-1]       # [B, S=1, D]
                camera_pose = pose_enc[:, 0, :]  # [B, D]

                # 3) depth
                depth, depth_conf = self.depth_head(aggregated_tokens, images=images, patch_start_idx=patch_start_idx)
                depth      = depth[:, 0]       # [B, H, W, 1]
                depth_conf = depth_conf[:, 0]  # [B, H, W]

                # 4) 3D points in world coords
                pts3d, pts3d_conf = self.point_head(aggregated_tokens, images=images, patch_start_idx=patch_start_idx)
                pts3d      = pts3d[:, 0]       # [B, H, W, 3]
                pts3d_conf = pts3d_conf[:, 0]  # [B, H, W]

                # 5) optional tracking across frames
                track = vis = track_conf = None
                if (self.track_head is not None) and (query_points is not None):
                    track_list, vis, conf = self.track_head(
                        aggregated_tokens,
                        images=images,
                        patch_start_idx=patch_start_idx,
                        query_points=query_points
                    )
                    track      = track_list[-1][:, 0]  # [B, N, 2]
                    vis        = vis[:, 0]             # [B, N]
                    track_conf = conf[:, 0]            # [B, N]
                    # feed next step:
                    query_points = track
                    state.last_query_points = track

                # 6) persist outputs (detach+cpu to control VRAM)
                _append_and_prune(state.camera_poses, camera_pose.detach().cpu(), state.max_keep)
                _append_and_prune(state.pts3d_list,   pts3d.detach().cpu(),        state.max_keep)
                _append_and_prune(state.depth_list,   depth.detach().cpu(),        state.max_keep)
                _append_and_prune(state.depth_conf_list, depth_conf.detach().cpu(), state.max_keep)

                # 7) per-frame result (optional)
                if return_per_frame:
                    out = {
                        'pts3d_in_other_view': pts3d,       # keep on GPU if you like
                        'conf'               : pts3d_conf,
                        'depth'              : depth,
                        'depth_conf'         : depth_conf,
                        'camera_pose'        : camera_pose,
                    }
                    if 'valid_mask' in f: out['valid_mask'] = f['valid_mask']
                    if track is not None:
                        out.update({'track': track, 'vis': vis, 'track_conf': track_conf})
                    outs.append(out)

                state.frame_idx += 1

        return outs, state
