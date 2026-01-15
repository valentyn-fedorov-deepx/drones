from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Union, Iterable

import cv2
import numpy as np
import torch
import trimesh

from streamvggt.models.streamvggt_live import (
    StreamVGGT, StreamState, OutsBuffer,
    new_outs_buffer, append_outs_inplace,
    buffer_to_points_and_colors_fixed_n, save_streamvggt_npz, load_npz_into_outsbuffer,
    extract_pts_cols_from_frame_result, ReservoirPointSampler, _apply_mask_to_preds_inplace
)
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


def iter_chunks(seq: List[Any], chunk: int = 3) -> Iterable[List[Any]]:
    """Yield consecutive chunks from a list."""
    for i in range(0, len(seq), chunk):
        yield seq[i:i+chunk]


class StreamVGGTProcessor:
    """
    High-level, stateful wrapper for StreamVGGT with:
      - single/batched/streamed inference
      - live subsampled viz via reservoir sampler
      - save/load whole sessions (incl. offline work without model)
      - convenient getters: points/conf/colors/depth/pose_enc/camera params
    """

    def __init__(
        self,
        device: str = "cuda",
        model_path: Optional[Union[str, Path]] = None,
        skip_model: bool = False,
        max_keep: Optional[int] = None,          # sliding window for internal histories
        live_capacity: int = 60_000,             # reservoir size for live sampling
        default_viz_conf_thresh: float = 0.50,   # percentile for per-frame point selection in live updates
    ):
        self.device = device
        self.model: Optional[StreamVGGT] = None
        self.state: Optional[StreamState] = None
        self.buf: OutsBuffer = new_outs_buffer()

        # live sampler used for incremental viz updates
        self.live_sampler = ReservoirPointSampler(capacity=int(live_capacity), seed=42)
        self.viz_conf_thresh = float(default_viz_conf_thresh)

        if not skip_model:
            self.model = StreamVGGT().to(device).eval()
            if model_path is not None:
                sd = torch.load(str(model_path), map_location=device)
                if isinstance(sd, dict) and "state_dict" in sd:
                    sd = sd["state_dict"]
                self.model.load_state_dict(sd, strict=False)
            self.state = self.model.new_state(max_keep=max_keep)

    # ---------------- Lifecycle / Buffer ----------------

    def has_model(self) -> bool:
        return self.model is not None

    def reset_state(self, max_keep: Optional[int] = None):
        """Reset streaming caches; keep buffer intact."""
        if not self.has_model():
            raise RuntimeError("Processor initialized with skip_model=True (no model to reset).")
        self.state = self.model.new_state(max_keep=max_keep)

    def clear_buffer(self):
        """Drop accumulated per-frame outputs & views."""
        self.buf = new_outs_buffer()
        self.live_sampler = ReservoirPointSampler(self.live_sampler.capacity, seed=42)

    # ---------------- Preprocess / Frames ----------------

    def preprocess_images(
        self,
        image_paths: List[Union[str, Path]],
        valid_masks: Optional[List[np.ndarray]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Load & preprocess paths → frames list for StreamVGGT.
        Each frame: {'img': CHW float tensor on device, 'path': str, optional 'valid_mask': HxW}
        """
        imgs_t = load_and_preprocess_images([str(p) for p in image_paths]).to(self.device)  # [N,C,H,W]
        frames: List[Dict[str, Any]] = []
        for i, p in enumerate(image_paths):
            f = {"img": imgs_t[i], "path": str(p)}
            if valid_masks is not None and i < len(valid_masks) and valid_masks[i] is not None:
                f["valid_mask"] = valid_masks[i]
            frames.append(f)
        return frames
    
    def _resize_mask_to_img(self, mask: np.ndarray, img_chw: torch.Tensor) -> np.ndarray:
        """
        Ensure mask matches the HxW of img_chw (C,H,W). Returns boolean HxW.
        """
        H, W = int(img_chw.shape[1]), int(img_chw.shape[2])
        m = mask
        if m.shape[:2] != (H, W):
            m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        return (m > 0).astype(np.bool_)

    def get_pose_encoding_stack(self) -> Optional[np.ndarray]:
        """Alias for your 9-D pose encoding per frame (what you store in 'camera_pose')."""
        return self.get_pose_enc_stack()

    def preprocess_image(
        self,
        image_path: str | Path,
        valid_mask: np.ndarray | None = None,
    ) -> dict:
        """
        Preprocess a SINGLE image path → frame dict compatible with StreamVGGT.
        Returns: {'img': CHW float tensor on device, 'path': str, optional 'valid_mask': HxW}
        """
        img_t = load_and_preprocess_images([str(image_path)]).to(self.device)  # [1,C,H,W]
        img_1 = img_t[0]  # [C,H,W]
        frame = {"img": img_1, "path": str(image_path)}
        if valid_mask is not None:
            frame["valid_mask"] = self._resize_mask_to_img(valid_mask, img_1)
        return frame


    # ---------------- Inference Modes ----------------

    def forward_views(
        self,
        views: List[Dict[str, Any]],
        query_points: Optional[torch.Tensor] = None,
        append_to_buffer: bool = True,
        store_device: torch.device = torch.device("cpu"),
        keep_images: bool = False,
        update_live: bool = True,
        conf_thresh: Optional[float] = None,
    ):
        """
        One-shot multi-view forward (no persistent caches).
        """
        if not self.has_model():
            raise RuntimeError("No model loaded (skip_model=True). Use load_npz(...) for offline work.")
        out = self.model.forward(views=views, query_points=query_points)
        if append_to_buffer:
            append_outs_inplace(self.buf, out.ress, frames=out.views,
                                store_device=store_device, keep_images=keep_images)
        if update_live:
            self._update_live_sampler_from_outs(out.ress, out.views, conf_thresh)
        return out

    def stream_frames(
        self,
        frames: List[Dict[str, Any]],
        query_points: Optional[torch.Tensor] = None,
        append_to_buffer: bool = True,
        return_per_frame: bool = True,
        store_device: torch.device = torch.device("cpu"),
        keep_images: bool = False,
        update_live: bool = True,
        conf_thresh: Optional[float] = None,
    ):
        """
        Stateful streaming over 1..K frames, updating caches and (optionally) live sampler.
        """
        if not self.has_model() or self.state is None:
            raise RuntimeError("Model/state not initialized.")
        outs, self.state = self.model.stream_step(
            frames=frames,
            state=self.state,
            query_points=query_points,
            return_per_frame=return_per_frame,
        )
        if outs:
            outs = self._mask_outs_for_buffer_only(outs, frames)
        if append_to_buffer and len(outs):
            append_outs_inplace(self.buf, outs, frames=frames,
                                store_device=store_device, keep_images=keep_images)
        if update_live and len(outs):
            self._update_live_sampler_from_outs(outs, frames, conf_thresh)
        return outs

    def step(
        self,
        frame: Dict[str, Any],
        query_points: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """Process a single frame in streaming mode."""
        return self.stream_frames([frame], query_points=query_points, **kwargs)

    def stream_in_chunks(
        self,
        frames_all: List[Dict[str, Any]],
        chunk: int = 3,
        query_points: Optional[torch.Tensor] = None,
        store_device: torch.device = torch.device("cpu"),
        keep_images: bool = False,
        conf_thresh: Optional[float] = None,
    ):
        """
        Convenience: stream frames in chunks of K (2-3-5...), updating buffer + live sampler.
        """
        for batch in iter_chunks(frames_all, chunk=chunk):
            self.stream_frames(
                batch,
                query_points=query_points,
                append_to_buffer=True,
                return_per_frame=True,
                store_device=store_device,
                keep_images=keep_images,
                update_live=True,
                conf_thresh=conf_thresh,
            )
        # return the current live sample for display
        return self.live_sampler.get()

    # ---------------- Live Subsampling & Viz ----------------
    def _mask_outs_for_buffer_only(
        self,
        outs: List[Dict[str, Any]],
        frames: Optional[List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Return outs where pts/depth/conf fields are masked IF frames[i]['valid_mask'] exists.
        Originals in `outs` are left untouched (we mask clones)."""
        if not outs or frames is None:
            return outs
        masked_outs: List[Dict[str, Any]] = []
        for i, res in enumerate(outs):
            m = frames[i].get("valid_mask") if (i < len(frames) and isinstance(frames[i], dict)) else None
            if m is None:
                masked_outs.append(res)
                continue

            # Ensure required fields exist
            p  = res.get("pts3d_in_other_view")
            cp = res.get("conf")
            d  = res.get("depth")
            cd = res.get("depth_conf")
            if p is None or cp is None or d is None or cd is None:
                masked_outs.append(res)
                continue

            # Clone to avoid mutating originals
            p_m  = p.clone()  if torch.is_tensor(p)  else p.copy()
            cp_m = cp.clone() if torch.is_tensor(cp) else cp.copy()
            d_m  = d.clone()  if torch.is_tensor(d)  else d.copy()
            cd_m = cd.clone() if torch.is_tensor(cd) else cd.copy()

            # Reuse proven in-place logic on the clones
            _apply_mask_to_preds_inplace(
                pts3d=p_m, conf_pts=cp_m, depth=d_m, conf_depth=cd_m, mask_any=m
            )

            r2 = dict(res)
            r2["pts3d_in_other_view"] = p_m
            r2["conf"]                = cp_m
            r2["depth"]               = d_m
            r2["depth_conf"]          = cd_m
            masked_outs.append(r2)
        return masked_outs

    def _update_live_sampler_from_outs(
        self,
        outs: List[Dict[str, Any]],
        frames: Optional[List[Dict[str, Any]]] = None,
        conf_thresh: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Incrementally feed new points/colors (filtered by confidence percentile)
        into the live reservoir sampler.
        """
        thr = float(self.viz_conf_thresh if conf_thresh is None else conf_thresh)
        for i, res in enumerate(outs):
            img_path = None
            if frames is not None and i < len(frames) and isinstance(frames[i], dict):
                img_path = frames[i].get("path", None)
            pts, cols = extract_pts_cols_from_frame_result(res, img_path, conf_thresh=thr)
            if pts.shape[0]:
                self.live_sampler.add_batch(pts, cols)
        return self.live_sampler.get()
    
    def rebuild_live_from_buffer(self, conf_thresh: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Rebuild the live reservoir strictly from the *current* buffer contents
        (after any masking/edits). This guarantees get_live_sample reflects filtering.
        """
        thr = float(self.viz_conf_thresh if conf_thresh is None else conf_thresh)
        # fresh reservoir with same capacity/seed for determinism
        self.live_sampler = ReservoirPointSampler(self.live_sampler.capacity, seed=42)

        # walk buffer and add points/colors the same way streaming does
        for t, res in enumerate(self.buf.ress):
            img_path = None
            if t < len(self.buf.views) and isinstance(self.buf.views[t], dict):
                img_path = self.buf.views[t].get("path", None)
            pts, cols = extract_pts_cols_from_frame_result(res, img_path, conf_thresh=thr)
            if pts.shape[0]:
                self.live_sampler.add_batch(pts, cols)

        return self.live_sampler.get()

    def get_live_sample(self, rebuild=False) -> Tuple[np.ndarray, np.ndarray]:
        """Get current (pts, cols) from the live reservoir sampler."""
        if rebuild:
            self.rebuild_live_from_buffer()
        return self.live_sampler.get()

    def fresh_sample_from_buffer(
        self, target_n: int = 60_000, conf_thresh: float = 0.50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Re-sample uniformly from the entire buffer (ignores the live sampler)."""
        return buffer_to_points_and_colors_fixed_n(self.buf, target_n=target_n, conf_thresh=conf_thresh)

    # ---------------- Save / Load “Everything” ----------------

    def save_npz(
        self,
        path: Union[str, Path],
        include_view_fields: Optional[List[str]] = ("path",),
        compress: bool = True,
    ) -> str:
        """Persist full per-frame outputs to a single NPZ (for offline use)."""
        return save_streamvggt_npz(
            str(path),
            buf=self.buf,
            include_view_fields=list(include_view_fields) if include_view_fields else None,
            compress=compress,
        )

    def save_all(
        self,
        out_dir: Union[str, Path],
        *,
        sample_n: int = 120_000,
        conf_thresh: float = 0.55,
        save_stream_npz_name: str = "stream_outputs.npz",
        save_glb_name: str = "reconstruction.glb",
        save_points_npy: str = "points.npy",
        save_colors_npy: str = "colors.npy",
        save_cam_npz_name: str = "camera_params.npz",
    ) -> Dict[str, str]:
        """
        Save: sampled (pts, cols), camera extrinsics/intrinsics, and full stream outputs NPZ.
        Returns dict of file paths.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1) sub-sampled cloud
        pts, cols = self.fresh_sample_from_buffer(target_n=sample_n, conf_thresh=conf_thresh)
        glb_path = out_dir / save_glb_name
        self.save_points_as_glb(pts, cols, glb_path)

        np.save(out_dir / save_points_npy, pts)
        np.save(out_dir / save_colors_npy, cols)

        # 2) camera parameters
        ex, intr = self.get_camera_extri_intri(as_numpy=True)
        np.savez(out_dir / save_cam_npz_name, extrinsics=ex, intrinsics=intr)

        # 3) full per-frame outputs (with view paths)
        npz_path = out_dir / save_stream_npz_name
        self.save_npz(npz_path, include_view_fields=("path",), compress=True)

        return {
            "glb": str(glb_path),
            "points": str(out_dir / save_points_npy),
            "colors": str(out_dir / save_colors_npy),
            "camera_params": str(out_dir / save_cam_npz_name),
            "stream_npz": str(npz_path),
        }

    def load_npz(self, path: Union[str, Path], to_torch: bool = False, device: Union[str, torch.device] = "cpu"):
        """
        Load a prior run into this processor's buffer (offline mode OK).
        """
        self.buf = load_npz_into_outsbuffer(str(path), to_torch=to_torch, device=device)
        # reset live sampler; you can rebuild it by calling fresh_sample_from_buffer or replaying outs.
        self.live_sampler = ReservoirPointSampler(self.live_sampler.capacity, seed=42)
        return self.buf

    @classmethod
    def from_npz_only(
        cls,
        npz_path: Union[str, Path],
        *,
        device: str = "cpu",
        live_capacity: int = 60_000,
    ) -> "StreamVGGTProcessor":
        """
        Offline constructor: no model loaded; just a buffer for analysis/exports.
        """
        proc = cls(device=device, skip_model=True, live_capacity=live_capacity)
        proc.load_npz(npz_path, to_torch=False, device="cpu")
        return proc

    # ---------------- Retrieval / Getters ----------------
    # All getters operate on the internal OutsBuffer (works online or offline)

    def _stack_over_time_numpy(self, key: str) -> Optional[np.ndarray]:
        """
        Stack [T,...] for a given key, converting to numpy.
        Returns None if key is missing for all frames.
        """
        vals = []
        for r in self.buf.ress:
            if r is None or key not in r:
                continue
            v = r[key]
            # v has shape [B,...], B==1
            if torch.is_tensor(v):
                vals.append(v[0].detach().cpu().numpy())
            else:
                # assume numpy-like
                vals.append(np.array(v)[0])
        if not vals:
            return None
        try:
            return np.stack(vals, axis=0)
        except Exception:
            # ragged; fallback to list -> object array
            return np.array(vals, dtype=object)

    def get_points_grid_stack(self) -> Optional[np.ndarray]:
        """[T, H, W, 3] world points per frame (NaNs possible for invalid)."""
        return self._stack_over_time_numpy("pts3d_in_other_view")

    def get_point_conf_stack(self) -> Optional[np.ndarray]:
        """[T, H, W] confidence for world points."""
        return self._stack_over_time_numpy("conf")

    def get_depth_stack(self) -> Optional[np.ndarray]:
        """[T, H, W, 1] depth per frame."""
        return self._stack_over_time_numpy("depth")

    def get_depth_conf_stack(self) -> Optional[np.ndarray]:
        """[T, H, W] depth confidence per frame."""
        return self._stack_over_time_numpy("depth_conf")

    def get_pose_enc_stack(self) -> Optional[np.ndarray]:
        """[T, D] pose encoding per frame (D usually 9)."""
        # stored as [B, D] per frame → stack to [T, D]
        vals = []
        for r in self.buf.ress:
            if r is None or "camera_pose" not in r:
                continue
            v = r["camera_pose"]
            v = v[0] if torch.is_tensor(v) else np.array(v)[0]
            v = v.detach().cpu().numpy() if torch.is_tensor(v) else np.array(v)
            vals.append(v)
        if not vals:
            return None
        return np.stack(vals, axis=0)

    def get_camera_extri_intri(
        self, as_numpy: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decode extrinsics (4x4) and intrinsics (3x3) for all frames using pose_enc & image size.
        """
        T = self.buf.n_frames
        if T == 0:
            raise RuntimeError("Buffer is empty.")
        pose_enc = self.get_pose_enc_stack()  # [T,D]
        if pose_enc is None:
            raise RuntimeError("No 'camera_pose' fields in buffer.")

        # try to get H,W from depth; otherwise from conf
        H = W = None
        for r in self.buf.ress:
            if "depth" in r:
                d = r["depth"][0]
                H, W = int(d.shape[0]), int(d.shape[1])
                break
        if H is None:
            for r in self.buf.ress:
                if "conf" in r:
                    c = r["conf"][0]
                    H, W = int(c.shape[0]), int(c.shape[1])
                    break
        if H is None or W is None:
            raise RuntimeError("Cannot infer (H,W) to decode camera parameters.")

        pose_enc_t = torch.from_numpy(pose_enc).unsqueeze(0)  # [1,T,D]
        ex, intr = pose_encoding_to_extri_intri(pose_enc_t, (H, W))  # [1,T,4,4], [1,T,3,3]
        ex = ex.squeeze(0)
        intr = intr.squeeze(0)
        if as_numpy:
            ex = ex.detach().cpu().numpy()
            intr = intr.detach().cpu().numpy()
        return ex, intr

    def get_colors_grid_stack(self) -> Optional[np.ndarray]:
        """
        [T, H, W, 3] colors per frame, re-read from original images saved in views['path'].
        Returns None if no paths are stored.
        """
        T = self.buf.n_frames
        if T == 0:
            return None
        pts0 = self.get_points_grid_stack()
        if pts0 is None:
            return None
        H, W = pts0.shape[1], pts0.shape[2]

        colors = []
        for t in range(T):
            img_path = None
            if t < len(self.buf.views) and isinstance(self.buf.views[t], dict):
                img_path = self.buf.views[t].get("path", None)
            if img_path is None:
                colors.append(np.zeros((H, W, 3), np.uint8))
                continue
            im_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if im_bgr is None:
                colors.append(np.zeros((H, W, 3), np.uint8))
                continue
            im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
            if im_rgb.shape[:2] != (H, W):
                im_rgb = cv2.resize(im_rgb, (W, H), interpolation=cv2.INTER_AREA)
            colors.append(im_rgb.astype(np.uint8))
        return np.stack(colors, axis=0)

    # ---------------- Exports ----------------

    @staticmethod
    def save_points_as_glb(pts: np.ndarray, cols: Optional[np.ndarray], output_path: Union[str, Path]) -> None:
        rgba = None
        if cols is not None and cols.size:
            rgba = np.hstack([cols.astype(np.uint8), 255 * np.ones((cols.shape[0], 1), np.uint8)])
        pc = trimesh.points.PointCloud(vertices=pts, colors=rgba)
        scene = trimesh.Scene()
        scene.add_geometry(pc)
        scene.export(file_obj=str(output_path), file_type="glb")
        
    def filter_buffer_with_masks(
        self,
        masks_per_frame: list[np.ndarray],  # each HxW bool/uint8, one per frame
    ):
        """
        In-place filter of points/depth/conf in self.buf using masks_per_frame[t].
        Masks are resized to the stored prediction grid with nearest-neighbor.
        """
        T = self.buf.n_frames
        if T == 0:
            return
        for t in range(min(T, len(masks_per_frame))):
            m = masks_per_frame[t]
            if m is None:
                continue
            res = self.buf.ress[t]
            if ("pts3d_in_other_view" not in res) or ("conf" not in res) or ("depth" not in res) or ("depth_conf" not in res):
                continue

            p  = res["pts3d_in_other_view"]  # [B, H, W, 3]
            cp = res["conf"]                 # [B, H, W]
            d  = res["depth"]                # [B, H, W, 1]
            cd = res["depth_conf"]           # [B, H, W]

            # infer H,W and resize mask
            H, W = int(p.shape[1]), int(p.shape[2])
            mm = np.asarray(m).astype(np.uint8)
            if mm.shape[:2] != (H, W):
                mm = cv2.resize(mm, (W, H), interpolation=cv2.INTER_NEAREST)
            mm = (mm > 0)

            if torch.is_tensor(p):
                mm_t   = torch.from_numpy(mm).to(p.device, dtype=torch.bool)  # [H,W]
                mm_bhw = mm_t.unsqueeze(0)        # [1,H,W]
                mm_bhw1= mm_bhw.unsqueeze(-1)     # [1,H,W,1]
                inv1   = ~mm_bhw1
                p[inv1.expand_as(p)] = torch.nan
                cp *= mm_bhw.float()
                cd *= mm_bhw.float()
                d[inv1.expand_as(d)] = 0
            else:
                # numpy fallback
                inv = ~mm
                p[0][inv]  = np.nan
                cp[0][inv] = 0
                cd[0][inv] = 0
                d[0][inv]  = 0
                
    def align_sheets_and_adjust(
        self,
        *,
        conf_thresh: float = 0.60,         # percentile used when extracting per-frame clouds
        max_points_per_frame: int = 80_000, # cap per-frame points used for registration
        ref_strategy: str = "densest",     # {"first","densest"}
        icp_iters: int = 30,
        trim: float = 0.20,                # robust trimming: drop top 20% residual matches each iter
        tol: float = 1e-5,
        extrinsic_mode: str = "c2w",       # {"c2w","w2c"} for how to adjust extrinsics
        update_buffer_inplace: bool = True,
        rebuild_live: bool = True,
    ) -> Dict[str, Any]:
        """
        Aligns per-frame point 'sheets' via Sim(3) ICP, updates stored points/depth in self.buf,
        and returns adjusted camera extrinsics along with a preview point sample.

        Returns:
            {
            "sim3_per_frame": np.ndarray [T,4,4],
            "extrinsics_adj": np.ndarray [T,4,4],
            "intrinsics":     np.ndarray [T,3,3],
            "points_preview": (pts, cols),          # sampled after adjustment
            "stats":          {"used_points_per_frame": List[int], "iters_per_frame": List[int]}
            }
        """
        # ---------------- helpers ----------------
        import numpy as _np
        import numpy.linalg as _LA
        try:
            from scipy.spatial import cKDTree as _KDTree
            _HAS_KD = True
        except Exception:
            _KDTree = None
            _HAS_KD = False

        def _random_downsample(P: _np.ndarray, M: int) -> _np.ndarray:
            if P.shape[0] <= M:
                return P
            idx = _np.random.default_rng(42).choice(P.shape[0], size=M, replace=False)
            return P[idx]

        def _flatten_from_frame_result(res: Dict[str, Any], img_path: str | None, thr: float) -> _np.ndarray:
            # Reuse your existing extractor for consistent filtering
            pts_np, _cols_np = extract_pts_cols_from_frame_result(res, img_path, conf_thresh=thr)
            # Drop NaNs
            if pts_np.size == 0:
                return pts_np
            ok = _np.isfinite(pts_np).all(axis=1)
            pts_np = pts_np[ok]
            if pts_np.size == 0:
                return pts_np
            return _random_downsample(pts_np, max_points_per_frame)

        def _umeyama(X: _np.ndarray, Y: _np.ndarray, with_scale: bool = True):
            """
            Returns s, R, t s.t.  Y ≈ s*R @ X + t
            """
            assert X.shape == Y.shape and X.ndim == 2 and X.shape[1] == 3
            mu_x = X.mean(axis=0)
            mu_y = Y.mean(axis=0)
            Xc = X - mu_x
            Yc = Y - mu_y
            C = (Yc.T @ Xc) / X.shape[0]
            U, S, Vt = _LA.svd(C)
            R = U @ Vt
            if _LA.det(R) < 0:
                Vt[-1] *= -1
                R = U @ Vt
            if with_scale:
                var_x = (Xc**2).sum() / X.shape[0]
                s = (S.sum() / var_x) if var_x > 0 else 1.0
            else:
                s = 1.0
            t = mu_y - s * (R @ mu_x)
            return float(s), R, t

        def _icp_sim3(src: _np.ndarray, tgt: _np.ndarray, iters: int, trim: float, tol: float):
            """
            Basic Sim(3)-ICP using NN matches (KDTree if available), robust trimmed Umeyama.
            Returns s, R, t, n_iter.
            """
            if src.shape[0] == 0 or tgt.shape[0] == 0:
                return 1.0, _np.eye(3), _np.zeros(3), 0
            X = src.copy()
            Y = tgt
            if _HAS_KD:
                kdt = _KDTree(Y)
            last_err = _np.inf
            for it in range(iters):
                if _HAS_KD:
                    dists, idx = kdt.query(X, k=1)
                    pairs_Y = Y[idx]
                else:
                    # fallback: brute force on a further downsampled target
                    Yd = _random_downsample(Y, min(20000, Y.shape[0]))
                    # compute NN by chunk to avoid huge memory; simple & coarse
                    # split X into chunks of ~50k
                    chunk = 50000
                    nnY = []
                    for i0 in range(0, X.shape[0], chunk):
                        Xi = X[i0:i0+chunk]
                        # Euclidean distance matrix (Xi x Yd)
                        D = ((Xi[:, None, :] - Yd[None, :, :])**2).sum(axis=2)
                        nnY.append(Yd[D.argmin(axis=1)])
                    pairs_Y = _np.vstack(nnY)

                # robust trimming
                res = _np.linalg.norm(X - pairs_Y, axis=1)
                keep = int(max(10, (1.0 - trim) * res.shape[0]))
                order = _np.argpartition(res, keep-1)[:keep]
                Xk, Yk = X[order], pairs_Y[order]

                s, R, t = _umeyama(Xk, Yk, with_scale=True)
                # apply update to X
                X = (s * (R @ X.T)).T + t

                err = _np.median(_np.linalg.norm(X - pairs_Y, axis=1))
                if abs(last_err - err) < tol:
                    return s, R, t, it+1
                last_err = err
            return s, R, t, iters

        def _sim3_to_44(s: float, R: _np.ndarray, t: _np.ndarray) -> _np.ndarray:
            T = _np.eye(4, dtype=_np.float64)
            T[:3, :3] = s * R
            T[:3, 3]  = t
            return T

        def _apply_sim3_to_buffer_frame_inplace(res: Dict[str, Any], s: float, R: np.ndarray, t: np.ndarray):
            """Apply Sim(3) to pts3d grid (3- or 4-channels) and scale depth accordingly."""
            p  = res.get("pts3d_in_other_view", None)  # [B,H,W,3] or [B,H,W,4]
            d  = res.get("depth", None)                # [B,H,W,1] (optional)
            if p is None:
                return

            is_torch = torch.is_tensor(p)
            P = p.detach().cpu().numpy() if is_torch else np.asarray(p)

            if P.ndim != 4 or P.shape[-1] not in (3, 4):
                # Unexpected layout; bail out safely
                return

            B, H, W, C = P.shape
            Pw = P.reshape(B, H * W, C)

            # Use only XYZ for transform; keep W (if present) as-is
            X = Pw[0, :, :3]
            ok = np.isfinite(X).all(axis=1)
            X_ok = X[ok]
            if X_ok.shape[0]:
                X_new = (s * (R @ X_ok.T)).T + t
                Pw[0, ok, :3] = X_new

            P_new = Pw.reshape(B, H, W, C)
            if is_torch:
                p.copy_(torch.from_numpy(P_new).to(p.device, dtype=p.dtype))
            else:
                p[...] = P_new

            # Scale depth if present
            if d is not None:
                if torch.is_tensor(d):
                    d.mul_(float(s))
                else:
                    d *= float(s)
        # ---------------- gather data ----------------
        T = self.buf.n_frames
        if T == 0:
            raise RuntimeError("Buffer is empty; nothing to align.")

        # Extract per-frame sparse clouds for registration
        imgs = []
        if len(self.buf.views):
            for v in self.buf.views:
                imgs.append(v.get("path", None) if isinstance(v, dict) else None)
        else:
            imgs = [None] * T

        used_counts = []
        clouds: list[_np.ndarray] = []
        for t in range(T):
            res = self.buf.ress[t]
            P = _flatten_from_frame_result(res, imgs[t] if t < len(imgs) else None, conf_thresh)
            clouds.append(P)
            used_counts.append(int(P.shape[0]))

        # Choose reference
        if ref_strategy == "first":
            ref_idx = 0
        else:
            # densest
            ref_idx = int(_np.argmax([c.shape[0] for c in clouds]))
        ref = clouds[ref_idx]
        if ref.shape[0] == 0:
            # fall back to first non-empty
            for i, c in enumerate(clouds):
                if c.shape[0]:
                    ref_idx, ref = i, c
                    break
        if ref.shape[0] == 0:
            raise RuntimeError("No valid points in any frame for alignment.")

        # Decode original camera parameters once
        ex_orig, intr = self.get_camera_extri_intri(as_numpy=True)  # [T,4,4], [T,3,3]

        # Register each frame -> reference (incrementally merge to strengthen reference)
        sim3_per_frame = _np.tile(_np.eye(4), (T, 1, 1)).astype(_np.float64)
        iters_per_frame = [0] * T

        ref_agg = ref.copy()
        ref_cap = 400_000  # avoid unbounded growth
        for t in range(T):
            if t == ref_idx or clouds[t].shape[0] == 0:
                sim3_per_frame[t] = _np.eye(4, dtype=_np.float64)
                continue
            s, R, tt, n_it = _icp_sim3(clouds[t], ref_agg, icp_iters, trim, tol)
            iters_per_frame[t] = n_it
            G = _sim3_to_44(s, R, tt)
            sim3_per_frame[t] = G

            # Apply to buffer now if requested (points+depth)
            if update_buffer_inplace:
                _apply_sim3_to_buffer_frame_inplace(self.buf.ress[t], s, R, tt)

            # Merge transformed cloud into reference (limit size)
            Xnew = (s * (R @ clouds[t].T)).T + tt
            if ref_agg.shape[0] < ref_cap:
                # simple uniform append (optional: voxel grid / random keep to cap)
                need = max(0, ref_cap - ref_agg.shape[0])
                if Xnew.shape[0] > need:
                    Xnew = _random_downsample(Xnew, need)
                ref_agg = _np.vstack([ref_agg, Xnew]) if Xnew.shape[0] else ref_agg

        # Ensure the reference frame is identity and (optionally) already in-place updated
        if update_buffer_inplace and ref_idx < T:
            # make sure ref frame remains as-is (identity)
            pass

        # Adjust extrinsics according to found Sim(3)
        ex_shape = ex_orig.shape[1:]
        ex_adj = np.empty_like(ex_orig)

        def _to_44(E: np.ndarray) -> tuple[np.ndarray, bool]:
            # Returns 4×4 matrix and a flag indicating whether the input was 3×4
            if E.shape == (4, 4):
                return E, False
            if E.shape == (3, 4):
                E44 = np.eye(4, dtype=E.dtype)
                E44[:3, :4] = E
                return E44, True
            raise ValueError(f"Unexpected extrinsic shape: {E.shape}")

        for t in range(self.buf.n_frames):
            G = sim3_per_frame[t]  # 4×4 Sim(3)
            E44, was_34 = _to_44(ex_orig[t])

            if extrinsic_mode.lower() == "c2w":
                # world' = G * world  =>  E' = G * E
                Eadj44 = G @ E44
            else:
                # w2c: x_cam = E * x_world ; x_world = G^{-1} * x_world' => E' = E * G^{-1}
                Eadj44 = E44 @ np.linalg.inv(G)

            # Write back in the same shape as original
            if was_34:
                ex_adj[t, :, :] = Eadj44[:3, :4]
            else:
                ex_adj[t, :, :] = Eadj44

        # Rebuild live sampler for your viewer
        pts_cols = None
        if rebuild_live:
            pts_cols = self.rebuild_live_from_buffer(conf_thresh=None)

        out = {
            "sim3_per_frame": sim3_per_frame,
            "extrinsics_adj": ex_adj,
            "intrinsics": intr,
            "stats": {
                "used_points_per_frame": used_counts,
                "iters_per_frame": iters_per_frame,
                "ref_idx": ref_idx,
            }
        }
        if pts_cols is not None:
            out["points_preview"] = pts_cols
        return out

    def get_camera_extri_intri_frame(
                self, frame_idx: int, as_numpy: bool = True
            ) -> Tuple[np.ndarray, np.ndarray]:
                """Decode extrinsics (3x4) and intrinsics (3x3) for a single frame."""
                ex, intr = self.get_camera_extri_intri(as_numpy=as_numpy)
                if frame_idx < 0 or frame_idx >= ex.shape[0]:
                    raise IndexError(f"frame_idx {frame_idx} out of bounds [0,{ex.shape[0]-1}]")
                return ex[frame_idx], intr[frame_idx]