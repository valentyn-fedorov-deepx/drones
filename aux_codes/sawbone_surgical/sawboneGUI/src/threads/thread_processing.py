import sys

import datetime
import logging
import os
import shutil
import secrets


from PyQt6.QtCore import QThread, pyqtSignal, pyqtSlot, QObject
import cv2
import numpy as np
import time
from collections import deque 

import torch

from pathlib import Path
import shutil
from src.reconstraction_pipe.visualization import filter_by_confidence

class ProcessingWorker(QObject):
    
    # For visual overlays when they ready
    vis_frame_ready = pyqtSignal(object)
    # For reconstraction when it is ready
    reconstruction_ready = pyqtSignal(object)

    def __init__(self,) -> None:
        super().__init__()
        
        self._frame = None
        self._frames_batch = deque(maxlen=5)

        self.mask_list = []
        self.camera_params = []
        self.vggt_frames = []
        self.vggt_shots = None

        self.proc = None
        self.segm = None
        self.reconstr_proc = None

        self._busy: bool = False
        self._points = []
        self._have_init: bool = False 
        self._have_track_init: bool = False        
        
        
    def initialize(self):
    
        cur_dir = Path(__file__).parent.resolve()
        root_dir = cur_dir.parent
        data_dir = root_dir / "data"
        reconstr_files_path = data_dir / "reconstr_files_tmp_stream1"
        # remove files inside temp
        temp_dir = Path("/mnt/c/Users/Demo5/Desktop/volodymyr/sawboneGUI/src/temp")
        shutil.rmtree(temp_dir)
        temp_dir.mkdir(exist_ok=True, parents=True)
        
        reconstr_stream_path = (root_dir / "thirdparty/StreamVGGT").resolve()
        src_stream = (root_dir / "thirdparty/StreamVGGT/src").resolve()
        dx_vyzai_path = cur_dir.parent.parent
        segment_path = (root_dir / "thirdparty/sam2").resolve()
        code_path = (root_dir / "reconstraction_pipe").resolve()
        reconstr_path = (root_dir / "thirdparty/vggt_official")

        for p in [reconstr_stream_path, src_stream, dx_vyzai_path, segment_path, code_path, reconstr_path]:
            sys.path.append(str(p))
        
        from streamvggt_processor import StreamVGGTProcessor
        from sam2_model import SegmentAnything2, union_binary_masks
        from vggt_processor import VGGTProcessor
        
        ckpt_path = reconstr_stream_path / "ckpt/checkpoints.pth"
        self.proc = StreamVGGTProcessor(
            device="cuda",
            model_path=str(ckpt_path),
            max_keep=64, 
            live_capacity=60_000, 
        )
        self.segm = SegmentAnything2()
        self.process_mask = union_binary_masks
        self.reconstr_proc = VGGTProcessor()

        self._have_init = True
        print('FINISH')

    @pyqtSlot(object)
    def on_frame(self, frame: np.ndarray) -> None:
        """Receive frames from camera thread (queued into this worker thread)"""
        if frame is None or frame.size == 0:
            return

        # Keep a short batch for the "every 30th frame" logic
        self._frames_batch.append(frame)
        self._frame = frame
        # If we're busy or not yet initialized or user hasn't placed points — do nothing
        if self._busy or not self._have_init or not self._have_track_init:
            return
        print(self._busy, self._have_init, self._have_track_init)
        # When we have a full batch, run the heavy step
        if len(self._frames_batch) == self._frames_batch.maxlen and self._have_track_init:
            self._busy = True
            try:
                self._function_step()
            finally:
                self._frames_batch.clear()
                self._busy = False

    def init_sam_track(self, points):
        clicks = []
        
        for x,y, (w,h) in points:
            clicks.append({'x':x, 'y':y})
        
        obj_ids, obj_masks = self.segm.init_camera(click_coords=clicks, frame=self._frame, return_masks=True)

        vis_frame = self._draw_masks_on_frame(self._frame, obj_masks, alpha_fill=0.25, contour_thickness=2)

        if vis_frame is not None:
                self.vis_frame_ready.emit(vis_frame)
                self._have_track_init = True
    
    def _function_step(self):
        out_dir = Path("/mnt/c/Users/Demo5/Desktop/volodymyr/sawboneGUI/src/temp")

        rand_name = f"{secrets.randbelow(10**10):010d}.jpg"
        out_path = out_dir / rand_name

        self.proc.reset_state()
        self.proc.clear_buffer()

        obj_ids, obj_masks = self.segm.propagate_binary(self._frame)
        union_mask = self.process_mask(obj_masks)

        self.mask_list.append(union_mask)
        self.proc.clear_buffer()

        save_path = "test.jpg"
        cv2.imwrite(save_path, self._frame)
        frame = self.proc.preprocess_image(save_path, valid_mask=union_mask)
        outs = self.proc.step(
            frame,
            store_device=torch.device("cpu"),
            keep_images=False,
            conf_thresh=0.6  # live-sampler threshold (percentile)
        )

        new_frame_extr, new_frame_intr = self.proc.get_camera_extri_intri_frame(0, as_numpy=True)
        self.camera_params.append((new_frame_extr, new_frame_intr))
        self.vggt_frames.append(self._frame)

        cv2.imwrite(str(out_path), self._frame)

        pts_show, cols_show = self.proc.get_live_sample()

        self.reconstruction_ready.emit((pts_show, cols_show))
        self._frames_batch.clear()
    
    def get_final_reconstraction(self, image_paths, conf_thresh=0.5):
        preds = self.reconstr_proc.infer(
            imgs=image_paths
        )

        images = []
        for p in image_paths:        
            im = cv2.imread(str(p), cv2.IMREAD_COLOR)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            images.append(im)
        imgs = np.stack(images, axis=0)
        pts = self.reconstr_proc.get_points(
            preds=preds, 
            as_numpy=True
        )
        pts_flat = pts.reshape(-1, 3)  

        cols_flat = self.reconstr_proc.get_points_colors(
            points=pts,
            imgs=imgs,
            flatten=True,
        )
        confs_flat = self.reconstr_proc.get_points_conf(
            preds=preds,
            as_numpy=True,
            flatten=True
        )

        pts_flat, cols_flat = filter_by_confidence(pts_flat, confs_flat, cols_flat, thresh=conf_thresh)

        return (pts_flat, cols_flat)

    def _draw_masks_on_frame(self, frame, masks, alpha_fill=0.0, contour_thickness=2):
        if frame is None:
            return None

        out = frame.copy()

        if hasattr(masks, "detach"): 
            masks = masks.detach().cpu().numpy()
        masks = np.asarray(masks)

        if masks.ndim == 2:
            masks = masks[None, ...]         
        elif masks.ndim == 3 and masks.shape[0] < masks.shape[1]:
            pass
        else:
            if masks.ndim == 3 and masks.shape[-1] < masks.shape[0]:
                masks = np.transpose(masks, (2, 0, 1))

        H, W = out.shape[:2]
        rng = np.random.default_rng(42)
        colors = rng.integers(0, 255, size=(masks.shape[0], 3), dtype=np.uint8)

        for i, m in enumerate(masks):
            m_u8 = (m > 0.5).astype(np.uint8) * 255
            if m_u8.max() == 0:
                continue

            contours, _ = cv2.findContours(m_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                continue

            color = tuple(int(c) for c in colors[i])  # BGR
            cv2.drawContours(out, contours, -1, color, thickness=contour_thickness, lineType=cv2.LINE_AA)

        return out
    
    def _normalize(self, v, eps=1e-9):
        v = np.asarray(v, dtype=np.float64)
        n = np.linalg.norm(v, axis=-1, keepdims=True)
        n = np.maximum(n, eps)
        return v / n

    def _extr_to_center_and_z(self, extr, extr_mode="w2c", z_sign=+1):
        """
        Returns (C_world (3,), z_world (3,)) from extrinsic.
        extr can be (3,4) or (4,4). 
        extr_mode: "w2c" = world->cam [R|t], "c2w" = cam->world [R|t]
        z_sign: +1 if +Z is forward in camera coords; set -1 if your forward is -Z.
        """
        extr = np.asarray(extr)
        if extr.shape == (3,4):
            R = extr[:, :3]
            t = extr[:, 3]
        elif extr.shape == (4,4):
            R = extr[:3, :3]
            t = extr[:3, 3]
        else:
            raise ValueError(f"Unexpected extr shape: {extr.shape}")

        ez = np.array([0.0, 0.0, 1.0], dtype=np.float64) * float(z_sign)

        if extr_mode.lower() == "w2c":
            # x_cam = R x_world + t
            # Camera center C satisfies R C + t = 0 => C = -R^T t
            C = -R.T @ t
            # Camera Z axis expressed in world coordinates: columns of R^T are camera axes in world.
            z_world = R.T @ ez
        else:
            # x_world = R x_cam + t
            C = t.copy()
            z_world = R @ ez

        return C.astype(np.float64), self._normalize(z_world).reshape(3,)

    def pick_five_canonical_shots_from_extrs(
        self,
        camera_params, 
        extr_mode="w2c", 
        z_sign=+1,
        hemisphere_dot_thresh=0.0    # keep only views with dot(z, front_dir) >= this
    ):
        """
        camera_params: list of (extr, intr) for each frame in order.
        Returns: dict with indices for front/left/right/top/bottom and an ordered list.
        """
        N = len(camera_params)
        if N == 0:
            raise ValueError("camera_params is empty.")

        # 1) Collect z_world for every frame (only orientations, no need for points/buffers).
        Z = []
        for k in range(N):
            extr, _ = camera_params[k]
            _, z_world = self._extr_to_center_and_z(extr, extr_mode=extr_mode, z_sign=z_sign)
            Z.append(z_world)
        Z = np.stack(Z, axis=0)                             # [N,3]
        Z = self._normalize(Z)

        # 2) Define the "front" direction as the mean Z across all frames.
        front_dir = self._normalize(Z.mean(axis=0))
        # Ensure we keep only the same-side (no other side) views.
        keep_mask = (Z @ front_dir) >= float(hemisphere_dot_thresh)
        if not np.any(keep_mask):
            # fall back to keeping all (degenerate case)
            keep_mask = np.ones(N, dtype=bool)

        idx_keep = np.nonzero(keep_mask)[0]
        Zk = Z[keep_mask]

        # 3) FRONT: the frame whose Z is most aligned with front_dir.
        front_scores = (Zk @ front_dir)
        idx_front_local = int(np.argmax(front_scores))
        idx_front = int(idx_keep[idx_front_local])

        # 4) Build an orthonormal basis: a1 (left-right axis in the plane ⟂ front), a2 (top-bottom).
        # Project Z onto the plane orthogonal to front_dir.
        Z_proj = Zk - (Zk @ front_dir)[:, None] * front_dir[None, :]
        norms = np.linalg.norm(Z_proj, axis=1, keepdims=True)
        # Avoid zero vectors (if all are nearly identical to front)
        Z_proj[(norms[:,0] < 1e-6), :] = 0.0

        # PCA on the projected directions to get dominant sideways axis
        if Z_proj.shape[0] >= 2 and np.any(np.linalg.norm(Z_proj, axis=1) > 0):
            C = (Z_proj.T @ Z_proj) / max(1, Z_proj.shape[0])
            evals, evecs = np.linalg.eigh(C)
            a1 = evecs[:, np.argmax(evals)]   # principal direction in the plane
            # Ensure a1 is perpendicular to front_dir (numerical safety)
            a1 = self._normalize(a1 - (a1 @ front_dir) * front_dir)
        else:
            # Degenerate: pick any axis perpendicular to front_dir
            tmp = np.array([1.0, 0.0, 0.0])
            if abs(tmp @ front_dir) > 0.9:
                tmp = np.array([0.0, 1.0, 0.0])
            a1 = self._normalize(np.cross(front_dir, tmp))

        a2 = self._normalize(np.cross(front_dir, a1))  # orthonormal triplet: (a1, a2, front_dir)

        # 5) Scores for each direction (within kept hemisphere, and avoiding duplicates).
        used = set([idx_front])

        def _best_index_for_dir(axis, prefer_more_frontal=True):
            # axis: desired direction in world coords (unit)
            # Score primarily by alignment with 'axis',
            # with a tiny tie-breaker favoring more frontal views if requested.
            alignment = Zk @ axis
            if prefer_more_frontal:
                # small bonus for being more aligned with front_dir
                alignment = alignment + 1e-3 * (Zk @ front_dir)
            order_local = np.argsort(-alignment)  # descending
            for j in order_local:
                cand = int(idx_keep[j])
                if cand not in used:
                    used.add(cand)
                    return cand
            # fallback (shouldn't happen unless N<5): allow duplicates
            return int(idx_keep[int(order_local[0])])

        idx_right  = _best_index_for_dir(+a1)
        idx_left   = _best_index_for_dir(-a1)
        idx_top    = _best_index_for_dir(+a2)
        idx_bottom = _best_index_for_dir(-a2)

        out = {
            "front":  idx_front,
            "left":   idx_left,
            "right":  idx_right,
            "top":    idx_top,
            "bottom": idx_bottom,
            "order": [("front", idx_front),
                    ("left", idx_left),
                    ("right", idx_right),
                    ("top", idx_top),
                    ("bottom", idx_bottom)],
            "basis": {
                "front_dir": front_dir,  # for debugging/inspection
                "left_right_axis": a1,
                "top_bottom_axis": a2,
            }
        }
        return out

    @pyqtSlot()
    def stop_reconstruction(self):
        import glob
        self._have_track_init = False
        print('Processor', self._have_track_init)
        shots = self.pick_five_canonical_shots_from_extrs(
            self.camera_params, extr_mode="w2c", z_sign=+1, hemisphere_dot_thresh=0.0
            )
        self.vggt_shots = shots
        
        temp_dir = Path('/mnt/c/Users/Demo5/Desktop/volodymyr/sawboneGUI/src/temp')
        img_paths = glob.glob(str(temp_dir / "*.jpg"))
        if not img_paths:
            return

        pts_show, cols_show = self.get_final_reconstraction(img_paths)

        self.reconstruction_ready.emit((pts_show, cols_show))

        for p in temp_dir.iterdir():
            if p.is_file() or p.is_symlink():
                p.unlink(missing_ok=True)
            elif p.is_dir():
                shutil.rmtree(p, ignore_errors=True)

# if __name__ == "__main__":
#     test_c = ProcesingThread()
        
#     test_c.init_sam_track([1,1,(2,3)])