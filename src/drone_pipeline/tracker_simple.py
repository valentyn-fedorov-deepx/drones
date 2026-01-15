from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from src.drone_pipeline.interfaces import BaseTracker, Detection, TrackedObjectState


@dataclass
class _Track:
    id: int
    bbox: np.ndarray  # (4,)
    score: float
    label: str
    timestamp: float
    misses: int = 0
    dormant: bool = False  # if True, track is hidden but may be re-activated for a short time


class SimpleTracker(BaseTracker):
    """Very lightweight IoU-based tracker.

    This implementation is intentionally simple and has no external
    dependencies (like norfair). It keeps a set of active tracks and performs
    greedy IoU matching between tracks and new detections each frame.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_misses: int = 10,
        reid_time_window_s: float = 15.0,
    ) -> None:
        """Simple IoU-based tracker with short-term re-identification.

        Args:
            iou_threshold: IoU threshold for matching detections to existing tracks.
            max_misses: number of consecutive frames without a match before a track
                becomes dormant (hidden from output, but still eligible for re-use).
            reid_time_window_s: maximum time gap (in seconds) during which a dormant
                track can be re-activated by a new detection (same drone re-entering).
        """
        self.iou_threshold = iou_threshold
        self.max_misses = max_misses
        self.reid_time_window_s = reid_time_window_s
        self._tracks: Dict[int, _Track] = {}
        self._next_id: int = 0

    @staticmethod
    def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        if ix1 >= ix2 or iy1 >= iy2:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return float(inter / max(area_a + area_b - inter, 1e-6))

    def _associate(self, det_bboxes: np.ndarray) -> Dict[int, Optional[int]]:
        """Greedy IoU-based association for ACTIVE tracks only.

        Returns mapping track_id -> det_idx (or None).
        """
        # Consider only non-dormant tracks for standard frame-to-frame association.
        active_items = [(tid, tr) for tid, tr in self._tracks.items() if not tr.dormant]
        if len(active_items) == 0 or det_bboxes.size == 0:
            return {tid: None for tid, _ in active_items}

        track_ids = [tid for tid, _ in active_items]
        track_bboxes = np.stack([tr.bbox for _, tr in active_items], axis=0)  # (T,4)

        T = track_bboxes.shape[0]
        D = det_bboxes.shape[0]
        ious = np.zeros((T, D), dtype=float)
        for t in range(T):
            for d in range(D):
                ious[t, d] = self._bbox_iou(track_bboxes[t], det_bboxes[d])

        assigned_det = {tid: None for tid in track_ids}
        used_dets = set()

        # Greedy: repeatedly pick best IoU above threshold
        while True:
            t_idx, d_idx = divmod(int(ious.argmax()), D)
            best_iou = ious[t_idx, d_idx]
            if best_iou < self.iou_threshold:
                break
            tid = track_ids[t_idx]
            if assigned_det[tid] is None and d_idx not in used_dets:
                assigned_det[tid] = d_idx
                used_dets.add(d_idx)
            ious[t_idx, :] = -1.0
            ious[:, d_idx] = -1.0

        return assigned_det

    def update(
        self,
        frame: np.ndarray,
        timestamp: float,
        detections: Optional[List[Detection]] = None,
    ) -> List[TrackedObjectState]:  # type: ignore[override]
        detections = detections or []
        det_bboxes = np.array([det.bbox for det in detections], dtype=float) if detections else np.zeros((0, 4))

        # 1) Associate detections to existing tracks
        assoc = self._associate(det_bboxes)

        # 2) Update matched tracks (only active ones are present in assoc)
        used_dets = set()
        for tid, d_idx in assoc.items():
            track = self._tracks[tid]
            if d_idx is not None:
                det = detections[d_idx]
                track.bbox = np.array(det.bbox, dtype=float)
                track.score = det.score
                track.label = det.label
                track.timestamp = det.timestamp
                track.misses = 0
                track.dormant = False
                used_dets.add(d_idx)
            else:
                # No matching detection this frame
                track.misses += 1
                if track.misses > self.max_misses:
                    # Mark as dormant: hidden from output, but eligible for short-term reactivation
                    track.dormant = True

        # 3) Create new tracks for unmatched detections OR re-activate a recent dormant track
        # Count active tracks (non-dormant) to decide when we can safely reuse an ID.
        has_active_tracks = any(not tr.dormant for tr in self._tracks.values())

        # Pre-compute list of recent dormant tracks that are candidates for re-identification.
        recent_dormant: List[tuple[int, _Track]] = [
            (tid, tr)
            for tid, tr in self._tracks.items()
            if tr.dormant and (timestamp - tr.timestamp) <= self.reid_time_window_s
        ]

        # Only attempt ID reuse when there are no active tracks and exactly one recent dormant track.
        can_reactivate = (not has_active_tracks) and len(recent_dormant) == 1

        # List of det indices that are still free after association.
        unmatched_det_indices = [d_idx for d_idx in range(len(detections)) if d_idx not in used_dets]

        for d_idx in unmatched_det_indices:
            det = detections[d_idx]

            # Case A: no active tracks and exactly one recent dormant track -> reuse its ID.
            if can_reactivate:
                dormant_tid, dormant_tr = recent_dormant[0]
                dormant_tr.bbox = np.array(det.bbox, dtype=float)
                dormant_tr.score = det.score
                dormant_tr.label = det.label
                dormant_tr.timestamp = det.timestamp
                dormant_tr.misses = 0
                dormant_tr.dormant = False
                used_dets.add(d_idx)
                # After re-activating once, do not reuse the same dormant track again in this frame.
                can_reactivate = False
                continue

            # Case B: standard behaviour – start a brand new track.
            tid = self._next_id
            self._next_id += 1
            self._tracks[tid] = _Track(
                id=tid,
                bbox=np.array(det.bbox, dtype=float),
                score=det.score,
                label=det.label,
                timestamp=det.timestamp,
                misses=0,
                dormant=False,
            )

        # 4) Remove very old dormant tracks that are beyond the re-identification window
        to_delete = [
            tid
            for tid, tr in self._tracks.items()
            if tr.dormant and (timestamp - tr.timestamp) > self.reid_time_window_s
        ]
        for tid in to_delete:
            self._tracks.pop(tid, None)

        # 5) Build output states: only non-dormant (currently active) tracks are visible downstream
        states: List[TrackedObjectState] = []
        for tid, tr in self._tracks.items():
            if tr.dormant:
                continue
            x1, y1, x2, y2 = tr.bbox.astype(int).tolist()
            states.append(
                TrackedObjectState(
                    track_id=tid,
                    bbox=(x1, y1, x2, y2),
                    score=float(tr.score),
                    label=str(tr.label),
                    timestamp=float(tr.timestamp),
                )
            )

        return states
