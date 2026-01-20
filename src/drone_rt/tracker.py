from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from src.drone_pipeline.interfaces import Detection, TrackedObjectState

from .config import RTTrackerConfig


@dataclass
class _RTTrack:
    id: int
    bbox: np.ndarray  # (4,) float, xyxy in image coords
    score: float
    label: str
    last_timestamp: float
    age: int = 0          # how many frames the track has existed
    hits: int = 0         # how many times it has been associated with a detection
    misses: int = 0       # consecutive frames without association


class SimpleIOUTracker:
    """Lightweight IOU-based tracker for realtime drone tracking.

    Design goals:
      * No external dependencies beyond NumPy.
      * Simple track lifecycle: create on detection, keep while it is matched
        or until `max_age_frames` misses elapse.
      * Optionally require a minimum number of hits before a track becomes
        visible (min_hits).
    """

    def __init__(self, config: RTTrackerConfig) -> None:
        self._cfg = config
        self._tracks: Dict[int, _RTTrack] = {}
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

    def _associate(self, det_boxes: np.ndarray) -> Dict[int, int]:
        """Greedy IOU-based association.

        Returns mapping: track_id -> det_index.
        """

        if not self._tracks or det_boxes.size == 0:
            return {}

        track_ids = list(self._tracks.keys())
        track_boxes = np.stack([self._tracks[tid].bbox for tid in track_ids], axis=0)

        iou_matrix = np.zeros((len(track_ids), det_boxes.shape[0]), dtype=float)
        for i, tb in enumerate(track_boxes):
            for j, db in enumerate(det_boxes):
                iou_matrix[i, j] = self._bbox_iou(tb, db)

        assigned_tracks: Dict[int, int] = {}
        used_dets: set[int] = set()

        # Greedy: always pick the highest IOU pair above threshold.
        while True:
            i, j = divmod(int(np.argmax(iou_matrix)), iou_matrix.shape[1])
            best_iou = iou_matrix[i, j]
            if best_iou < self._cfg.iou_match_threshold:
                break
            tid = track_ids[i]
            if tid in assigned_tracks or j in used_dets:
                iou_matrix[i, j] = -1.0
                continue
            assigned_tracks[tid] = j
            used_dets.add(j)
            iou_matrix[i, :] = -1.0
            iou_matrix[:, j] = -1.0

        return assigned_tracks

    def update(self, frame: np.ndarray, timestamp: float, detections: Optional[List[Detection]] = None) -> List[TrackedObjectState]:
        """Update tracker with optional detections and return active tracks.

        The frame itself is currently unused (kept for future extensions and
        API compatibility).
        """

        detections = detections or []

        # Age and decay existing tracks
        for tr in self._tracks.values():
            tr.age += 1
            tr.misses += 1

        # Build detection array
        det_boxes_list: List[np.ndarray] = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            det_boxes_list.append(np.array([x1, y1, x2, y2], dtype=float))
        det_boxes = np.stack(det_boxes_list, axis=0) if det_boxes_list else np.zeros((0, 4), dtype=float)

        # Associate detections to existing tracks
        assigned = self._associate(det_boxes)
        used_det_indices: set[int] = set()

        for tid, d_idx in assigned.items():
            used_det_indices.add(d_idx)
            det = detections[d_idx]
            tr = self._tracks[tid]
            tr.bbox = det_boxes[d_idx]
            tr.score = float(det.score)
            tr.label = det.label
            tr.last_timestamp = timestamp
            tr.misses = 0
            tr.hits += 1

        # Create new tracks for unmatched detections
        for d_idx, det in enumerate(detections):
            if d_idx in used_det_indices:
                continue
            if len(self._tracks) >= self._cfg.max_tracks:
                break
            box = det_boxes[d_idx] if det_boxes.size else np.array(det.bbox, dtype=float)
            tid = self._next_id
            self._next_id += 1
            self._tracks[tid] = _RTTrack(
                id=tid,
                bbox=box,
                score=float(det.score),
                label=det.label,
                last_timestamp=timestamp,
                age=1,
                hits=1,
                misses=0,
            )

        # Remove stale tracks
        to_delete = [
            tid
            for tid, tr in self._tracks.items()
            if tr.misses > self._cfg.max_age_frames
        ]
        for tid in to_delete:
            self._tracks.pop(tid, None)

        # Build visible states (only sufficiently confirmed tracks)
        states: List[TrackedObjectState] = []
        for tid, tr in self._tracks.items():
            if tr.hits < self._cfg.min_hits:
                continue
            x1, y1, x2, y2 = tr.bbox.astype(int).tolist()
            states.append(
                TrackedObjectState(
                    track_id=tid,
                    bbox=(x1, y1, x2, y2),
                    score=tr.score,
                    label=tr.label,
                    timestamp=tr.last_timestamp,
                )
            )

        return states
