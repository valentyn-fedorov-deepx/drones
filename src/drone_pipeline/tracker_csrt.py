from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from src.drone_pipeline.interfaces import BaseTracker, Detection, TrackedObjectState


def _create_csrt_tracker() -> Any:
    """Create a CSRT tracker instance, handling OpenCV legacy namespaces."""
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()  # type: ignore[attr-defined]
    raise RuntimeError("CSRT tracker is not available in this OpenCV build")


@dataclass
class _CSRTTrack:
    id: int
    tracker: Optional[Any]
    bbox_xyxy: np.ndarray  # (4,) float
    label: str
    score: float
    last_timestamp: float
    misses: int = 0
    dormant: bool = False  # if True, track is hidden but may be re-activated for a short time
    hits: int = 0  # кількість разів, коли трек оновлювався по детекції
    static_frames: int = 0  # скільки кадрів поспіль bbox майже не рухається (ризик "гілки")


class CSRTTracker(BaseTracker):
    """OpenCV CSRT-based tracker.

    Логіка під твою задачу:
    - Використовує YOLO-детекції для ініціалізації треків.
    - CSRT оновлює bbox на кожному кадрі без додаткового детекту.
    - Якщо трек тимчасово втрачається, він переходить у "сплячий" стан і може
      бути ре-активований тією ж ID, якщо дрон повернувся протягом короткого часу.
    """

    def __init__(
        self,
        max_misses: int = 10,
        reid_time_window_s: float = 15.0,
        match_iou_threshold: float = 0.3,
        match_center_dist: float = 60.0,
        motion_threshold: float = 3.0,
        max_static_frames: int = 8,
    ) -> None:
        """Create CSRT-based tracker with short-term re-identification.

        Args:
            max_misses: consecutive CSRT update failures before a track becomes
                dormant (hidden from output, but still eligible for re-use).
            reid_time_window_s: maximum time gap (in seconds) during which a dormant
                track can be re-activated by a new detection (same drone re-entering).
            match_iou_threshold: IoU threshold for associating detections with
                existing CSRT tracks.
            match_center_dist: максимальна дистанція між центрами bbox при
                асоціації (потрібна для випадків з малим IoU, як у дронів).
            motion_threshold: поріг середнього руху (0-255) усередині bbox,
                нижче якого вважаємо, що там майже нічого не рухається.
            max_static_frames: скільки кадрів поспіль можна тримати статичний
                bbox, перш ніж вважати, що це "гілка" і вимкнути трек.
        """
        self.max_misses = max_misses
        self.reid_time_window_s = reid_time_window_s
        self.match_iou_threshold = match_iou_threshold
        self.match_center_dist = match_center_dist
        self.motion_threshold = motion_threshold
        self.max_static_frames = max_static_frames
        self._tracks: Dict[int, _CSRTTrack] = {}
        self._next_id: int = 0
        # Попередній кадр у grayscale для оцінки руху.
        self._prev_gray: Optional[np.ndarray] = None

    def _xyxy_to_xywh(self, box: np.ndarray) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = box.astype(float)
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        return int(x1), int(y1), int(w), int(h)

    def _init_from_detection(self, frame: np.ndarray, det: Detection) -> Optional[_CSRTTrack]:
        """Create a new CSRT track from a single detection."""
        tracker = _create_csrt_tracker()
        x1, y1, x2, y2 = det.bbox
        x, y, w, h = self._xyxy_to_xywh(np.array([x1, y1, x2, y2], dtype=float))
        ok = tracker.init(frame, (x, y, w, h))
        if not ok:
            return None
        tid = self._next_id
        self._next_id += 1
        track = _CSRTTrack(
            id=tid,
            tracker=tracker,
            bbox_xyxy=np.array([x1, y1, x2, y2], dtype=float),
            label=det.label,
            score=det.score,
            last_timestamp=det.timestamp,
            misses=0,
            dormant=False,
        )
        self._tracks[tid] = track
        return track

    def _update_existing(
        self,
        frame: np.ndarray,
        gray: np.ndarray,
        prev_gray: Optional[np.ndarray],
        timestamp: float,
    ) -> None:
        """Update all existing ACTIVE CSRT trackers on the current frame.

        Додатково рахуємо "motion" усередині bbox між prev_gray та gray, щоб
        виявляти випадки, коли трек залипає на статичній гілці.
        """
        H, W = frame.shape[:2]

        for tid, tr in self._tracks.items():
            if tr.dormant or tr.tracker is None:
                continue
            ok, box = tr.tracker.update(frame)
            if not ok:
                tr.misses += 1
                if tr.misses > self.max_misses:
                    # move to dormant state: keep last bbox/time, drop tracker instance
                    tr.dormant = True
                    tr.tracker = None
                continue

            x, y, w, h = box
            x1, y1, x2, y2 = x, y, x + w, y + h
            # Clamp to image bounds
            x1 = max(0.0, min(float(W - 1), float(x1)))
            y1 = max(0.0, min(float(H - 1), float(y1)))
            x2 = max(0.0, min(float(W), float(x2)))
            y2 = max(0.0, min(float(H), float(y2)))

            tr.bbox_xyxy = np.array([x1, y1, x2, y2], dtype=float)
            # ВАЖЛИВО: timestamp тут оновлюємо як "момент останнього frame-оновлення",
            # але логіку re-ID прив'язуємо до моменту останньої детекції, тому hits
            # оновлюється тільки при асоціації з YOLO.
            tr.last_timestamp = timestamp
            tr.misses = 0

            # Оцінка руху всередині bbox для виявлення "залипання" на гілці.
            if prev_gray is not None:
                x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
                x1i = max(0, min(W - 1, x1i))
                y1i = max(0, min(H - 1, y1i))
                x2i = max(x1i + 1, min(W, x2i))
                y2i = max(y1i + 1, min(H, y2i))
                if x2i > x1i and y2i > y1i:
                    g_curr = gray[y1i:y2i, x1i:x2i]
                    g_prev = prev_gray[y1i:y2i, x1i:x2i]
                    if g_curr.shape == g_prev.shape and g_curr.size > 0:
                        motion = float(np.mean(cv2.absdiff(g_curr, g_prev)))
                        if motion < self.motion_threshold:
                            tr.static_frames += 1
                        else:
                            tr.static_frames = 0
                        if tr.static_frames > self.max_static_frames:
                            # Вважаємо, що це static background (швидше за все гілка).
                            tr.dormant = True
                            tr.tracker = None
                            continue

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

    @staticmethod
    def _center_distance(a: np.ndarray, b: np.ndarray) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        acx, acy = (ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0
        bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
        return float(np.hypot(acx - bcx, acy - bcy))

    def _reset_track_from_detection(self, tr: _CSRTTrack, frame: np.ndarray, det: Detection) -> bool:
        """Hard-correct CSRT track to align with a fresh YOLO detection."""
        tracker = _create_csrt_tracker()
        x1, y1, x2, y2 = det.bbox
        x, y, w, h = self._xyxy_to_xywh(np.array([x1, y1, x2, y2], dtype=float))
        ok = tracker.init(frame, (x, y, w, h))
        if not ok:
            return False
        tr.tracker = tracker
        tr.bbox_xyxy = np.array([x1, y1, x2, y2], dtype=float)
        tr.label = det.label
        tr.score = det.score
        tr.last_timestamp = det.timestamp
        tr.misses = 0
        tr.dormant = False
        tr.hits += 1
        return True

    def update(
        self,
        frame: np.ndarray,
        timestamp: float,
        detections: Optional[List[Detection]] = None,
    ) -> List[TrackedObjectState]:  # type: ignore[override]
        detections = detections or []

        # 1) Update existing active CSRT trackers based on motion only.
        # Створюємо grayscale-версію кадру для motion-оцінки.
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        self._update_existing(frame, gray, self._prev_gray, timestamp)

        used_dets = set()

        # 2) Detection-driven correction of ACTIVE tracks
        active_ids = [tid for tid, tr in self._tracks.items() if not tr.dormant and tr.tracker is not None]
        if active_ids and detections:
            active_boxes = np.stack([self._tracks[tid].bbox_xyxy for tid in active_ids], axis=0)
            det_boxes = np.stack([np.array(det.bbox, dtype=float) for det in detections], axis=0)

            for d_idx, det in enumerate(detections):
                db = det_boxes[d_idx]
                best_tid: Optional[int] = None
                best_score = 0.0
                for i, tid in enumerate(active_ids):
                    tb = active_boxes[i]
                    iou = self._bbox_iou(tb, db)
                    dist = self._center_distance(tb, db)
                    if not (iou >= self.match_iou_threshold or dist <= self.match_center_dist):
                        continue
                    score = iou + max(0.0, (self.match_center_dist - dist) / max(self.match_center_dist, 1e-6))
                    if score > best_score:
                        best_score = score
                        best_tid = tid

                if best_tid is not None:
                    tr = self._tracks[best_tid]
                    if self._reset_track_from_detection(tr, frame, det):
                        used_dets.add(d_idx)

        # 3) Short-term re-identification для dormant-треків + створення нових.
        #    На цьому етапі частина детекцій вже могла бути використана для
        #    корекції активних треків (used_dets). Тепер пробуємо прив'язати
        #    решту детекцій до недавніх dormant-треків, щоб відновити той самий ID
        #    після короткого зникнення дрона.
        recent_dormant_ids: List[int] = [
            tid
            for tid, tr in self._tracks.items()
            if tr.dormant and (timestamp - tr.last_timestamp) <= self.reid_time_window_s
        ]
        dormant_boxes = (
            np.stack([self._tracks[tid].bbox_xyxy for tid in recent_dormant_ids], axis=0)
            if recent_dormant_ids
            else np.zeros((0, 4), dtype=float)
        )

        used_dormant: set[int] = set()

        # Спочатку пробуємо відновити dormant-треки для ще не використаних детекцій.
        if recent_dormant_ids and detections:
            det_boxes = np.stack([np.array(det.bbox, dtype=float) for det in detections], axis=0)
            for d_idx, det in enumerate(detections):
                if d_idx in used_dets:
                    continue
                db = det_boxes[d_idx]
                best_tid: Optional[int] = None
                best_score = 0.0
                for i, tid in enumerate(recent_dormant_ids):
                    if tid in used_dormant:
                        continue
                    tb = dormant_boxes[i]
                    iou = self._bbox_iou(tb, db)
                    dist = self._center_distance(tb, db)
                    if not (iou >= self.match_iou_threshold or dist <= self.match_center_dist):
                        continue
                    score = iou + max(0.0, (self.match_center_dist - dist) / max(self.match_center_dist, 1e-6))
                    if score > best_score:
                        best_score = score
                        best_tid = tid

                if best_tid is not None:
                    tr = self._tracks[best_tid]
                    if self._reset_track_from_detection(tr, frame, det):
                        used_dets.add(d_idx)
                        used_dormant.add(best_tid)

        # 4) Створити нові треки для детекцій, які все ще не використані
        for d_idx, det in enumerate(detections):
            if d_idx in used_dets:
                continue
            created = self._init_from_detection(frame, det)
            if created is not None:
                used_dets.add(d_idx)

        # 5) Remove very old dormant tracks that are beyond the re-identification window.
        to_delete = [
            tid
            for tid, tr in self._tracks.items()
            if tr.dormant and (timestamp - tr.last_timestamp) > self.reid_time_window_s
        ]
        for tid in to_delete:
            self._tracks.pop(tid, None)

        # 6) Build output states: only non-dormant (currently active) tracks are visible.
        states: List[TrackedObjectState] = []
        for tid, tr in self._tracks.items():
            if tr.dormant:
                continue
            x1, y1, x2, y2 = tr.bbox_xyxy.astype(int).tolist()
            states.append(
                TrackedObjectState(
                    track_id=tid,
                    bbox=(x1, y1, x2, y2),
                    score=float(tr.score),
                    label=str(tr.label),
                    timestamp=float(tr.last_timestamp),
                )
            )

        # Оновити prev_gray для наступного кадру
        self._prev_gray = gray

        return states
