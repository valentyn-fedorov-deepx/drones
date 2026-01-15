from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock
from typing import List, Optional

import numpy as np
from loguru import logger

from src.drone_pipeline.interfaces import (
    BaseClassifier,
    BaseDetector,
    BaseSegmenter,
    BaseTracker,
    ClassificationResult,
    Detection,
    TrackedObjectState,
)


@dataclass
class RefinedDetection:
    """Detection enriched with mask + classification (slow path output)."""

    detection: Detection
    mask: np.ndarray
    classification: ClassificationResult


class DronePipelineManager:
    """Latency-aware orchestrator for drone detection, tracking, and classification.

    Design principles:
      * Fast path (per-frame, caller thread):
          - advances tracker state every frame;
          - optionally ingests the latest completed refined detections;
          - never performs heavy model inference.
      * Slow path (background threads):
          - runs detection + segmentation + classification periodically;
          - produces `RefinedDetection` objects that the fast path consumes.
    """

    def __init__(
        self,
        detector: BaseDetector,
        segmenter: BaseSegmenter,
        classifier: BaseClassifier,
        tracker: BaseTracker,
        detection_interval: int = 5,
        detection_interval_tracked: Optional[int] = None,
        max_workers: int = 2,
        recovery_interval: Optional[int] = None,
        stale_detection_frames: int = 15,
    ) -> None:
        self._detector = detector
        self._segmenter = segmenter
        self._classifier = classifier
        self._tracker = tracker

        self._detection_interval = max(1, detection_interval)
        self._detection_interval_tracked = max(1, detection_interval_tracked or detection_interval)
        # Коли довго немає нових детекцій, переходимо в recovery-режим і
        # запускаємо YOLO частіше. Якщо recovery_interval не заданий, беремо
        # detection_interval.
        self._recovery_interval = max(1, recovery_interval or detection_interval)
        self._stale_detection_frames = max(1, stale_detection_frames)

        self._frame_idx = 0

        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._pending_future: Optional[Future[List[RefinedDetection]]] = None

        self._lock = Lock()
        self._pending_refined: Optional[List[RefinedDetection]] = None

        self._latest_tracks: List[TrackedObjectState] = []
        # Скільки кадрів пройшло з моменту останньої успішної детекції YOLO.
        self._frames_since_last_detection: int = 1_000_000

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update(self, frame: np.ndarray, timestamp: float) -> List[TrackedObjectState]:
        """Main per-frame entry point (FAST path).

        This call must be lightweight and non-blocking w.r.t. heavy models.
        """
        self._frame_idx += 1

        # 1) Incorporate any finished slow-path job
        self._gather_slow_path_results()

        # 2) Build detections for tracker from the last refined batch, if any
        refined_for_this_frame: Optional[List[RefinedDetection]] = None
        with self._lock:
            if self._pending_refined is not None:
                refined_for_this_frame = self._pending_refined
                self._pending_refined = None

        if refined_for_this_frame is not None and len(refined_for_this_frame) > 0:
            detections = [rd.detection for rd in refined_for_this_frame]
            tracks = self._tracker.update(frame, timestamp, detections=detections)
            # Attach classification + masks to tracks by IoU matching
            self._attach_classification_to_tracks(tracks, refined_for_this_frame)
        else:
            # No fresh detections → purely tracking
            tracks = self._tracker.update(frame, timestamp, detections=None)

        self._latest_tracks = tracks

        # 3) Decide if we should schedule a new slow-path job
        # Якщо на цьому кадрі не було нових детекцій, збільшуємо лічильник
        # давності детекцій. Якщо були — він був скинутий у _gather_slow_path_results.
        self._frames_since_last_detection += 1
        self._maybe_schedule_slow_path(frame, timestamp)

        return tracks

    # ------------------------------------------------------------------
    # Slow path orchestration
    # ------------------------------------------------------------------
    def _slow_path_job(self, frame: np.ndarray, timestamp: float) -> List[RefinedDetection]:
        """Heavy job: detection + segmentation + classification on a single frame."""
        logger.debug("Slow path started")

        detections = self._detector.detect(frame, timestamp)
        refined: List[RefinedDetection] = []

        for det in detections:
            try:
                mask = self._segmenter.segment(frame, det)
            except Exception as e:  # pragma: no cover - defensive
                logger.warning(f"Segmentation failed for detection {det}: {e}")
                # Fallback: empty mask
                x1, y1, x2, y2 = det.bbox
                h = max(1, y2 - y1)
                w = max(1, x2 - x1)
                mask = np.zeros((h, w), dtype=np.uint8)

            try:
                cls = self._classifier.classify(mask)
            except Exception as e:  # pragma: no cover - defensive
                logger.warning(f"Classification failed for detection {det}: {e}")
                cls = ClassificationResult(drone_type="unknown", confidence=0.0)

            refined.append(RefinedDetection(detection=det, mask=mask, classification=cls))

        logger.debug(f"Slow path finished with {len(refined)} refined detections")
        return refined

    def _maybe_schedule_slow_path(self, frame: np.ndarray, timestamp: float) -> None:
        """Decide when to schedule a new YOLO+segm+class job.

        Логіка:
          * Якщо взагалі немає треків → агресивний пошук (detection_interval).
          * Якщо треки є, але давно не було нових детекцій → recovery-режим
            (recovery_interval).
          * Якщо треки свіже підтверджені YOLO → детектимо рідко
            (detection_interval_tracked), CSRT/інші трекери тягнуть по кадрах.
        """
        has_tracks = bool(self._latest_tracks)

        if not has_tracks:
            interval = self._detection_interval
        else:
            # Якщо давно не було нових детекцій, форсимо recovery-режим.
            if self._frames_since_last_detection >= self._stale_detection_frames:
                interval = self._recovery_interval
            else:
                interval = self._detection_interval_tracked

        if self._frame_idx % interval != 0:
            return

        # Only one outstanding job at a time
        if self._pending_future is not None and not self._pending_future.done():
            return

        # Copy frame to avoid lifetime issues
        frame_copy = frame.copy()
        ts_copy = float(timestamp)
        logger.debug("Scheduling slow path job")
        self._pending_future = self._executor.submit(self._slow_path_job, frame_copy, ts_copy)

    def _gather_slow_path_results(self) -> None:
        if self._pending_future is None:
            return
        if not self._pending_future.done():
            return

        try:
            refined = self._pending_future.result()
        except Exception as e:  # pragma: no cover - defensive
            logger.error(f"Slow path job failed: {e}")
            refined = []

        # Оновити лічильник frames_since_last_detection: якщо YOLO щось знайшов,
        # скидаємо в нуль; якщо ні — просто залишимо як є (буде наростати в update).
        if refined:
            self._frames_since_last_detection = 0

        with self._lock:
            self._pending_refined = refined

        self._pending_future = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
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

    def _attach_classification_to_tracks(
        self,
        tracks: List[TrackedObjectState],
        refined_dets: List[RefinedDetection],
    ) -> None:
        if not tracks or not refined_dets:
            return

        # Precompute detection bboxes
        det_bboxes = [np.array(rd.detection.bbox, dtype=float) for rd in refined_dets]

        for tr in tracks:
            tb = np.array(tr.bbox, dtype=float)
            best_iou = 0.0
            best_idx: Optional[int] = None
            for i, db in enumerate(det_bboxes):
                iou = self._bbox_iou(tb, db)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i

            if best_idx is not None and best_iou > 0.1:  # simple threshold
                rd = refined_dets[best_idx]
                tr.classification = rd.classification
                tr.mask = rd.mask

    @property
    def latest_tracks(self) -> List[TrackedObjectState]:
        return self._latest_tracks
