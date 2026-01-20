from __future__ import annotations

from typing import List, Optional

import os
import sys
from pathlib import Path

import cv2 as cv
import numpy as np

from src.drone_pipeline.interfaces import BaseTracker, Detection, TrackedObjectState

# Try to import C++ extension; if it fails, attempt to add local build path
# and required OpenCV DLL path automatically.
SingleCsrtTracker = None

try:  # first attempt: normal import
    from csrt_tracker_ext import SingleCsrtTracker  # type: ignore[assignment]
except ImportError:
    # Try to add local build/Release directories to sys.path (both build_gpu and build)
    this_file = Path(__file__).resolve()
    project_root = this_file.parents[2]  # .../src/drone_pipeline -> project root
    candidate_builds = [
        project_root / "src" / "cpp" / "build_gpu" / "Release",
        project_root / "src" / "cpp" / "build" / "Release",
    ]
    for candidate_build in candidate_builds:
        if not candidate_build.exists():
            continue
        if str(candidate_build) not in sys.path:
            sys.path.insert(0, str(candidate_build))
        try:
            from csrt_tracker_ext import SingleCsrtTracker  # type: ignore[assignment]
            break
        except ImportError:
            SingleCsrtTracker = None

    # Also try to prepend OpenCV DLL directory to PATH (specific to this setup)
    # so that dependent DLLs are found when loading csrt_tracker_ext.
    if SingleCsrtTracker is None:
        opencv_dll_dir = Path(r"C:\opencv\opencv\build_msvc\install\x64\vc17\bin")
        if opencv_dll_dir.exists():
            # On Windows (Python 3.8+), the recommended way to make additional
            # DLL search paths visible is os.add_dll_directory. Updating PATH
            # alone is often not sufficient.
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(str(opencv_dll_dir))
            # Keep PATH update as a secondary fallback.
            os.environ["PATH"] = str(opencv_dll_dir) + os.pathsep + os.environ.get("PATH", "")
            try:
                from csrt_tracker_ext import SingleCsrtTracker  # type: ignore[assignment]
            except ImportError:
                SingleCsrtTracker = None


class CppCSRTTracker(BaseTracker):
    """Wrapper around C++ SingleCsrtTracker implementing the same external
    interface as our Python CSRTTracker, but restricted to a single drone.

    Semантика під поточну задачу:
    - один трек з ID=0;
    - якщо є детекція, ініціалізуємо/переініціалізуємо CSRT по цій bbox;
    - між детекціями CSRT тягне трек по motion;
    - anti-stick реалізовано всередині C++ (motion_threshold, max_static_frames);
    - якщо C++ трекер переходить у dormant, ми перестаємо повертати об'єкт.
    """

    def __init__(
        self,
        max_misses: int = 10,
        motion_threshold: float = 3.0,
        max_static_frames: int = 8,
        label: str = "drone",
        scale: float = 1.0,
        output_grace_frames: int = 45,
        motion_pred_frames: int = 30,
        motion_pred_alpha: float = 0.6,
    ) -> None:
        if SingleCsrtTracker is None:
            raise RuntimeError(
                "csrt_tracker_ext (C++ CSRT extension) is not available. "
                "Build it first, then run with --tracker-backend cpp."
            )

        self._tracker = SingleCsrtTracker(
            max_misses,
            float(motion_threshold),
            int(max_static_frames),
        )
        self._label = label
        self._has_track: bool = False
        self._last_score: float = 1.0
        self._last_bbox: Optional[tuple[int, int, int, int]] = None
        self._lost_frames: int = 0
        self._output_grace_frames: int = int(output_grace_frames)
        # Motion prediction state
        self._last_center: Optional[tuple[float, float]] = None
        self._velocity: Optional[tuple[float, float]] = None
        self._motion_pred_frames: int = int(motion_pred_frames)
        self._motion_pred_alpha: float = float(motion_pred_alpha)

        # Optional global downscaling factor for CSRT to speed up tracking.
        # If scale < 1.0, we resize the frame before passing it to C++ and
        # map all bounding boxes between original and scaled coordinates.
        self._scale: float = float(scale)
        if self._scale <= 0:
            self._scale = 1.0

    def _choose_detection(self, detections: List[Detection]) -> Optional[Detection]:
        if not detections:
            return None
        # Для одного дрона беремо детекцію з найбільшим score
        return max(detections, key=lambda d: d.score)

    def update(
        self,
        frame: np.ndarray,
        timestamp: float,
        detections: Optional[List[Detection]] = None,
    ) -> List[TrackedObjectState]:  # type: ignore[override]
        detections = detections or []

        # 1) Optional downscaling of frame for faster CSRT
        if self._scale != 1.0:
            h, w = frame.shape[:2]
            new_w = max(1, int(w * self._scale))
            new_h = max(1, int(h * self._scale))
            frame_for_track = cv.resize(frame, (new_w, new_h), interpolation=cv.INTER_LINEAR)
        else:
            frame_for_track = frame

        # 2) Якщо є детекції, прив'язуємо CSRT до найсильнішої
        det = self._choose_detection(detections)
        if det is not None:
            bbox = det.bbox
            if self._scale != 1.0:
                sx = self._scale
                sy = self._scale
                bbox_scaled = (
                    int(bbox[0] * sx),
                    int(bbox[1] * sy),
                    int(bbox[2] * sx),
                    int(bbox[3] * sy),
                )
            else:
                bbox_scaled = bbox

            # Використовуємо init() для першого разу, reset() для подальших кадрів
            if not self._has_track or not self._tracker.has_track():
                self._tracker.init(frame_for_track, bbox_scaled, float(timestamp))
                self._has_track = True
            else:
                self._tracker.reset(frame_for_track, bbox_scaled, float(timestamp))
            self._last_score = float(det.score)
            # Reset lost counter on a fresh detection
            self._lost_frames = 0

        # 3) Оновлюємо CSRT по поточному кадру
        if not self._has_track:
            return []

        ok, dormant, x1, y1, x2, y2, motion = self._tracker.update(frame_for_track, float(timestamp))

        if (not ok) or dormant:
            # Трек пішов у dormant або CSRT не зміг оновитись
            self._lost_frames += 1
            if self._last_bbox is not None and self._lost_frames <= self._output_grace_frames:
                x1_i, y1_i, x2_i, y2_i = self._last_bbox
                # Predict forward using last velocity for a short window
                if self._velocity is not None and self._last_center is not None and self._lost_frames <= self._motion_pred_frames:
                    cx, cy = self._last_center
                    vx, vy = self._velocity
                    cx_p = cx + vx
                    cy_p = cy + vy
                    w_box = max(1, x2_i - x1_i)
                    h_box = max(1, y2_i - y1_i)
                    H, W = frame.shape[:2]
                    x1_i = int(max(0, min(W - 1, cx_p - w_box / 2)))
                    y1_i = int(max(0, min(H - 1, cy_p - h_box / 2)))
                    x2_i = int(max(0, min(W, x1_i + w_box)))
                    y2_i = int(max(0, min(H, y1_i + h_box)))
                    self._last_bbox = (x1_i, y1_i, x2_i, y2_i)
                    self._last_center = (cx_p, cy_p)
                state = TrackedObjectState(
                    track_id=0,
                    bbox=(x1_i, y1_i, x2_i, y2_i),
                    score=self._last_score,
                    label=self._label,
                    timestamp=float(timestamp),
                    tracking_lost=True,
                )
                return [state]
            return []

        # 4) Масштабуємо координати назад у систему full-res, якщо потрібно
        if self._scale != 1.0:
            inv = 1.0 / self._scale
            x1 *= inv
            y1 *= inv
            x2 *= inv
            y2 *= inv

        # 5) Побудувати TrackedObjectState з фіксованим ID=0
        x1_i, y1_i, x2_i, y2_i = int(x1), int(y1), int(x2), int(y2)
        # Update velocity based on bbox center (EMA)
        cx = (x1_i + x2_i) / 2.0
        cy = (y1_i + y2_i) / 2.0
        if self._last_center is not None:
            dx = cx - self._last_center[0]
            dy = cy - self._last_center[1]
            if self._velocity is None:
                self._velocity = (dx, dy)
            else:
                self._velocity = (
                    self._motion_pred_alpha * dx + (1.0 - self._motion_pred_alpha) * self._velocity[0],
                    self._motion_pred_alpha * dy + (1.0 - self._motion_pred_alpha) * self._velocity[1],
                )
        self._last_center = (cx, cy)
        self._last_bbox = (x1_i, y1_i, x2_i, y2_i)
        self._lost_frames = 0
        state = TrackedObjectState(
            track_id=0,
            bbox=(x1_i, y1_i, x2_i, y2_i),
            score=self._last_score,
            label=self._label,
            timestamp=float(timestamp),
            tracking_lost=False,
        )
        return [state]
