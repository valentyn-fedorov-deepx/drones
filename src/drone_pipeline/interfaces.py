from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Protocol, runtime_checkable

import numpy as np


BBox = Tuple[int, int, int, int]  # x1, y1, x2, y2


@dataclass
class Detection:
    """Single-frame detector output.

    Note: detector is responsible for mapping model-specific outputs to this
    minimal, backend-agnostic representation.
    """

    bbox: BBox
    score: float
    label: str
    timestamp: float
    frame_index: Optional[int] = None


@dataclass
class ClassificationResult:
    """Result of mask-based classification for a single object."""

    drone_type: str
    confidence: float


@dataclass
class TrackedObjectState:
    """Tracker-centric representation used by the main pipeline.

    This is intentionally separate from any particular tracker/measurement
    implementation (Norfair, ByteTrack, etc.).
    """

    track_id: int
    bbox: BBox
    score: float
    label: str
    timestamp: float

    # Optional rich fields
    mask: Optional[np.ndarray] = None  # bbox-local binary mask (H, W)
    classification: Optional[ClassificationResult] = None

    # Optional kinematics / 3D info (can be filled by existing measurement stack)
    world_xyz: Optional[Tuple[float, float, float]] = None
    velocity_mps: Optional[Tuple[float, float, float]] = None


@runtime_checkable
class BaseDetector(Protocol):
    """Abstract interface for detectors.

    Implementations MUST be stateless w.r.t. individual calls (no tracking here).
    """

    def detect(self, frame: np.ndarray, timestamp: float) -> List[Detection]:  # pragma: no cover - interface
        ...


@runtime_checkable
class BaseSegmenter(Protocol):
    """Abstract interface for mask refinement / segmentation backends."""

    def segment(self, frame: np.ndarray, detection: Detection) -> np.ndarray:  # pragma: no cover - interface
        """Return a bbox-local binary mask with shape (H, W) and dtype uint8.

        The mask is expected to align with `detection.bbox` when pasted back
        into the full image. Implementations may use SAM, YOLOv8-Seg, etc.
        """

        ...


@runtime_checkable
class BaseClassifier(Protocol):
    """Abstract interface for silhouette-based classifiers."""

    def classify(self, mask: np.ndarray) -> ClassificationResult:  # pragma: no cover - interface
        ...


@runtime_checkable
class BaseTracker(Protocol):
    """Abstract interface for trackers that operate every frame.

    The tracker maintains its own internal state across calls. On frames
    where no fresh detections are available, call `update()` with an empty
    list to advance tracks based on motion only.
    """

    def update(
        self,
        frame: np.ndarray,
        timestamp: float,
        detections: Optional[List[Detection]] = None,
    ) -> List[TrackedObjectState]:  # pragma: no cover - interface
        ...
