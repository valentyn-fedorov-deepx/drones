from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger

from src.drone_pipeline.interfaces import BaseClassifier, ClassificationResult
from src.utils.common import resource_path


@dataclass
class SilhouetteRef:
    name: str
    mask: np.ndarray  # normalized binary mask (H,W)
    hu: np.ndarray    # shape (7,)
    edge: np.ndarray  # binary edge map (H,W)


def _normalize_mask(mask: np.ndarray, out_size: int = 128) -> np.ndarray:
    """Normalize a binary mask to a centered square canvas of size out_size.

    - Binarizes mask.
    - Crops tight bounding box.
    - Rescales preserving aspect ratio to fit inside out_size×out_size.
    - Centers in the canvas.
    """
    if mask is None:
        return np.zeros((out_size, out_size), dtype=np.uint8)

    m = (mask > 0).astype(np.uint8)
    ys, xs = np.where(m > 0)
    if len(xs) == 0 or len(ys) == 0:
        return np.zeros((out_size, out_size), dtype=np.uint8)

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    crop = m[y1 : y2 + 1, x1 : x2 + 1]

    h, w = crop.shape
    scale = min(out_size / max(h, 1), out_size / max(w, 1))
    new_h, new_w = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    canvas = np.zeros((out_size, out_size), dtype=np.uint8)
    y_off = (out_size - new_h) // 2
    x_off = (out_size - new_w) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized
    return canvas


def _compute_hu(mask: np.ndarray) -> np.ndarray:
    m = cv2.moments(mask.astype(np.uint8))
    hu = cv2.HuMoments(m).flatten()
    # Log transform for better numerical stability & comparability
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)
    return hu.astype(np.float32)


def _compute_edge(mask: np.ndarray) -> np.ndarray:
    # Simple Canny edge on binary mask
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    edges = cv2.Canny(mask_u8, 50, 150)
    return (edges > 0).astype(np.uint8)


class SilhouetteClassifier(BaseClassifier):
    """Multi-stage silhouette classifier.

    Stages:
      1. Hu-moments distance to filter reference silhouettes.
      2. 2D similarity via IoU + Chamfer distance on normalized masks.
      3. Optional hook for 3D-model-based refinement (not implemented here).
    """

    def __init__(
        self,
        silhouettes_dir: str,
        image_size: int = 128,
        top_k: int = 3,
        chamfer_weight: float = 0.5,
    ) -> None:
        self.image_size = image_size
        self.top_k = max(1, top_k)
        self.chamfer_weight = float(np.clip(chamfer_weight, 0.0, 1.0))

        base_dir = Path(resource_path(silhouettes_dir))
        if not base_dir.exists():
            logger.warning(f"Silhouette directory {base_dir} does not exist; classifier will be degenerate")

        self._refs: List[SilhouetteRef] = self._load_refs(base_dir)
        if not self._refs:
            logger.warning("No reference silhouettes loaded; classifier will always return 'unknown'")

    # ------------------------------------------------------------------
    # Reference loading
    # ------------------------------------------------------------------
    def _load_refs(self, base_dir: Path) -> List[SilhouetteRef]:
        exts = {".png", ".jpg", ".jpeg", ".bmp"}
        refs: List[SilhouetteRef] = []
        if not base_dir.exists():
            return refs

        for p in base_dir.rglob("*"):
            if p.suffix.lower() not in exts:
                continue

            # Class name from parent directory
            cls_name = p.parent.name
            try:
                img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                mask = (img > 127).astype(np.uint8)
                norm = _normalize_mask(mask, self.image_size)
                hu = _compute_hu(norm)
                edge = _compute_edge(norm)
                refs.append(SilhouetteRef(name=cls_name, mask=norm, hu=hu, edge=edge))
            except Exception as e:  # pragma: no cover - defensive
                logger.warning(f"Failed to load silhouette {p}: {e}")

        logger.info(f"Loaded {len(refs)} reference silhouettes from {base_dir}")
        return refs

    # ------------------------------------------------------------------
    # Stage 1: Hu moments filtering
    # ------------------------------------------------------------------
    def _stage1_candidates(self, hu: np.ndarray) -> List[Tuple[SilhouetteRef, float]]:
        if not self._refs:
            return []
        dists = []
        for ref in self._refs:
            d = float(np.linalg.norm(hu - ref.hu))
            dists.append((ref, d))
        dists.sort(key=lambda x: x[1])
        return dists[: self.top_k]

    # ------------------------------------------------------------------
    # Stage 2: IoU + Chamfer similarity
    # ------------------------------------------------------------------
    @staticmethod
    def _iou(a: np.ndarray, b: np.ndarray) -> float:
        a_bin = a > 0
        b_bin = b > 0
        inter = np.logical_and(a_bin, b_bin).sum()
        union = np.logical_or(a_bin, b_bin).sum()
        if union == 0:
            return 0.0
        return float(inter / union)

    @staticmethod
    def _chamfer_similarity(a_edge: np.ndarray, b_edge: np.ndarray) -> float:
        """Symmetric Chamfer-based similarity in [0,1]."""
        a = (a_edge > 0).astype(np.uint8)
        b = (b_edge > 0).astype(np.uint8)
        if a.shape != b.shape:
            b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Distance transforms
        da = cv2.distanceTransform(1 - a, cv2.DIST_L2, 3)
        db = cv2.distanceTransform(1 - b, cv2.DIST_L2, 3)

        # Average distance from A edges to B and vice versa
        a_pts = np.where(a > 0)
        b_pts = np.where(b > 0)
        if len(a_pts[0]) == 0 or len(b_pts[0]) == 0:
            return 0.0

        d_ab = float(db[a_pts].mean())
        d_ba = float(da[b_pts].mean())
        d = 0.5 * (d_ab + d_ba)

        # Convert distance to similarity in [0, 1]
        # Heuristic: assume distances up to ~20 pixels are meaningful
        s = np.exp(-d / 20.0)
        return float(s)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def classify(self, mask: np.ndarray) -> ClassificationResult:  # type: ignore[override]
        if mask is None or mask.size == 0 or not self._refs:
            return ClassificationResult(drone_type="unknown", confidence=0.0)

        norm = _normalize_mask(mask, self.image_size)
        hu = _compute_hu(norm)
        edge = _compute_edge(norm)

        # Stage 1: Hu moments
        candidates = self._stage1_candidates(hu)
        if not candidates:
            return ClassificationResult(drone_type="unknown", confidence=0.0)

        # Stage 2: IoU + Chamfer
        best_name = "unknown"
        best_score = 0.0
        scores_per_class: Dict[str, float] = {}

        for ref, hu_dist in candidates:
            iou = self._iou(norm, ref.mask)
            chamfer_sim = self._chamfer_similarity(edge, ref.edge)
            # Combine: weighted sum, both already in [0,1]
            combined = (1.0 - self.chamfer_weight) * iou + self.chamfer_weight * chamfer_sim

            # Track best per class (multiple ref views per class allowed)
            prev = scores_per_class.get(ref.name, 0.0)
            if combined > prev:
                scores_per_class[ref.name] = combined

        if scores_per_class:
            best_name, best_score = max(scores_per_class.items(), key=lambda kv: kv[1])

        # Map similarity [0,1] to a soft confidence; keep it simple here
        confidence = float(np.clip(best_score, 0.0, 1.0))
        return ClassificationResult(drone_type=best_name, confidence=confidence)
