from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from src.drone_pipeline.interfaces import BaseTracker, Detection, TrackedObjectState


@dataclass
class _TemplateState:
    """Internal state for the GPU template tracker."""

    bbox: tuple[float, float, float, float]
    template: torch.Tensor  # shape (1, 1, Ht, Wt), float32 on device
    h: int
    w: int


class GpuTemplateTracker(BaseTracker):
    """Very simple single-object tracker based on template matching on GPU.

    Семантика під нашу задачу (один дрон):
      * Під час наявності детекції ми ініціалізуємо/оновлюємо шаблон з bbox.
      * Між детекціями трекаємо дрона шляхом пошуку кращої відповідності шаблону
        у локальному вікні довкола попереднього bbox (крос-кореляція як conv2d).
      * Усе обчислення match'у робиться в PyTorch на обраному девайсі (cuda/cpu).
    """

    def __init__(
        self,
        device: str = "cuda",
        search_expansion: float = 2.0,
        min_search_radius: int = 16,
    ) -> None:
        """Args:
        device: 'cuda' або 'cpu'. Якщо GPU доступний, краще 'cuda'.
        search_expansion: у скільки разів розширюємо bbox по ширині/висоті
            для зони пошуку (наприклад, 2.0 = x2).
        min_search_radius: мінімальний радіус пошуку в пікселях від центру bbox.
        """
        self._device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self._state: Optional[_TemplateState] = None
        self._search_expansion = float(search_expansion)
        self._min_search_radius = int(min_search_radius)
        self._label = "drone"
        self._last_score: float = 1.0

    # ------------------------------
    # Helpers
    # ------------------------------
    def _frame_to_gray(self, frame: np.ndarray) -> torch.Tensor:
        """Convert HxWx3 uint8 RGB frame to 1x1xH xW float32 tensor on device."""
        if frame.ndim == 2:
            gray_np = frame.astype(np.float32) / 255.0
        else:
            # RGB to gray
            r = frame[..., 0].astype(np.float32)
            g = frame[..., 1].astype(np.float32)
            b = frame[..., 2].astype(np.float32)
            gray_np = 0.299 * r + 0.587 * g + 0.114 * b
            gray_np /= 255.0
        gray_t = torch.from_numpy(gray_np).to(self._device, dtype=torch.float32)
        return gray_t.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

    def _init_from_detection(self, frame: np.ndarray, det: Detection) -> None:
        x1, y1, x2, y2 = det.bbox
        x1_i, y1_i = max(0, int(x1)), max(0, int(y1))
        x2_i, y2_i = int(x2), int(y2)
        h = max(1, y2_i - y1_i)
        w = max(1, x2_i - x1_i)

        gray = self._frame_to_gray(frame)  # (1,1,H,W)
        # crop template
        template = gray[:, :, y1_i:y2_i, x1_i:x2_i]
        # normalize template to zero-mean, unit-variance (roughly)
        t_mean = template.mean()
        t_std = template.std()
        if t_std < 1e-6:
            t_std = torch.tensor(1e-6, device=self._device)
        template_norm = (template - t_mean) / t_std

        self._state = _TemplateState(
            bbox=(float(x1_i), float(y1_i), float(x2_i), float(y2_i)),
            template=template_norm,
            h=h,
            w=w,
        )
        self._last_score = det.score

    def _track_on_frame(self, frame: np.ndarray) -> Optional[tuple[float, float, float, float]]:
        if self._state is None:
            return None

        H, W = frame.shape[:2]
        gray = self._frame_to_gray(frame)

        x1, y1, x2, y2 = self._state.bbox
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)

        # search window size
        radius_x = max(self._min_search_radius, int(0.5 * bw * self._search_expansion))
        radius_y = max(self._min_search_radius, int(0.5 * bh * self._search_expansion))

        sx1 = max(0, int(cx - radius_x))
        sy1 = max(0, int(cy - radius_y))
        sx2 = min(W, int(cx + radius_x))
        sy2 = min(H, int(cy + radius_y))

        if sx2 <= sx1 + 2 or sy2 <= sy1 + 2:
            return None

        search = gray[:, :, sy1:sy2, sx1:sx2]  # (1,1,Hs,Ws)
        tmpl = self._state.template  # (1,1,Ht,Wt)

        Hs = search.shape[-2]
        Ws = search.shape[-1]
        Ht = tmpl.shape[-2]
        Wt = tmpl.shape[-1]
        if Hs < Ht or Ws < Wt:
            return None

        # Cross-correlation via conv2d
        # (no padding, so output size is (Hs-Ht+1, Ws-Wt+1))
        corr = F.conv2d(search, tmpl)
        # Find max response
        max_idx = torch.argmax(corr)
        max_y, max_x = divmod(int(max_idx), int(corr.shape[-1]))

        # Top-left in full frame coords
        new_x1 = sx1 + max_x
        new_y1 = sy1 + max_y
        new_x2 = new_x1 + Wt
        new_y2 = new_y1 + Ht

        # Clamp to frame
        new_x1 = float(max(0, min(W - 1, new_x1)))
        new_y1 = float(max(0, min(H - 1, new_y1)))
        new_x2 = float(max(0, min(W, new_x2)))
        new_y2 = float(max(0, min(H, new_y2)))

        return new_x1, new_y1, new_x2, new_y2

    # ------------------------------
    # BaseTracker API
    # ------------------------------
    def update(
        self,
        frame: np.ndarray,
        timestamp: float,
        detections: Optional[List[Detection]] = None,
    ) -> List[TrackedObjectState]:  # type: ignore[override]
        detections = detections or []

        # 1) Якщо є детекції, ініціалізуємося з найкращої.
        if detections:
            best = max(detections, key=lambda d: d.score)
            self._init_from_detection(frame, best)

        # 2) Якщо немає шаблону — нічого трекати.
        if self._state is None:
            return []

        # 3) Оновити bbox через template matching на поточному кадрі.
        tracked_bbox = self._track_on_frame(frame)
        if tracked_bbox is None:
            return []

        x1, y1, x2, y2 = tracked_bbox
        self._state.bbox = (x1, y1, x2, y2)

        state = TrackedObjectState(
            track_id=0,
            bbox=(int(x1), int(y1), int(x2), int(y2)),
            score=float(self._last_score),
            label=self._label,
            timestamp=float(timestamp),
        )
        return [state]
