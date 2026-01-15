import datetime as dt
import asyncio
from fractions import Fraction

import numpy as np
import cv2
from aiortc import VideoStreamTrack
from av import VideoFrame

from src.threads.capture_thread import CaptureThread
from configs.data_pixel_tensor import DATA_PIXEL_TENSOR_BACKEND

TIME_FORMAT = "%Y-%m-%d_%H-%M-%S"
RESOLUTION_W_H_FACTOR = 4
MIN_OUT_SIZE = 256  # ensure downsampled width/height never below this


class NumpyVideoTrack(VideoStreamTrack):
    """
    Custom video track that generates frames from numpy arrays
    """
    def __init__(self, cap_thread: CaptureThread, fps=30):
        super().__init__()
        self.cap_thread = cap_thread
        self.fps = fps
        # No camera-based width/height here; computed per-frame in recv

    async def next_timestamp(self):
        """
        Sleep so frames are produced at self.fps and return (pts, time_base).
        time_base ~= 1 / fps (supports non-integer fps like 29.97).
        """
        loop = asyncio.get_running_loop()

        if not hasattr(self, "_ts_start"):
            self._ts_start = loop.time()
            self._frame_idx = 0
            den = max(1, int(round(self.fps * 1000)))
            self._time_base = Fraction(1000, den)  # ≈ 1 / fps

        target_elapsed = self._frame_idx / float(self.fps)
        now_elapsed = loop.time() - self._ts_start
        sleep_for = target_elapsed - now_elapsed
        if sleep_for > 0:
            await asyncio.sleep(sleep_for)

        pts = self._frame_idx
        self._frame_idx += 1
        return pts, self._time_base

    async def recv(self):
        """
        Generate and return video frames from numpy arrays
        """
        pts, time_base = await self.next_timestamp()

        pxi_frame = self.cap_thread.latest_data
        frame_array = pxi_frame["view_img"]

        if DATA_PIXEL_TENSOR_BACKEND == 'torch':
            frame_array = frame_array.cpu().numpy()

        # Compute safe downsample based on *actual* frame dims
        h, w = frame_array.shape[:2]
        # max factor that still keeps both dims >= MIN_OUT_SIZE
        max_allowed_factor = max(1, min(w // MIN_OUT_SIZE, h // MIN_OUT_SIZE))
        ds = min(RESOLUTION_W_H_FACTOR, max_allowed_factor)

        if ds != 1:
            # Prefer INTER_AREA for better quality when shrinking
            new_w = max(1, w // ds)
            new_h = max(1, h // ds)
            frame_array = cv2.resize(frame_array, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Lazy-export resulting dimensions (optional)
        if not hasattr(self, "width"):
            self.width = frame_array.shape[1]
        if not hasattr(self, "height"):
            self.height = frame_array.shape[0]

        if frame_array.dtype != np.uint8:
            denom = getattr(pxi_frame, "max_value", None)
            if denom is None:
                # fallback to per-frame max; guard against zero
                denom = float(np.max(frame_array)) or 1.0
            normalized = frame_array / float(denom)
            frame_array = np.uint8(np.clip(normalized, 0, 1) * 255)

        if frame_array.ndim == 2:
            frame_array = cv2.cvtColor(frame_array, cv2.COLOR_GRAY2RGB)  # fixed RBG→RGB

        # Convert numpy array to VideoFrame
        frame = VideoFrame.from_ndarray(frame_array, format="rgb24")
        frame.pts = pts
        frame.time_base = time_base

        return frame
