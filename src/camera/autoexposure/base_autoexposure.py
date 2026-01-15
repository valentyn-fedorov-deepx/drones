import numpy as np
from loguru import logger
import cv2

from configs.data_pixel_tensor import DATA_PIXEL_TENSOR_BACKEND


class BaseAutoExposure:
    def __init__(self, max_possible_value, max_exposure_value, step_coef: float = 0.05):
        self._roi_mask = None
        self._xyxyn = None
        self.optim_value = 0.18 * max_possible_value 
        self.max_exposure_value = max_exposure_value
        self.step_coef = step_coef
        self.prev_mean = 0

    def set_roi_from_xyxyn(self, xyxyn):
        self._xyxyn = xyxyn

    def set_roi_from_mask(self, mask):
        self._roi_mask = mask.astype(bool)

    def __call__(self, raw_frame, current_exposure):
        if DATA_PIXEL_TENSOR_BACKEND == "torch":
            if isinstance(raw_frame, np.ndarray):
                raw_frame = raw_frame
            else:
                raw_frame = raw_frame.cpu().numpy()

        # --- Create the ROI mask (same as before) ---
        if self._xyxyn is not None:
            x1, y1, x2, y2 = self._xyxyn
            h, w = raw_frame.shape[:2]
            x1, x2 = int(x1 * w), int(x2 * w)
            y1, y2 = int(y1 * h), int(y2 * h)
            roi_mask = np.zeros(raw_frame.shape[:2], dtype=bool)
            roi_mask[y1:y2, x1:x2] = 1
        else:
            roi_mask = np.ones(raw_frame.shape[:2], dtype=bool)

        if self._roi_mask is not None:
            roi_mask = np.bitwise_or(roi_mask, self._roi_mask)

        # Check if the image is color (has 3 dimensions)
        if raw_frame.ndim == 3 and raw_frame.shape[2] in [3, 4]:
            # Assuming BGR format, common in OpenCV. Use COLOR_RGB2GRAY if your data is RGB.
            # We work on a copy to avoid modifying the original frame if it's used elsewhere.
            luminance_frame = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2GRAY)
        else:
            # It's already monochrome, just use it as is.
            luminance_frame = raw_frame

        # --- Calculate mean on the single-channel luminance frame ---
        pixel_values = luminance_frame[roi_mask]
        
        # Avoid division by zero if ROI is empty
        if pixel_values.size == 0:
            return current_exposure
            
        current_mean = pixel_values.mean()
        self.prev_mean = current_mean
        
        # Avoid division by zero if the frame is black
        if current_mean < 1e-5:
             # If the image is black, we can't determine a ratio, maybe increase exposure slightly
            ratio = 2.0 
        else:
            ratio = self.optim_value / current_mean

        # --- Exposure adjustment logic (same as before) ---
        if 0.95 < ratio < 1.05:
            return current_exposure

        new_exposure = current_exposure * (ratio ** self.step_coef)
        new_exposure = min(new_exposure, self.max_exposure_value)
        new_exposure = max(new_exposure, 1)

        logger.debug(f"Changed exposure from {current_exposure} to {new_exposure}")
        logger.debug(f"Optim value: {self.optim_value}, current mean: {current_mean:.2f}, ratio: {ratio:.2f}")

        return new_exposure