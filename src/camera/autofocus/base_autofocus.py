import cv2
import numpy as np
from loguru import logger

from src.utils.im_transform import resize_with_reflect_padding


class AutofocusController:
    def __init__(self, min_possible_value, max_possible_value):
        self._roi_mask = None
        self.min_possible_value = min_possible_value
        self.max_possible_value = max_possible_value
        
        # --- State for the hill-climbing autofocus algorithm ---
        self.last_score = 0.0
        self.direction = 1
        self.bad_move_counter = 0
        self.patience = 2

        # --- Parameters for Inverse Proportional Step Size ---
        self.scaling_factor_K = 850.0
        self.epsilon = 1.0
        self.max_step = (max_possible_value - min_possible_value) // 8
        self.min_step = 1

        # --- "Reduce on Plateau" Scheduler Parameters ---
        self.best_score_so_far = 0.0
        # How many direction reversals without improvement before reducing K
        self.scheduler_patience = 3
        # The factor to reduce K by (e.g., 0.8 = 20% reduction)
        self.scheduler_factor = 0.8 
        # Counter for reversals without improvement
        self.reversal_counter = 0

        self._xyxyn = None

    def set_roi_from_xyxyn(self, xyxyn):
        self._xyxyn = xyxyn

    def set_region(self, mask):
        self._roi_mask = mask.astype(bool)
    
    def _calculate_focus_score(self, frame, roi_mask):
        """Calculates the focus score (variance of Laplacian) for the ROI."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = resize_with_reflect_padding(gray, (1024, 1024))
        roi_mask = resize_with_reflect_padding(roi_mask.astype(np.uint8), (1024, 1024),
                                               interpolation=cv2.INTER_NEAREST).astype(bool)
        laplacian = cv2.Laplacian(gray, ddepth=cv2.CV_64F, ksize=5)
        laplacian_values = laplacian[roi_mask]
        
        if laplacian_values.size > 0:
            return laplacian_values.std()
        return 0


    def __call__(self, raw_frame, current_focus):
        """
        Implements hill-climbing with a dynamic step size and a "Reduce on Plateau"
        scheduler for the scaling factor K.
        """
        if self._xyxyn is not None:
            x1, y1, x2, y2 = self._xyxyn
            h, w = raw_frame.shape[:2]
            x1 = int(x1 * w)
            x2 = int(x2 * w)
            y1 = int(y1 * h)
            y2 = int(y2 * h)
            roi_mask = np.zeros(raw_frame.shape[:2], dtype=bool)
            roi_mask[y1:y2, x1:x2] = 1
        else:
            roi_mask = np.ones(raw_frame.shape[:2], dtype=bool)

        if self._roi_mask is not None:
            roi_mask = np.bitwise_or(roi_mask, self._roi_mask)

        score = self._calculate_focus_score(raw_frame, roi_mask)

        dynamic_step_size = self.scaling_factor_K / (score + self.epsilon)
        step_size = np.clip(dynamic_step_size, self.min_step, self.max_step)

        if score > self.last_score:
            self.bad_move_counter = 0
        else:
            self.bad_move_counter += 1

        if self.bad_move_counter >= self.patience:
            # --- This is a direction reversal, where a peak has been found ---
            # Check if this peak is the best one we've ever seen
            if self.last_score > self.best_score_so_far:
                self.best_score_so_far = self.last_score
                # Reset counter because we found a better peak
                self.reversal_counter = 0
            else:
                # We failed to find a better peak, increment the counter
                self.reversal_counter += 1

            # --- Scheduler Logic ---
            # If we've had too many reversals without improvement, reduce K
            if self.reversal_counter >= self.scheduler_patience:
                new_K = self.scaling_factor_K * self.scheduler_factor
                self.scaling_factor_K = max(new_K, 50.0) # Don't let K get too small
                logger.debug(f"Scheduler: Reducing K to {self.scaling_factor_K:.2f}") # For debugging
                self.reversal_counter = 0 # Reset after reducing

            # Finally, reverse direction and reset the move counter
            self.direction *= -1
            self.bad_move_counter = 0

        new_focus = current_focus + self.direction * step_size
        new_focus = np.clip(new_focus, self.min_possible_value, self.max_possible_value)

        self.last_score = score
        
        logger.debug(f"Changed focus from the {current_focus} to the {new_focus}, with score: {score:.3f} and scaling factor: {self.scaling_factor_K:.3f}")
        return int(new_focus)