import cv2
import numpy as np


def label_ridges(input_image: np.ndarray, half_kernel_size: int = 9) -> np.ndarray:
    """
    Compute a binary “ridge map” from a single‐channel or 3‐channel image.
    Steps:
      1. Convert to 8‐bit grayscale if needed.
      2. Build a mask of valid (non‐background) pixels: mask = (pixel < 255).
      3. Find the maximum intensity within the masked region.
      4. Zero‐out any “background” pixels, then fill them with max intensity.
      5. Adaptive‐threshold (mean‐based, inverted) to highlight ridges.
      6. Multiply threshold result by original mask to clear outside regions.
      7. Erode the original mask to remove small border artifacts.
      8. Invert the eroded mask so "true background" becomes white.
      9. OR the inverted‐mask with the thresholded result, then invert final image
         so that ridges are black (0) on white (255).

    Args:
        input_image (np.ndarray): Input image array (grayscale or BGR). May be >8‐bit.
        half_kernel_size (int): half‐size for adaptive‐threshold block & erosion.

    Returns:
        np.ndarray: 8‐bit single‐channel image (ridges=0, background=255).
    """
    img = input_image.copy()

    if img.dtype == np.uint16:
        img = img / img.max()
        img = np.uint8(img * 255)
    elif img.dtype != np.uint8:
        img = cv2.convertScaleAbs(img)

    if img.ndim == 3 and img.shape[2] == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img.ndim == 2:
        img_gray = img
    else:
        raise ValueError("Unsupported image shape. Expect 1‐ or 3‐channel array.")

    _, mask = cv2.threshold(img_gray, thresh=254, maxval=255, type=cv2.THRESH_BINARY_INV)
    min_val, max_val, _, _ = cv2.minMaxLoc(img_gray, mask=mask)

    img_masked = cv2.bitwise_and(img_gray, mask)
    mask_inv = cv2.bitwise_not(mask)
    img_normalized = img_masked.copy()
    img_normalized[mask_inv == 255] = max_val

    block_size = 2 * half_kernel_size + 1
    thresh_map = cv2.adaptiveThreshold(
        img_normalized,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=block_size,
        C=0
    )

    thresh_masked = cv2.bitwise_and(thresh_map, mask)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (2 * half_kernel_size + 1, 2 * half_kernel_size + 1)
    )
    mask_eroded = cv2.erode(mask, kernel, iterations=1)
    mask_eroded_inv = cv2.bitwise_not(mask_eroded)

    combined = cv2.bitwise_or(thresh_masked, mask_eroded_inv)
    ridge_map = cv2.bitwise_not(combined)

    return ridge_map
