import numpy as np
import cv2

from configs.data_pixel_tensor import ENABLE_COLOR_CORRECTION, COLOR_CORRECTION_MODEL_PATH

if ENABLE_COLOR_CORRECTION:
    from src.utils.color_correction import ColorCorrectionNumpy
    CC_MODEL = ColorCorrectionNumpy.load(COLOR_CORRECTION_MODEL_PATH)


def dstack(elements):
    return np.stack(elements, axis=-1)


def change_dtype(x, target_dtype):
    return x.astype(target_dtype)


def copy_element(x):
    return x.copy()


def decode_mono12_contiguous_byteswapped(raw_data: bytes, width: int, height: int) -> np.ndarray:
    """
    Decodes a contiguous 12-bit format assuming a BGR-like byte order.
    (e.g., [B_7 ... B_0][B_11 ... B_8, A_3 ... A_0][A_11 ... A_4])
    """
    expected_bytes = int(width * height * 1.5)
    if len(raw_data) != expected_bytes:
        raise ValueError(f"Incorrect data size. Should be {expected_bytes}, got {len(raw_data)}")

    data = np.frombuffer(raw_data, dtype=np.uint8)
    byte0 = data[0::3].astype(np.uint16)
    byte1 = data[1::3].astype(np.uint16)
    byte2 = data[2::3].astype(np.uint16)

    pixel_a = (byte2 << 4) | (byte1 >> 4)
    pixel_b = ((byte1 & 0x0F) << 8) | byte0

    image = np.empty((width * height,), dtype=np.uint16)
    image[0::2] = pixel_b
    image[1::2] = pixel_a

    return image.reshape((height, width))


def custom_jet_colormap(img):
    """Custom implementation of jet colormap"""
    if img.shape == 3:
        img = img.squeeze(0)

    # Normalize to [0, 1]
    img = (img - img.min()) / (img.max() - img.min())

    # Create RGB channels
    r = np.clip(1.5 - np.abs(4.0 * img - 3.0), 0, 1)
    g = np.clip(1.5 - np.abs(4.0 * img - 2.0), 0, 1)
    b = np.clip(1.5 - np.abs(4.0 * img - 1.0), 0, 1)

    # Stack to create RGB tensor
    colored_img = np.stack([r, g, b], axis=-1)

    return colored_img


def gamma_correct(image, gamma=1.5):
    """Applies a standard gamma correction for display."""
    return np.power(image, 1.0 / gamma)


def gray_world_white_balance(img, max_value: int = 255):
    wb = cv2.xphoto.createLearningBasedWB()
    wb.setSaturationThreshold(0.5)
    img_wb = wb.balanceWhite(img)

    img_wb_float = img_wb / max_value

    img_final = gamma_correct(np.clip(img_wb_float, 0, 1)) * max_value

    img_final = img_final.round().astype(img.dtype)

    return img_final


def conv2x2_stride2(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolves a 4D NumPy array `image` with a 4D `kernel` (2x2) using stride=2.

    image shape: (N, 1, H, W)
    kernel shape: (1, 1, 2, 2)
      - 1 output channel,
      - 1 input channel,
      - kernel height = 2,
      - kernel width = 2.

    Returns a 4D array: (N, 1, outH, outW)
    """
    # Extract individual kernel elements
    k00 = kernel[0, 0, 0, 0]
    k01 = kernel[0, 0, 0, 1]
    k10 = kernel[0, 0, 1, 0]
    k11 = kernel[0, 0, 1, 1]

    # With stride=2, each output pixel is computed from a 2x2 patch:
    out = (image[:, :, 0::2, 0::2] * k00 +
           image[:, :, 0::2, 1::2] * k01 +
           image[:, :, 1::2, 0::2] * k10 +
           image[:, :, 1::2, 1::2] * k11)
    return out


def colorize(raw_image, max_value=None, for_display=True):
    """
    Colorizes a polarization image by performing a 2x2 convolution,
    demosaicing, normalization, and gray-world white balance.

    raw_image is expected as a NumPy array.
    """
    layout = cv2.COLOR_BAYER_RGGB2RGB
    original_dtype = raw_image.dtype
    if max_value is None:
        max_value = np.iinfo(original_dtype).max
    # Ensure the image is in floating point format
    if isinstance(raw_image, np.ndarray):
        raw_image = raw_image.astype(np.float32)

    # If the image is 2D, add batch and channel dimensions to get shape (1, 1, H, W)
    if raw_image.ndim == 2:
        raw_image = np.expand_dims(np.expand_dims(raw_image, axis=0), axis=0)

    # Convolve with a 2x2 kernel of ones (using stride 2)
    kernel = np.ones((1, 1, 2, 2)) / 4
    polarization_summed = conv2x2_stride2(raw_image, kernel)

    # Remove extra dimensions (if any)
    polarization_summed = np.squeeze(polarization_summed)

    # Demosaic using OpenCV (the image is cast to uint16 for cvtColor)
    demosaiced = cv2.cvtColor(polarization_summed.astype(np.uint16), layout)

    demosaiced = cv2.resize(demosaiced, dsize=(raw_image.shape[2:][::-1]),
                            interpolation=cv2.INTER_LINEAR)

    demosaiced = demosaiced.astype(original_dtype)

    if for_display:
        if ENABLE_COLOR_CORRECTION:
            demosaiced = CC_MODEL.process(demosaiced, max_value)
        else:
            demosaiced = gray_world_white_balance(demosaiced, max_value)

    return demosaiced


def upsample_nn(x: np.ndarray, factor: int = 2) -> np.ndarray:
    """
    Performs nearest-neighbor upsampling on a 4D array (N, C, H, W)
    by repeating each element 'factor' times along the height and width.
    """

    return np.repeat(np.repeat(x, factor, axis=2), factor, axis=3)


def upsample_linear(x: np.ndarray, factor: int = 2) -> np.ndarray:
    """
    Performs bilinear upsampling on a 4D array (N, C, H, W) by a given integer factor.
    Coordinate mapping uses align_corners=False (common in DL libs): 
        src = (dst + 0.5) / factor - 0.5
    """
    assert x.ndim == 4, "Input must be (N, C, H, W)"
    assert factor >= 1 and isinstance(factor, int), "factor must be an integer >= 1"
    if factor == 1:
        return x.copy()

    N, C, H, W = x.shape
    H2, W2 = H * factor, W * factor

    # map each output coord to a (clamped) source float coord
    ys = (np.arange(H2, dtype=np.float32) + 0.5) / factor - 0.5
    xs = (np.arange(W2, dtype=np.float32) + 0.5) / factor - 0.5
    ys = np.clip(ys, 0.0, H - 1.0)
    xs = np.clip(xs, 0.0, W - 1.0)

    # integer neighbors and fractional parts
    y0 = np.floor(ys).astype(np.int32)
    x0 = np.floor(xs).astype(np.int32)
    y1 = np.minimum(y0 + 1, H - 1)
    x1 = np.minimum(x0 + 1, W - 1)

    wy = ys - y0  # shape (H2,)
    wx = xs - x0  # shape (W2,)

    # weights (H2, W2) -> broadcast to (N, C, H2, W2)
    wy0 = (1.0 - wy)[:, None]
    wy1 = wy[:, None]
    wx0 = (1.0 - wx)[None, :]
    wx1 = wx[None, :]

    wa = wy0 * wx0  # top-left
    wb = wy0 * wx1  # top-right
    wc = wy1 * wx0  # bottom-left
    wd = wy1 * wx1  # bottom-right

    # gather 4 neighbors with np.take for vectorization
    Ia = np.take(np.take(x, y0, axis=2), x0, axis=3)
    Ib = np.take(np.take(x, y0, axis=2), x1, axis=3)
    Ic = np.take(np.take(x, y1, axis=2), x0, axis=3)
    Id = np.take(np.take(x, y1, axis=2), x1, axis=3)

    # ensure float math for interpolation
    out_dtype = x.dtype if np.issubdtype(x.dtype, np.floating) else np.float32
    out = (Ia * wa[None, None, :, :] +
           Ib * wb[None, None, :, :] +
           Ic * wc[None, None, :, :] +
           Id * wd[None, None, :, :]).astype(out_dtype)

    return out


def demosaicing_polarization(img_bayer):
    """
    Demosaics a polarization Bayer image into four images corresponding to different
    polarization angles.

    If img_bayer is 2D, it is reshaped to (1, 1, H, W).
    The four angles are extracted as follows:
      - img_000: no roll applied,
      - img_045: roll applied on rows,
      - img_090: roll applied on both rows and columns,
      - img_135: roll applied on columns.

    Returns a tuple of four images (i0, i45, i90, i135) each with shape (H, W, 1).
    """
    if isinstance(img_bayer, np.ndarray):
        img_bayer = img_bayer.astype(np.float32)

    # If the image is 2D, add batch and channel dimensions
    if img_bayer.ndim == 2:
        img_bayer = np.expand_dims(np.expand_dims(img_bayer, axis=0), axis=0)  # shape (1, 1, H, W)

    # Extract sub-images by slicing and then upsample using nearest-neighbor interpolation
    img_090 = upsample_linear(img_bayer[:, :, ::2, ::2], factor=2)
    img_090 = np.roll(img_090, shift=(-1, -1), axis=(2, 3))

    img_045 = upsample_linear(img_bayer[:, :, ::2, 1::2], factor=2)
    img_045 = np.roll(img_045, shift=-1, axis=2)

    img_135 = upsample_linear(img_bayer[:, :, 1::2, ::2], factor=2)
    img_135 = np.roll(img_135, shift=-1, axis=3)

    img_000 = upsample_linear(img_bayer[:, :, 1::2, 1::2], factor=2)
    img_000 = np.roll(img_000, shift=-1, axis=3)

    # Convert from shape (1, 1, H, W) to (H, W)
    i0 = np.transpose(img_000[0], (1, 2, 0)).squeeze()
    i45 = np.transpose(img_045[0], (1, 2, 0)).squeeze()
    i90 = np.transpose(img_090[0], (1, 2, 0)).squeeze()
    i135 = np.transpose(img_135[0], (1, 2, 0)).squeeze()

    return i0, i45, i90, i135


def get_normals(s1, s2, s0, scale_factor, dolp_factor):
    """
    Computes surface normals from polarization measurements.

    Parameters:
      s1, s2, s0 : arrays (or values convertible to arrays)
      scale_factor : scales the degree-of-linear-polarization (dolp)
      dolp_factor : factor used in computing the z-component

    Returns:
      n_z  : Scaled and clipped dolp (between 0 and 1)
      n_xy : Angle in normalized degrees (0-1 range)
      n_xyz: Stacked normal vector components in the order [z, y, x]
              with shape (..., 3)
    """
    s1 = np.asarray(s1, dtype=np.float32)
    s2 = np.asarray(s2, dtype=np.float32)
    s0 = np.asarray(s0, dtype=np.float32)

    theta = 0.5 * np.arctan2(s2, -s1)
    s0[s0 == 0] = 1e-12
    dolp = np.sqrt(s1**2 + s2**2) / s0

    n_z = np.clip(dolp * scale_factor, 0.0, 1.0)

    theta_deg = ((theta * 180 / np.pi) + 90) / 180
    n_xy = theta_deg

    sin_d2 = np.sin(dolp) ** 2
    sin_t2 = np.sin(theta) ** 2

    x2 = sin_d2 * sin_t2
    y2 = sin_d2 - x2
    z2 = (1 - sin_d2) * (dolp_factor ** 2)

    norm = x2 + y2 + z2
    x2, y2, z2 = x2 / norm, y2 / norm, z2 / norm

    x = np.sqrt(x2)
    y = np.sqrt(y2)
    z = np.sqrt(z2)

    # Stack in the order [x, y, z] to mimic concatenation along a channel dimension.
    n_xyz = np.stack((x, y, z), axis=-1)

    return n_z, n_xy, n_xyz


def get_stokes(i0, i45, i90, i135):
    s0 = i0.astype(float) + i90
    s1 = i90.astype(float) - i0
    s2 = i45.astype(float) - i135

    return s0, s1, s2


def process_raw(img_bayer, scale_factor, dolp_factor):
    polarization = i0, i45, i90, i135 = demosaicing_polarization(img_bayer)

    stokes_parameters = (s0, s1, s2) = get_stokes(i0, i45, i90, i135)

    normals = n_z, n_xy, n_xyz = get_normals(s1, s2, s0,
                                             scale_factor,
                                             dolp_factor)

    return polarization, stokes_parameters, normals


def demosaicing_color(img_cpfa: np.ndarray, suffix: str = "") -> np.ndarray:
    """Color-Polarization demosaicing for np.uint8 or np.uint16 type"""
    height, width = img_cpfa.shape[:2]

    # 1. Get the correct demosaicing code from OpenCV
    code = getattr(cv2, f"COLOR_BayerBG2RGB{suffix}")

    # 2. Create the final output image
    img_mpfa_bgr = np.empty((height, width, 3), dtype=img_cpfa.dtype)

    # 3. Separate all 4 channels using vectorized slicing
    # This replaces the down-sampling part of the loop
    # (i, j) = (0, 0) -> 90 deg
    # (i, j) = (0, 1) -> 45 deg
    # (i, j) = (1, 0) -> 135 deg
    # (i, j) = (1, 1) -> 0 deg
    img_bayer_00 = img_cpfa[0::2, 0::2]
    img_bayer_01 = img_cpfa[0::2, 1::2]
    img_bayer_10 = img_cpfa[1::2, 0::2]
    img_bayer_11 = img_cpfa[1::2, 1::2]

    # 4. Demosaic each channel individually
    # cv2.cvtColor still needs to be called on each image
    bgr_00 = cv2.cvtColor(img_bayer_00, code)
    bgr_01 = cv2.cvtColor(img_bayer_01, code)
    bgr_10 = cv2.cvtColor(img_bayer_10, code)
    bgr_11 = cv2.cvtColor(img_bayer_11, code)

    # 5. Reconstruct the full image using vectorized slicing
    img_mpfa_bgr[0::2, 0::2] = bgr_00
    img_mpfa_bgr[0::2, 1::2] = bgr_01
    img_mpfa_bgr[1::2, 0::2] = bgr_10
    img_mpfa_bgr[1::2, 1::2] = bgr_11

    return cv2.split(img_mpfa_bgr)
