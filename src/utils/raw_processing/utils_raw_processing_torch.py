import enum

import numpy as np
import torch
import torch.nn.functional as F

from configs.data_pixel_tensor import TORCH_DEVICE


def dstack(elements):
    return torch.stack(elements, dim=-1)


def change_dtype(x, target_dtype):
    target_dtype = getattr(torch, target_dtype)
    return x.to(target_dtype)


def copy_element(x):
    return x.clone()


@torch.inference_mode
def decode_mono12_contiguous_byteswapped(raw_data: bytes, width: int, height: int) -> torch.Tensor:
    """
    Decodes a contiguous 12-bit format assuming a BGR-like byte order.
    (e.g., [B_7 ... B_0][B_11 ... B_8, A_3 ... A_0][A_11 ... A_4])
    """
    expected_bytes = int(width * height * 1.5)
    if len(raw_data) != expected_bytes:
        raise ValueError("Incorrect data size.")
        
    data = torch.frombuffer(raw_data, dtype=torch.uint8)
    byte0 = data[0::3].to(torch.uint16)
    byte1 = data[1::3].to(torch.uint16)
    byte2 = data[2::3].to(torch.uint16)

    # Same logic as above, but with byte0 and byte2 swapped.
    pixel_a = ((byte2.to(torch.int32) << 4) | (byte1.to(torch.int32) >> 4)).to(torch.uint16)
    pixel_b = (((byte1.to(torch.int32) & 0x0F) << 8) | byte0.to(torch.int32)).to(torch.uint16)
    
    image = torch.empty((width * height,), dtype=torch.uint16)
    image[0::2] = pixel_b
    image[1::2] = pixel_a
    
    return image.reshape((height, width))


@torch.inference_mode
def custom_jet_colormap(tensor: torch.Tensor) -> torch.Tensor:
    # bring everything into a floating type first
    t = tensor.to(torch.float32)

    # now min/max are implemented and safe
    t_min, t_max = t.min(), t.max()
    denom = t_max - t_min
    if denom == 0:
        t_norm = torch.zeros_like(t)
    else:
        t_norm = (t - t_min) / denom

    # ramp functions
    r = torch.clamp(1.5 - torch.abs(4.0 * t_norm - 3.0), 0.0, 1.0)
    g = torch.clamp(1.5 - torch.abs(4.0 * t_norm - 2.0), 0.0, 1.0)
    b = torch.clamp(1.5 - torch.abs(4.0 * t_norm - 1.0), 0.0, 1.0)

    # restore original shape & device
    rgb = torch.stack([r, g, b], dim=-1)
    return rgb.to(tensor.dtype).to(tensor.device)

@torch.inference_mode()
def gamma_correct(image, gamma=1.5):
    """Applies a standard gamma correction for display."""
    return torch.pow(image, 1.0 / gamma)


@torch.inference_mode()
def gray_world_white_balance(img, max_value: int = 255):
    avg_R = img[:, :, 0].mean()
    avg_G = img[:, :, 1].mean()
    avg_B = img[:, :, 2].mean()

    avg_gray = (avg_R + avg_G + avg_B) / 3

    scale_R = avg_gray / avg_R
    scale_G = avg_gray / avg_G
    scale_B = avg_gray / avg_B

    final_image = img.clone()

    final_image[:, :, 0] = torch.clip(img[:, :, 0] * scale_R, 0, max_value)
    final_image[:, :, 1] = torch.clip(img[:, :, 1] * scale_G, 0, max_value)
    final_image[:, :, 2] = torch.clip(img[:, :, 2] * scale_B, 0, max_value)

    img_wb_float = final_image / max_value

    img_final = gamma_correct(torch.clip(img_wb_float, 0, 1)) * max_value

    img_final = img_final.round().to(img.dtype)

    return img_final


class Layout(enum.Enum):
    """Possible Bayer color filter array layouts.

    The value of each entry is the color index (R=0,G=1,B=2)
    within a 2x2 Bayer block.
    """

    RGGB = (0, 1, 1, 2)
    GRBG = (1, 0, 2, 1)
    GBRG = (1, 2, 0, 1)
    BGGR = (2, 1, 1, 0)


class Debayer3x3(torch.nn.Module):
    """Demosaicing of Bayer images using 3x3 convolutions.

    Compared to Debayer2x2 this method does not use upsampling.
    Instead, we identify five 3x3 interpolation kernels that
    are sufficient to reconstruct every color channel at every
    pixel location.

    We convolve the image with these 5 kernels using stride=1
    and a one pixel reflection padding. Finally, we gather
    the correct channel values for each pixel location. Todo so,
    we recognize that the Bayer pattern repeats horizontally and
    vertically every 2 pixels. Therefore, we define the correct
    index lookups for a 2x2 grid cell and then repeat to image
    dimensions.
    """

    def __init__(self, layout: Layout = Layout.RGGB):
        super(Debayer3x3, self).__init__()
        self.layout = layout
        # fmt: off
        self.kernels = torch.nn.Parameter(
            torch.tensor(
                [
                    [0, 0.25, 0],
                    [0.25, 0, 0.25],
                    [0, 0.25, 0],

                    [0.25, 0, 0.25],
                    [0, 0, 0],
                    [0.25, 0, 0.25],

                    [0, 0, 0],
                    [0.5, 0, 0.5],
                    [0, 0, 0],

                    [0, 0.5, 0],
                    [0, 0, 0],
                    [0, 0.5, 0],
                ]
            ).view(4, 1, 3, 3),
            requires_grad=False,
        )
        # fmt: on

        self.index = torch.nn.Parameter(
            self._index_from_layout(layout),
            requires_grad=False,
        )

    def forward(self, x):
        """Debayer image.

        Parameters
        ----------
        x : Bx1xHxW tensor
            Images to debayer

        Returns
        -------
        rgb : Bx3xHxW tensor
            Color images in RGB channel order.
        """
        B, C, H, W = x.shape

        xpad = torch.nn.functional.pad(x, (1, 1, 1, 1))
        c = torch.nn.functional.conv2d(xpad, self.kernels, stride=1)
        c = torch.cat((c, x), 1)  # Concat with input to give identity kernel Bx5xHxW

        rgb = torch.gather(
            c,
            1,
            self.index.repeat(
                1,
                1,
                torch.div(H, 2, rounding_mode="floor"),
                torch.div(W, 2, rounding_mode="floor"),
            ).expand(
                B, -1, -1, -1
            ),  # expand in batch is faster than repeat
        )
        return rgb

    def _index_from_layout(self, layout: Layout) -> torch.Tensor:
        """Returns a 1x3x2x2 index tensor for each color RGB in a 2x2 bayer tile.

        Note, the index corresponding to the identity kernel is 4, which will be
        correct after concatenating the convolved output with the input image.
        """
        #       ...
        # ... b g b g ...
        # ... g R G r ...
        # ... b G B g ...
        # ... g r g r ...
        #       ...
        # fmt: off
        rggb = torch.tensor(
            [
                # dest channel r
                [4, 2],  # pixel is R,G1
                [3, 1],  # pixel is G2,B
                # dest channel g
                [0, 4],  # pixel is R,G1
                [4, 0],  # pixel is G2,B
                # dest channel b
                [1, 3],  # pixel is R,G1
                [2, 4],  # pixel is G2,B
            ]
        ).view(1, 3, 2, 2)
        # fmt: on
        return {
            Layout.RGGB: rggb,
            Layout.GRBG: torch.roll(rggb, 1, -1),
            Layout.GBRG: torch.roll(rggb, 1, -2),
            Layout.BGGR: torch.roll(rggb, (1, 1), (-1, -2)),
        }.get(layout)


class Debayer2x2(torch.nn.Module):
    """Fast demosaicing of Bayer images using 2x2 convolutions.

    This method uses 3 kernels of size 2x2 and stride 2. Each kernel
    corresponds to a single color RGB. For R and B the corresponding
    value from each 2x2 Bayer block is taken according to the layout.
    For G, G1 and G2 are averaged. The resulting image has half width/
    height and is upsampled by a factor of 2.
    """

    def __init__(self, layout: Layout = Layout.RGGB):
        super(Debayer2x2, self).__init__()
        self.layout = layout

        self.kernels = torch.nn.Parameter(
            self._kernels_from_layout(layout),
            requires_grad=False,
        )

    def forward(self, x):
        """Debayer image.

        Parameters
        ----------
        x : Bx1xHxW tensor
            Images to debayer

        Returns
        -------
        rgb : Bx3xHxW tensor
            Color images in RGB channel order.
        """
        x = torch.nn.functional.conv2d(x, self.kernels, stride=2)

        x = torch.nn.functional.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=True
        )
        return x

    def _kernels_from_layout(self, layout: Layout) -> torch.Tensor:
        v = torch.tensor(layout.value).reshape(2, 2)
        r = torch.zeros(2, 2)
        r[v == 0] = 1.0

        g = torch.zeros(2, 2)
        g[v == 1] = 0.5

        b = torch.zeros(2, 2)
        b[v == 2] = 1.0

        k = torch.stack((r, g, b), 0).unsqueeze(1)  # 3x1x2x2
        return k


class DebayerSplit(torch.nn.Module):
    """Demosaicing of Bayer images using 3x3 green convolution and red,blue upsampling.
    Requires Bayer layout `Layout.RGGB`.
    """

    def __init__(self, layout: Layout = Layout.RGGB):
        super().__init__()
        if layout != Layout.RGGB:
            raise NotImplementedError("DebayerSplit only implemented for RGGB layout.")
        self.layout = layout

        self.pad = torch.nn.ReflectionPad2d(1)
        self.kernel = torch.nn.Parameter(
            torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]])[None, None] * 0.25
        )

    def forward(self, x):
        """Debayer image.

        Parameters
        ----------
        x : Bx1xHxW tensor
            Images to debayer

        Returns
        -------
        rgb : Bx3xHxW tensor
            Color images in RGB channel order.
        """
        B, _, H, W = x.shape
        red = x[:, :, ::2, ::2]
        blue = x[:, :, 1::2, 1::2]

        green = torch.nn.functional.conv2d(self.pad(x), self.kernel)
        green[:, :, ::2, 1::2] = x[:, :, ::2, 1::2]
        green[:, :, 1::2, ::2] = x[:, :, 1::2, ::2]

        return torch.cat(
            (
                torch.nn.functional.interpolate(
                    red, size=(H, W), mode="bilinear", align_corners=False
                ),
                green,
                torch.nn.functional.interpolate(
                    blue, size=(H, W), mode="bilinear", align_corners=False
                ),
            ),
            dim=1,
        )


class Debayer5x5(torch.nn.Module):
    """Demosaicing of Bayer images using Malver-He-Cutler algorithm.

    Requires BG-Bayer color filter array layout. That is,
    the image[1,1]='B', image[1,2]='G'. This corresponds
    to OpenCV naming conventions.

    Compared to Debayer2x2 this method does not use upsampling.
    Compared to Debayer3x3 the algorithm gives sharper edges and
    less chromatic effects.

    ## References
    Malvar, Henrique S., Li-wei He, and Ross Cutler.
    "High-quality linear interpolation for demosaicing of Bayer-patterned
    color images." 2004
    """

    def __init__(self, layout: Layout = Layout.RGGB):
        super(Debayer5x5, self).__init__()
        self.layout = layout
        # fmt: off
        self.kernels = torch.nn.Parameter(
            torch.tensor(
                [
                    # G at R,B locations
                    # scaled by 16
                    [ 0,  0, -2,  0,  0], # noqa
                    [ 0,  0,  4,  0,  0], # noqa
                    [-2,  4,  8,  4, -2], # noqa
                    [ 0,  0,  4,  0,  0], # noqa
                    [ 0,  0, -2,  0,  0], # noqa

                    # R,B at G in R rows
                    # scaled by 16
                    [ 0,  0,  1,  0,  0], # noqa
                    [ 0, -2,  0, -2,  0], # noqa
                    [-2,  8, 10,  8, -2], # noqa
                    [ 0, -2,  0, -2,  0], # noqa
                    [ 0,  0,  1,  0,  0], # noqa

                    # R,B at G in B rows
                    # scaled by 16
                    [ 0,  0, -2,  0,  0], # noqa
                    [ 0, -2,  8, -2,  0], # noqa
                    [ 1,  0, 10,  0,  1], # noqa
                    [ 0, -2,  8, -2,  0], # noqa
                    [ 0,  0, -2,  0,  0], # noqa

                    # R at B and B at R
                    # scaled by 16
                    [ 0,  0, -3,  0,  0], # noqa
                    [ 0,  4,  0,  4,  0], # noqa
                    [-3,  0, 12,  0, -3], # noqa
                    [ 0,  4,  0,  4,  0], # noqa
                    [ 0,  0, -3,  0,  0], # noqa

                    # R at R, B at B, G at G
                    # identity kernel not shown
                ]
            ).view(4, 1, 5, 5).float() / 16.0,
            requires_grad=False,
        )
        # fmt: on

        self.index = torch.nn.Parameter(
            # Below, note that index 4 corresponds to identity kernel
            self._index_from_layout(layout),
            requires_grad=False,
        )

    def forward(self, x):
        """Debayer image.

        Parameters
        ----------
        x : Bx1xHxW tensor
            Images to debayer

        Returns
        -------
        rgb : Bx3xHxW tensor
            Color images in RGB channel order.
        """
        B, C, H, W = x.shape

        xpad = torch.nn.functional.pad(x, (2, 2, 2, 2), mode="reflect")
        planes = torch.nn.functional.conv2d(xpad, self.kernels, stride=1)
        planes = torch.cat(
            (planes, x), 1
        )  # Concat with input to give identity kernel Bx5xHxW
        rgb = torch.gather(
            planes,
            1,
            self.index.repeat(
                1,
                1,
                torch.div(H, 2, rounding_mode="floor"),
                torch.div(W, 2, rounding_mode="floor"),
            ).expand(
                B, -1, -1, -1
            ),  # expand for singleton batch dimension is faster
        )
        return torch.clamp(rgb, 0, 1)

    def _index_from_layout(self, layout: Layout) -> torch.Tensor:
        """Returns a 1x3x2x2 index tensor for each color RGB in a 2x2 bayer tile.

        Note, the index corresponding to the identity kernel is 4, which will be
        correct after concatenating the convolved output with the input image.
        """
        #       ...
        # ... b g b g ...
        # ... g R G r ...
        # ... b G B g ...
        # ... g r g r ...
        #       ...
        # fmt: off
        rggb = torch.tensor(
            [
                # dest channel r
                [4, 1],  # pixel is R,G1
                [2, 3],  # pixel is G2,B
                # dest channel g
                [0, 4],  # pixel is R,G1
                [4, 0],  # pixel is G2,B
                # dest channel b
                [3, 2],  # pixel is R,G1
                [1, 4],  # pixel is G2,B
            ]
        ).view(1, 3, 2, 2)
        # fmt: on
        return {
            Layout.RGGB: rggb,
            Layout.GRBG: torch.roll(rggb, 1, -1),
            Layout.GBRG: torch.roll(rggb, 1, -2),
            Layout.BGGR: torch.roll(rggb, (1, 1), (-1, -2)),
        }.get(layout)


def conv2x2_stride2(
    image: torch.Tensor,
    kernel: torch.Tensor
) -> torch.Tensor:
    """
    Convolves a 4D PyTorch Tensor `image` with a 4D `kernel` (2x2) using stride=2.
    image shape: (N, 1, H, W)
    kernel shape: (1, 1, 2, 2)
      - 1 output channel,
      - 1 input channel,
      - kernel height = 2,
      - kernel width = 2.
    Returns a 4D tensor: (N, 1, outH, outW)
    """
    # Perform 2D convolution with stride=2
    # (bias=None if you don't need bias)
    out = F.conv2d(image, kernel, bias=None, stride=2)
    return out


@torch.inference_mode()
def colorize(raw_image, max_value: int = None,
             for_display: bool = True, device: str = TORCH_DEVICE):
    if isinstance(raw_image, np.ndarray):
        raw_image = torch.tensor(raw_image).to(device)

    if len(raw_image.shape) == 2:
        raw_image = raw_image.unsqueeze(0).unsqueeze(0)

    original_dtype = raw_image.dtype
    if max_value is None:
        max_value = torch.iinfo(raw_image.dtype).max

    kernel = torch.tensor([
        [1, 1],
        [1, 1]
    ], device=TORCH_DEVICE,
       dtype=raw_image.dtype).reshape((1, 1, 2, 2)) / 4

    raw_image = raw_image.to(TORCH_DEVICE)

    polarization_summed = conv2x2_stride2(raw_image.to(torch.float32), kernel)

    demosaiced = Debayer3x3().to(device)(polarization_summed)

    raw_width = raw_image.shape[-1]
    raw_height = raw_image.shape[-2]
    final_size = torch.Size([raw_height, raw_width])

    demosaiced = demosaiced
    demosaiced = F.interpolate(demosaiced, size=final_size,
                               mode='bilinear').squeeze().permute(1, 2, 0)
    if for_display:
        demosaiced = gray_world_white_balance(demosaiced, max_value)
    return demosaiced.to(original_dtype)


@torch.inference_mode()
def demosaicing_polarization(img_bayer, device=TORCH_DEVICE):
    if isinstance(img_bayer, np.ndarray):
        img_bayer = torch.tensor(img_bayer.astype(np.float32))

    if len(img_bayer.shape) == 2:
        img_bayer = img_bayer.unsqueeze(0).unsqueeze(0)
    img_bayer = img_bayer.to(device)
    original_dtype = img_bayer.dtype

    img_090 = torch.nn.functional.interpolate(img_bayer[:, :, ::2, ::2].to(float),
                                              scale_factor=2, mode="bilinear")
    img_090 = torch.roll(img_090, (-1, -1), dims=(2, 3)).to(original_dtype)

    img_045 = torch.nn.functional.interpolate(img_bayer[:, :, ::2, 1::2].to(float),
                                              scale_factor=2, mode="bilinear")
    img_045 = torch.roll(img_045, (-1,), dims=(2,)).to(original_dtype)

    img_135 = torch.nn.functional.interpolate(img_bayer[:, :, 1::2, ::2].to(float),
                                              scale_factor=2, mode="bilinear")
    img_135 = torch.roll(img_135, (-1,), dims=(3,)).to(original_dtype)

    img_000 = torch.nn.functional.interpolate(img_bayer[:, :, 1::2, 1::2].to(float),
                                              scale_factor=2, mode="bilinear")
    img_000 = torch.roll(img_000, (-1,), dims=(3,)).to(original_dtype)

    i0, i45, i90, i135 = img_000[0].permute(1, 2, 0), img_045[0].permute(1, 2, 0), img_090[0].permute(1, 2, 0), img_135[0].permute(1, 2, 0)
    return i0.squeeze(), i45.squeeze(), i90.squeeze(), i135.squeeze()


@torch.inference_mode()
def get_normals(s1, s2, s0, scale_factor, dolp_factor, device=TORCH_DEVICE):
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

    if isinstance(s1, np.ndarray):
        s1 = torch.tensor(s1)

    if isinstance(s2, np.ndarray):
        s2 = torch.tensor(s2)

    if isinstance(s0, np.ndarray):
        s0 = torch.tensor(s0)

    theta = 0.5 * torch.arctan2(s2, -s1)
    s0[s0 == 0] = 1e-12
    dolp = torch.sqrt(s1**2 + s2**2) / s0

    n_z = torch.clip(dolp * scale_factor, 0.0, 1.0)

    theta_deg = ((theta * 180 / torch.pi) + 90) / 180
    n_xy = theta_deg

    sin_d2 = torch.sin(dolp) ** 2
    sin_t2 = torch.sin(theta) ** 2

    x2 = sin_d2 * sin_t2
    y2 = sin_d2 - x2
    z2 = (1 - sin_d2) * (dolp_factor ** 2)

    norm = x2 + y2 + z2
    x2, y2, z2 = x2 / norm, y2 / norm, z2 / norm

    x = torch.sqrt(x2)
    y = torch.sqrt(y2)
    z = torch.sqrt(z2)

    # Stack in the order [x, y, z] to mimic concatenation along a channel dimension.
    n_xyz = torch.stack((x, y, z), dim=-1)

    return n_z, n_xy, n_xyz


@torch.inference_mode()
def get_stokes(i0, i45, i90, i135):
    s0 = i0.to(float) + i90
    s1 = i90.to(float) - i0
    s2 = i45.to(float) - i135

    return s0, s1, s2


@torch.inference_mode()
def process_raw(img_bayer, scale_factor, dolp_factor):
    i0, i45, i90, i135 = demosaicing_polarization(img_bayer)
    # polarization = (i0.cpu().numpy(), i45.cpu().numpy(), i90.cpu().numpy(), i135.cpu().numpy())
    polarization = (i0, i45, i90, i135)

    # stokes_parameters = (s0.cpu().numpy(), s1.cpu().numpy(), s2.cpu().numpy())
    stokes_parameters = (s0, s1, s2) = get_stokes(i0, i45, i90, i135)

    n_z, n_xy, n_xyz = get_normals(s1, s2, s0,
                                   scale_factor, dolp_factor)

    # normals = (n_z.cpu().numpy(), n_xy.cpu().numpy(), n_xyz.cpu().numpy())
    normals = (n_z, n_xy, n_xyz)

    return polarization, stokes_parameters, normals


@torch.inference_mode()
def demosaicing_color(img_cpfa, suffix: str = ""):
    """Color-Polarization demosaicing for uint8 or uint16 type"""
    if isinstance(img_cpfa, np.ndarray):
        img_cpfa = torch.tensor(img_cpfa).to(TORCH_DEVICE)

    height, width = img_cpfa.shape[:2]

    # 2. Create the final output image
    img_mpfa_bgr = torch.zeros((height, width, 3), dtype=img_cpfa.dtype)

    # 3. Separate all 4 channels using vectorized slicing
    # This replaces the down-sampling part of the loop
    # (i, j) = (0, 0) -> 90 deg
    # (i, j) = (0, 1) -> 45 deg
    # (i, j) = (1, 0) -> 135 deg
    # (i, j) = (1, 1) -> 0 deg
    img_bayer_00 = img_cpfa[0::2, 0::2].to(torch.float32).to(TORCH_DEVICE)
    img_bayer_01 = img_cpfa[0::2, 1::2].to(torch.float32).to(TORCH_DEVICE)
    img_bayer_10 = img_cpfa[1::2, 0::2].to(torch.float32).to(TORCH_DEVICE)
    img_bayer_11 = img_cpfa[1::2, 1::2].to(torch.float32).to(TORCH_DEVICE)

    # 4. Demosaic each channel individually
    bgr_00 = Debayer3x3(Layout.RGGB).to(TORCH_DEVICE)(img_bayer_00.unsqueeze(0).unsqueeze(0)).squeeze().permute(1, 2, 0)
    bgr_01 = Debayer3x3(Layout.RGGB).to(TORCH_DEVICE)(img_bayer_01.unsqueeze(0).unsqueeze(0)).squeeze().permute(1, 2, 0)
    bgr_10 = Debayer3x3(Layout.RGGB).to(TORCH_DEVICE)(img_bayer_10.unsqueeze(0).unsqueeze(0)).squeeze().permute(1, 2, 0)
    bgr_11 = Debayer3x3(Layout.RGGB).to(TORCH_DEVICE)(img_bayer_11.unsqueeze(0).unsqueeze(0)).squeeze().permute(1, 2, 0)

    # 5. Reconstruct the full image using vectorized slicing
    # This replaces the up-sampling part of the loop
    img_mpfa_bgr[0::2, 0::2] = bgr_00
    img_mpfa_bgr[0::2, 1::2] = bgr_01
    img_mpfa_bgr[1::2, 0::2] = bgr_10
    img_mpfa_bgr[1::2, 1::2] = bgr_11

    return torch.unbind(img_mpfa_bgr, dim=-1)
