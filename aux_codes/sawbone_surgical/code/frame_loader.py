import cv2
import numpy as np
from pathlib import Path

from src.data_pixel_tensor import DataPixelTensor


class FrameLoader:
    """Loads and saves frames from PXI files."""
    def __init__(self, bit_depth: int = 8, color_data: bool = True):
        self.bit_depth = bit_depth
        self.color_data = color_data
    
    def load_bgr_frompxi(self, path: Path) -> np.ndarray:
        dt = DataPixelTensor.from_file(str(path), color_data_main_channel=0,
                                       color_data=self.color_data, bit_depth=self.bit_depth,
                                       for_display=True)
        if self.color_data:
            dt.color_data = True
            dt._cache.pop('view_img', None)
        img = dt.view_img.astype(np.float32) / (2**self.bit_depth - 1) * 255
        img = img.astype(np.uint8)
        if img.ndim==3 and img.shape[2]==3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    
    def load_nxyz_bgr_frompxi(self, path: Path) -> np.ndarray:
        dt = DataPixelTensor.from_file(str(path), color_data_main_channel=0,
                                       color_data=self.color_data, bit_depth=self.bit_depth,
                                       for_display=True)
        if self.color_data:
            dt.color_data = True
            dt._cache.pop('view_img', None)
        img = dt.n_xyz.astype(np.float32) / (2**self.bit_depth - 1) * 255
        img = img.astype(np.uint8)
        if img.ndim==3 and img.shape[2]==3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def select_frames(self, files: list[Path], start: int, end: int, count: int) -> list[Path]:
        subset = files[start:end]
        step = max(1, (len(subset)-1)//(count-1))
        return [subset[i*step] for i in range(count-1)] + [subset[-1]]

    def save_frames(
        self,
        paths: list[Path],
        out_dir: Path, extension=".jpg",
        scale_factor: float = 1.,
        brightening_factor: float = 1.,
        view="view_img",
        ) -> list[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        outs = []
        for i,p in enumerate(paths):
            if view == "nxyz":
                img = self.load_nxyz_bgr_frompxi(p)
            else:
                img = self.load_bgr_frompxi(p)
            if scale_factor != 1.:
                img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
            if brightening_factor != 1.:
                img = cv2.convertScaleAbs(img, alpha=brightening_factor, beta=0)
            op = out_dir/f"{i:05d}{extension}"
            cv2.imwrite(str(op), img)
            outs.append(op)
        return outs