# refactored_params.py

from dataclasses import dataclass, field
from typing import Tuple, Optional, Union
import numpy as np
import cv2
import warnings
from processing3d.transforms import get_camera_rotation


@dataclass
class CameraParams:
    """
    Hold intrinsic & distortion parameters and provide image undistortion.
    """
    K_pinhole: np.ndarray                   # shape (3,3)
    K_distorted: np.ndarray                 # shape (3,3)
    dist_coeffs: np.ndarray                 # shape (5,) or (1,5)
    dist_shape: Tuple[int, int]             # (height, width) of distorted image
    final_shape: Tuple[int, int]            # (height, width) of undistorted output

    _map: Tuple[np.ndarray, np.ndarray] = field(init=False, repr=False)

    def __post_init__(self):
        # ensure arrays
        self.K_pinhole = np.asarray(self.K_pinhole, dtype=float)
        self.K_distorted = np.asarray(self.K_distorted, dtype=float)
        self.dist_coeffs = np.asarray(self.dist_coeffs, dtype=float)

        self._validate_shapes()
        # Precompute the undistort/remap once
        self._map = cv2.initUndistortRectifyMap(
            self.K_distorted,
            self.dist_coeffs,
            None,
            self.K_pinhole,
            self.final_shape,
            cv2.CV_32FC1
        )

    def _validate_shapes(self):
        if self.K_pinhole.shape != (3, 3):
            raise ValueError("K_pinhole must be shape (3,3)")
        if self.K_distorted.shape != (3, 3):
            raise ValueError("K_distorted must be shape (3,3)")
        if self.dist_coeffs.ndim != 1 or self.dist_coeffs.size not in (4,5,8):
            raise ValueError("dist_coeffs must be a 1D array of length 4, 5, or 8")
        if len(self.dist_shape) != 2 or len(self.final_shape) != 2:
            raise ValueError("dist_shape and final_shape must be (height, width) tuples")

    def undistort(self, image: np.ndarray) -> np.ndarray:
        """
        Undistort the given image using the stored camera parameters.
        """
        map_x, map_y = self._map
        return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)


@dataclass
class ImageParams:
    """
    Hold extrinsic parameters and provide 3D→2D projection & visibility utilities.
    """
    image_id: int
    image_name: str
    center_world: Union[np.ndarray, Tuple[float, float, float]]    # shape (3,)
    K_pinhole: Union[np.ndarray, Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]]  # shape (3,3)
    qvec: Union[np.ndarray, Tuple[float, float, float, float]]      # shape (4,) [w, x, y, z]
    tvec: Union[np.ndarray, Tuple[float, float, float]]             # shape (3,)
    rpy_deg: Union[np.ndarray, Tuple[float, float, float]]         # shape (3,)

    def __post_init__(self):
        # convert to arrays
        self.center_world = np.asarray(self.center_world, dtype=float)
        self.K_pinhole = np.asarray(self.K_pinhole, dtype=float)
        self.qvec = np.asarray(self.qvec, dtype=float)
        self.tvec = np.asarray(self.tvec, dtype=float)
        self.rpy_deg = np.asarray(self.rpy_deg, dtype=float)

        # shape checks
        if self.center_world.shape != (3,):
            raise ValueError("center_world must be shape (3,)")
        if self.K_pinhole.shape != (3, 3):
            raise ValueError("K_pinhole must be shape (3,3)")
        if self.qvec.shape != (4,):
            raise ValueError("qvec must have 4 elements")
        if self.tvec.shape != (3,):
            raise ValueError("tvec must be shape (3,)")
        if self.rpy_deg.shape != (3,):
            raise ValueError("rpy_deg must be shape (3,)")

    @property
    def R(self) -> np.ndarray:
        """
        Rotation matrix from world→camera using the stored quaternion.
        """
        return get_camera_rotation(self.qvec)

    def project(self, points: Union[np.ndarray, list]) -> np.ndarray:
        """
        Projects 3D world points into 2D pixel coordinates (u,v).
        Points behind the camera are set to NaN.
        """
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        if pts.shape[1] != 3:
            raise ValueError("Input points must have shape (N,3) or (3,)")

        # World→camera
        pts_cam = pts @ self.R.T + self.tvec
        z = pts_cam[:, 2]

        if np.any(z <= 0):
            warnings.warn("Some points are behind the camera; their projections will be NaN")

        uv = np.full((pts.shape[0], 2), np.nan, dtype=np.float64)
        valid = z > 0
        if np.any(valid):
            x_norm = pts_cam[valid, 0] / z[valid]
            y_norm = pts_cam[valid, 1] / z[valid]
            fx = self.K_pinhole[0, 0]
            fy = self.K_pinhole[1, 1]
            cx = self.K_pinhole[0, 2]
            cy = self.K_pinhole[1, 2]
            uv[valid, 0] = fx * x_norm + cx
            uv[valid, 1] = fy * y_norm + cy

        return uv


@dataclass
class ReconstructionPoint:
    """
    A single 3D point with an ID and color.
    """
    id: int
    world_coords: np.ndarray    # shape (3,)
    color: np.ndarray           # shape (3,)


@dataclass
class ProjPoint:
    """
    A single 2D projection record.
    """
    id: int
    point_id: int
    image_id: int
    prev_image_id: Optional[int]
    image_name: str
    img_coords: np.ndarray      # shape (2,) (u, v)
    Nz_rgb: np.ndarray          # e.g. unnormalized grayscale
    Nxy_rgb: np.ndarray
    Nxyz_rgb: np.ndarray


def filter_visible_uv(
    uv: np.ndarray,
    image_shape: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Keep only points whose UV fall inside [0,width)x[0,height),
    and whose coordinates are not NaN.
    Returns (uv_visible, mask_boolean).
    """
    h, w = image_shape
    mask = (
        (~np.isnan(uv).any(axis=1)) &
        (uv[:, 0] >= 0) & (uv[:, 0] < w) &
        (uv[:, 1] >= 0) & (uv[:, 1] < h)
    )
    return uv[mask], mask


def drop_duplicates_from_lists(
    proj_points: np.ndarray,
    points3d: np.ndarray,
    center_world: np.ndarray,
    leave_closest: bool = True,
    include_mask: bool = False
) -> Union[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray]
]:
    """
    Remove duplicate pixel coords by (rounded) UV. For each duplicate,
    keep the 3D point closest to center_world if leave_closest=True.
    """
    # Round and integerize UV to group
    keys = np.round(proj_points).astype(int)
    # unique rows + inverse mapping
    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)

    best_indices = []
    for k in range(len(unique_keys)):
        idxs = np.where(inverse == k)[0]
        if leave_closest:
            dists = np.linalg.norm(points3d[idxs] - center_world, axis=1)
            best = idxs[np.argmin(dists)]
        else:
            best = idxs[0]
        best_indices.append(best)
    best_indices = np.array(best_indices, dtype=int)

    uv_unique = proj_points[best_indices]
    pts_unique = points3d[best_indices]

    if include_mask:
        mask = np.zeros(len(proj_points), dtype=bool)
        mask[best_indices] = True
        return uv_unique, pts_unique, mask

    return uv_unique, pts_unique
