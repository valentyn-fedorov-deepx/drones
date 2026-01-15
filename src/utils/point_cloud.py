import numpy as np
import open3d as o3d


def create_point_cloud_from_depth(K, depth, mask=None, color_image=None):
    """
    Creates an Open3D PointCloud object from a depth image and camera intrinsics.

    Args:
        K (np.ndarray): Camera intrinsic matrix (3x3).
        depth (np.ndarray): Depth image (HxW). Values represent depth, often in meters or mm.
        mask (np.ndarray, optional): Boolean or binary mask (HxW). If provided,
                                     only points where the mask is True are included.
                                     Defaults to None (all points are used).
        color_image (np.ndarray, optional): Color image (HxWx3) corresponding to the
                                            depth image. If provided, colors are
                                            assigned to the point cloud points.
                                            Values should ideally be RGB.
                                            Defaults to None.

    Returns:
        o3d.geometry.PointCloud: The resulting point cloud object.

    Raises:
        ValueError: If color_image is provided but its dimensions (HxW) do not match
                    the depth image, or if it's not a 3-channel image.
    """

    # Input validation for color image dimensions
    if color_image is not None:
        if color_image.shape[:2] != depth.shape[:2]:
            raise ValueError(
                f"Color image shape {color_image.shape[:2]} must match "
                f"depth image shape {depth.shape[:2]}."
            )
        if color_image.ndim != 3 or color_image.shape[2] != 3:
            raise ValueError(
                f"Color image must be an HxWx3 array, but got shape {color_image.shape}."
            )

    # Determine the pixel coordinates (ys, xs) and corresponding depth values
    if mask is not None:
        mask_bool = mask.astype(bool)
        if mask_bool.shape != depth.shape[:2]:
            raise ValueError(
                f"Mask shape {mask_bool.shape} must match "
                f"depth image shape {depth.shape[:2]}."
            )
        ys, xs = np.where(mask_bool)
        depth_values = depth[mask_bool]
    else:
        # Create coordinate arrays for all pixels when no mask is provided
        ys, xs = np.indices(depth.shape[:2])
        ys = ys.flatten()
        xs = xs.flatten()
        depth_values = depth.flatten()  # Get all depth values

    # Extract corresponding colors if color_image is provided
    colors = None
    if color_image is not None:
        colors = color_image[ys, xs]  # Shape: (N, 3)

        # Normalize colors to [0, 1] range for Open3D
        # Common case: uint8 image (0-255)
        if colors.dtype == np.uint8:
            colors = colors.astype(np.float64) / 255.0
        # Handle other cases (e.g., float images already in [0,1] or > 1)
        elif colors.max() > 1.0:
            # Simple heuristic: if max value is > 1, assume it's like 0-255
            # You might need more sophisticated normalization depending on the source
            print("Warning: Color image maximum value > 1.0. "
                  "Assuming range needs normalization (e.g., dividing by 255).")
            # Attempt normalization, checking for potential division by zero if max is 0
            max_val = colors.max()
            if max_val > 0:
                # A common case might be integer types other than uint8 scaled to 255
                if np.issubdtype(colors.dtype, np.integer):
                    colors = colors.astype(np.float64) / 255.0
                else:  # Or float types scaled higher than 1
                    colors = colors.astype(np.float64) / max_val  # Normalize by max
            else:
                colors = colors.astype(np.float64)  # Already zeros


    # Perform back-projection to 3D
    # Create homogeneous image coordinates (x, y, 1)
    im_points = np.stack([xs, ys, np.ones_like(ys)], axis=-1) # Shape: (N, 3)

    # Apply inverse intrinsics
    # Note: This specific calculation method might assume a certain coordinate system
    # or be a simplification. A common alternative involves scaling the direction
    # vector (inv(K) @ [x, y, 1]) by the depth value.
    # Using the method from the original snippet:
    inv_K = np.linalg.inv(K)
    points_3d = inv_K @ im_points.T  # Shape: (3, N)
    points_3d = points_3d.T  # Shape: (N, 3)

    # Replace the Z coordinate (calculated assuming Z=1 pre-scaling) with the actual depth
    # This step is specific to the original function's logic.
    points_3d[:, 2] = depth_values

    # Create Open3D point cloud object
    pcd = o3d.geometry.PointCloud()

    # Set the points from the numpy array
    pcd.points = o3d.utility.Vector3dVector(points_3d)

    # Set the colors if available
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd