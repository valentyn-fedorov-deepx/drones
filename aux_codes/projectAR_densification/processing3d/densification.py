import numpy as np
import cv2
from sklearn.cluster import MeanShift, estimate_bandwidth
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from datawraps.pcd_classes import ProjPoint

def cluster_colors(
    image_nxyz: np.ndarray,
    sample_size: int = 5000,
    qt: float = 0.1
) -> tuple[np.ndarray, list[np.ndarray], dict[int, np.ndarray]]:
    """Cluster the colors in the given image using MeanShift with specified hyperparameters.
        Args:
            image_nxyz (np.ndarray): The input image in NXYZ format.
            sample_size (int): The number of pixels to sample for bandwidth estimation.
            qt (float): The quantile for bandwidth estimation.
        
        Returns:
            tuple: A tuple containing:
                - labels (np.ndarray): The cluster labels for each pixel in the image.
                - centers_rgb (list[np.ndarray]): The RGB values of the cluster centers.
                - cluster_masks (dict[int, np.ndarray]): A dictionary mapping cluster labels to binary masks.
    """
    h, w, _ = image_nxyz.shape

    # convert image to LAB color space and flatten
    lab = cv2.cvtColor(image_nxyz, cv2.COLOR_RGB2LAB)
    pixels = lab.reshape(-1, 3)

    # sample n pixels from the image, estimate bandwidth
    n_pixels = pixels.shape[0]
    idx = np.random.choice(n_pixels, min(sample_size, n_pixels), replace=False)
    sample = pixels[idx]
    bw = estimate_bandwidth(sample, quantile=qt, n_samples=sample.shape[0])

    # fit MeanShift and predict labels
    ms = MeanShift(bandwidth=bw, bin_seeding=True)
    ms.fit(sample)
    labels_flat = ms.predict(pixels)
    labels = labels_flat.reshape(h, w)
    centers_lab = ms.cluster_centers_.astype(np.uint8)

    # convert cluster centers to RGB
    centers_rgb = [
        cv2.cvtColor(c.reshape(1, 1, 3), cv2.COLOR_LAB2RGB)[0, 0]
        for c in centers_lab
    ]

    # get the cluster labels and masks
    cluster_masks = {
        k: (labels == k).astype(np.uint8) * 255
        for k in range(len(centers_rgb))
    }

    return labels, centers_rgb, cluster_masks



def visualize_clusters(
    centers_rgb: list[np.ndarray],
    cluster_masks: dict[int, np.ndarray]
) -> plt.Figure:
    """
    Visualize the clusters and their spatial distribution using matplotlib. Returns the figure object for display.
    
    Args:
        centers_rgb (list[np.ndarray]): The RGB values of the cluster centers.
        cluster_masks (dict[int, np.ndarray]): A dictionary mapping cluster labels to binary masks.
        
    Returns:
        plt.Figure: The figure object containing the visualized clusters.
    """
    n_clusters = len(centers_rgb)
    n_cols = 4
    n_rows = int(np.ceil(n_clusters / n_cols))

    # plot grid
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 6 * n_rows))
    axes = axes.flatten()

    for k in range(n_clusters):
        # show color patch
        patch = np.zeros((50, 50, 3), dtype=np.uint8)
        patch[:] = centers_rgb[k]
        axes[k].imshow(patch)
        axes[k].axis('off')
        axes[k].set_title(f"Cluster {k}\nRGB={centers_rgb[k]}")

        # overlay the mask below the patch + small inset axis for the mask
        mask = cluster_masks[k]
        inset = axes[k].inset_axes([0.1, -0.6, 0.8, 0.5])
        inset.imshow(mask, cmap='gray')
        inset.axis('off')

    # hide any extra subplots
    for ax in axes[n_clusters:]:
        ax.axis('off')

    plt.tight_layout()
    return fig


def densify_points(
    proj_pts: np.ndarray,     # (N,2) pixel coords
    pts3d: np.ndarray,        # (N,3) 3D points
    image_colors: np.ndarray, # (H,W,3) per‐pixel colors
    mask: np.ndarray,         # (H,W) binary mask of pixels to fill
    k_neighbors: int = 3,     # number of neighbors for interp
    distance_thresh: float = None,  # max 2D distance (pixels)
    max_3d_gap: float  = None       # max 3D distance among neighbors
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Densify sparse 3D points by interpolating new points within a mask region.
     
    Args:
        proj_pts (np.ndarray): Projected pixel coordinates of the 3D points.
        pts3d (np.ndarray): 3D points corresponding to the projected pixel coordinates.
        image_colors (np.ndarray): Per-pixel colors of the image.
        mask (np.ndarray): Binary mask indicating the region to fill.
        k_neighbors (int): Number of neighbors to consider for interpolation.
        distance_thresh (float): Maximum 2D distance for interpolation.
        max_3d_gap (float): Maximum 3D distance among neighbors for interpolation.
        
    Returns:
        tuple: A tuple containing:
            - proj_pts (np.ndarray): Updated projected pixel coordinates.
            - pts3d (np.ndarray): Updated 3D points.
            - cols (np.ndarray): Updated colors corresponding to the projected pixel coordinates.
    """
    
    # 1) collect all mask pixels
    ys, xs = np.nonzero(mask > 0)
    targets = np.stack([xs, ys], axis=1)  # (M,2)

    # 2) drop pixels we already have
    existing = set(map(tuple, proj_pts.tolist()))
    keep_new = [tuple(p) not in existing for p in targets]
    targets = targets[keep_new]
    if len(targets) == 0:
        cols = image_colors[proj_pts[:,1], proj_pts[:,0]]
        return proj_pts, pts3d, cols

    # 3) find at most k_neigh neighbors k nearest
    tree = cKDTree(proj_pts)
    k_eff = min(k_neighbors, proj_pts.shape[0])
    dists, idxs = tree.query(targets, k=k_eff)
    # if k_eff==1 you get 1d arrays; make them 2D for consistency
    if k_eff == 1:
        dists = dists[:,None]
        idxs  = idxs[:,None]

    # 4) optional: filter by 2D distance
    if distance_thresh is not None:
        ok = dists[:,0] <= distance_thresh
        targets, dists, idxs = targets[ok], dists[ok], idxs[ok]
        if len(targets) == 0:
            cols = image_colors[proj_pts[:,1], proj_pts[:,0]]
            return proj_pts, pts3d, cols

    # 5) optional: filter by 3D spread
    if max_3d_gap is not None:
        neigh3d = pts3d[idxs] # (M,k,3)
        ref3d   = neigh3d[:,0] # nearest neighbor
        diff3d  = np.linalg.norm(neigh3d - ref3d[:,None,:], axis=-1)  # (M,k)
        ok3d    = np.max(diff3d, axis=1) <= max_3d_gap
        targets, dists, idxs = targets[ok3d], dists[ok3d], idxs[ok3d]
        if len(targets) == 0:
            cols = image_colors[proj_pts[:,1], proj_pts[:,0]]
            return proj_pts, pts3d, cols

    # 6) compute inverse‐distance weights
    eps = 1e-6
    w = 1.0 / (dists + eps)
    w /= w.sum(axis=1, keepdims=True) # normalize

    # 7) interpolate new 3D points
    neigh_pts = pts3d[idxs] # (M,k,3)
    new_pts3d = np.einsum('mk,mkd->md', w, neigh_pts)

    # 8) get colors at new pixels
    new_cols = image_colors[targets[:,1], targets[:,0]]

    # 9) join original + new
    all_proj  = np.vstack([proj_pts,   targets])
    all_pts3d = np.vstack([pts3d,      new_pts3d])
    orig_cols = image_colors[proj_pts[:,1], proj_pts[:,0]]
    all_cols  = np.vstack([orig_cols, new_cols])

    return all_proj, all_pts3d, all_cols


def extract_cluster_data(
    mask: np.ndarray,
    image_proj_points: dict[str, ProjPoint],
    points: np.ndarray,
    image_nxyz: np.ndarray,
    image_idx: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract original projected pixel coords, 3D points, and colors for a cluster.
    
    Args:
        mask (np.ndarray): Binary mask of the cluster.
        image_proj_points (dict): Dictionary mapping image IDs to projected points.
        points (np.ndarray): 3D points corresponding to the projected pixel coordinates.
        image_nxyz (np.ndarray): Per-pixel colors of the image.
        image_idx (int): Index of the image being processed.
    
    Returns:
        tuple: A tuple containing:
            - proj_pts (np.ndarray): Projected pixel coordinates of the 3D points.
            - pts3d (np.ndarray): 3D points corresponding to the projected pixel coordinates.
            - colors (np.ndarray): Colors corresponding to the projected pixel coordinates.    
    """
    proj_pts = []
    pts3d = []
    colors = []
    # iterate over all projected points
    for pp in image_proj_points.values():
        x, y = pp.img_coords
        if pp.image_id == image_idx and mask[y, x] > 0:
            # check if the projected point is within the mask
            proj_pts.append((x, y))
            pts3d.append(points[pp.point_id])
            colors.append(image_nxyz[y, x])
    
    # convert to numpy arrays
    proj_pts = np.array(proj_pts, dtype=int)
    pts3d    = np.array(pts3d, dtype=float)
    colors   = np.array(colors, dtype=np.uint8)
    return proj_pts, pts3d, colors


def process_cluster(
    k: int,
    mask: np.ndarray,
    image_proj_points: dict,
    points: np.ndarray,
    image_nxyz: np.ndarray,
    image_idx: int,
    k_neighbors: int = 4,
    distance_thresh: int = 5,
    max_3d_gap: float = 0.3,
    verbose: bool = False
) -> dict:
    """
    Process a single cluster: extract data, densify points, and combine results.
    
    Args:
        k (int): Cluster ID.
        mask (np.ndarray): Binary mask of the cluster.
        image_proj_points (dict): Dictionary mapping image IDs to projected points.
        points (np.ndarray): 3D points array.
        image_nxyz (np.ndarray): Per-pixel colors of the image.
        image_idx (int): Index of the image being processed.
        k_neighbors (int): Number of neighbors for interpolation.
        distance_thresh (int): Maximum 2D distance for interpolation.
        max_3d_gap (float): Maximum 3D distance among neighbors for interpolation.
        
    Returns:
        dict: A dictionary containing:
            - orig_points_3d: Original 3D points.
            - orig_colors: Original colors.
            - new_points_3d: New 3D points after densification.
            - new_proj_points: Projected pixel coordinates of the new points.
            - new_colors: Colors of the new points.
            - all_points_3d: Combined original and new 3D points.
            - all_colors: Combined original and new colors.
    """
    proj_pts, pts3d, colors = extract_cluster_data(mask, image_proj_points, points, image_nxyz, image_idx)

    # if there are points in the cluster, densify them 
    if proj_pts.size:
        new_px, new_3d, new_cols = densify_points(
            proj_pts,
            pts3d,
            image_colors=image_nxyz,
            mask=(mask > 0),
            k_neighbors=k_neighbors,
            distance_thresh=distance_thresh,
            max_3d_gap=max_3d_gap
        )
    else: # if no points in the cluster, create empty arrays
        new_px   = np.empty((0,2), dtype=int)
        new_3d   = np.empty((0,3), dtype=float)
        new_cols = np.empty((0,3), dtype=np.uint8)

    # get all points and from the densification
    all_pts3d  = np.vstack([pts3d,   new_3d])   if new_3d.size else pts3d
    all_colors = np.vstack([colors, new_cols])  if new_cols.size else colors
    if verbose:
        print(f"Cluster {k}: +{new_3d.shape[0]} new points (≤{distance_thresh}px), total {all_pts3d.shape[0]} points")

    return {
        'orig_points_3d': pts3d,
        'orig_colors':   colors,
        'new_points_3d': new_3d,
        'new_proj_points': new_px,
        'new_colors':    new_cols,
        'all_points_3d': all_pts3d,
        'all_colors':    all_colors
    }


def process_all_clusters(
    cluster_masks: dict[int, np.ndarray],
    image_proj_points: dict,
    points: np.ndarray,
    image_nxyz: np.ndarray,
    image_idx: int,
    k_neighbors: int = 4,
    distance_thresh: int = 5,
    max_3d_gap: float = 0.3,
    verbose: bool = False
) -> dict[int, dict]:
    """Process all clusters and return dense data per cluster."""
    cluster_dense = {}
    for k, mask in cluster_masks.items():
        if np.sum(mask > 0) == 0:
            if verbose:
                print(f"Cluster {k}: empty mask")
            continue
        cluster_dense[k] = process_cluster(
            k,
            mask,
            image_proj_points,
            points,
            image_nxyz,
            image_idx,
            k_neighbors,
            distance_thresh,
            max_3d_gap,
            verbose=verbose
        )
    return cluster_dense
