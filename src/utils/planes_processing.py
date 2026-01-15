import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN
import plotly.graph_objects as go
import cv2


def visualize_principal_axes_on_image(
    image: np.ndarray,
    pcd: o3d.geometry.PointCloud,
    K: np.ndarray,
    axis_length_scale: float = 0.1,
    axis_thickness: int = 2
):
    """
    Visualize the principal axes of a 3D point cloud on the original RGB image.
    
    :param image:  The original color image as a NumPy array (H x W x 3, dtype=uint8).
    :param pcd:    Open3D PointCloud in the *camera coordinate system*.
    :param K:      The 3x3 intrinsic camera matrix. 
                   E.g. [[fx,  0, cx],
                         [ 0, fy, cy],
                         [ 0,  0,  1]]
    :param axis_length_scale: Factor to control how long to draw the principal axes, 
                              relative to the point cloud size.
    :param axis_thickness: Thickness (in pixels) for drawing the axes lines on the image.
    :return: A copy of the image with the principal axes drawn.
    """
    # 1) Convert Open3D PointCloud to a NumPy array
    points = np.asarray(pcd.points)  # shape: (N, 3)
    if points.shape[0] < 3:
        raise ValueError("Need at least 3 points in the cloud to compute principal axes.")

    # 2) Compute the centroid
    centroid = np.mean(points, axis=0)  # shape: (3,)

    # 3) Center the points
    centered_pts = points - centroid

    # 4) Compute covariance and do SVD => principal components
    #    Cov = Q^T Q, or we can do SVD of Q directly
    Cov = centered_pts.T @ centered_pts  # shape: (3,3)
    U, S, Vt = np.linalg.svd(Cov)       # Cov = U * diag(S) * V^T
    # Columns of V = principal directions; 
    # but here we get them from V^T. So the rows of Vt are the principal directions.
    # The largest singular value => first row of Vt, second largest => second row, etc.
    principal_axes = Vt  # shape: (3, 3). Each row is an axis direction.

    # Let's order them by descending variance if needed:
    # but typically np.linalg.svd returns them in descending order of S already.
    # principal_axes[0,:] => direction of largest variance
    # principal_axes[1,:] => second largest
    # principal_axes[2,:] => third

    # 5) Decide how far to draw each axis. 
    #    We'll base it on the sqrt of the largest eigenvalue or bounding box dimension
    largest_singular_val = np.sqrt(S[0])  # a measure of the scale
    axis_length = axis_length_scale * largest_singular_val

    # 6) Project lines for each principal axis onto the image
    #    We'll draw from centroid -> centroid + axis_length * principal_axes[i]
    #    For clarity, let's do +/- direction if you like, but one direction is often enough.

    # We copy the original image so we don't overwrite it
    img_out = image.copy()

    # We'll pick a distinct color for each axis:
    axis_colors = [(0, 0, 255),    # Red
                   (0, 255, 0),    # Green
                   (255, 0, 0)]    # Blue

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    def project_3d_to_2d(X, Y, Z):
        # Simple pinhole projection
        if Z <= 0:
            return None  # behind camera or invalid
        u = (fx * X) / 1 + cx
        v = (fy * Y) / 1 + cy
        return (int(round(u)), int(round(v)))

    for i in range(3):
        axis_dir = principal_axes[i, :]  # shape: (3,)
        color = axis_colors[i % len(axis_colors)]

        # Start: centroid
        start_3D = centroid - (axis_length * axis_dir)

        # End: centroid + axis_length * direction
        end_3D = centroid + (axis_length * axis_dir)

        # Project them to 2D
        start_2D = project_3d_to_2d(start_3D[0], start_3D[1], start_3D[2])
        end_2D = project_3d_to_2d(end_3D[0], end_3D[1], end_3D[2])

        if start_2D is not None and end_2D is not None:
            # Draw a line or arrow on img_out
            cv2.arrowedLine(
                img_out,
                start_2D, end_2D,
                color,
                thickness=axis_thickness,
                tipLength=0.05  # fraction of the arrow length
            )

            cv2.arrowedLine(
                img_out,
                end_2D, start_2D,
                color,
                thickness=axis_thickness,
                tipLength=0.05  # fraction of the arrow length
            )

    return img_out


def visualize_point_clouds(point_clouds, marker_size=2):
    """
    Visualize multiple point clouds in different colors.

    Args:
        point_clouds (list): List of numpy arrays, each with shape (N_points, 3)
        marker_size (float): Size of markers in the visualization

    Returns:
        None (displays interactive plot)
    """
    # Define some distinct colors for different point clouds
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow']

    # Create figure
    fig = go.Figure()

    # Add each point cloud as a separate scatter3d trace
    for idx, points in enumerate(point_clouds):
        # Ensure points is numpy array with correct shape
        points = np.asarray(points)
        if points.shape[1] != 3:
            raise ValueError(f"Point cloud {idx} has incorrect shape. Expected (N, 3), got {points.shape}")

        # Add scatter3d trace
        fig.add_trace(go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=marker_size,
                color=colors[idx % len(colors)],  # Cycle through colors if more point clouds than colors
            ),
            name=f'Cloud {idx}'  # Label in legend
        ))

    # Update layout for better visualization
    fig.update_layout(
        scene=dict(
            aspectmode='data',  # Preserve aspect ratio
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z')
        ),
        showlegend=True
    )

    fig.show()


def cluster_plane_inliers(inliers_cloud: o3d.geometry.PointCloud,
                         inliers: np.ndarray,
                         eps: float = 0.02,
                         min_points: int = 10) -> tuple[np.ndarray, o3d.geometry.PointCloud]:
    """
    Cluster plane inliers using sklearn's DBSCAN and return the largest cluster.
    
    Args:
        inliers_cloud: Point cloud containing only the inlier points
        inliers: Original indices of inlier points in the full cloud
        eps: DBSCAN epsilon parameter (maximum distance between points in a cluster)
        min_points: Minimum number of points to form a cluster
        
    Returns:
        tuple containing:
        - indices of largest cluster in original point cloud
        - point cloud containing only the largest cluster points
    """
    # Convert Open3D point cloud to numpy array for sklearn
    points = np.asarray(inliers_cloud.points)
    
    # Run DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_points).fit(points)
    labels = clustering.labels_
    
    # Handle case where no clusters were found
    if len(np.unique(labels[labels >= 0])) == 0:
        return inliers, inliers_cloud
        
    # Find the largest cluster
    unique_labels = np.unique(labels)
    largest_cluster_label = max(unique_labels, key=lambda x: np.sum(labels == x) if x >= 0 else 0)
    
    # Get indices of points in largest cluster
    largest_cluster_indices = np.where(labels == largest_cluster_label)[0]
    
    # Convert cluster indices to original point cloud indices
    # Ensure inliers is a numpy array
    inliers_array = np.asarray(inliers)
    largest_cluster_global_idx = inliers_array[largest_cluster_indices]
    
    # Create point cloud with only largest cluster points
    largest_cluster_cloud = inliers_cloud.select_by_index(largest_cluster_indices)
    
    return largest_cluster_global_idx, largest_cluster_cloud


def plane_normals_from_parameters(planes):
    """
    Given a list of planes from sequential_ransac_plane_segmentation, 
    where each entry is (plane_model, inlier_cloud) and plane_model = [A, B, C, D],
    return a list of unit normal vectors, one for each plane.

    :param planes: list of (plane_model, inlier_cloud),
                   where plane_model is [A, B, C, D].
    :return: list of numpy arrays (3,) representing normalized normals.
    """
    normals = []
    for plane_model, _ in planes:
        A, B, C, D = plane_model
        # Unnormalized normal
        n = np.array([A, B, C], dtype=float)
        # Compute length
        length = np.linalg.norm(n)
        if length < 1e-9:
            # Degenerate or invalid plane; produce a zero normal
            n_unit = np.zeros(3, dtype=float)
        else:
            n_unit = n / length
        normals.append(n_unit)
    return normals


def plane_fit_metric(plane_model, pcd):
    """
    Compute the average distance of points in 'pcd' to the plane given by plane_model=[A,B,C,D].
    Returns the mean distance (float). If no points assigned, return None or 0.
    """
    points = np.asarray(pcd.points)
    n_points = points.shape[0]
    if n_points == 0:
        return None  # or 0.0, depending on how you want to handle empty assignments

    A, B, C, D = plane_model
    norm_len = np.sqrt(A**2 + B**2 + C**2)
    if norm_len < 1e-9:
        # Degenerate plane, treat as infinite error or skip
        return float("inf")

    # distances = |(A*x + B*y + C*z + D)| / norm_len
    numerator = np.abs(A*points[:,0] + B*points[:,1] + C*points[:,2] + D)
    distances = numerator / norm_len
    mean_dist = np.mean(distances)
    return mean_dist


def sort_planes_by_fit(planes):
    """
    Sort planes by how well they fit their assigned points (ascending mean distance).
    Each element in 'planes' is (plane_model, assigned_pcd).
    Returns a new list of (plane_model, assigned_pcd, mean_distance).
    """
    plane_entries = []
    for (plane_model, assigned_cloud) in planes:
        mdist = plane_fit_metric(plane_model, assigned_cloud)
        plane_entries.append((plane_model, assigned_cloud, mdist))

    # Filter out None or inf if you want
    plane_entries_valid = [p for p in plane_entries if p[2] is not None and np.isfinite(p[2])]

    # Sort by mean distance ascending
    plane_entries_valid.sort(key=lambda x: x[2])

    return plane_entries_valid


def merge_similar_planes(planes, angle_threshold_degs=5, offset_threshold=0.01):
    """
    Merge planes that are within angle_threshold_degs and offset_threshold of each other.
    Return a new list of merged planes.
    """

    # 1) Preprocess planes => get unit normals and offsets
    planes_info = []  # we'll store (unit_normal, offset, index)
    for i, (model, assigned_cloud) in enumerate(planes):
        A,B,C,D = model
        norm_len = np.sqrt(A*A + B*B + C*C)
        if norm_len < 1e-9:
            norm_len = 1e-9
        # unit normal
        n = np.array([A,B,C]) / norm_len
        # offset
        d = D / norm_len
        planes_info.append((n, d, i))

    # We'll do a union-find or adjacency-based grouping
    # Convert angle_threshold_degs to radians
    angle_threshold = np.radians(angle_threshold_degs)

    n_planes = len(planes_info)
    parent = np.arange(n_planes)  # union-find parent

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx = find(x)
        ry = find(y)
        if rx != ry:
            parent[ry] = rx

    # 2) Compare every pair
    for i in range(n_planes):
        n_i, d_i, idx_i = planes_info[i]
        for j in range(i+1, n_planes):
            n_j, d_j, idx_j = planes_info[j]

            # angle between normals
            dot_ij = n_i.dot(n_j)
            dot_ij = np.clip(dot_ij, -1.0, 1.0)  # numerical safety
            angle_ij = np.arccos(abs(dot_ij))   # abs => handle parallel or antiparallel
            if angle_ij <= angle_threshold:
                # plane offset difference
                # if dot_ij >= 0 => same direction => check |d_i - d_j|
                # if dot_ij < 0  => opposite => check |d_i + d_j|
                if dot_ij >= 0:
                    offset_diff = abs(d_i - d_j)
                else:
                    offset_diff = abs(d_i + d_j)

                if offset_diff <= offset_threshold:
                    # Merge these two planes
                    union(i, j)

    # 3) group by find(x)
    clusters = {}
    for i in range(n_planes):
        r = find(i)
        if r not in clusters:
            clusters[r] = []
        clusters[r].append(i)

    merged_planes = []
    # 4) For each cluster, combine assigned points & refit or pick a rep
    for r, idxs in clusters.items():
        # collect all points from the planes in idxs
        all_points = []
        for i_idx in idxs:
            plane_idx_in_original_list = planes_info[i_idx][2]
            _, assigned_pcd = planes[plane_idx_in_original_list]
            all_points.append(np.asarray(assigned_pcd.points))
        if len(all_points) == 0:
            continue
        big_points = np.concatenate(all_points, axis=0)

        if big_points.shape[0] < 3:
            # not enough points to define a plane => skip
            continue

        # Option A: refit plane by SVD/least-squares
        # For brevity, just pick the first plane as the "representative"
        # Or do:
        # best_plane_model = fit_plane_svd(big_points)
        representative_index = idxs[0]
        plane_idx_in_original_list = planes_info[representative_index][2]
        rep_plane_model, _ = planes[plane_idx_in_original_list]

        # Construct new merged plane entry
        merged_pcd = o3d.geometry.PointCloud()
        merged_pcd.points = o3d.utility.Vector3dVector(big_points)
        merged_planes.append((rep_plane_model, merged_pcd))

    return merged_planes


def assign_points_to_planes_vectorized(
    original_pcd: o3d.geometry.PointCloud,
    planes: list,
    assignment_threshold: float = 0.01
):
    """
    Vectorized assignment of each point in 'original_pcd' to the plane with the smallest distance,
    if that distance <= assignment_threshold. Otherwise the point is unassigned (-1).

    :param original_pcd:       Full point cloud in original coords (N points).
    :param planes:             List of (plane_model, inlier_cloud) where plane_model = [A,B,C,D].
    :param assignment_threshold: Max allowed distance to assign a point to a plane.
    :return: List of (plane_model, assigned_pcd) for each plane, in original coords.
    """
    if not planes:
        return []

    # Extract plane models [A,B,C,D] => separate arrays
    plane_models = [p[0] for p in planes]  # M planes
    plane_coefs = np.array([m[:3] for m in plane_models])  # shape (M,3)
    plane_offsets = np.array([m[3] for m in plane_models]) # shape (M,)

    # Norms for each plane normal
    plane_norms = np.linalg.norm(plane_coefs, axis=1)
    plane_norms[plane_norms < 1e-9] = 1e-9  # avoid zero division

    points = np.asarray(original_pcd.points) # shape (N,3)
    n_points = points.shape[0]
    m_planes = plane_coefs.shape[0]

    if n_points == 0 or m_planes == 0:
        return [(pm, o3d.geometry.PointCloud()) for (pm, _) in planes]

    # 1) Dot products: shape => (N,M)
    #    We do (N,3) dot (3,M) by transposing plane_coefs to shape (3,M).
    dot_vals = points @ plane_coefs.T  # shape (N,M)

    # 2) Add offsets
    # Broadcasting offset: (N,M) + (M,) => (N,M)
    dist_numerators = dot_vals + plane_offsets  # shape (N,M)

    # 3) Absolute value
    dist_numerators = np.abs(dist_numerators)

    # 4) Divide by plane_norms => broadcast (N,M) / (M,) => (N,M)
    distances = dist_numerators / plane_norms

    # 5) For each point, find the plane with min distance
    best_plane_idx = np.argmin(distances, axis=1)       # shape (N,) -> plane index
    best_plane_dist = np.min(distances, axis=1)         # shape (N,) -> distance value

    # 6) Check threshold
    # any point with best_plane_dist > assignment_threshold is unassigned
    unassigned_mask = best_plane_dist > assignment_threshold
    # We'll set those plane indices to -1
    best_plane_idx[unassigned_mask] = -1

    # Now group point indices by plane
    plane_assigned_indices = [[] for _ in range(m_planes)]
    for i in range(n_points):
        idx = best_plane_idx[i]
        if idx >= 0:
            plane_assigned_indices[idx].append(i)

    # Build the assigned clouds
    final_planes = []
    for i, (plane_model, _) in enumerate(planes):
        assigned_indices = plane_assigned_indices[i]
        if len(assigned_indices) == 0:
            assigned_pcd = o3d.geometry.PointCloud()
        else:
            assigned_pcd = original_pcd.select_by_index(assigned_indices)
        final_planes.append((plane_model, assigned_pcd))

    return final_planes


def normalize_point_cloud(pcd: o3d.geometry.PointCloud):
    """
    Normalize 'pcd' so that its axis-aligned bounding box diagonal ~ 1.0.
    Returns: (pcd_normalized, center, diag_len)
    """
    pcd_normalized = o3d.geometry.PointCloud(pcd)

    aabb = pcd_normalized.get_axis_aligned_bounding_box()
    min_bound = aabb.min_bound
    max_bound = aabb.max_bound
    center = aabb.get_center()
    diag_vec = max_bound - min_bound
    diag_len = np.linalg.norm(diag_vec)

    if diag_len < 1e-9:
        diag_len = 1.0

    # Translate so center is (0,0,0)
    pcd_normalized.translate(-center)
    # Scale so diagonal ~ 1
    pcd_normalized.scale(1.0/diag_len, center=(0, 0, 0))

    return pcd_normalized, center, diag_len


def denormalize_plane_model(plane_model, center, diag):
    """
    Convert plane [a', b', c', d'] in normalized coords to [A, B, C, D] in original coords.
    """
    a_prime, b_prime, c_prime, d_prime = plane_model

    # plane in normalized coords: a'( (x - cx)/diag ) + b'( ... ) + c'( ... ) + d' = 0
    # => A = a'/diag, B = b'/diag, C = c'/diag
    #    D = d' - (a'*cx + b'*cy + c'*cz)/diag
    A = a_prime / diag
    B = b_prime / diag
    C = c_prime / diag
    shift_term = - (a_prime*center[0] + b_prime*center[1] + c_prime*center[2]) / diag
    D = d_prime + shift_term

    return [A, B, C, D]


def denormalize_inlier_cloud(inlier_cloud_normalized, center, diag):
    """
    Convert an inlier cloud from normalized -> original coordinates.
    """
    cloud_original = o3d.geometry.PointCloud(inlier_cloud_normalized)
    cloud_original.scale(diag, center=(0, 0, 0))
    cloud_original.translate(center)
    return cloud_original


# ---------------------------------------------------------------------------
# Subsampling Utility
# ---------------------------------------------------------------------------
def subsample_point_cloud_random(pcd: o3d.geometry.PointCloud, max_points: int) -> o3d.geometry.PointCloud:
    """
    Randomly subsample the point cloud so it does not exceed 'max_points' points.
    """
    n_points = np.asarray(pcd.points).shape[0]
    if n_points <= max_points:
        return pcd  # no need to subsample

    indices = np.arange(n_points)
    np.random.shuffle(indices)
    chosen_indices = indices[:max_points]
    return pcd.select_by_index(chosen_indices)


# ---------------------------------------------------------------------------
# Assign All Original Points to the "Best-Fitting" Plane
# ---------------------------------------------------------------------------
def assign_points_to_planes(
    original_pcd: o3d.geometry.PointCloud,
    planes: list,
    assignment_threshold: float = 0.01
):
    """
    For each point in 'original_pcd', compute its distance to each plane in 'planes'
    and assign it to the plane with the smallest distance, if that distance <= assignment_threshold.

    :param original_pcd:      The FULL (unsubsampled) point cloud in original coords
    :param planes:            List of (plane_model, inlier_cloud), where plane_model is [A,B,C,D]
    :param assignment_threshold: Max distance for a point to be considered on that plane
    :return: new list of (plane_model, assigned_pcd)
             - assigned_pcd is an Open3D PointCloud of all points assigned to that plane
    """
    if not planes:
        return []

    # Extract plane models
    plane_models = [p[0] for p in planes]

    points = np.asarray(original_pcd.points)  # shape: (N, 3)
    n_points = points.shape[0]

    # We'll keep an array "assign_idx" of shape (N,) with the plane index each point belongs to
    # or -1 if not assigned
    assign_idx = np.full((n_points,), -1, dtype=np.int32)

    # We'll precompute plane normal magnitudes for distance denominator
    plane_norms = []
    for (A, B, C, D) in plane_models:
        norm_len = np.sqrt(A**2 + B**2 + C**2)
        plane_norms.append(norm_len if norm_len > 1e-9 else 1e-9)

    # For each point, find the plane with minimal distance
    for i in range(n_points):
        x, y, z = points[i]
        best_plane = -1
        best_dist = float("inf")

        # Check each plane
        for plane_i, (A, B, C, D) in enumerate(plane_models):
            dist = abs(A*x + B*y + C*z + D) / plane_norms[plane_i]
            if dist < best_dist:
                best_dist = dist
                best_plane = plane_i

        # If best_dist <= threshold, assign
        if best_dist <= assignment_threshold:
            assign_idx[i] = best_plane

    # Now group indices by plane
    plane_assigned_indices = [[] for _ in range(len(planes))]
    for i in range(n_points):
        pidx = assign_idx[i]
        if pidx >= 0:
            plane_assigned_indices[pidx].append(i)

    # Build the assigned clouds
    new_planes = []
    for plane_i, (plane_model, _) in enumerate(planes):
        assigned_inds = plane_assigned_indices[plane_i]
        if len(assigned_inds) == 0:
            # no points assigned
            assigned_pcd = o3d.geometry.PointCloud()
        else:
            assigned_pcd = original_pcd.select_by_index(assigned_inds)
        new_planes.append((plane_model, assigned_pcd))

    return new_planes


# ---------------------------------------------------------------------------
# Main: Sequential RANSAC + Final Reassignment
# ---------------------------------------------------------------------------
def sequential_ransac_plane_segmentation(
    pcd: o3d.geometry.PointCloud,
    distance_threshold: float = 0.01,
    ransac_n: int = 3,
    num_iterations: int = 1000,
    min_inliers: int = 1000,
    normalize: bool = True,
    subsample: bool = False,
    max_points: int = 10000,
    cluster_inliers: bool = False,
    dbscan_eps: float = 0.02,
    dbscan_min_points: int = 10,
    final_assignment_threshold: float = 0.01,
    merge_planes: bool = False,
    reassign_points: bool = False
):
    """
    Perform sequential RANSAC plane detection, then reassign ALL original points
    to whichever plane they fit best (if within 'final_assignment_threshold').

    Returns a list of (plane_model, assigned_cloud) in original coords.

    :param pcd:   Full, original-scale point cloud (will be used later for final assignment).
    :param distance_threshold, ransac_n, num_iterations, min_inliers:
        RANSAC parameters (in normalized coords if 'normalize' is True).
    :param normalize:   Whether to normalize the cloud (scale invariance).
    :param subsample:   Whether to random-subsample the cloud to 'max_points' for RANSAC.
    :param cluster_inliers:  If True, optionally cluster inliers and remove only the largest cluster.
    :param dbscan_eps, dbscan_min_points: Clustering parameters for DBSCAN.
    :param final_assignment_threshold:  Distance threshold for final assignment of each original point to planes.
    """
    # Save reference to the full original cloud
    original_pcd = pcd

    # ----------------------------------------
    # 1. (Optional) Subsample for RANSAC
    # ----------------------------------------
    if subsample:
        pcd = subsample_point_cloud_random(pcd, max_points)

    # ----------------------------------------
    # 2. (Optional) Normalize
    # ----------------------------------------
    if normalize:
        pcd_norm, center, scale = normalize_point_cloud(pcd)
        remaining_cloud = pcd_norm
    else:
        center = np.array([0, 0, 0], dtype=np.float64)
        scale = 1.0
        remaining_cloud = pcd

    planes_in_original_coords = []

    # ----------------------------------------
    # 3. Sequentially detect planes
    # ----------------------------------------
    while True:
        if len(remaining_cloud.points) < min_inliers:
            print("[SequentialRANSAC] Not enough points left to detect further planes.")
            break

        plane_model_norm, inliers = remaining_cloud.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations
        )

        inlier_count = len(inliers)
        if inlier_count < min_inliers:
            print(f"[SequentialRANSAC] Found a plane with only {inlier_count} inliers; stopping.")
            break

        print(f"[SequentialRANSAC] Detected plane {plane_model_norm} with {inlier_count} inliers.")

        inliers_cloud_norm = remaining_cloud.select_by_index(inliers)

        # ----------------------------------------
        # (Optional) DBSCAN clustering among inliers
        # ----------------------------------------
        if cluster_inliers:
            print("Running clustering for detected plane")
            biggest_cluster_global_idx, inliers_cloud_norm = cluster_plane_inliers(
                inliers_cloud_norm,
                inliers,
                eps=dbscan_eps,
                min_points=dbscan_min_points
            )
            
            # Remove only biggest cluster
            remaining_cloud = remaining_cloud.select_by_index(biggest_cluster_global_idx, invert=True)
            inlier_count = len(inliers_cloud_norm.points)
            print(f"[Clustering] Largest cluster has {inlier_count} points.")
        else:
            # Remove all inliers
            remaining_cloud = remaining_cloud.select_by_index(inliers, invert=True)

        # ----------------------------------------
        # 4. Denormalize plane + inliers
        # ----------------------------------------
        if normalize:
            plane_model_original = denormalize_plane_model(plane_model_norm, center, scale)
            inliers_cloud_original = denormalize_inlier_cloud(inliers_cloud_norm, center, scale)
        else:
            plane_model_original = plane_model_norm
            inliers_cloud_original = inliers_cloud_norm

        planes_in_original_coords.append((plane_model_original, inliers_cloud_original))

    if merge_planes:
        # 2) Merge similar planes
        print("[SequentialRANSAC] Merging similar planes.")
        merged_planes = merge_similar_planes(planes_in_original_coords,
                                             angle_threshold_degs=15,
                                             offset_threshold=0.05)
 
    # ----------------------------------------
    # 5. Reassign ALL original points
    # ----------------------------------------
    if reassign_points:
        print("[SequentialRANSAC] Reassigning points.")
        final_planes = assign_points_to_planes_vectorized(
            original_pcd,
            merged_planes,
            assignment_threshold=final_assignment_threshold
        )

    return final_planes
