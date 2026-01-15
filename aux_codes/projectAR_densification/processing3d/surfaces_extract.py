import numpy as np


def project_to_plane(v, n):
    """
    Project a vector to a plane defined by its normal.
    
    :param v: vector to project
    :param n: normal vector of the plane
    
    :return: projected vector
    """
    n = n / np.linalg.norm(n)
    return v - np.dot(v, n) * n

# use PCA to choose other plane vectors
def get_xy_plane_vectors(normal: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Get plane vectors from the normal and points.
    
    :param normal: normal vector of the plane
    :param points: numpy array of 3D points
    
    :return: list of two orthogonal vectors in the plane
    """
    cov = np.cov(points.T)
    eigs, vecs = np.linalg.eigh(cov)
    order = np.argsort(eigs)[::-1]
    pc1, pc2 = vecs[:, order[0]], vecs[:, order[1]]
    
    axis_x = project_to_plane(pc1, normal)
    axis_x = axis_x / np.linalg.norm(axis_x)
    
    axis_y = project_to_plane(pc2, normal)
    axis_y = axis_y/ np.linalg.norm(axis_y)
    
    # enforce right‐handedness: x × y should align with normal
    if np.dot(np.cross(axis_x, axis_y), normal) < 0:
        axis_y = -axis_y
        
    return axis_x, axis_y

