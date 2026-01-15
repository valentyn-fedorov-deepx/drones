import numpy as np
from scipy.spatial.transform import Rotation as R

def qvec_to_scipy(qvec: np.ndarray) -> np.ndarray:
    # Convert COLMAP quaternion format [q_w, q_x, q_y, q_z] to SciPy format [q_x, q_y, q_z, q_w]
    return np.array([qvec[1], qvec[2], qvec[3], qvec[0]])  # [x, y, z, w]

def get_camera_rotation(qvec: np.ndarray) -> np.ndarray:
    # Get rotation matrix from quaternion (converting from world-to-camera)
    R_mat = R.from_quat(qvec_to_scipy(qvec)).as_matrix()
    return R_mat

def get_camera_center(qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    # Get rotation matrix from quaternion (converting from world-to-camera)
    R_mat = get_camera_rotation(qvec)
    
    # Camera center in world coordinates
    center = - R_mat.T @ tvec  
    return center

def get_camera_axes(qvec: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Invert the rotation to get camera orientation in world coordinates
    R_world = get_camera_rotation(qvec).T  
    
    # Camera axes in world coordinates
    x_axis = R_world[:, 0]
    y_axis = R_world[:, 1]
    z_axis = R_world[:, 2]
    return x_axis, y_axis, z_axis

def get_camera_rpy(qvec: np.ndarray) -> tuple[float, float, float]:
    # Using 'yxz' order: Pitch around Y, Roll around X, Yaw around Z
    roll, pitch, yaw = R.from_quat(qvec_to_scipy(qvec)).as_euler('xyz', degrees=True)
    return roll, pitch, yaw

def apply_manual_rot(points: np.ndarray, x_deg: float, y_deg: float, z_deg: float) -> np.ndarray:
    """
    Apply manual rotation to a set of 3D points using Euler angles (in degrees).
    
    Parameters:
        points (np.ndarray): An array of shape (N, 3) representing N points in 3D space.
        x_deg (float): Rotation angle around the X-axis in degrees.
        y_deg (float): Rotation angle around the Y-axis in degrees.
        z_deg (float): Rotation angle around the Z-axis in degrees.
    
    Returns:
        np.ndarray: The rotated points as an array of shape (N, 3).
    """
    # Convert degrees to radians
    x_rad = np.radians(x_deg)
    y_rad = np.radians(y_deg)
    z_rad = np.radians(z_deg)

    # Create rotation matrices for each axis
    Rx = R.from_euler('x', x_rad).as_matrix()
    Ry = R.from_euler('y', y_rad).as_matrix()
    Rz = R.from_euler('z', z_rad).as_matrix()

    # Combined rotation matrix
    R_combined = Rz @ Ry @ Rx

    # Apply rotation
    rotated_points = points @ R_combined.T
    return rotated_points

def compute_camera_rpy_in_basis(
    camera_axes: tuple[np.ndarray, np.ndarray, np.ndarray],
    new_basis: tuple[np.ndarray, np.ndarray, np.ndarray],
    euler_seq: str = 'xyz'
) -> tuple[float, float, float]:
    """
    Given the camera's X, Y, Z unit‐axes (in world coords) and a new orthonormal basis
    (also in world coords), return the camera's roll, pitch, and yaw *relative* to that basis.

    Parameters:
        camera_axes (tuple): A tuple of three numpy arrays representing the camera's X, Y, Z axes.
        new_basis (tuple): A tuple of three numpy arrays representing the new orthonormal basis.
        euler_seq (str): The sequence of rotations to apply. Default is 'xyz'.
    
    Returns:
        tuple: A tuple containing the roll, pitch, and yaw angles in degrees.
        
    """
    
    # normalize the camera axes
    normal_cam_axes = np.array(camera_axes)
    normal_cam_axes = normal_cam_axes / np.linalg.norm(normal_cam_axes, axis=1)[:, np.newaxis]
    x_cam, y_cam, z_cam = normal_cam_axes
    
    # normalize the new basis
    normal_new_basis = np.array(new_basis)
    normal_new_basis = normal_new_basis / np.linalg.norm(normal_new_basis, axis=1)[:, np.newaxis]
    b1, b2, b3 = normal_new_basis
    # assert orthonormality - sum of dot products should be 0
    dot_sum = np.dot(b1, b2) + np.dot(b1, b3) + np.dot(b2, b3)
    assert np.isclose(dot_sum, 0), f"New basis vectors are not orthonormal: {dot_sum}"
    
    # build 3×3 rotation matrices whose columns are the axis vectors
    cam_to_world   = np.column_stack((x_cam, y_cam, z_cam))
    basis_to_world = np.column_stack((b1, b2, b3))

    # compute the camera frame in the new basis:
    rel_rotation = basis_to_world.T @ cam_to_world

    # convert relative rotation into Euler angles
    rotation_obj = R.from_matrix(rel_rotation)
    angles = rotation_obj.as_euler(euler_seq, degrees=True)
    
    return angles


