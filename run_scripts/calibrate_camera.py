import os
import glob

import cv2
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

from src.data_pixel_tensor import DataPixelTensor


class CameraCalibrator:
    def __init__(self, chessboard_size=None, square_size=None, mtx=None, dist=None, rvecs=None, tvecs=None):
        """
        Initializes the CameraCalibrator.

        Args:
            chessboard_size (tuple): The inner dimensions of the chessboard (width, height).
            square_size (float): The size of each square on the chessboard in meters or another unit.
        """
        self.chessboard_size = chessboard_size
        self.square_size = square_size
        self.mtx = mtx
        self.dist = dist
        self.rvecs = rvecs
        self.tvecs = tvecs
        self.objpoints = []  # 3d point in real world space
        self.imgpoints = []  # 2d points in image plane.
        self.objp = None
        self.image_size = None

    def find_chessboard_corners_from_images(self, images_path, square_size=None,resize_factor=None, criteria=None, subpix_options=None):
        """
        Finds chessboard corners in a set of images.

        Args:
            images_path (str): The path to the directory containing the calibration images (e.g., 'calibration_images/*.jpg').
        """
        if square_size is not None:
            self.square_size = square_size
        images = glob.glob(images_path)
        if not images:
            raise FileNotFoundError(f"No images found at {images_path}")

        self.objp = np.zeros((np.prod(self.chessboard_size), 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1, 2) * self.square_size
        for fname in tqdm(images, desc="Looping through images"):
            data_tensor = DataPixelTensor.from_file(fname)
            # img = cv2.imread(fname)
            img = cv2.cvtColor(data_tensor.view_img, cv2.COLOR_RGB2BGR)

            if img.dtype == 'uint16':
                img = np.uint8((img / data_tensor.max_value) * 255)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # gray = self.
            ret, corners = find_corners(gray, self.chessboard_size,
                                        criteria=criteria,
                                        resize_factor=resize_factor,
                                        subpix_options=subpix_options)
            if ret:
                self.objpoints.append(self.objp)
                self.imgpoints.append(corners)
                if self.image_size is None:
                    self.image_size = gray.shape[::-1]
            else:
                print(f"Chessboard not found in {fname}")

    def calibrate_camera(self, use_fisheye=False):
        """
        Calibrates the camera using the found chessboard corners.
        """
        if not self.objpoints or not self.imgpoints:
            raise ValueError("No chessboard corners found. Run find_chessboard_corners_from_images first.")

        if use_fisheye:
            # For fisheye calibration, we need to reshape our object points
            # Fisheye calibration requires object points to be (N,1,3) instead of (N,3)
            objpoints_fisheye = [np.reshape(obj, (obj.shape[0], 1, 3)) for obj in self.objpoints]
            imgpoints_fisheye = [np.reshape(img, (img.shape[0], 1, 2)) for img in self.imgpoints]

            # Fisheye calibration parameters
            # K is the camera matrix, D is the distortion coefficients (k1,k2,k3,k4)
            K = np.zeros((3, 3))
            D = np.zeros((4, 1))

            # Flags for fisheye calibration
            calibration_flags = (
                cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC +
                cv2.fisheye.CALIB_CHECK_COND +
                cv2.fisheye.CALIB_FIX_SKEW
            )

            ret, self.K, self.D, self.rvecs, self.tvecs = cv2.fisheye.calibrate(
                objpoints_fisheye,
                imgpoints_fisheye,
                self.image_size,
                K,
                D,
                flags=calibration_flags
            )

            # Store the results using the same attribute names for compatibility
            self.mtx = self.K
            self.dist = self.D
        else:
            ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(
                self.objpoints, self.imgpoints, self.image_size, None, None
            )
        if ret:
            print("Camera calibration successful.")
        else:
            print("Camera calibration failed.")
        return ret

    def display_reprojection_error(self):
        """
        Displays the reprojection error after camera calibration.
        """
        mean_error = 0
        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv2.projectPoints(self.objpoints[i], self.rvecs[i], self.tvecs[i], self.mtx, self.dist)
            error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error
        print(f"Total error: {mean_error / len(self.objpoints)}")

    def display_all_calibration_points(self, img=None):
        """
        Displays all the calibration points on an image.

        Args:
            img (numpy.ndarray): The image to display the calibration points on.
        """
        all_points = np.array(self.imgpoints).squeeze().reshape(-1, 2)
        x = all_points[:, 0]
        y = all_points[:, 1]
        plt.figure()
        if img is not None:
            plt.imshow(img)
        plt.plot(x, y, 'r.')
        plt.show()

    def undistort_image(self, image, get_roi=False):
        """
        Undistorts an image using the calibration parameters.

        Args:
            image (numpy.ndarray): The image to undistort.

        Returns:
            numpy.ndarray: The undistorted image.
        """
        if self.mtx is None or self.dist is None:
            raise ValueError("Camera not calibrated. Run calibrate_camera first or load the parameters from the file.")
        if get_roi:
            h, w = image.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w, h), 1, (w, h))
            dst = cv2.undistort(image, self.mtx, self.dist, None, newcameramtx)
            x, y, w, h = roi
            dst = dst[y:y + h, x:x + w]
            return dst
        else:
            return cv2.undistort(image, self.mtx, self.dist, None)

    def save_calibration_parameters(self, filename):
        """
        Saves the calibration parameters to a file.

        Args:
            filename (str): The filename to save the parameters to.
        """
        if self.mtx is None or self.dist is None:
            raise ValueError("Camera not calibrated. Run calibrate_camera first.")
        np.savez(filename, mtx=self.mtx, dist=self.dist, rvecs=self.rvecs, tvecs=self.tvecs)

    @classmethod
    def calibrate_from_folder(cls, images_path, chessboard_size, square_size=None, calibration_file=None, criteria=None, resize_factor=0.1, use_fisheye=False):
        """
        Finds chessboard corners in a set of images, calibrates the camera, and saves the calibration parameters to a file.

        Args:
            images_path (str): The path to the directory containing the calibration images (e.g., 'calibration_images/*.jpg').
            chessboard_size (tuple): The inner dimensions of the chessboard (width, height).
            square_size (float): The size of each square on the chessboard in mm or another unit.
            calibration_file (str): The filename to save the calibration parameters to.
            criteria (tuple): The criteria for refining the corners.
            resize_factor (float): The factor to resize the image by before finding the corners.
            use_fisheye (bool): Whether to use fisheye calibration.
        """

        # Create a new instance of the class
        calibrator = cls(chessboard_size=chessboard_size, square_size=square_size)
        # Find the chessboard corners in all images
        calibrator.find_chessboard_corners_from_images(images_path, resize_factor=resize_factor, criteria=criteria)

        # Calibrate the camera
        if calibrator.calibrate_camera(use_fisheye=use_fisheye):
            # Save the calibration parameters to a file if specified
            if calibration_file:
                print(f"Saving the calibration at: {calibration_file}")
                calibrator.save_calibration_parameters(calibration_file)
            else:
                print("No saving the calibration file")
            return calibrator

    @classmethod
    def load_from_file(cls, filename):
        """
        Loads calibration parameters from a file and returns a CameraCalibrator instance.

        Args:
            filename (str): The filename to load the parameters from.

        Returns:
            CameraCalibrator: A CameraCalibrator instance with loaded parameters, or None if loading fails.
        """
        if not os.path.exists(filename):
            print(f"Calibration file {filename} not found.")
            return None

        try:
            data = np.load(filename)
            return cls(
                mtx=data['mtx'],
                dist=data['dist'],
                rvecs=data['rvecs'],
                tvecs=data['tvecs']
            )
        except Exception as e:
            print(f"Error loading calibration parameters: {e}")
            return None


class Homography:
    def __init__(self, homography_matrix=None, width=None, height=None, pattern_size=None):
        """
        Initializes the Homography class.

        Args:
            homography_matrix (numpy.ndarray): The homography matrix.
        """
        self.homography_matrix = homography_matrix
        self.width = width
        self.height = height
        self.pattern_size = pattern_size
        self.object_points_2d = None
        self.corners = None

    def warp_image(self, image):
        """
        Warps an image using the homography matrix.

        Args:
            image (numpy.ndarray): The image to warp.

        Returns:
            numpy.ndarray: The warped image.
        """
        if self.homography_matrix is None:
            raise ValueError("Homography matrix not found.")
        if self.width is None or self.height is None:
            raise ValueError("Width and/or height not found.")
        warped_image = cv2.warpPerspective(image, self.homography_matrix, (self.width, self.height))
        return warped_image

    @classmethod
    def correct_image(cls, image, pattern_size, square_size, pattern_type='checkerboard'):
        """
        Corrects an image using a homography matrix.

        Args:
            image (numpy.ndarray): The image to correct.
            pattern_size (tuple): The inner dimensions of the checkerboard (width, height).
            square_size (float): The size of each square on the checkerboard in mm or another unit.

        Returns:
            numpy.ndarray: The corrected image.
        """
        # Generate the real-world 2D coordinates of the corners
        HC = cls(width=image.shape[1], height=image.shape[0], pattern_size=pattern_size)

        ret, HC.corners = find_corners(image, pattern_size,resize_factor=0.5,type=pattern_type)
        if not ret:
            print("Corners not found.")
            return None, None
        object_points_2d = generate_checkerboard_corners(pattern_size, square_size)
        # Remove the z-coordinate for 2D homography calculation
        object_points_2d = object_points_2d[:, :2]
        HC.object_points_2d = object_points_2d + np.array([HC.corners[0,0,0],HC.corners[0,0,1]])

        HC.homography_matrix,status = cv2.findHomography(HC.corners, HC.object_points_2d)

        corrected_image = HC.warp_image(image)
        return corrected_image, HC

    def save_homography_matrix(self, filename):
        """
        Saves the homography matrix to a file.

        Args:
            filename (str): The filename to save the homography matrix to.
        """
        np.savez(filename, homography_matrix=self.homography_matrix)

    @classmethod
    def load_homography_file(cls, filename):
        """
        Loads the homography matrix from a file and returns a Homography instance.

        Args:
            filename (str): The filename to load the homography matrix from.

        Returns:
            Homography: A Homography instance with the loaded matrix, or None if loading fails.
        """
        if not os.path.exists(filename):
            print(f"File {filename} not found.")
            return None

        try:
            data = np.load(filename)
            return cls(homography_matrix=data['homography_matrix'])
        except Exception as e:
            print(f"Error loading homography matrix: {e}")
            return None


def generate_checkerboard_corners(grid_pattern, square_size):
    """
    Generates the real-world 2D coordinates of the corners of a checkerboard.

    Args:
        grid_pattern (tuple): The inner dimensions of the checkerboard (width, height).
        square_size (float): The size of each square on the checkerboard in meters or another unit.
    """
    x_coords, y_coords = np.meshgrid(
        np.arange(grid_pattern[0]) * square_size,
        np.arange(grid_pattern[1]) * square_size
    )
    object_points = np.dstack((x_coords, y_coords)).reshape(-1, 2)
    return object_points


def find_corners(img, chessboard_size, resize_factor=None, criteria=None, subpix_options=None,type='checkerboard',flags=cv2.CALIB_CB_SYMMETRIC_GRID):
    """
    Finds chessboard corners in a single image.

    Args:
        img (numpy.ndarray): The image to find the chessboard corners in.
        pattern_size (tuple): The inner dimensions of the chessboard (width, height).
        resize_factor (float): The factor to resize the image by before finding the corners.
        criteria (tuple): The criteria for refining the corners.
        subpix_options (dict): The options for refining the corners. ['winSize'] - size of the search window, ['zeroZone'] - half of the size of the dead region in the middle of the search zone.
        type (str): The type of the pattern to find ('checkerboard' or 'circular').
    """
    if criteria is None:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.00001)

    if resize_factor is None:
        resize_factor = 0.1

    if subpix_options is None:
        subpix_options = {}
        subpix_options['winSize'] = (11,11) 
        subpix_options['zeroZone'] = (-1,-1)
    img_small = cv2.resize(img, (0, 0), fx = resize_factor, fy = resize_factor, interpolation= cv2.INTER_LINEAR)
    if type == 'checkerboard':
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(img_small, chessboard_size, None)
        if ret == True:
            corners2 = cv2.cornerSubPix(img, corners * (1 / resize_factor), subpix_options['winSize'], subpix_options['zeroZone'], criteria)
            return ret, corners2
        else:
            return ret, None
    elif type == 'circular':
        ret, corners = cv2.findCirclesGrid(img_small, chessboard_size, None, flags=flags)
        if ret == True:
            corners2 = corners * (1 / resize_factor)
            return ret, corners2
        else:
            return ret, None
    else:
        raise ValueError("Invalid pattern type. Use 'checkerboard' or 'circular'.")


def load_image(image_path):
    """
    Loads an image from a file.

    Args:
        image_path (str): The path to the image file.

    Returns:
        numpy.ndarray: The loaded image.
    """
    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    if len(img.shape) == 3:  # Check if it's a color image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


# Example usage:
if __name__ == "__main__":
    # Camera calibration
    chessboard_size = (9, 6)  # Inner corners of the chessboard
    square_size = 0.025  # Size of each square in meters

    # calibration_path = r'C:\Users\Kashchuk\Google Drive\Documents\Deepxhub\Work\Photo\Calibration'
    # calibration_path = r'G:\My Drive\Documents\Deepxhub\Work\Photo\Calibration'
    calibration_path = '/sdb-disk/vyzai/data/pxi_source/2025.07.22_CheckboardCamera1'
    # calibration_path = '/sdb-disk/vyzai/data/pxi_source/2025.07.11_Calibration_IDS_12mm/params'

    calibration_file = os.path.join(calibration_path,'camera_calibration.npz')
    calibration_image_files_path = os.path.join(calibration_path,'*.pxi')
    calibrator = CameraCalibrator.calibrate_from_folder(calibration_image_files_path, chessboard_size, square_size,
                                                        calibration_file)

    # Correct image

    # calibration_file = os.path.join(calibration_path,'camera_calibration.npz')
    # # load image
    # # img = load_image(r"G:\My Drive\Documents\Deepxhub\Work\Photo\20250217_145846.jpg")
    # img = load_image(r"C:\Users\Kashchuk\Google Drive\Documents\Deepxhub\Work\Photo\20250217_145846.jpg")

    # # Load camera calibration parameters
    # calibrator = CameraCalibrator.load_from_file(calibration_file)

    # # Undistort the image
    # img_undistorted = calibrator.undistort_image(img)

    # # Apply homography
    # grid_pattern = (11, 8)
    # square_size = 8.4*30
    # corrected_image, HC = Homography.correct_image(img_undistorted, grid_pattern, square_size, pattern_type='circular')
    # plt.imshow(corrected_image)
    # plt.show()

    # thresh = cv2.adaptiveThreshold(corrected_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # # Find contours in the binary image
    # contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
