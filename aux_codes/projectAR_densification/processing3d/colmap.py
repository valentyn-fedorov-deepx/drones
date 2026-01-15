import numpy as np

class ColmapProcessor:
    def __init__(self):
        self.model_params = {
            0: 3,  # SIMPLE_PINHOLE: [f, cx, cy]
            1: 4,  # PINHOLE: [fx, fy, cx, cy]
            2: 4,  # SIMPLE_RADIAL: [f, cx, cy, k]
            3: 5,  # RADIAL: [f, cx, cy, k1, k2]
            4: 8,  # OPENCV: [fx, fy, cx, cy, k1, k2, p1, p2]
            5: 8,  # OPENCV_FISHEYE (example; actual meaning may vary)
            6: 12, # FULL_OPENCV (example)
            7: 4,  # FOV (example)
            8: 3,  # SIMPLE_RADIAL_FISHEYE (example)
            9: 5,  # RADIAL_FISHEYE (example)
            # Add any additional models if needed.
        }
        
    def process_colmap_opencv_data(self, reconstruction_dir: str):
        """
        Reads COLMAP camera and image data from binary files and converts them to OpenCV format.
        
        Parameters:
            reconstruction_path (str): Path to the COLMAP reconstruction directory.
        """
        dist_cameras_bin = reconstruction_dir / "distorted" / "sparse" / "0" / "cameras.bin"
        cameras_bin = reconstruction_dir / "sparse" / "0" / "cameras.bin"
        images_bin = reconstruction_dir / "sparse" / "0" / "images.bin"
        
        # Read the cameras and images from the binary files.
        dist_cameras = self.read_cameras_binary(dist_cameras_bin)
        cameras = self.read_cameras_binary(cameras_bin)
        images = self.read_images_binary(images_bin)
        
        return dist_cameras, cameras, images

    def read_cameras_binary(self, path: str) -> dict:
        cameras = {}
        with open(path, "rb") as f:
            # First, read the number of cameras (stored as a uint64).
            num_cameras = np.fromfile(f, np.uint64, 1)[0]
            
            for _ in range(num_cameras):
                # Read camera ID and model (each as int32).
                camera_id = np.fromfile(f, np.int32, 1)[0]
                model = np.fromfile(f, np.int32, 1)[0]
                
                # Read image dimensions (width and height as int64).
                width = np.fromfile(f, np.int64, 1)[0]
                height = np.fromfile(f, np.int64, 1)[0]
                
                # Number of parameters depends on the camera model.
                num_params = self.model_params.get(model)
                if num_params is None:
                    raise ValueError(f"Unknown camera model: {model}")
                
                # Read the camera parameters (stored as float64).
                params = np.fromfile(f, np.float64, num_params)
                
                # Store the camera information in a dictionary.
                cameras[camera_id] = {
                    "model": model,
                    "width": width,
                    "height": height,
                    "params": params.tolist()  # convert NumPy array to list for easier handling
                }
        
        return cameras
    
    def read_images_binary(self, path: str) -> dict:
        images = {}
        with open(path, "rb") as f:
            # Read the number of images (stored as a uint64).
            num_images = int(np.fromfile(f, np.uint64, 1)[0])
            
            for _ in range(num_images):
                # Read the image ID (int32).
                image_id = int(np.fromfile(f, np.int32, 1)[0])
                # Read the quaternion (4 float64 values).
                qvec = np.fromfile(f, np.float64, 4)
                # Read the translation vector (3 float64 values).
                tvec = np.fromfile(f, np.float64, 3)
                # Read the camera ID (int32).
                camera_id = int(np.fromfile(f, np.int32, 1)[0])
                
                # Read the image name (null-terminated string).
                image_name_bytes = []
                while True:
                    c = f.read(1)
                    if c == b'\x00':
                        break
                    image_name_bytes.append(c)
                image_name = b"".join(image_name_bytes).decode("utf-8")
                
                # Read the number of 2D point observations (stored as a uint64).
                num_points = int(np.fromfile(f, np.uint64, 1)[0])
                # Each observation consists of 2 float64 values (x, y) and 1 int64 value (point3D id)
                # which amounts to 2*8 + 8 = 24 bytes per observation.
                f.seek(num_points * 24, 1)
                
                images[image_id] = {
                    "qvec": qvec,
                    "tvec": tvec,
                    "camera_id": camera_id,
                    "image_name": image_name,
                }
        
        return images
    
    def colmap_to_opencv_intrinsics(self, camera) -> tuple[np.ndarray, np.ndarray]:
        """
        Converts COLMAP camera parameters (for an OpenCV camera model) to OpenCV intrinsic matrix and distortion coefficients.
        
        Parameters:
            camera (dict): A dictionary with keys 'model', 'width', 'height', and 'params'. For an OpenCV camera (model 4),
                        the params array is expected to be:
                        [fx, fy, cx, cy, k1, k2, p1, p2]
        
        Returns:
            K (np.ndarray): The 3x3 camera intrinsic matrix.
            dist_coeffs (np.ndarray): The distortion coefficients vector.
        """
        if camera['model'] != 4:
            raise ValueError("Camera model is not OpenCV (model 4).")
        # Extract the first four parameters for intrinsic matrix
        fx, fy, cx, cy = camera['params'][:4]
        
        # Construct the camera intrinsic matrix K
        K = np.array([[fx, 0,  cx],
                    [0,  fy, cy],
                    [0,  0,  1]], dtype=np.float64)
        
        # The remaining parameters are the distortion coefficients (k1, k2, p1, p2)
        dist_coeffs = np.array(camera['params'][4:], dtype=np.float64)
        
        return K, dist_coeffs
    
    def colmap_to_pinhole_intrinsics(self, camera) -> np.ndarray:
        """
        Converts COLMAP camera parameters (for a pinhole camera model) to OpenCV intrinsic matrix.
        
        Parameters:
            camera (dict): A dictionary with keys 'model', 'width', 'height', and 'params'. For a pinhole camera (model 1),
                        the params array is expected to be:
                        [fx, fy, cx, cy]
        
        Returns:
            K (np.ndarray): The 3x3 camera intrinsic matrix.
        """
        if camera['model'] != 1:
            raise ValueError("Camera model is not Pinhole (model 1).")
        
        # Extract the parameters for intrinsic matrix
        fx, fy, cx, cy = camera['params']
        
        # Construct the camera intrinsic matrix K
        K = np.array([[fx, 0,  cx],
                    [0,  fy, cy],
                    [0,  0,  1]], dtype=np.float64)
        
        return K

    def __repr__(self):
        return f"ColmapParser()"