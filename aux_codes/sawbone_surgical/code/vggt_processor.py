import cv2
from sympy import flatten
import torch
from pathlib import Path
import numpy as np
from huggingface_hub import hf_hub_download
import shutil
import trimesh
#! ATTENTION: ADD VGGT PATH BEFOREHAND IN YOUR CODE
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

class VGGTProcessor:
    """Loads VGGT model and runs inference."""
    REPO_ID = "facebook/VGGT-1B"
    HF_FILENAME = "model.pt"
    CHECKPOINT_SUBDIR = Path(__file__).parent / "data" / "checkpoints" / "VGGT"

    def __init__(self, model_path: Path = None, device: str='cuda', skip_model: bool = False):
        self.device = device
        self.model = None
        if not skip_model:
            if model_path is None:
                model_path = self._ensure_local_checkpoint()
            self.model = VGGT().to(device).eval()
            state = torch.load(model_path, map_location=device)
            self.model.load_state_dict(state)
        
    def _ensure_local_checkpoint(self) -> Path:
        """
        Download from HF (if not already cached), resolve any internal symlinks,
        then copy to checkpoints/VGGT/model.pt so torch.load() always works.
        """
        # create the cache root
        ckpt_root = Path(__file__).parent / self.CHECKPOINT_SUBDIR
        ckpt_root.mkdir(parents=True, exist_ok=True)

        # this returns the actual downloaded file path (possibly inside HF cache tree)
        downloaded = hf_hub_download(
            repo_id   = self.REPO_ID,
            filename  = self.HF_FILENAME,
            cache_dir = str(ckpt_root),
            library_name="vggt_processor"
        )

        # resolve any symlinks, get the true blob location
        real_blob = Path(downloaded).resolve()

        # flatten into a simple file for torch.load
        flat_ckpt = ckpt_root / self.HF_FILENAME
        if not flat_ckpt.exists() or flat_ckpt.resolve() != real_blob:
            shutil.copy2(real_blob, flat_ckpt)

        return flat_ckpt
    
    def _as_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Convert a torch tensor to a numpy array, ensuring it's on CPU and float32.
        """
        if isinstance(tensor, np.ndarray):
            return tensor
        return tensor.detach().cpu().to(torch.float32).numpy()
    
    def _output_handler(self, array: torch.Tensor, squeezed: bool, as_numpy: bool):
        """
        Handle output conversion: squeeze and convert to numpy if needed.
        """
        if squeezed:
            array = array.squeeze()
        if as_numpy:
            array = self._as_numpy(array)
        return array
    
    def preprocess(self, imgs: list[Path]) -> torch.Tensor:
        """
        Turn a list of image-paths into a batched torch tensor on self.device.
        """
        img_t = load_and_preprocess_images(imgs).to(self.device)
        return img_t

    def run_model(self, img_t: torch.Tensor) -> dict:
        """
        Run the VGGT forward pass. Returns raw torch output dict.
        """
        if self.model is None:
            raise RuntimeError("VGGTProcessor was initialized with skip_model=True")
        
        # choose dtype based on device compute capability
        cap = torch.cuda.get_device_capability(self.device)[0] if "cuda" in self.device else 0
        dtype = torch.bfloat16 if cap >= 8 else torch.float16

        # cast input once
        img_t = img_t.to(dtype)

        with torch.no_grad():
            out = self.model(img_t)
        return out

    def infer(self, imgs: list[Path]) -> dict:
        """
        Shortcut: preprocess → run_model
        """
        if self.model is None:
            raise RuntimeError("VGGTProcessor was initialized with skip_model=True")
        img_t = self.preprocess(imgs)
        raw = self.run_model(img_t)
        return raw

    def print_shapes(self, preds: dict) -> None:
        """
        Print the shapes of all tensors in the preds dict.
        Useful for debugging and understanding model output.
        """
        for k, v in preds.items():
            if isinstance(v, torch.Tensor):
                print(f"{k}: {v.shape} (dtype: {v.dtype})")
            elif isinstance(v, np.ndarray):
                print(f"{k}: {v.shape} (dtype: {v.dtype})")
            else:
                print(f"{k}: {type(v)}")

    def get_depth(
        self,
        preds: dict,
        squeezed=False,
        as_numpy=False
        ) -> np.ndarray | torch.Tensor:
        """
        Extract the raw depth map from preds. [1, N, H, W, 1]
        Apply squeezing and numpy conversion if requested.
        """
        depth = preds["depth"]
        depth = self._output_handler(depth, squeezed, as_numpy)
        return depth

    def get_pose_enc(
        self,
        preds: dict,
        squeezed=False,
        as_numpy=False,
        ) -> np.ndarray | torch.Tensor:
        """
        Extract the raw pose encoding. [1, N, 9]
        Apply squeezing and numpy conversion if requested.
        """
        pose_enc = preds["pose_enc"]
        pose_enc = self._output_handler(pose_enc, squeezed, as_numpy)
        return pose_enc

    def get_camera_params(
        self, 
        preds: dict, 
        squeezed=False, 
        as_numpy=False
        ) -> tuple[np.ndarray, np.ndarray] | tuple[torch.Tensor, torch.Tensor]:
        """
        From a preds dict (either raw torch tensors or numpy arrays),
        compute (extrinsic 4×4, intrinsic 3×3) as numpy arrays.
        """
        pose_enc_t = self.get_pose_enc(preds)
        depth_t = self.get_depth(preds)
        depth_shape = tuple(depth_t.shape[2:4]) 
        
        # [1, N, 4, 4] extrinsics, [1, N, 3, 3] intrinsics
        ex_list, in_list = pose_encoding_to_extri_intri(pose_enc_t, depth_shape)
        ex_list = self._output_handler(ex_list, squeezed, as_numpy)
        in_list = self._output_handler(in_list, squeezed, as_numpy)
        
        return ex_list, in_list
    
    def get_camera_poses(
        self,
        preds: dict,
        as_numpy: bool = True
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract camera poses for all frames in the sequence.
        
        Parameters:
            preds: Prediction dictionary from VGGT model
            as_numpy: Whether to return numpy arrays (True) or torch tensors (False)
            
        Returns:
            Tuple of (positions, x_axes, y_axes, z_axes) where:
            - positions: (N, 3) array of camera positions in world coordinates
            - x_axes: (N, 3) array of camera x-axis direction vectors
            - y_axes: (N, 3) array of camera y-axis direction vectors
            - z_axes: (N, 3) array of camera z-axis direction vectors (viewing direction)
        """
        # Get extrinsics for all frames
        ex_all, _ = self.get_camera_params(preds=preds, squeezed=True, as_numpy=as_numpy)
        
        # Number of frames
        N = ex_all.shape[0] if len(ex_all.shape) > 2 else 1
        ex_all = ex_all.reshape(N, 3, 4)
        
        # Extract rotation matrices and translation vectors
        rot_matrices = ex_all[:, :3, :3]  # (N, 3, 3)
        t_vecs = ex_all[:, :3, 3]         # (N, 3)
        
        # Calculate camera positions in world coordinates
        # cam_pos = -R^T * t
        if isinstance(rot_matrices, np.ndarray):
            # Using numpy
            cam_positions = -np.einsum('nij,nj->ni', np.transpose(rot_matrices, (0, 2, 1)), t_vecs)
            
            # Camera axes (columns of R^T)
            x_axes = np.transpose(rot_matrices, (0, 2, 1))[:, :, 0]  # First column of R^T
            y_axes = np.transpose(rot_matrices, (0, 2, 1))[:, :, 1]  # Second column of R^T
            z_axes = np.transpose(rot_matrices, (0, 2, 1))[:, :, 2]  # Third column of R^T
        else:
            # Using torch
            cam_positions = -torch.bmm(torch.transpose(rot_matrices, 1, 2), 
                                    t_vecs.unsqueeze(-1)).squeeze(-1)
            
            # Camera axes (columns of R^T)
            R_t = torch.transpose(rot_matrices, 1, 2)
            x_axes = R_t[:, :, 0]
            y_axes = R_t[:, :, 1]
            z_axes = R_t[:, :, 2]
        
        return cam_positions, x_axes, y_axes, z_axes

    def get_camera_pose(
        self,
        preds: dict,
        frame_idx: int = 0,
        as_numpy: bool = True
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract camera pose for a specific frame.
        
        Parameters:
            preds: Prediction dictionary from VGGT model
            frame_idx: Index of the frame to extract pose for
            as_numpy: Whether to return numpy arrays (True) or torch tensors (False)
            
        Returns:
            Tuple of (position, x_axis, y_axis, z_axis) where:
            - position: (3,) array of camera position in world coordinates
            - x_axis: (3,) array of camera x-axis direction vector
            - y_axis: (3,) array of camera y-axis direction vector
            - z_axis: (3,) array of camera z-axis direction vector (viewing direction)
        """
        # Get extrinsics for all frames
        ex_all, _ = self.get_camera_params(preds=preds, squeezed=False, as_numpy=as_numpy)
        
        # Extract the specific frame
        ex = ex_all[:, frame_idx, :, :].reshape(3, 4)
        
        # Extract rotation matrix and translation vector
        rot_matrix = ex[:3, :3]  # (3, 3)
        t_vec = ex[:3, 3]        # (3,)
        
        # Calculate camera position in world coordinates
        # cam_pos = -R^T * t
        if isinstance(rot_matrix, np.ndarray):
            # Using numpy
            cam_position = -np.matmul(rot_matrix.T, t_vec)
            
            # Camera axes (columns of R^T)
            x_axis = rot_matrix.T[:, 0]  # First column of R^T
            y_axis = rot_matrix.T[:, 1]  # Second column of R^T
            z_axis = rot_matrix.T[:, 2]  # Third column of R^T
        else:
            # Using torch
            cam_position = -torch.matmul(rot_matrix.T, t_vec)
            
            # Camera axes (columns of R^T)
            x_axis = rot_matrix.T[:, 0]
            y_axis = rot_matrix.T[:, 1]
            z_axis = rot_matrix.T[:, 2]
        
        return cam_position, x_axis, y_axis, z_axis
    
    def get_points(
        self,
        preds: dict,
        squeezed: bool = False,
        as_numpy: bool = False,
        flatten: bool = False
        ) -> np.ndarray:
        """
        Extract the world points from preds. [1, N, H, W, 3]
        Apply squeezing and numpy conversion if requested.
        """
        pts = preds["world_points"]
        pts = self._output_handler(pts, squeezed, as_numpy)
        if flatten:
            pts = pts.reshape(-1, 3)  # Flatten to (N, 3) if requested
        return pts
    
    def get_points_conf(
        self,
        preds: dict,
        squeezed: bool = False,
        as_numpy: bool = False,
        flatten: bool = False,
        ) -> np.ndarray:
        """
        Extract the world points confidence from preds. [1, N, H, W]
        Apply squeezing and numpy conversion if requested.
        """
        conf = preds["world_points_conf"]
        conf = self._output_handler(conf, squeezed, as_numpy)
        if flatten:
            conf = conf.reshape(-1)
        return conf
    
    def get_points_colors(
        self,
        points: np.ndarray | torch.Tensor,
        imgs: np.ndarray,
        squeezed: bool = False,
        flatten: bool = False,
        as_hex = False,
        ) -> np.ndarray | torch.Tensor:
        """
        Extract the world points colors from preds. [1, N, H, W, 3]
        Apply squeezing if requested.
        Output type matches input type (torch.Tensor or np.ndarray).
        """
        # do not allow flattened point array
        if points.ndim == 2:
            raise ValueError("Passed flattened points array.")
        N_img, H_img, W_pts, _ = imgs.shape
        if points.shape[0] == 1:
            pts_shape = points.squeeze(0).shape
        else:
            pts_shape = points.shape
        N_pts, H_pts, W_pts, _ = pts_shape
        assert N_img == N_pts, "Number of images and points must match."
        
        # resize imgs to match points shape
        if (H_img, W_pts) != (H_pts, W_pts):
            resized = [cv2.resize(im, (W_pts, H_pts), interpolation=cv2.INTER_AREA)
                       for im in imgs]
            imgs = np.stack(resized, axis=0)
        
        # get colors as pts_shape
        colors = imgs.reshape(*points.shape)
        colors = torch.tensor(colors) if isinstance(points, torch.Tensor) else np.array(colors)
        if squeezed:
            colors = colors.squeeze()    
        if flatten:
            colors = colors.reshape(-1, 3)
        if as_hex:
            # Convert RGB to hex format
            colors = colors.astype(np.uint8)
            # change last dimension (3) to hex string inside the np array
            colors = np.apply_along_axis(
                lambda x: f"#{x[0]:02x}{x[1]:02x}{x[2]:02x}",
                axis=-1,
                arr=colors
            )
            
        return colors

    def preds_to_numpy(self, preds: dict) -> dict:
        """
        Convert all tensor/array outputs in preds to NumPy arrays.
        Supports:
          - torch.Tensor → float32 NumPy
          - np.ndarray    → as-is
          - list of tensors/arrays → stacked into one array
        Skips any other types.
        """
        np_preds = {}
        for k, v in preds.items():
            # 1) Single tensor → NumPy
            if isinstance(v, torch.Tensor):
                np_preds[k] = self._as_numpy(v)
            # 2) Already a NumPy array
            elif isinstance(v, np.ndarray):
                np_preds[k] = v
            # 3) List of tensors or arrays → stack
            elif isinstance(v, list):
                converted = []
                for elem in v:
                    if isinstance(elem, torch.Tensor):
                        converted.append(self._as_numpy(elem))
                    elif isinstance(elem, np.ndarray):
                        converted.append(elem)
                    else:
                        raise TypeError(f"Cannot convert element of type {type(elem)} in list for key '{k}'")
                # stack along first axis
                np_preds[k] = np.stack(converted, axis=0)
            # 4) Otherwise skip
            else:
                # you can log or silently ignore unsupported types
                continue

        return np_preds

    
    def segment_pointcloud(
        self,
        preds: dict,
        imgs: np.ndarray,
        masks: list[np.ndarray],
        return_sel: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (pts, cols) for *one* object mask-list:
          - pts:  (N,3) world points
          - cols: (N,3) uint8 RGB
        """
        # 1) grab world_points & colors
        pts = self.get_points(preds, squeezed=True, as_numpy=True)           # (T,H_d,W_d,3)
        cols = self.get_points_colors(points=pts, imgs=imgs, squeezed=True) # (T,H_d,W_d,3)

        T, H_d, W_d, _ = pts.shape
        pts_flat  = pts.reshape(-1, 3)
        cols_flat = cols.reshape(-1, 3).astype(np.uint8)

        # 2) build & resize mask stack → flatten
        m = np.stack(masks, axis=0).astype(np.uint8)  # (T, H_img, W_img)
        if (m.shape[1], m.shape[2]) != (H_d, W_d):
            m = np.stack([
                cv2.resize(frame, (W_d, H_d),
                           interpolation=cv2.INTER_NEAREST)
                for frame in m
            ], axis=0)
        sel = (m > 0).reshape(-1)
        if return_sel:
            return pts_flat, cols_flat, sel
        return pts_flat[sel], cols_flat[sel]
    


    def depth_to_point_cloud(self,
                             depth: np.ndarray | torch.Tensor,
                             extrinsic: np.ndarray,
                             intrinsic: np.ndarray,
                             squeezed: bool=False,
                             flatten: bool=False
                            ) -> np.ndarray:
        """
        Convert a depth map to a point cloud using the provided extrinsic and intrinsic matrices.
        depth: (H, W) or (1, H, W)
        extrinsic: (3, 4) or (N, 3, 4)
        intrinsic: (3, 3) or (N, 3, 3)
        Returns: (N, H*W, 3) point cloud in world coordinates.
        """
        if depth.ndim == 2:
            depth = depth[None, ..., None]  # Add batch dimension
        if extrinsic.ndim == 2:
            extrinsic = extrinsic[None]
        if intrinsic.ndim == 2:
            intrinsic = intrinsic[None]
        
        # Unproject depth map to point map
        point_map = unproject_depth_map_to_point_map(
            depth,
            extrinsic,
            intrinsic
        )
        
        if squeezed:
            point_map = point_map.squeeze()
        if flatten:
            point_map = point_map.reshape(-1, 3)
            
        return point_map  # Flatten to (N, H*W, 3)
    
    @staticmethod
    def save_bones_as_glb(femur_pts: np.ndarray,
                          tibia_pts: np.ndarray,
                          output_path: str,
                          femur_colors: np.ndarray=None,
                          tibia_colors: np.ndarray=None):
        """
        Merge femur and tibia 3D points (and optional RGB colors) into a scene,
        then export the combined scene as a binary GLTF (.glb) file using trimesh.
        """
        # Create trimesh PointCloud objects for each bone
        # If color arrays are given, ensure they are RGBA (uint8)
        femur_pc = trimesh.points.PointCloud(vertices=femur_pts, colors=None)
        tibia_pc = trimesh.points.PointCloud(vertices=tibia_pts, colors=None)
        
        if femur_colors is not None:
            # Convert (N,3) RGB to (N,4) RGBA with full opacity
            rgba = np.hstack((femur_colors.astype(np.uint8),
                              255 * np.ones((femur_colors.shape[0],1), np.uint8)))
            femur_pc = trimesh.points.PointCloud(vertices=femur_pts, colors=rgba)
        if tibia_colors is not None:
            rgba = np.hstack((tibia_colors.astype(np.uint8),
                              255 * np.ones((tibia_colors.shape[0],1), np.uint8)))
            tibia_pc = trimesh.points.PointCloud(vertices=tibia_pts, colors=rgba)

        # Create a scene and add both point clouds
        scene = trimesh.Scene()
        scene.add_geometry(femur_pc, node_name='Femur')
        scene.add_geometry(tibia_pc, node_name='Tibia')

        # Export the scene as binary GLTF (GLB); this returns bytes or writes to file
        scene.export(file_obj=output_path, file_type='glb')
        
    @staticmethod
    def save_points_as_glb(pts: np.ndarray, cols: np.ndarray, output_path: str) -> None:
        """
        Save a point cloud with colors as a GLB file.
        pts: (N,3) array of 3D points.
        cols: (N,3) array of RGB colors.
        output_path: path to save the GLB file.
        """
        rgba = np.hstack((cols.astype(np.uint8),
                          255 * np.ones((cols.shape[0],1), np.uint8)))
        pc = trimesh.points.PointCloud(vertices=pts, colors=rgba)
        scene = trimesh.Scene()
        scene.add_geometry(pc)
        scene.export(file_obj=output_path, file_type='glb')
                              
        

