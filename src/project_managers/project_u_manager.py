import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import seed_everything 

from scipy import sparse
import open3d as o3d

from src.cv_module.height_map import estimate_height_map
from src.cv_module.lino_model import LiNo_UniPS
from src.cv_module.lino_model.inference_data import InferenceData


class ProjectUManager:
    def __init__(self, weights_path="weights/lino.pth"):
        self.model = self.load_model_with_weights(weights_path)

    def get_normal_map(self, image_folder, name_card="*.png"):
        testdata = self.get_dataset(image_folder, numofimages=4, name_card=name_card)

        test_loader = DataLoader(testdata, batch_size=1)
        trainer = pl.Trainer(accelerator="auto", devices=1,precision="bf16-mixed")
        normals = trainer.predict(model=self.model, dataloaders=test_loader)
        self.normals = normals[0]
        return self.normals
    
    def load_model_with_weights(self, model_weights_path):
        state_dict = torch.load(model_weights_path, weights_only=False, map_location=torch.device("cpu"))
        model = LiNo_UniPS()
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model
    
    def get_dataset(self, image_folder, numofimages=4, name_card="*.png"):
        inference_data = InferenceData(image_folder, numofimages, name_card)
        return inference_data
    
    def save_normals_map(self, normals_map, save_folder_path):
        # Normalize normals map to [0, 255] for saving as an image
        np.save(os.path.join(save_folder_path,'normals.npy'),normals_map)

        normals_map_normalized = ( ( (normals_map + 1) / 2) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(save_folder_path,'normals.png'), cv2.cvtColor(normals_map_normalized, cv2.COLOR_RGB2BGR) )
    
    def compute_depth(self,mask,N):
        """
        compute the depth picture
        """
        Nin = N.copy()
        N = np.zeros_like(Nin, dtype=np.float32)
        N[:, :, 0] = Nin[:, :, 2]
        N[:, :, 1] = Nin[:, :, 1]
        N[:, :, 2] = Nin[:, :, 0]
        im_h, im_w = mask.shape
        N = np.reshape(N, (im_h, im_w, 3))

        # =================get the non-zero index of mask=================
        obj_h, obj_w = np.where(mask != 0)
        no_pix = np.size(obj_h) #37244
        full2obj = np.zeros((im_h, im_w))
        for idx in range(np.size(obj_h)):
            full2obj[obj_h[idx], obj_w[idx]] = idx
        full2obj = full2obj.astype(int)
        M = sparse.lil_matrix((2*no_pix, no_pix))
        v = np.zeros((2*no_pix, 1))

        # ================= fill the M&V =================
        for idx in range(no_pix):
            # obtain the 2D coordinate
            h = obj_h[idx]
            w = obj_w[idx]
            # obtian the surface normal vector
            n_x = N[h, w, 0]
            n_y = N[h, w, 1]
            n_z = N[h, w, 2]
            
            row_idx = idx * 2
            if mask[h, w+1]:
                idx_horiz = full2obj[h, w+1]
                M[row_idx, idx] = -1
                M[row_idx, idx_horiz] = 1
                if n_z==0:
                    v[row_idx] = 0
                else:
                    v[row_idx] = -n_x / n_z
            elif mask[h, w-1]:
                idx_horiz = full2obj[h, w-1]
                M[row_idx, idx_horiz] = -1
                M[row_idx, idx] = 1
                if n_z==0:
                    v[row_idx] = 0
                else:
                    v[row_idx] = -n_x / n_z

            row_idx = idx * 2 + 1
            if mask[h+1, w]:
                idx_vert = full2obj[h+1, w]
                M[row_idx, idx] = 1
                M[row_idx, idx_vert] = -1
                if n_z==0:
                    v[row_idx] = 0
                else:
                    v[row_idx] = -n_y / n_z
            elif mask[h-1, w]:
                idx_vert = full2obj[h-1, w]
                M[row_idx, idx_vert] = 1
                M[row_idx, idx] = -1
                if n_z==0:
                    v[row_idx] = 0
                else:
                    v[row_idx] = -n_y / n_z

        # =================solving the linear equations Mz = v=================
        MtM = M.T @ M
        Mtv = M.T @ v
        z = sparse.linalg.spsolve(MtM, Mtv)

        Z = np.zeros_like(mask, dtype=np.float32)
        for idx in range(no_pix):
            h = obj_h[idx]
            w = obj_w[idx]
            Z[h, w] = z[idx]

        # std_z = np.std(z, ddof=1)
        # mean_z = np.mean(z)
        # z_zscore = (z - mean_z) / std_z
        # outlier_ind = np.abs(z_zscore) > 10
        # z_min = np.min(z[~outlier_ind])
        # z_max = np.max(z[~outlier_ind])

        # Z = mask.astype('float')
        # for idx in range(no_pix):
        #     # obtain the position in 2D picture 
        #     h = obj_h[idx]
        #     w = obj_w[idx]
        #     Z[h, w] = (z[idx] - z_min) / (z_max - z_min) * 255

        depth = Z - np.min(Z)
        return depth

    def save_depthmap(self,depth,save_path):
        np.save(os.path.join(save_path,'depthmap.npy'),depth)
        # normals_map_normalized = (depth * 255).astype(np.uint8)
        # cv2.imwrite(os.path.join(save_folder_path,'normals.png'), cv2.cvtColor(normals_map_normalized, cv2.COLOR_RGB2BGR) )
    

    def get_depth_map(self, normals_map, mask=None):
        if mask is None:
            mask = np.ones(normals_map.shape[:2], dtype=np.uint8)
            mask[:, -1] = 0  # Set the last column to 0
            mask[-1, :] = 0  # Set the last row to 0
            mask[:,0] = 0  # Set the first column to 0
            mask[0,:] = 0  # Set the first row to 0
        
        depth_map = self.compute_depth(mask=mask.copy(),N=normals_map.copy())
        return depth_map
    
    def get_pcd_from_depth(self, depth_np, params=None):
        if params is None:
            params = {
                "depth_scale": 1000.0,  # Assuming depth values are in millimeters (common for sensors)
                "depth_trunc": 50.0,     # Example: Truncate points beyond 5 meters
                "intrinsics": [150,150,depth_np.shape[0]/2,depth_np.shape[1]/2]  # fx,fy,cx,cy
            }
        depth_o3d_image = o3d.geometry.Image(depth_np.astype(np.float32))
        camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width=depth_np.shape[1],
            height=depth_np.shape[0],
            fx=params['intrinsics'][0],  # Focal length in x (in pixels)
            fy=params['intrinsics'][1],  # Focal length in y (in pixels)
            cx=depth_np.shape[1] / 2,  # Principal point x (usually image center)
            cy=depth_np.shape[0] / 2   # Principal point y (usually image center)
        )

        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            depth_o3d_image,
            camera_intrinsics,
            depth_scale=params['depth_scale'],  # Assuming depth values are in millimeters (common for sensors)
            # depth_trunc=50.0,     # Example: Truncate points beyond 5 meters
            project_valid_depth_only=True
        )

        return pcd

    def get_pcd_from_height_map(self, height_map, pixel_size_mm=0.1):
        """
        Convert a height map to a point cloud.
        """
        h, w = height_map.shape
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        zz = height_map

        # Apply scaling
        xx = xx * pixel_size_mm
        yy = yy * pixel_size_mm
        zz = zz * pixel_size_mm 

        # Flatten
        points = np.vstack((xx.ravel(), yy.ravel(), zz.ravel())).T

        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd

    def save_pcd(self, pcd, save_path=None,):
        """
        Save the point cloud to a PLY file.
        """
        # Convert to ply and save
        if save_path.endswith('.ply'):
            o3d.io.write_point_cloud(save_path, pcd)
        else:
            raise ValueError("Save path must end with .ply")

    def process_images(self, image_folder, save_folder_path=None, name_card="*.png",params=None, pixel_size_mm=0.1):
        if save_folder_path is None:
            save_folder_path = os.path.join(image_folder,"output")
            os.makedirs(save_folder_path, exist_ok=True)
        else:
            os.makedirs(save_folder_path, exist_ok=True)
        normals_map = self.get_normal_map(image_folder, name_card=name_card)
        self.save_normals_map(normals_map, save_folder_path)

        # depth_map = self.get_depth_map(normals_map)
        n = cv2.imread(os.path.join(save_folder_path,'normals.png'))
        n = cv2.cvtColor(n, cv2.COLOR_BGR2RGB)
        depth_map = estimate_height_map(n, raw_values=True)
        self.save_depthmap(depth_map,save_path=save_folder_path)

        # pcd = self.get_pcd_from_depth(depth_map,params=params)
        pcd = self.get_pcd_from_height_map(depth_map,pixel_size_mm=pixel_size_mm)
        # import pdb; pdb.set_trace()
        normals_map_scaled = (normals_map + 1) / 2
        pcd.colors = o3d.utility.Vector3dVector(normals_map_scaled.reshape(-1, 3))

        self.save_pcd(pcd, os.path.join(save_folder_path,"output.ply"))
        print(f"Output saved to {save_folder_path}")
        return normals_map, depth_map


