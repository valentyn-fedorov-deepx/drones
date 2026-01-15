import h5py
from pcd_classes import CamerasParams, ImageParams, ReconstrPoint

class HDF5Parser:
    @staticmethod
    def write_db(filename: str,
                camera_params: CamerasParams,
                image_params_list: list,
                recon_points_list: list) -> None:
        """
        Write one CamerasParams object along with lists of ImageParams and ReconstrPoint objects
        into an HDF5 database file with three top-level groups.
        """
        with h5py.File(filename, "w") as f:
            # Group 1: camera_params
            gp_cam = f.create_group("camera_params")
            gp_cam.create_dataset("K_pinhole", data=camera_params.K_pinhole)
            gp_cam.create_dataset("K_distorted", data=camera_params.K_distorted)
            gp_cam.create_dataset("dist_coeffs", data=camera_params.dist_coeffs)
            if camera_params.dist_shape is not None:
                gp_cam.create_dataset("dist_shape", data=camera_params.dist_shape)
            if camera_params.final_shape is not None:
                gp_cam.create_dataset("final_shape", data=camera_params.final_shape)

            # Group 2: image_params
            gp_images = f.create_group("image_params")
            for i, img in enumerate(image_params_list):
                gp_img = gp_images.create_group(f"image_{i}")
                gp_img.attrs["image_id"] = img.image_id
                gp_img.attrs["image_name"] = img.image_name
                gp_img.create_dataset("center_world", data=img.center_world)
                gp_img.create_dataset("K_pinhole", data=img.K_pinhole)
                gp_img.create_dataset("qvec", data=img.qvec)
                gp_img.create_dataset("tvec", data=img.tvec)
                gp_img.create_dataset("RPY_deg", data=img.RPY_deg)

            # Group 3: reconstruction_points
            gp_recon = f.create_group("reconstruction_points")
            for i, pt in enumerate(recon_points_list):
                gp_pt = gp_recon.create_group(f"point_{i}")
                gp_pt.attrs["id"] = pt.id
                gp_pt.create_dataset("world_coords", data=pt.world_coords)
                gp_pt.create_dataset("color", data=pt.color)

    @staticmethod
    def read_db(filename: str):
        """
        Read the HDF5 file and return a tuple containing:
        - a CamerasParams object,
        - a list of ImageParams objects,
        - a list of ReconstrPoint objects.
        """
        with h5py.File(filename, "r") as f:
            # Read from Group 1: camera_params
            gp_cam = f["camera_params"]
            K_pinhole = gp_cam["K_pinhole"][:]
            K_distorted = gp_cam["K_distorted"][:]
            dist_coeffs = gp_cam["dist_coeffs"][:]
            dist_shape = gp_cam["dist_shape"][:] if "dist_shape" in gp_cam else None
            final_shape = gp_cam["final_shape"][:] if "final_shape" in gp_cam else None
            camera_params = CamerasParams(K_pinhole, K_distorted, dist_coeffs, dist_shape, final_shape)

            # Read from Group 2: image_params
            image_params_list = []
            gp_images = f["image_params"]
            for key in gp_images:
                gp_img = gp_images[key]
                image_id = int(gp_img.attrs["image_id"])
                image_name = gp_img.attrs["image_name"]
                center_world = gp_img["center_world"][:]
                K_pinhole_img = gp_img["K_pinhole"][:]
                qvec = gp_img["qvec"][:]
                tvec = gp_img["tvec"][:]
                RPY_deg = gp_img["RPY_deg"][:]
                image_obj = ImageParams(image_id, image_name, center_world,
                                        K_pinhole_img, qvec, tvec, RPY_deg)
                image_params_list.append(image_obj)

            # Read from Group 3: reconstruction_points
            recon_points_list = []
            gp_recon = f["reconstruction_points"]
            for key in gp_recon:
                gp_pt = gp_recon[key]
                point_id = int(gp_pt.attrs["id"])
                world_coords = gp_pt["world_coords"][:]
                color = gp_pt["color"][:]
                point_obj = ReconstrPoint(point_id, world_coords, color)
                recon_points_list.append(point_obj)

            return camera_params, image_params_list, recon_points_list