import open3d as o3d
from PIL import Image
import numpy as np
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from transformers import AutoModelForImageSegmentation
from torchvision import transforms
import torch


from src.utils.measure_time import measure_time
from src.utils.plane_tracking import PlaneTracker
from src.utils.planes_processing import sequential_ransac_plane_segmentation, visualize_principal_axes_on_image

torch.set_float32_matmul_precision(['medium', 'high',  'highest'][0])


def get_tight_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """
    Finds the tight bounding box of an object in a binary mask.
    Returns coordinates in xyxy format (x_min, y_min, x_max, y_max).

    Args:
        mask: Binary numpy array where the object is represented by non-zero values

    Returns:
        tuple (x_min, y_min, x_max, y_max) representing the bounding box coordinates

    Example:
        >>> mask = np.zeros((100, 100))
        >>> mask[30:45, 50:75] = 1  # Create object
        >>> x_min, y_min, x_max, y_max = get_tight_bbox(mask)
        >>> tight_bbox = mask[y_min:y_max, x_min:x_max]  # Get cropped region
    """
    # Find non-zero points
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    # Find the boundaries
    row_idx = np.where(rows)[0]
    col_idx = np.where(cols)[0]

    # Handle empty mask
    if len(row_idx) == 0 or len(col_idx) == 0:
        return (0, 0, 0, 0)

    # Return in xyxy format (x_min, y_min, x_max, y_max)
    # Note: x corresponds to columns, y corresponds to rows
    return (int(col_idx[0]), int(row_idx[0]), int(col_idx[-1] + 1), int(row_idx[-1] + 1))


class SawbonesManager:
    def __init__(self, depth_model_name: str = "depth-anything/Depth-Anything-V2-Small-hf",
                 device: str = "cuda"):
        self.depth_image_processor = AutoImageProcessor.from_pretrained(depth_model_name,
                                                                        device=device)

        self.depth_model = AutoModelForDepthEstimation.from_pretrained(depth_model_name).to(device).eval()
        # self.rmbg_session = new_session("briaai/RMBG-2.0")

        self.bg_model = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet_lite",
                                                                      trust_remote_code=True).to(device).eval()
        self.device = device
        self.K = np.array([[2730.22290039, 0.00000000, 1125.36999512],
                           [0.00000000, 2730.22290039, 1117.56738281],
                           [0.00000000, 0.00000000, 1.00000000]])
        self.plane_tracker = PlaneTracker(angle_threshold_degs=15, offset_threshold=0.5, 
                                          max_missed_frames=1)

    def _preprocess_image_depth(self, img: np.ndarray) -> np.ndarray:
        # if len(img.shape) == 3:
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # img_equalized = cv2.equalizeHist(img)
        # img_equalized = cv2.cvtColor(img_equalized, cv2.COLOR_GRAY2RGB)

        # img_equalized = Image.fromarray(img_equalized)
        img_equalized = Image.fromarray(img)

        biggest_side = 512
        scale = biggest_side / max(img_equalized.size)
        new_size = (int(img_equalized.size[0] * scale), int(img_equalized.size[1] * scale))

        img_equalized = img_equalized.resize(new_size)
        return img_equalized

    def _preprocess_image_segmentation(self, img: np.ndarray) -> np.ndarray:
        image_size = (512, 512)
        transform_image = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # if len(img.shape) == 3:
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # img_equalized = cv2.equalizeHist(img)
        # img_equalized = cv2.cvtColor(img_equalized, cv2.COLOR_GRAY2RGB)

        # img_equalized = Image.fromarray(img_equalized)
        img_equalized = Image.fromarray(img)

        input_images = transform_image(img_equalized).unsqueeze(0).to('cuda')

        return input_images

    def _create_point_cloud(self, depth, mask):
        ys, xs = np.where(mask.astype(bool))
        depth_values = depth[mask.astype(bool)]
        im_points = np.stack([xs, ys, np.ones_like(ys)]).T
        points_3d = np.linalg.inv(self.K).dot(im_points.T).T
        points_3d[:, -1] = depth_values / 255
        pcd = o3d.geometry.PointCloud()

        # Set the points from numpy array
        pcd.points = o3d.utility.Vector3dVector(points_3d)
        return pcd

    @measure_time(message="Finding cut planes")
    def _find_cut_planes(self, depth, mask):
        pcd = self._create_point_cloud(depth, mask)
        planes = sequential_ransac_plane_segmentation(pcd, distance_threshold=0.008,
                                                      num_iterations=500, min_inliers=1000,
                                                      subsample=True, max_points=20000,
                                                      cluster_inliers=True, merge_planes=True,
                                                      reassign_points=True,
                                                      dbscan_min_points=100, dbscan_eps=0.02)

        return planes, np.asarray(pcd.points).copy()

    def _vizualize_cut_planes(self, img, planes):
        rgb_planes_viz = img.copy()

        # Add each point cloud as a separate scatter3d trace
        # for idx, (plane_params, pcd) in enumerate(planes):
        for plane in planes:
            # import ipdb; ipdb.set_trace()
            # Ensure points is numpy array with correct shape
            # plane_points = np.asarray(pcd.points).copy()
            plane_points = np.asarray(plane.point_cloud.points).copy()
            color = [int(c*255) for c in plane.plane_color]

            plane_points[:, -1] = np.ones_like(plane_points[:, 0])
            projected_plane_points = self.K.dot(plane_points.T).T
            projected_plane_points = projected_plane_points[:, :2] / projected_plane_points[:, -1, None]
            projected_plane_points = projected_plane_points.round().astype(int)

            # color = [random.randint(0, 255) for _ in range(3)]
            # color = (206, 206, 206)
            rgb_planes_viz[projected_plane_points[:, 1], projected_plane_points[:, 0]] = color

        return rgb_planes_viz

    @measure_time(message="Segmentation sawbones processor and model")
    def _get_mask(self, img) -> np.ndarray:
        input_images = self._preprocess_image_segmentation(img)
        with torch.inference_mode():
            preds = self.bg_model(input_images)[-1].sigmoid().cpu()

        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize((img.shape[1], img.shape[0]))
        mask = np.asarray(mask).copy()
        return mask

    @measure_time(message="Depth prediction sawbones processor and model")
    def _get_depth(self, img) -> np.ndarray:
        input_img = self._preprocess_image_depth(img)
        # import ipdb; ipdb.set_trace()
        inputs = self.depth_image_processor(images=input_img, return_tensors="pt")

        with torch.inference_mode():
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            outputs = self.depth_model(**inputs)
            predicted_depth = outputs.predicted_depth

        post_processed_output = self.depth_image_processor.post_process_depth_estimation(outputs,
                                                                                         target_sizes=[(img.shape[0], img.shape[1])])

        # visualize the prediction
        predicted_depth = post_processed_output[0]["predicted_depth"]
        depth = predicted_depth * 255 / predicted_depth.max()
        depth = depth.detach().cpu().numpy().round().astype(np.uint8)
        return depth

    @measure_time(message="Total neural part in sawbones project")
    def _process_neural_part(self, img):
        mask = self._get_mask(img)
        if not mask.any():
            return mask, None
        x1, y1, x2, y2 = get_tight_bbox(mask)
        object_crop = img[y1:y2, x1:x2].copy()
        depth_crop = self._get_depth(object_crop)
        depth_full = np.zeros_like(mask)
        depth_full[y1:y2, x1:x2] = depth_crop

        return mask, depth_full

    def _process_principal_axes(self, img, mask, depth):
        pcd = self._create_point_cloud(depth, mask)
        viz = visualize_principal_axes_on_image(img, pcd, self.K,
                                                axis_length_scale=0.0001,
                                                axis_thickness=5)

        return viz, np.asarray(pcd.points).copy()

    def process(self, img: np.ndarray) -> np.ndarray:
        mask, depth = self._process_neural_part(img)
        # import ipdb; ipdb.set_trace()

        # depth_masked = cv2.bitwise_and(depth, depth, mask=mask)
        # vizualization = np.stack([depth_masked, depth_masked, depth_masked], axis=-1)

        if depth is None:
            return img, []

        planes, points_3d = self._find_cut_planes(depth, mask)

        self.plane_tracker.update(planes)
        planes_tracked = self.plane_tracker.get_tracked_planes()
        if planes_tracked:
            # planes_tracked = sorted(planes_tracked, key=lambda x: np.asarray(x.point_cloud.points).size, reverse=True)[:3]
            vizualization = self._vizualize_cut_planes(img, planes_tracked)
        # vizualization, points_3d = self._process_principal_axes(img, mask,
        #                                                         depth)
        else:
            vizualization = img.copy()

        return vizualization, points_3d


if __name__ == "__main__":
    pass
