import sys

import numpy as np
import cv2
import os

import torch
import groundingdino.datasets.transforms as T

from groundingdino.util.inference import load_model, predict
from groundingdino.util import box_ops
from segment_anything import build_sam_vit_b, SamPredictor, SamAutomaticMaskGenerator

from PIL import Image
from skimage.measure import label
import networkx as nx

from .obstacle import Obstacle


########################################################################################################################
def print_it(a, name: str = ''):
    m = a.float().mean() if isinstance(a, torch.Tensor) else a.mean()
    # m = np.array(a, dtype='float64').mean()
    # m = a.mean()
    print(name, a.shape, a.dtype, a.min(), m, a.max())


########################################################################################################################
class StaticObstacleDetector:

    def __init__(self, groundingdino_dir, model_dir, imsize):

        self.H, self.W = imsize
        self.min_obstacle_y = int(self.H * 0.0)
        self.mask_overlap_thr = 0.5
        self.mask_merging_phrase_groups = {
            "post": 0,
            "tree trunk": 0,
            "obstacle": 1,
            "hedge": 2,
            "other": 3,
        }
        self.everything_mode_merging_overlap_threshold = 0.01

        sam = build_sam_vit_b(checkpoint=os.path.join(model_dir , "sam_vit_b_01ec64.pth")).to(device="cuda")
        self.segment_anything = SamPredictor(sam)
        self.segment_everything = SamAutomaticMaskGenerator(sam)

        p_gd_py = os.path.join(groundingdino_dir, 'groundingdino/config/GroundingDINO_SwinT_OGC.py')
        p_gd_pth = os.path.join(groundingdino_dir, 'weights/groundingdino_swint_ogc.pth')

        self.grounding_dino = load_model(p_gd_py, p_gd_pth)

        self.params = {
            "global_level": {
                "crops": [
                    (0.0, 0.0, 1.0, 1.0),
                ],
                "phrases": {
                    "post": 0.2,
                    "tree trunk": 0.2,
                    "hedge": 0.2,
                    "obstacle": 0.2,
                },
                "proximity_to_bottom_threshold": -1,
            },
            "local_level_1": {
                "crops": [
                    (0.0, 0.0, 0.5, 0.6),
                    (0.25, 0.0, 0.75, 0.6),
                    (0.5, 0.0, 1.0, 0.6),
                ],
                "phrases": {
                    "post": 0.2,
                    "tree trunk": 0.2,
                    "hedge": 0.2,
                },
                "proximity_to_bottom_threshold": 10,
            },
            "local_level_2": {
                "crops": [
                    (0.0, 0.25, 0.25, 0.5),
                    (0.25, 0.25, 0.5, 0.5),
                    (0.5, 0.25, 0.75, 0.5),
                    (0.75, 0.25, 1.0, 0.5),
                ],
                "phrases": {
                    "post": 0.2,
                    "tree trunk": 0.2,
                    "hedge": 0.2,
                },
                "proximity_to_bottom_threshold": 10,
            },
        }

    @staticmethod
    def postprocess_phrases(phrases, classes):
        phrases_new = []
        for phrase in phrases:
            for class_ in classes:
                if class_ in phrase:
                    phrases_new.append(class_)
                    break
        return phrases_new

    @staticmethod
    def preprocess_phrases(params_detect):
        phrases_list = []
        box_thr = 1.0
        for phrase, conf_thr in params_detect.items():
            phrases_list.append(phrase)
            box_thr = min(box_thr, conf_thr)
        text_prompt = ". ".join(phrases_list)
        text_thr = box_thr
        return text_prompt, phrases_list, box_thr, text_thr

    @staticmethod
    def getLargestCC(segmentation):
        labels = label(segmentation)
        assert( labels.max() != 0 )
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
        return largestCC

    @staticmethod
    def postprocess_masks(masks):
        masks_new = []
        for mask in masks:
            mask = StaticObstacleDetector.getLargestCC(mask)
            masks_new.append(mask)
        return masks_new

    @staticmethod
    def mask2bbox(mask):
        a = np.where(mask > 0)
        bbox = np.min(a[1]), np.min(a[0]), np.max(a[1]), np.max(a[0])
        return bbox

    @staticmethod
    def show_mask(mask, image, random_color=True):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

        annotated_frame_pil = Image.fromarray(image).convert("RGBA")
        mask_image_pil = Image.fromarray((mask_image * 255).astype(np.uint8)).convert("RGBA")

        return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))

    @staticmethod
    def prepare_image(image):
        image = Image.fromarray(image).convert("RGB")
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image, _ = transform(image, None)
        return image

    @staticmethod
    def crop_image(image, box):
        h, w = image.shape[:2]
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(w*x0), int(h*y0), int(w*x1), int(h*y1)
        return image[y0:y1,x0:x1], (x0, y0, x1, y1)

    @staticmethod
    def bbox_intersect(bbox_1, bbox_2):
        x1 = max(bbox_1[0], bbox_2[0])
        x2 = min(bbox_1[2], bbox_2[2])
        y1 = max(bbox_1[1], bbox_2[1])
        y2 = min(bbox_1[3], bbox_2[3])

        return (x1 < x2) and (y1 < y2), (x1, y1, x2, y2)

    def merge_overlapping_masks(self, detections):

        if len(detections) == 0:
            return detections

        masks, boxes, confs, phrases = zip(*detections)

        n = len(masks)
        G = nx.Graph()

        for i in range(n):
            G.add_node(i)

        def if_intersect(idx_1, idx_2):
            bbox_1, bbox_2 = boxes[idx_1], boxes[idx_2]
            mask_1, mask_2 = masks[idx_1], masks[idx_2]
            intersect, (x1,y1,x2,y2) = self.bbox_intersect(bbox_1, bbox_2)
            if self.mask_merging_phrase_groups[phrases[idx_1]] != self.mask_merging_phrase_groups[phrases[idx_2]]:
                return False
            if not intersect:
                return False

            mask_1, mask_2 = masks[idx_1], masks[idx_2]
            area_1 = mask_1[bbox_1[1]:bbox_1[3],bbox_1[0]:bbox_1[2]].sum()
            area_2 = mask_2[bbox_2[1]:bbox_2[3],bbox_2[0]:bbox_2[2]].sum()
            area_12 = (mask_1[y1:y2,x1:x2] * mask_2[y1:y2,x1:x2]).sum()
            overlap_ratio = area_12 / min(area_1, area_2)

            return overlap_ratio > self.mask_overlap_thr

        for i in range(n):
            for j in range(i+1, n):
                if if_intersect(i, j):
                    G.add_edges_from([[i,j],[j,i]])
        
        G = G.to_undirected()
        ccs = [list(cc) for cc in nx.connected_components(G)]

        detections_new = []
        for cc in ccs:
            conf = 0
            phrase = phrases[cc[0]]
            bbox = [np.inf,np.inf,-np.inf,-np.inf]
            mask = np.zeros(masks[0].shape[:2], dtype=bool)

            for idx in cc:
                conf = max(conf, confs[idx])
                bbox[0], bbox[1] = min(bbox[0], boxes[idx][0]), min(bbox[1], boxes[idx][1])
                bbox[2], bbox[3] = max(bbox[2], boxes[idx][2]), max(bbox[3], boxes[idx][3])
                mask = mask | (masks[idx]>0)
            bbox = [int(x) for x in bbox]
            
            detections_new.append([mask, bbox, conf, phrase])

        return detections_new

    @staticmethod
    def filter_boxes_by_confidence(boxes, confs, phrases, params_detect):
        boxes_new = []
        confs_new = []
        phrases_new = []
        for box, conf, phrase in zip(boxes, confs, phrases):
            if conf > params_detect[phrase]:
                boxes_new.append(box)
                confs_new.append(conf)
                phrases_new.append(phrase)
        return boxes_new, confs_new, phrases_new

    def predict_gdino(self, image, params_detect):

        text_prompt, phrases_list, box_thr, text_thr = self.preprocess_phrases(params_detect)

        h, w = image.shape[:2]
        image_tensor = self.prepare_image(image)
        boxes, logits, phrases = predict(
            model=self.grounding_dino,
            image=image_tensor,
            caption=text_prompt,
            box_threshold=box_thr,
            text_threshold=text_thr
        )

        boxes = box_ops.box_cxcywh_to_xyxy(boxes).numpy() * np.array([w, h, w, h])
        confs = logits.numpy()
        phrases = self.postprocess_phrases(phrases, phrases_list)
        boxes, confs, phrases = self.filter_boxes_by_confidence(boxes, confs, phrases, params_detect)

        return boxes, phrases

    def predict_sam(self, image, boxes):
        self.segment_anything.set_image(image)
        boxes = torch.Tensor(boxes)
        transformed_boxes = self.segment_anything.transform.apply_boxes_torch(boxes, image.shape[:2]).to("cuda")
        masks, confidence, _ = self.segment_anything.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )

        masks = [mask.cpu().numpy()[0] for mask in masks]
        masks = self.postprocess_masks(masks)

        boxes_new = []
        for mask in masks:
            bbox = np.array(self.mask2bbox(mask))
            boxes_new.append(bbox)

        return boxes_new, masks, confidence

    def process_image(self, image, params_detect):

        boxes, phrases = self.predict_gdino(image, params_detect)
        boxes, masks, confs = self.predict_sam(image, boxes)

        return boxes, masks, phrases, confs

    def process_level(self, image, params_level):

        detections = []

        for xyxyn in params_level["crops"]:
            crop, xyxy = self.crop_image(image, xyxyn)
            x0, y0, x1, y1 = xyxy

            boxes, masks, phrases, confs = self.process_image(crop, params_level["phrases"])
            for box, mask, conf, phrase in zip(boxes, masks, confs, phrases):
                box = box[0]+x0, box[1]+y0, box[2]+x0, box[3]+y0
                mask_new = np.zeros(image.shape[:2])
                mask_new[y0:y1,x0:x1] = mask

                if box[3] > y1-params_level["proximity_to_bottom_threshold"]:
                    continue
                if box[3] < self.min_obstacle_y:
                    continue

                detections.append([mask_new, box, conf, phrase])

        detections = self.merge_overlapping_masks(detections)
        return detections

    def process_in_everything_mode(self, image):
        detections = []
        for output in self.segment_everything.generate(image):
            mask = output["segmentation"]
            mask = self.postprocess_masks([mask])[0]
            bbox = self.mask2bbox(mask)
            conf = output["predicted_iou"]
            phrase = "other"
            if 2*10**5 > mask.sum() > 10**3 and bbox[3] >= self.min_obstacle_y:
                detections.append([mask, bbox, conf, phrase])
        detections = self.merge_overlapping_masks(detections)
        return detections

    def merge_with_everything_mode_detections(self, detections, detections_everything):
        detections_new = []

        global_occupancy_mask = np.zeros((self.H, self.W))
        for detection in detections:
            detections_new.append(detection)
            global_occupancy_mask[detection[0]] = 1

        for detection in detections_everything:
            detection_mask = detection[0]
            detection_area = (detection_mask > 0).sum()
            overlap_area = (detection_mask * global_occupancy_mask > 0).sum()
            if overlap_area / detection_area < self.everything_mode_merging_overlap_threshold:
                detections_new.append(detection)

        return detections_new

    def process(self, image):
        detections = []
        for level, params in self.params.items():
            detections_level = self.process_level(image, params_level=params)
            detections.extend(detections_level)
        detections = self.merge_overlapping_masks(detections)

        # detections_everything = self.process_in_everything_mode(image)
        # detections = self.merge_with_everything_mode_detections(detections, detections_everything)

        obstacles = []
        for mask, box, conf, phrase in detections:
            obstacles.append(Obstacle(mask, phrase, conf, (self.H, self.W), bbox=box))

        return obstacles


if __name__ == "__main__":
    image = cv2.imread("/root/pdst/data_decoded/500mRoadside/500mRoadside-5000.png")
    processor = StaticObstacleDetector()
    obstacles = processor.process(image)

    print(len(obstacles))

    for obstacle in obstacles:
        print(obstacle.area)
        image = processor.show_mask(obstacle.get_full_mask(), image)

    cv2.imwrite("obstacles.png", image)
