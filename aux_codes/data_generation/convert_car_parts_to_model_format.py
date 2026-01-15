import numpy as np
import cv2
from pathlib import Path
import shutil
import json
from typing import List, Dict
from tqdm import tqdm


def mask_representation_from_image(mask: np.ndarray):
    contours, hier = cv2.findContours(mask, cv2.RETR_TREE,
                                      cv2.CHAIN_APPROX_SIMPLE)
    return [contour.tolist() for contour in contours]


def read_masks_info(mask_path: str):
    results = list()

    mask_path = Path(mask_path)
    mask_json_file = mask_path / "mask.json"

    image_path = mask_path / "mask.png"
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if not mask_json_file.exists():
        print(mask_json_file)
        return results

    with open(mask_json_file, "r") as f:
        labels = json.load(f)

    for instance_info in labels:
        if instance_info["label"] == "background":
            continue

        x1, y1, x2, y2 = list(map(int, instance_info["box"]))
        instance_crop_mask = np.uint8(image == instance_info["value"]) * 255

        instance_mask = mask_representation_from_image(instance_crop_mask)

        if len(instance_mask) == 0:
            continue

        instance_info_new_format = dict(class_name=instance_info["label"],
                                        bbox=[x1, y1, x2, y2],
                                        instance_mask=instance_mask,
                                        item_name=mask_path.parent.name,
                                        parts=list())
        results.append(instance_info_new_format)

    return results


def get_iou(box1, box2):
    x1_1, y1_1, x1_2, y1_2 = box1
    x2_1, y2_1, x2_2, y2_2 = box2

    x_left = max(x1_1, x2_1)
    y_top = max(y1_1, y2_1)
    x_right = min(x1_2, x2_2)
    y_bottom = min(y1_2, y2_2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    bb1_area = (x1_2 - x1_1) * (y1_2 - y1_1)
    bb2_area = (x2_2 - x2_1) * (y2_2 - y2_1)

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    return iou


def get_relative_intersection(box1, box2):
    x1_1, y1_1, x1_2, y1_2 = box1
    x2_1, y2_1, x2_2, y2_2 = box2

    x_left = max(x1_1, x2_1)
    y_top = max(y1_1, y2_1)
    x_right = min(x1_2, x2_2)
    y_bottom = min(y1_2, y2_2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    box2_area = (x2_2 - x2_1) * (y2_2 - y2_1)

    relative_intersection = intersection_area / box2_area
    return relative_intersection


def group_car_parts(raw_image: np.ndarray, main_objects: List[Dict], possible_parts: List[Dict],
                    intersection_thresh: float = 0.9, iou_thresh: float = 0.9):
    for main_object in main_objects:
        main_bbox = main_object["bbox"]
        main_mask = np.zeros_like(raw_image)
        contours = [np.array(contour, dtype=int) for contour in main_object["instance_mask"]]
        main_mask = cv2.drawContours(main_mask, contours, -1,
                                     (255, 255, 255), -1)[:, :, 0]

        part_masks = np.zeros_like(raw_image)

        for possible_part in possible_parts:
            part_bbox = possible_part["bbox"]
            intersection = get_relative_intersection(main_bbox, part_bbox)
            iou = get_iou(main_bbox, part_bbox)

            # print("-------------------------")
            # print(possible_part["class_name"])
            # print(iou, intersection)
            # print(len(possible_part["instance_mask"]))

            # checking that the possible part instance is largely inside in the main object with relative intersection
            # also we need to check if the object is mostly
            if intersection > intersection_thresh and iou < iou_thresh:
                main_object["parts"].append(dict(label=possible_part["class_name"],
                                                 mask=possible_part["instance_mask"]))

                part_contours = [np.array(contour, dtype=int) for contour in possible_part["instance_mask"]]
                part_masks = cv2.drawContours(part_masks, part_contours, -1,
                                              (255, 255, 255), -1)

        part_masks = part_masks[:, :, 0]
        body_mask = cv2.bitwise_xor(main_mask, part_masks)
        body_label = dict(label="body",
                          mask=mask_representation_from_image(body_mask))

        main_object["parts"].append(body_label)

    return main_objects


if __name__ == "__main__":
    parts_object_dir_to_class_name = {
        "window": "window",
        "wheel": "wheel",
        "car_window": "window"
    }

    samples_path = Path("Grounded-Segment-Anything/outputs/full_car_classification_dataset_fixed")
    save_path = Path("output/car_part_labels_fixed")
    save_path_labels = save_path / "labels"
    save_path_images = save_path / "images"

    save_path_labels.mkdir(parents=True, exist_ok=True)
    save_path_images.mkdir(exist_ok=True)

    all_samples_path_iter = samples_path.glob("*")
    # all_samples_path_iter = [Path("/media/sviatoslav/MainVolume/Projects/deepxhub/people-track/aux_codes/data_generation/Grounded-Segment-Anything/outputs/full_car_classification_dataset/2889")]
    for sample_path in tqdm(all_samples_path_iter):

        main_part_path = sample_path / "car"
        raw_image_path = sample_path / "raw_image.jpg"

        raw_image = cv2.imread(str(raw_image_path))

        main_part_masks = read_masks_info(main_part_path)

        for path_name, class_name in parts_object_dir_to_class_name.items():
            part_object_path = sample_path / path_name
            if not part_object_path.exists():
                continue
            part_object_masks = read_masks_info(part_object_path)

            main_part_masks = group_car_parts(raw_image,
                                              main_part_masks,
                                              part_object_masks)

        with open(save_path_labels / f"{sample_path.name}.json", "w") as f:
            json.dump(main_part_masks, f)

        shutil.copy(raw_image_path, save_path_images / f"{sample_path.name}.jpg")
