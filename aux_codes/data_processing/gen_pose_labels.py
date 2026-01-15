import cv2
import numpy as np
import os
import albumentations as A
from tqdm import tqdm
from argparse import ArgumentParser
import random

from src.cv_module.people.people_pose_estimation import PeoplePoseEstimator


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--images-path")
    parser.add_argument("--labels-path")
    parser.add_argument("--save-path")
    parser.add_argument("--ignore-intersections", action='store_true',
                        default=False)
    parser.add_argument("--min-allowed-side-size", type=int, default=125)
    parser.add_argument("--samples-per-image", type=int, default=3)

    return parser.parse_args()


def intersects(box1, box2):
    x1, y1, x1_2, y1_2 = box1
    w1 = x1_2 - x1
    h1 = y1_2 - y1

    x2, y2, x2_2, y2_2 = box2
    w2 = x2_2 - x2
    h2 = y2_2 - y2

    return not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1)


def find_non_intersecting_boxes(boxes):
    non_intersecting = []
    for i, box in enumerate(boxes):
        if all([not intersects(box, other_box) for j, other_box in enumerate(boxes) if i != j]):
            non_intersecting.append(i)

    return non_intersecting


def prepare(images_path, labels_path,
            save_path, ignore_intersections,
            min_allowed_bbox_size=45, min_points=4,
            samples_per_image=3):
    images = os.listdir(images_path)
    labels = os.listdir(labels_path)

    transform = A.Compose([
        A.ToGray(p=1.0),
        A.AdvancedBlur(blur_limit=(7, 13), p=0.7),
        A.ImageCompression(1, 45, compression_type='jpeg',
                           p=0.7),
        A.RandomScale((-0.9, -0.1), interpolation=cv2.INTER_LINEAR,
                      p=0.8)
    ])

    for label in tqdm(labels):
        image = cv2.imread(f'{images_path}/{label[:-4]}.jpg')

        if image is None:
            continue

        height, width, _ = image.shape

        for im_sample_idx in range(samples_per_image):
            with open(f'{labels_path}/{label}') as file:
                bboxes = list()
                orig_bboxes = list()
                poses = list()
                crops = list()
                for line in file:
                    pose_label = line.strip().split()
                    pose_label = np.array(pose_label).astype(float)

                    class_id = int(pose_label[0])
                    orig_bbox = pose_label[1:5]
                    orig_keypoints = pose_label[5:]

                    orig_keypoints = np.array(orig_keypoints).reshape(-1, 3)

                    if np.count_nonzero(orig_keypoints[:, -1] != 0) < min_points:
                        continue

                    new_keypoints = orig_keypoints.copy()
                    new_keypoints[:, 0] = new_keypoints[:, 0] * width
                    new_keypoints[:, 1] = new_keypoints[:, 1] * height

                    new_keypoints_filtered = new_keypoints[new_keypoints[:, -1] > 0]
                    keypoints_min_x, keypoints_min_y, _ = new_keypoints_filtered.min(0).round().astype(int)
                    keypoints_max_x, keypoints_max_y, _ = new_keypoints_filtered.max(0).round().astype(int)

                    x = orig_bbox[0] * width
                    y = orig_bbox[1] * height
                    x_width = orig_bbox[2] * width
                    y_height = orig_bbox[3] * height

                    x_min = min(int(round(x - (x_width / 2))), keypoints_min_x)
                    y_min = min(int(round(y - (y_height / 2))), keypoints_min_y)
                    x_max = max(int(round(x + (x_width / 2))), keypoints_max_x)
                    y_max = max(int(round(y + (y_height / 2))), keypoints_max_y)

                    bbox = [x_min, y_min, x_max, y_max]
                    x_min, y_min, x_max, y_max = bbox

                    pad_y_im = random.random() * 0.2
                    pad_x_im = random.random() * 0.2

                    x_min_padded, y_min_padded, x_max_padded, y_max_padded = padded_bbox = PeoplePoseEstimator.pad_bbox_relative(bbox, pad_x_im, pad_y_im,
                                                                                                                                 image.shape)

                    pad_left = x_min - x_min_padded
                    pad_right = x_max_padded - x_max
                    pad_top = y_min - y_min_padded
                    pad_bottom = y_max_padded - y_max

                    # import ipdb; ipdb.set_trace()

                    crop = image[y_min_padded:y_max_padded, x_min_padded:x_max_padded]

                    crop_height, crop_width, _ = crop.shape

                    pad_y_zeros = random.random() * 0.3
                    pad_x_zeros = random.random() * 0.3
                    pad_x_zeros = int(round((pad_x_zeros * crop_width) / 2))
                    pad_y_zeros = int(round((pad_y_zeros * crop_height) / 2))

                    padded_crop = cv2.copyMakeBorder(crop, pad_y_zeros,
                                                     pad_y_zeros, pad_x_zeros,
                                                     pad_x_zeros, cv2.BORDER_CONSTANT,
                                                     value=0)

                    padded_crop_height, padded_crop_width, _ = padded_crop.shape

                    transformed = transform(image=padded_crop)
                    transformed_crop = transformed['image']

                    if np.any(np.array(transformed_crop.shape[:2]) < min_allowed_bbox_size):
                        continue

                    crops.append(transformed_crop)

                    if crop_height < min_allowed_bbox_size or crop_width < min_allowed_bbox_size:
                        continue

                    if crop is None or crop_height == 0 or crop_width == 0:
                        continue

                    crop_x_min = pad_left + pad_x_zeros
                    crop_y_min = pad_top + pad_y_zeros
                    crop_x_max = padded_crop_width - pad_left - pad_x_zeros
                    crop_y_max = padded_crop_height - pad_top - pad_y_zeros

                    crop_bbox_width = crop_x_max - crop_x_min
                    crop_bbox_height = crop_y_max - crop_y_min
                    crop_center_x = (crop_x_min + crop_x_max) / 2
                    crop_center_y = (crop_y_min + crop_y_max) / 2
                    # cv2.circle(crop, (int(crop_center_x),int(crop_center_y)), 1, (0,255,0),-1)

                    crop_bbox = [crop_center_x / padded_crop_width,
                                 crop_center_y / padded_crop_height,
                                 crop_bbox_width / padded_crop_width,
                                 crop_bbox_height / padded_crop_height]

                    # import ipdb; ipdb.set_trace()

                    bboxes.append(crop_bbox)
                    orig_bboxes.append(padded_bbox)

                    # cv2.rectangle(padded_crop, (crop_x_min, crop_y_min), (crop_x_max, crop_y_max), (0,0,255), 1)

                    new_crop_pose = []
                    for point in new_keypoints:
                        x, y, visability = point

                        if x != 0 or y != 0:
                            crop_x = x - x_min_padded + pad_x_zeros if x != 0 else 0
                            crop_y = y - y_min_padded + pad_y_zeros if y != 0 else 0
                        else:
                            crop_x = x
                            crop_y = y

                        if visability == 1 and (crop_x > 1 or crop_x < 0 or crop_y > 1 or crop_y < 0):
                            crop_y = 0
                            crop_x = 0
                            visability = 0

                        # cv2.circle(padded_crop, (int(round(crop_x)),int(crop_y)), 1, (0,0,255),-1)
                        new_crop_pose.append([crop_x / padded_crop_width,
                                              crop_y / padded_crop_height,
                                              visability])

                        # import ipdb; ipdb.set_trace()

                    new_crop_pose = np.array(new_crop_pose).flatten()
                    poses.append(new_crop_pose)

                if ignore_intersections:
                    indices = find_non_intersecting_boxes(orig_bboxes)
                    bboxes = [bboxes[idx] for idx in indices]
                    poses = [poses[idx] for idx in indices]
                    crops = [crops[idx] for idx in indices]

                for i, (transformed_crop, crop_bbox, new_crop_pose) in enumerate(zip(crops, bboxes, poses)):
                    cv2.imwrite(f'{save_path}/images/{label[:-4]}_{i}_{im_sample_idx}.jpg',
                                transformed_crop)

                    if np.any(np.logical_or(np.array(crop_bbox) > 1, np.array(crop_bbox) < 0)):
                        import ipdb; ipdb.set_trace()

                    if np.any(np.logical_or(np.array(new_crop_pose).reshape(-1, 3)[:, :2] > 1, np.array(new_crop_pose).reshape(-1, 3)[:, :2] < 0)):
                        import ipdb; ipdb.set_trace()

                    with open(f'{save_path}/labels/{label[:-4]}_{i}_{im_sample_idx}.txt', 'w') as f:
                        crop_bbox = map(lambda x: "{:.6f}".format(x), crop_bbox)
                        new_crop_pose = map(lambda x: "{:.6f}".format(x),
                                            new_crop_pose)

                        bbox_str = " ".join(map(str, crop_bbox))
                        keypoints_str = " ".join(map(str, new_crop_pose))

                        f.write(f"{class_id} {bbox_str} {keypoints_str}\n")


if __name__ == "__main__":
    args = parse_args()

    os.makedirs(f"{args.save_path}/labels", exist_ok=True)
    os.makedirs(f"{args.save_path}/images", exist_ok=True)

    prepare(args.images_path, args.labels_path,
            args.save_path, ignore_intersections=args.ignore_intersections,
            samples_per_image=args.samples_per_image)
