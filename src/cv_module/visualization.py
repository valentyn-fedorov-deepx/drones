import cv2
import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation, binary_fill_holes
from typing import List, Optional
import re

from src.cv_module.people.person import Person
from src.cv_module.detected_object import DetectedObject


# color_bg, color_fg, alpha
DEFAULT_ANNOTATION_SETTING = ((50, 50, 50), (0xff, 0xff, 0xff), 0.7)

ANNOTATION_SETTINGS = {
    # "person": ((44, 57, 48), (63, 79, 68), 0.9),
    "person": ((44, 57, 48), (255, 255, 255), 0.9),
    "car": ((32, 87, 129), (79, 149, 157), 0.9),
    "drone": ((98, 111, 71), (164, 180, 101), 0.9)
}


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    # color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255],
                    thickness=tf, lineType=cv2.LINE_AA)


def blend_images_with_mask(img1, img2, mask):
    # Ensure all images have the same dimensions
    if img1.shape != img2.shape:
        # Resize second image to match first image
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # Resize mask to match image dimensions if necessary
    if mask.shape != img1.shape[:2]:
        mask = cv2.resize(mask, (img1.shape[1], img1.shape[0]))

    # Normalize mask to range 0-1
    mask = mask.astype(float) / 255

    # Expand mask to 3 channels to match image dimensions
    mask_3channel = np.stack([mask] * 3, axis=2)

    # Blend images using the mask
    blended = cv2.multiply(mask_3channel, img1.astype(float)) + \
                cv2.multiply(1.0 - mask_3channel, img2.astype(float))

    # Convert back to uint8
    result = np.uint8(blended)

    return result


def get_contour(mask):
    contour = mask > binary_erosion(mask, iterations=2)
    return contour


def postprocess_mask(mask, holes=False):
    mask = binary_erosion(mask, iterations=2)
    mask = binary_dilation(mask, iterations=2)
    if holes:
        mask = binary_fill_holes(mask)
    return mask


def get_masks_for_visualization(mask):
    mask = postprocess_mask(mask)
    h, w = mask.shape[:2]
    padded = np.zeros((h + 2, w + 2))
    padded[1:-1, 1:-1] = mask
    contour = get_contour(padded)[1:-1, 1:-1]

    return mask, contour


def draw_human_pose(frame, pose, color=(0, 255, 0)):
    skeleton = [
        (0, 1), (1, 3), (0, 2), (2, 4),
        (0, 5), (0, 6),
        (5, 7), (7, 9),
        (6, 8), (8, 10),
        (5, 11), (6, 12),
        (11, 13), (13, 15),
        (12, 14), (14, 16)
    ]

    for point in pose:
        if point[0] == 0 or point[1] == 0:
            continue

        cv2.circle(frame, point, 2, color, -1)

    for joint in skeleton:
        partA = joint[0]
        partB = joint[1]

        if pose[partA][0] == 0 or pose[partA][1] == 0 or pose[partB][0] == 0 or pose[partB][1] == 0:
            continue
        cv2.line(frame, pose[partA], pose[partB], color, 2)

    return frame


def draw_person(img_to_draw: np.ndarray, person: Person,
                mask_alpha: int = 125, overlay_img: Optional[np.ndarray] = None) -> np.ndarray:
    img_to_draw = img_to_draw.copy()
    if person.mask is not None and person.mask.sum() > 0:
        mask, contour = get_masks_for_visualization(person.mask)

        if overlay_img is None:
            overlay_img = np.full_like(img_to_draw, fill_value=(255, 0, 0))

        img_to_draw = blend_images_with_mask(overlay_img, img_to_draw,
                                             mask * mask_alpha)
        img_to_draw[contour] = (0, 0, 255)

    if person.has_pose:
        img_to_draw = draw_human_pose(img_to_draw, person.pose.round().astype(int))

    x1, y1, x2, y2 = person.bbox.round().astype(int)
    img_to_draw = cv2.rectangle(img_to_draw, (x1, y1), (x2, y2),
                                (0, 0, 255), 4)

    return img_to_draw


def annotate(frame, bbox, labels, ann_settings, velocity=None):
    """Draws the annotation box with labels and a velocity arrow."""
    color_bg, color_fg, alpha = ann_settings
    font = cv2.FONT_HERSHEY_SIMPLEX

    lw = 2
    tf = max(lw - 1, 1)  # font thickness
    sf = lw / 3  # font scale

    # --- Position calculation for the annotation box ---
    x_center_bbox = (bbox[0] + bbox[2]) // 2
    p_marker = (int(x_center_bbox), int(bbox[1]) - 30)
    p1 = (int(x_center_bbox), int(bbox[1]) - 60)

    marker_size = 12
    marker_thickness = 5
    cv2.drawMarker(frame, p_marker, color_bg, cv2.MARKER_TRIANGLE_DOWN,
                   marker_size, marker_thickness)

    if not labels:
        return frame

    # --- Calculate box dimensions based on label text size ---
    pad = 10
    # Note: cv2.getTextSize handles empty strings correctly
    w = max([cv2.getTextSize(label, font, fontScale=sf, thickness=tf + 1)[0][0] for label in labels])
    h = cv2.getTextSize(labels[0], font, fontScale=sf, thickness=tf + 1)[0][1] + pad
    x, y = p1
    w_box, h_box = w + 2 * pad, len(labels) * h + pad

    # --- Create the colored label box ---
    label_box = np.zeros((h_box, w_box, 3), dtype=np.uint8)
    label_box[:] = color_bg

    # --- Draw text labels on the box ---
    for i, label in enumerate(labels):
        # This will skip drawing text for our empty placeholder row
        if not label.strip():
            continue
        ww = cv2.getTextSize(label, font, fontScale=sf, thickness=tf + 1)[0][0]
        x_la = w_box // 2 - ww // 2
        y_la = (i + 1) * h
        cv2.putText(label_box, label, (x_la, y_la), font, sf, color_fg,
                    thickness=tf + 1, lineType=cv2.LINE_AA)

    # --- Draw velocity arrow in the dedicated last row --- HERE_HERE
    if velocity is not None:
        arrow_len = 15
        arrow_thickness = 2

        vx, vy = velocity[0], velocity[1]
        magnitude = np.sqrt(vx**2 + vy**2)

        if magnitude > 0.1:
            dir_x = vx / magnitude
            dir_y = vy / magnitude

            # Position the arrow in the center of the last row
            start_x = w_box // 2
            start_y = h_box - (h // 2) - (pad // 2) # Fine-tuned vertical position

            end_x = int(start_x + dir_x * arrow_len)
            end_y = int(start_y + dir_y * arrow_len)

            # Draw the arrow ➔
            cv2.arrowedLine(label_box, (start_x, start_y), (end_x, end_y),
                            color_fg, arrow_thickness, line_type=cv2.LINE_AA,
                            tipLength=0.3)

    # --- Blend the label box with the main frame ---
    y1 = min(max(y - h_box, 0), frame.shape[0] - h_box)
    y2 = y1 + h_box
    x1 = min(max(x - w_box // 2, 0), frame.shape[1] - w_box)
    x2 = x1 + w_box

    box_img = frame[y1:y2, x1:x2]
    box_alpha = alpha * label_box + (1 - alpha) * box_img
    frame[y1:y2, x1:x2] = box_alpha.astype(np.uint8)

    return frame

def plot_object_with_distance(img_to_draw, detected_objects, overlay_img=None,
                              blend_alpha=0.9):
    full_image_mask = np.zeros(img_to_draw.shape[:2], dtype=np.uint8)

    for detected_object in detected_objects:
        measurements = detected_object.meas
        labels = [f'{detected_object.name.capitalize()} #{detected_object.id}']
        velocity_vector = None

        if measurements:
            labels += [
                f"Xo: {measurements.X:.1f} m",
                f"Yo: {measurements.Y:.1f} m",
                f"Zo: {measurements.Z:.1f} m",
                f"Range: {measurements.dist_property:.1f} m",
            ]
            if measurements.velocity is not None:
                velocity_vector = measurements.velocity
                magnitude_3d = np.linalg.norm(velocity_vector)
                # 1. Add the magnitude text
                labels.append(f"V {magnitude_3d:.2f} m/s")
                # 2. Add an empty placeholder to reserve a row for the arrow
                labels.append("")

        annotation_setting = ANNOTATION_SETTINGS.get(detected_object.name, DEFAULT_ANNOTATION_SETTING)

        img_to_draw = annotate(img_to_draw, detected_object.bbox,
                               labels, annotation_setting,
                               velocity=velocity_vector)

        if detected_object.mask is not None and detected_object.mask.size:
            # Process mask for full-image visualization; be robust to shape mismatches
            mask, contour = get_masks_for_visualization(detected_object.mask)
            x1, y1, x2, y2 = detected_object.bbox

            # Extract sub-region and ensure mask has the same spatial shape
            sub = full_image_mask[y1:y2, x1:x2]
            if sub.shape[:2] != mask.shape[:2]:
                try:
                    # Resize boolean mask to fit the bbox region
                    resized = cv2.resize(mask.astype(np.uint8), (sub.shape[1], sub.shape[0]),
                                         interpolation=cv2.INTER_NEAREST)
                    mask_bool = resized.astype(bool)
                except Exception:
                    # If anything goes wrong, skip this mask to avoid crashing the whole pipeline
                    mask_bool = None
            else:
                mask_bool = mask.astype(bool)

            if mask_bool is not None:
                sub[mask_bool] = 255
                full_image_mask[y1:y2, x1:x2] = sub

            if overlay_img is not None:
                img_to_draw = blend_images_with_mask(overlay_img, img_to_draw,
                                                     full_image_mask * blend_alpha)
    return img_to_draw


class ResultsAnnotator:
    """
    Visualize results of the pipeline

    TODO mask2bbox and morphological operations can definitely be optimized
    """

    def __init__(self, devmode=False, roi=[0, 0, 0, 0]):
        self.devmode = devmode

        self.lw = 4
        self.tf = max(self.lw - 1, 1)  # font thickness
        self.sf = self.lw / 3  # font scale
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.draw_obstacle_dist = True
        self.roi = roi

        self.annotation_settings_human = ((50, 50, 50), (0xff, 0xff, 0xff), 0.7, True)
        self.annotation_settings_obstacle = ((0, 215, 0), (0xff, 0xff, 0xff), 0.8, False)
        self.annotation_settings_car = ((215, 0, 0), (0xff, 0xff, 0xff), 0.8, False)

    @staticmethod
    def mask2bbox(mask):
        a = np.where(mask > 0)
        bbox = np.min(a[1]), np.min(a[0]), np.max(a[1]), np.max(a[0])
        return bbox

    def draw_frame_index(self, frame, frame_idx):
        cv2.putText(frame, f"Frame: {frame_idx}", (100, 100), self.font, fontScale=self.sf * 2, color=(106, 0, 255),
                    thickness=(self.tf + 1) * 2, lineType=cv2.LINE_AA)

    def annotate(self, frame, box, pose, labels, ann_settings, crop_fov: bool = False):
        color_bg, color_fg, alpha, is_human = ann_settings

        if crop_fov:
            self.lw = 2
            self.tf = max(self.lw - 1, 1)  # font thickness
            self.sf = self.lw / 3  # font scale
        else:
            self.lw = 4
            self.tf = max(self.lw - 1, 1)  # font thickness
            self.sf = self.lw / 3  # font scale

        x = (box[0] + box[2]) // 2
        if is_human:
            p1 = int(x), int(box[1]) - 39
        else:
            p1 = int(x), max(int(box[1]) - 200, 400)

        p_m = (int(x), p1[1] + 30)
        marker_size = 20
        marker_thickness = 7
        if crop_fov:
            marker_size // 2
            marker_thickness // 2

        cv2.drawMarker(frame, p_m, color_bg, cv2.MARKER_TRIANGLE_DOWN,
                       marker_size, marker_thickness)

        n_l = len(labels)
        if n_l == 0:
            return

        # h is the approximate row height estimate, w is max width
        pad = 10
        w = max([cv2.getTextSize(label, self.font, fontScale=self.sf, thickness=self.tf + 1)[0][0] for label in labels])
        h = cv2.getTextSize(labels[0], self.font, fontScale=self.sf, thickness=self.tf + 1)[0][1] + pad
        x, y = p1
        w_box, h_box = w + 2 * pad, n_l * h + 2 * pad

        box = np.zeros((h_box, w_box, 3), dtype=np.uint8)
        box[:] = color_bg
        for i, label in enumerate(labels):
            # Center row
            ww = cv2.getTextSize(label, self.font, fontScale=self.sf, thickness=self.tf + 1)[0][0]
            x_la = w_box // 2 - ww // 2
            y_la = (i + 1) * h
            cv2.putText(box, label, (x_la, y_la), self.font, self.sf, color_fg, thickness=self.tf + 1,
                        lineType=cv2.LINE_AA)

        # frame[y - h_box: y, x - w_box // 2: x - w_box // 2 + w_box] = box

        y1 = min(max(y - h_box, 0), frame.shape[0] - h_box)
        y2 = y1 + h_box

        x1 = min(max(x - w_box // 2, 0), frame.shape[1] - w_box)
        x2 = x1 + w_box

        box_img = frame[y1:y2, x1:x2]

        box_alpha = alpha * box + (1 - alpha) * box_img
        box_img[...] = box_alpha.astype(np.uint8)

        if self.devmode and pose is not None:
            draw_human_pose(frame, pose)
        # cv2.circle(frame, p1, 10, (0xff, 0, 0), -1)
        return frame

    def draw_obstacle(self, frame, obstacle, nxy,
                      color=(0, 255, 0)):
        x1, y1, x2, y2 = obstacle.bbox

        cv2.rectangle(frame, (x1, y1), (x2, y2),
                      (0, 0, 255), 2)

        return frame

    def draw_plate(self, frame, car, color=(0, 255, 0)):
        if car.plate is not None:
            x1, y1, x2, y2 = car.plate

            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            cv2.rectangle(frame, (x1, y1),
                          (x2, y2), (0, 0, 255), 2)

        return frame

    def draw_human(self, frame, mask, bbox, nxyz):
        x1, y1, x2, y2 = bbox
        mask, contour = get_masks_for_visualization(mask)

        frame[y1:y2, x1:x2][mask] = nxyz[y1:y2, x1:x2][mask]
        frame[y1:y2, x1:x2][contour] = (0, 0, 255)

        return frame

    def process(self, frame, frame_idx, people: List[Person], cars: List["Car"], 
                obstacles: List[DetectedObject], nxy, nxyz, crop_fov: bool = False):

        for person in people:
            bbox = x1, y1, x2, y2 = person.bbox.astype(int)
            mask = person.mask

            if mask is not None:
                mask = mask[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                if mask.sum() == 0:
                    continue
                self.draw_human(frame, mask, bbox, nxyz)

        for obstacle in obstacles:
            self.draw_obstacle(frame, obstacle, nxy)

        for car in cars:
            self.draw_obstacle(frame, car.detection, nxy,
                               color=self.annotation_settings_car[0])
            self.draw_plate(frame, car)
        # print(f"draw obstacle: {time.time()-t1}")

        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # frame = frame[:,:,::-1]
        # print(f"convert 2: {time.time()-t1}")

        cv2.rectangle(frame, (self.roi[0], self.roi[1]), (self.roi[2], self.roi[3]), (255,0,0), 3)

        for obstacle in obstacles:
            labels = []
            if self.draw_obstacle_dist:
                if obstacle.dist < 500:
                    labels.append(f"R={obstacle.dist:.1f}")
                else:
                    labels.append(f"R=>{500.0:.1f}")
            labels.append(f"{str.upper(obstacle.name.split()[0])}")
            # self.annotate_obstacle_old(frame, obstacle.bbox, labels)
            self.annotate(frame, obstacle.bbox, None, labels[::-1], self.annotation_settings_obstacle, crop_fov)
        # print(f"annotate obstacle: {time.time()-t1}")

        for person in people:
            track_id = person.id
            box = person.bbox
            if person.has_pose:
                pose = person.pose.astype(int)
            else:
                pose = None

            measurements = person.meas

            labels = [
                f'PERSON #{track_id}'
            ]
            if measurements and not np.isnan(measurements['dist']):
                labels += [f"Xo: {measurements['X']:.1f} m",
                           f"Yo: {measurements['Y']:.1f} m",
                           f"Zo: {measurements['Z']:.1f} m",
                           f"Range: {measurements['dist']:.1f} m",]

            self.annotate(frame, box, pose, labels, self.annotation_settings_human, crop_fov)

        for car in cars:
            box = car.detection.bbox

            labels = [
                f'{car.detection.name} #{car.id}',
                # f"Range: {car.dist:.1f} m"
            ]

            if car.ocr is not None:
                try:
                    # sorted_data = sorted(car.ocr, key=lambda x: x[1], reverse=True)
                    # #sorted_names = [re.sub(r'\W+', '', i[1]) for i in sorted_data]

                    # if len(sorted_names[0]) >= 6:
                    #     labels.append(f"OCR: {sorted_names[0]}")
                    # else:
                    #     labels.append(f"OCR: {sorted_names[:2]}")
                    filtered_ocr = re.sub(r"[^А-Яа-яA-Za-z0-9]", "", car.ocr[0][0][0])
                    labels.append(f"OCR: {filtered_ocr.upper()}")
                except:
                    print()

            if car.velocity is not None and not np.isnan(car.velocity) and car.velocity != 0:
                labels.append(f"Speed: {car.velocity:.1f} mph")

            # if car.ocr is not None:
            #     cv2.rectangle(frame, (2560 - 300, 0 + 90), (2560 - 5, 0 + 200), (255,0,0), -1)
            #     cv2.putText(frame, f'{car.detection.name} #{car.id}', (2560 - 280, 0 + 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
            #     cv2.putText(frame, f'OCR: {car.ocr}', (2560 - 280, 0 + 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)

            self.annotate(frame, box, None, labels, self.annotation_settings_car, crop_fov)

            # self.annotate_human_old(frame, box, labels)
        # print(f"annotate human: {time.time()-t1}")

        self.draw_frame_index(frame, frame_idx)
        # print(f"draw frame index: {time.time()-t1}")

        return frame


def visualize_pose_vectors_opencv(img, person,
                                  head_to_ankle_vector,
                                  side_orientation_vector,
                                  conf_thresh=0.3):
    """
    Visualizes a COCO-format pose and two computed vectors using OpenCV.

    The function:
      - Computes the person center as the mean of keypoints with confidence >= conf_thresh.
      - Shifts all keypoints so that the person center is mapped to the center of the image.
      - Draws keypoints as circles and labels them with their index.
      - Draws the head-to-ankle vector (if available) in green and the side orientation
        vector (if available) in purple, both starting at the person center.

    Parameters:
      pose: np.ndarray of shape (17, 2) with keypoint coordinates.
      pose_confidence: np.ndarray of shape (17,) with keypoint confidences.
      head_to_ankle_vector: np.ndarray of shape (2,) or None.
      side_orientation_vector: np.ndarray of shape (2,) or None.
      conf_thresh: minimum confidence to consider a keypoint valid.
      img_size: tuple (width, height) for the output image.

    Returns:
      The image (numpy.ndarray) with the visualizations.
    """
    img = img.copy()
    # Define the center of the image.
    x1, y1, x2, y2 = person.bbox
    x_c = (x1 + x2) / 2
    y_c = (y1 + y2) / 2
    center_img = np.array([x_c, y_c], dtype=np.float32)

    # Draw the computed person center (should map to the center of the image).
    cv2.circle(img, tuple(center_img.astype(int)), radius=6, color=(0, 0, 255), thickness=-1)

    # For drawing arrows, we start at the person center (i.e. origin in the shifted coordinate system).
    origin = tuple(center_img.astype(int))

    # Draw the head-to-ankle vector in green.
    if head_to_ankle_vector is not None:
        # The vector is a displacement. The arrow tip is origin + vector.
        tip = (int(origin[0] + head_to_ankle_vector[0]),
               int(origin[1] + head_to_ankle_vector[1]))
        cv2.arrowedLine(img, origin, tip, color=(0, 255, 0), thickness=10,
                        tipLength=0.1)

    # Draw the side orientation vector in purple.
    if side_orientation_vector is not None:
        tip = (int(origin[0] + side_orientation_vector[0]),
               int(origin[1] + side_orientation_vector[1]))
        cv2.arrowedLine(img, origin, tip, color=(255, 0, 255), thickness=10,
                        tipLength=0.1)

    return img
