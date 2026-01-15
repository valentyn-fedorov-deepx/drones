import numpy as np
import cv2
from itertools import product
from typing import Optional, List

from src.cv_module.people.person import Person
from src.cv_module.people.people_pose_classifier import POSE_LABELS, POSE_LABELS_TO_SIDE, LEFT_LABELS, RIGHT_LABELS, HEAD_LABELS


MISSING_POSE_POINTS_WE_CARE_ABOUT = ["l_shoulder",
                                     "r_shoulder",
                                    #  "l_elbow",
                                    #  "r_elbow",
                                    #  "l_wrist",
                                    #  "r_wrist",
                                     "l_hip",
                                     "r_hip",
                                     "l_knee",
                                     "r_knee",
                                     "l_ankle",
                                     "r_ankle"]

MISSING_POSE_POINTS_WE_CARE_ABOUT_INDICES = [POSE_LABELS[point_name] for point_name in MISSING_POSE_POINTS_WE_CARE_ABOUT]

POSE_LABELS_OCCLUSION_WEIGTH = {
        "nose": 1,
        "l_eye": 1,
        "r_eye": 1,
        "l_ear": 1,
        "r_ear": 1,
        "l_shoulder": 2,
        "r_shoulder": 2,
        "l_elbow": 1,
        "r_elbow": 1,
        "l_wrist": 1,
        "r_wrist": 1,
        "l_hip": 2,
        "r_hip": 2,
        "l_knee": 13,
        "r_knee": 14,
        "l_ankle": 15,
        "r_ankle": 16
    }


def check_mask_contains_pose(person: Person, conf_thresh: float = 0.85):
    if person.mask is None or not person.has_pose:
        return False

    x1, y1, x2, y2 = person.bbox
    w, h = x2 - x1, y2 - y1
    area = w * h
    coef = np.array([3.40432538e-05, 7.94829929e+00])
    poly1d_fn = np.poly1d(coef)
    estimated_kernel_size = int(poly1d_fn(area))
    kernel = np.ones((estimated_kernel_size, estimated_kernel_size),
                     np.uint8)
    dilated_mask = cv2.dilate(person.mask, kernel,
                              iterations=3)

    person_pose_points_scaled = person.pose.round().astype(int) - np.array([x1, y1])
    person_pose_points = person_pose_points_scaled[person.pose_conf > conf_thresh]
    points_inside_mask = dilated_mask[person_pose_points[:, 1], person_pose_points[:, 0]]

    # from src.cv_module.visualization import draw_human_pose
    # vis = draw_human_pose(dilated_mask * 255, person_pose_points_scaled, color=(126))
    # cv2.imwrite('debug_dilated_mask.png', vis)

    return points_inside_mask.all()


def check_missing_pose_points(person: Person, conf_thresh: float = 0.7):
    if not person.has_pose:
        return False

    vizible_points = person.pose_conf < conf_thresh
    vizible_points_filtered = vizible_points[MISSING_POSE_POINTS_WE_CARE_ABOUT_INDICES]

    missing_points = vizible_points_filtered.sum()

    return missing_points > 0


def classify_occlusion(person: Person):
    # import ipdb; ipdb.set_trace()
    if check_missing_pose_points(person):
        return True

    if not check_mask_contains_pose(person):
        return True

    return False


def calculate_body_orientation(pose: np.ndarray, 
                               pose_confidence: np.ndarray,
                               min_conf: float = 0.5):
    """
    Calculate normalized vertical and horizontal orientation vectors of a person
    from COCO-format pose estimation.

    **Vertical Orientation:**
    Primary: Use shoulder-to-hip vectors on each side.
      - Left vertical: from left shoulder (5) to left hip (11)
      - Right vertical: from right shoulder (6) to right hip (12)
    If both sides are available, the two unit vectors are averaged.
    If not, we fall back to a top-to-bottom method (using shoulders, head,
    ankles, hips, or knees). Finally, if no vertical orientation is found, we
    attempt a face-based fallback: if face keypoints (eyes and ears) are detected,
    we compute the vector between them, rotate that vector by 90° (i.e. take the
    perpendicular), and adjust its sign so that the result points from top to bottom.

    **Horizontal Orientation:**
    Primary: Use left vs. right keypoints (shoulder and hip) to form a vector.
    If insufficient keypoints are available, try leg keypoints (knees/ankles),
    and finally fall back to the perpendicular of the vertical vector (adjusted so
    that its x component is positive).

    Both returned vectors are normalized.

    Args:
      pose: np.ndarray of shape (17, 2) containing keypoint (x,y) coordinates.
      pose_confidence: np.ndarray of shape (17,) with confidence scores for each keypoint.
      min_conf: float, minimum confidence threshold to consider a keypoint valid.

    Returns:
      vertical_vec: np.ndarray of shape (2,) representing the vertical direction (top -> bottom)
      horizontal_vec: np.ndarray of shape (2,) representing the horizontal direction (left -> right)
    """

    # -------------------------------
    # 1. Compute vertical orientation from shoulder-to-hip pairs.
    # -------------------------------
    vertical_candidates = []

    if pose is None:
        return None, None

    # Left side: left shoulder (5) -> left hip (11)
    if pose_confidence[5] > min_conf and pose_confidence[11] > min_conf:
        left_vec = pose[11] - pose[5]
        norm = np.linalg.norm(left_vec)
        if norm > 1e-5:
            vertical_candidates.append(left_vec / norm)

    # Right side: right shoulder (6) -> right hip (12)
    if pose_confidence[6] > min_conf and pose_confidence[12] > min_conf:
        right_vec = pose[12] - pose[6]
        norm = np.linalg.norm(right_vec)
        if norm > 1e-5:
            vertical_candidates.append(right_vec / norm)

    if vertical_candidates:
        vertical_vec = np.mean(vertical_candidates, axis=0)
        norm = np.linalg.norm(vertical_vec)
        if norm > 1e-5:
            vertical_vec = vertical_vec / norm
        else:
            vertical_vec = np.array([0, 1], dtype=float)
    else:
        # Fallback: try to use top-to-bottom approach using other keypoints.
        top_points = []
        bottom_points = []
        # Try to get top from shoulders first.
        for i in [5, 6]:
            if pose_confidence[i] > min_conf:
                top_points.append(pose[i])
        if not top_points:
            # Fall back to head keypoints: nose, eyes, ears (indices 0-4)
            for i in range(5):
                if pose_confidence[i] > min_conf:
                    top_points.append(pose[i])
        # Try bottom from ankles.
        for i in [15, 16]:
            if pose_confidence[i] > min_conf:
                bottom_points.append(pose[i])
        if not bottom_points:
            # Fall back to hips/knees (indices 11-16)
            for i in [11, 12, 13, 14, 15, 16]:
                if pose_confidence[i] > min_conf:
                    bottom_points.append(pose[i])
        if top_points and bottom_points:
            top_mean = np.mean(top_points, axis=0)
            bottom_mean = np.mean(bottom_points, axis=0)
            v = bottom_mean - top_mean
            norm = np.linalg.norm(v)
            if norm > 1e-5:
                vertical_vec = v / norm
            else:
                vertical_vec = np.array([0, 1], dtype=float)
        else:
            vertical_vec = None

    # -------------------------------
    # 1.b Additional fallback: Use face keypoints.
    #     If vertical_vec is still not found, check for face keypoints (eyes and ears).
    #     For each side, compute the vector from eye to ear, then take its perpendicular,
    #     and adjust the sign so that the y-component is positive (downwards).
    # -------------------------------
    if vertical_vec is None:
        face_candidates = []
        # Left face: left eye (1) and left ear (3)
        if pose_confidence[1] > min_conf and pose_confidence[3] > min_conf:
            vec = pose[3] - pose[1]
            # Rotate vec by +90°: (x,y) -> (y, -x)
            candidate = np.array([vec[1], -vec[0]])
            # Ensure the candidate points downward (y positive in image coordinates)
            if candidate[1] < 0:
                candidate = -candidate
            norm = np.linalg.norm(candidate)
            if norm > 1e-5:
                face_candidates.append(candidate / norm)
        # Right face: right eye (2) and right ear (4)
        if pose_confidence[2] > min_conf and pose_confidence[4] > min_conf:
            vec = pose[4] - pose[2]
            candidate = np.array([vec[1], -vec[0]])
            if candidate[1] < 0:
                candidate = -candidate
            norm = np.linalg.norm(candidate)
            if norm > 1e-5:
                face_candidates.append(candidate / norm)
        if face_candidates:
            vertical_vec = np.mean(face_candidates, axis=0)
            norm = np.linalg.norm(vertical_vec)
            if norm > 1e-5:
                vertical_vec = vertical_vec / norm
            else:
                vertical_vec = np.array([0, 1], dtype=float)

    # -------------------------------
    # 2. Compute horizontal orientation.
    # -------------------------------
    # Primary: use shoulder/hip keypoints to form left and right groups.
    # left_points = []
    # right_points = []
    horizontal_test_points = list()

    left_points = [5, 11]
    right_points = [6, 12]
    for left_point, right_point in zip(left_points, right_points):
        if pose_confidence[left_point] > min_conf and pose_confidence[right_point] > min_conf:
            horizontal_test_points.append(pose[right_point] - pose[left_point])

    if horizontal_test_points:
        horizontal_test_points = np.array(horizontal_test_points).reshape(-1, 2)
        h = horizontal_test_points.mean(0)
        norm = np.linalg.norm(h)
        if norm > 1e-5:
            horizontal_vec = h / norm
        else:
            horizontal_vec = np.array([1, 0], dtype=float)
    else:
        # Fallback to face points
        left_points = [1, 3]
        right_points = [2, 4]
        for left_point, right_point in zip(left_points, right_points):
            if pose_confidence[left_point] > min_conf and pose_confidence[right_point] > min_conf:
                horizontal_test_points.append(pose[right_point] - pose[left_point])

        if horizontal_test_points:
            horizontal_test_points = np.array(horizontal_test_points).reshape(-1, 2)
            h = horizontal_test_points.mean(0)
            norm = np.linalg.norm(h)
            if norm > 1e-5:
                horizontal_vec = h / norm
            else:
                horizontal_vec = np.array([1, 0], dtype=float)
        else:
            horizontal_vec = None

        # Final fallback: if horizontal vector cannot be computed, use the perpendicular of vertical.
        if horizontal_vec is None and vertical_vec is not None:
            candidate = np.array([vertical_vec[1], -vertical_vec[0]])

            if candidate[0] < 0:
                candidate = -candidate
            horizontal_vec = candidate

    # Ensure horizontal vector is normalized.
    if horizontal_vec is not None:
        norm = np.linalg.norm(horizontal_vec)
        if norm > 1e-5:
            horizontal_vec = horizontal_vec / norm
        else:
            horizontal_vec = np.array([1, 0], dtype=float)

    if vertical_vec is not None:
        vertical_vec_normed = vertical_vec / np.linalg.norm(vertical_vec)
    else:
        vertical_vec_normed = None

    if horizontal_vec is not None:
        horizontal_vec_normed = horizontal_vec / np.linalg.norm(horizontal_vec)
    else:
        horizontal_vec_normed = None

    return vertical_vec_normed, horizontal_vec_normed


def check_missing_parts(pose: np.ndarray,
                        pose_conf: np.ndarray,
                        pose_info: List[str],
                        conf_thresh: float = 0.5):
    """
    Iteratively checks for occlusion in the top/bottom and left/right regions of a pose.

    Parameters:
      pose (np.ndarray): Array of shape (17, 2) containing (x, y) keypoint coordinates (COCO order).
      pose_conf (np.ndarray): Array of shape (17,) with confidence values for each keypoint.
      conf_thresh (float): Confidence threshold; keypoints with confidence below this are treated as missing.

    Returns:
      missing_side (str or None): "left", "right", "both", or None (if neither lateral side is occluded).
      missing_top_bottom (str or None): "top", "bottom", "both", or None (if neither vertical region is occluded).

    Iterative logic:
      - Top: Check head keypoints first ([0,1,2,3,4]). If none are detected, then the top is occluded.
      - Bottom: Check the ankles ([15,16]). If neither is detected, then the bottom is occluded.
      - Left/Right: For each side, check two chains:
            * Left arm chain: [9, 7, 5] (wrist, elbow, shoulder)
            * Left leg chain: [15, 13, 11] (ankle, knee, hip)
            * Right arm chain: [10, 8, 6] (wrist, elbow, shoulder)
            * Right leg chain: [16, 14, 12] (ankle, knee, hip)
        A side is considered visible if at least one keypoint in either chain is detected.
    """

    if pose_conf is None:
        return None, None

    # ---------- TOP/BOTTOM CHECK ----------
    # Top: Check head keypoints (nose, eyes, ears)
    head_indices = [0, 1, 2, 3, 4]
    head_visible = any(pose_conf[i] >= conf_thresh for i in head_indices)
    top_occluded = not head_visible

    # Bottom: Check ankle keypoints
    ankle_indices = [15, 16]
    ankle_visible = any(pose_conf[i] >= conf_thresh for i in ankle_indices)
    bottom_occluded = not ankle_visible

    if top_occluded and bottom_occluded:
        missing_top_bottom = ["top", "bottom"]
    elif top_occluded:
        missing_top_bottom = ["top"]
    elif bottom_occluded:
        missing_top_bottom = ["bottom"]
    else:
        missing_top_bottom = None

    # ---------- LEFT/RIGHT CHECK ----------
    def chain_visible(chain):
        """Return True if any keypoint in the provided chain is detected."""
        for idx in chain:
            if pose_conf[idx] >= conf_thresh:
                return True
        return False

    # Left side: use an arm chain and a leg chain.
    # left_arm_chain = [9, 7, 5]      # left wrist, left elbow, left shoulder
    # left_leg_chain = [15, 13, 11]    # left ankle, left knee, left hip
    left_limbs_chain = [9, 7, 15, 13]
    if 'sideways' in pose_info:
        left_limbs_chain += [11, 5]
    # left_visible = chain_visible(left_arm_chain) or chain_visible(left_leg_chain)
    left_visible = chain_visible(left_limbs_chain)

    # Right side: similarly, define two chains.
    # right_arm_chain = [10, 8, 6]     # right wrist, right elbow, right shoulder
    # right_leg_chain = [16, 14, 12]   # right ankle, right knee, right hip
    right_limbs_chain = [10, 8, 16, 14]
    if 'sideways' in pose_info:
        right_limbs_chain += [6, 12]
    # right_visible = chain_visible(right_arm_chain) or chain_visible(right_leg_chain)x
    right_visible = chain_visible(right_limbs_chain)

    if (not left_visible) and (not right_visible):
        missing_side = ["left", "right"]
    elif not left_visible:
        missing_side = ["left"]
    elif not right_visible:
        missing_side = ["right"]
    else:
        missing_side = None

    return missing_side, missing_top_bottom


def bresenham_line(y1: int, x1: int, y2: int, x2: int):
    """
    Returns a list of (row, col) integer coordinates along the line
    from (y1, x1) to (y2, x2) using Bresenham's algorithm.
    """
    points = []

    # Differences and step directions
    dx = abs(x2 - x1)
    sx = 1 if x1 < x2 else -1
    dy = -abs(y2 - y1)
    sy = 1 if y1 < y2 else -1

    # Error term
    err = dx + dy

    cur_x, cur_y = x1, y1

    while True:
        points.append((cur_y, cur_x))

        # Check if finished
        if cur_x == x2 and cur_y == y2:
            break

        e2 = 2 * err
        if e2 >= dy:
            err += dy
            cur_x += sx
        if e2 <= dx:
            err += dx
            cur_y += sy

    return points


def find_uncovered_line_midpoints(keypoints: np.ndarray,
                                  mask: np.ndarray,
                                  confs: np.ndarray = None,
                                  conf_threshold: float = 0.0) -> list:
    """
    Find midpoints of uncovered (outside) segments on two lines:
      - Left shoulder to right shoulder
      - Left hip to right hip

    :param keypoints: np.ndarray of shape (17, 3) in COCO format
                      or (17, 2) if confidence is separate.
                      Indices follow COCO:
                         left_shoulder = 5
                         right_shoulder = 6
                         left_hip = 11
                         right_hip = 12
                      keypoints[i] = [x, y, (optional) visibility/confidence]
    :param mask: Binary np.ndarray (H x W) where 1 = covered, 0 = not covered
    :param confs: (Optional) np.ndarray of shape (17,) with confidence scores
                  per keypoint. If provided, only use points above conf_threshold.
    :param conf_threshold: Minimum confidence to consider a keypoint valid.
    :return: A list of midpoints (x, y), one per contiguous uncovered segment found.
    """
    # COCO indices of interest
    left_shoulder_idx  = 5
    right_shoulder_idx = 6
    left_hip_idx       = 11
    right_hip_idx      = 12

    # Lines to check: (index1, index2)
    lines_to_check = [
        (left_shoulder_idx, right_shoulder_idx),
        (left_hip_idx, right_hip_idx),
    ]

    uncovered_midpoints = []

    for (idx1, idx2) in lines_to_check:
        # If we have per-keypoint confidences, skip line if either keypoint
        # has confidence below threshold
        if confs is not None:
            if confs[idx1] < conf_threshold or confs[idx2] < conf_threshold:
                continue

        # Extract keypoints: (x1, y1), (x2, y2)
        x1, y1 = keypoints[idx1][0], keypoints[idx1][1]
        x2, y2 = keypoints[idx2][0], keypoints[idx2][1]

        # Round to integer pixel coords
        x1_i, y1_i = int(round(x1)), int(round(y1))
        x2_i, y2_i = int(round(x2)), int(round(y2))

        # Get the list of (row, col) using Bresenham
        line_points = bresenham_line(y1_i, x1_i, y2_i, x2_i)

        # Separate out contiguous segments of points outside the mask
        contiguous_outside_segments = []
        current_segment = []

        for (r, c) in line_points:
            # Check bounds
            if not (0 <= r < mask.shape[0] and 0 <= c < mask.shape[1]):
                # Out-of-bounds => treat as uncovered
                current_segment.append((r, c))
            else:
                if mask[r, c] == 0:  # uncovered
                    current_segment.append((r, c))
                else:
                    # We just finished a segment of uncovered
                    if len(current_segment) > 0:
                        contiguous_outside_segments.append(current_segment)
                        current_segment = []

        # Close the last segment if it ends uncovered
        if len(current_segment) > 0:
            contiguous_outside_segments.append(current_segment)

        # For each uncovered segment, compute the midpoint
        for seg in contiguous_outside_segments:
            seg_arr = np.array(seg)
            mean_r = np.mean(seg_arr[:, 0])
            mean_c = np.mean(seg_arr[:, 1])

            # Return midpoint in (x, y) format => (mean_c, mean_r)
            uncovered_midpoints.append(np.array((mean_c, mean_r), dtype=int))

    return uncovered_midpoints


def get_proposed_occlusion_points(person: Person,
                                  vertical_orientation: np.ndarray,
                                  horizontal_orientation: np.ndarray,
                                  missing_side: Optional[List[str]],
                                  missing_top_bottom: Optional[List[str]]):
    points = list()
    bboxes = list()
    if not person.has_pose:
        return points

    # Ignoring missing top points
    if missing_top_bottom is not None:
        if "top" in missing_top_bottom:
            missing_top_bottom.remove("top")
            if not missing_top_bottom:
                missing_top_bottom = None

    # Adding keypoints that are not included in the mask
    x1, y1, x2, y2 = person.bbox
    w, h = x2 - x1, y2 - y1
    area = w * h
    coef = np.array([3.40432538e-05, 7.94829929e+00])
    poly1d_fn = np.poly1d(coef)
    estimated_kernel_size = int(poly1d_fn(area))
    kernel = np.ones((estimated_kernel_size, estimated_kernel_size),
                     np.uint8)
    dilated_mask = cv2.dilate(person.mask, kernel,
                              iterations=1)

    detected_keypoints_idx = np.where(person.pose_conf > 0.8)[0]
    detected_keypoints = person.pose[detected_keypoints_idx].round().astype(int)
    detected_points_in_mask = ~dilated_mask[detected_keypoints[:, 1], detected_keypoints[:, 0]].astype(bool)
    if detected_points_in_mask.any():
        keypoints_outside_mask = detected_keypoints[detected_points_in_mask]
        for point_outside_mask in keypoints_outside_mask:
            points.append(point_outside_mask)
            bboxes.append(create_bbox_for_occlusion_point(point_outside_mask, person))

    # Adding keypoints for gap in mask
    gap_points = find_uncovered_line_midpoints(person.pose, person.mask, person.pose_conf)
    if gap_points:
        points += gap_points
        gap_points_bboxes = [create_bbox_for_occlusion_point(point, person) for point in gap_points]
        bboxes += gap_points_bboxes

    # Projecting points from the bbox
    if missing_side is None and missing_top_bottom is None:
        return points

    if vertical_orientation is None:
        vertical_orientation = np.array([0, 1])

    if horizontal_orientation is None:
        horizontal_orientation = np.array([-1, 0])

    x1, y1, x2, y2 = person.bbox.round().astype(int)
    w = x2 - x1
    h = y2 - y1
    x_c = (x1 + x2) // 2
    y_c = (y1 + y2) // 2

    vertical_move = h * 0.1
    horizontal_move = w * 0.1

    if missing_top_bottom is not None:
        for name in missing_top_bottom:
            if name == "bottom":
                origin_point = np.array([x_c, y2])
                moved_point = origin_point + vertical_orientation*vertical_move
            elif name == 'top':
                origin_point = np.array([x_c, y1])
                moved_point = origin_point - vertical_orientation*vertical_move

            points.append(moved_point.round().astype(int))

    if missing_side is not None:
        for name in missing_side:
            if name == 'left':
                if "front" in person.pose_info:
                    origin_point = np.array([x2, y_c])
                    moved_point = origin_point + horizontal_orientation * horizontal_move * -1
                elif "back" in person.pose_info:
                    origin_point = np.array([x1, y_c])
                    moved_point = origin_point + horizontal_orientation * horizontal_move * -1
                else:
                    continue

            elif name == "right":
                if "front" in person.pose_info:
                    origin_point = np.array([x1, y_c])
                    moved_point = origin_point + horizontal_orientation * horizontal_move
                elif "back" in person.pose_info:
                    origin_point = np.array([x2, y_c])
                    moved_point = origin_point + horizontal_orientation * horizontal_move

                else:
                    continue

            points.append(moved_point.round().astype(int))

    if missing_side is not None and missing_top_bottom is not None:
        for vertical_name in missing_top_bottom:
            if vertical_name == 'bottom':
                y_origin = y2
                move_vector_vertical_part = vertical_orientation * vertical_move
            elif vertical_name == 'top':
                y_origin = y1
                move_vector_vertical_part = vertical_orientation * vertical_move * -1

            for horizontal_name in missing_side:
                if horizontal_name == 'left':
                    if "front" in person.pose_info:
                        x_origin = x2
                        move_vector_horizontal_part = horizontal_orientation * horizontal_move
                    elif "back" in person.pose_info:
                        x_origin = x1
                        move_vector_horizontal_part = horizontal_orientation * horizontal_move * -1
                    else:
                        continue
                else:
                    if "front" in person.pose_info:
                        x_origin = x1
                        move_vector_horizontal_part = horizontal_orientation * horizontal_move * -1
                    elif "back" in person.pose_info:
                        x_origin = x2
                        move_vector_horizontal_part = horizontal_orientation * horizontal_move
                    else:
                        continue

                origin_point = np.array([x_origin, y_origin])
                moved_point = origin_point + move_vector_vertical_part + move_vector_horizontal_part

                points.append(moved_point.round().astype(int))

    # Double check if points are inside the image
    mask_h, mask_w = person.mask.shape[:2]
    filtered_points = list()
    for point in points:
        x, y = point
        if x >= 0 and x < mask_w and y >= 0 and y < mask_h:
            filtered_points.append(point)

    return filtered_points


def create_bbox_for_occlusion_point(occlusion_point, person, w=128, h=128):
    # w, h = 64, 64
    # w, h = 128, 128
    half_w = w // 2
    half_h = h // 2
    x_c, y_c = occlusion_point

    x1 = max(x_c - half_w, 0)
    y1 = max(y_c - half_h, 0)
    x2 = x_c + half_w
    y2 = y_c + half_h

    bbox = x1, y1, x2, y2
    return bbox
