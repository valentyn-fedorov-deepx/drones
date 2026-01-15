import numpy as np
from typing import List, Optional

from .person import Person

POSE_LABELS = {
        "nose": 0,
        "l_eye": 1,
        "r_eye": 2,
        "l_ear": 3,
        "r_ear": 4,
        "l_shoulder": 5,
        "r_shoulder": 6,
        "l_elbow": 7,
        "r_elbow": 8,
        "l_wrist": 9,
        "r_wrist": 10,
        "l_hip": 11,
        "r_hip": 12,
        "l_knee": 13,
        "r_knee": 14,
        "l_ankle": 15,
        "r_ankle": 16
    }


HEAD_LABELS = {
    "nose": 0,
    "l_eye": 1,
    "r_eye": 2,
    "l_ear": 3,
    "r_ear": 4,
}

# Left side labels
LEFT_LABELS = {
    "l_eye": 1,
    "l_ear": 3,
    "l_shoulder": 5,
    "l_elbow": 7,
    "l_wrist": 9,
    "l_hip": 11,
    "l_knee": 13,
    "l_ankle": 15
}

# Right side labels
RIGHT_LABELS = {
    "r_eye": 2,
    "r_ear": 4,
    "r_shoulder": 6,
    "r_elbow": 8,
    "r_wrist": 10,
    "r_hip": 12,
    "r_knee": 14,
    "r_ankle": 16
}


POSE_LABELS_TO_SIDE_FROM_NAMES = {
        "nose": None,
        "l_eye": 'left',
        "r_eye": 'right',
        "l_ear": 'left',
        "r_ear": 'right',
        "l_shoulder": 'left',
        "r_shoulder": 'right',
        "l_elbow": 'left',
        "r_elbow": 'right',
        "l_wrist": 'left',
        "r_wrist": 'right',
        "l_hip": 'left',
        "r_hip": 'right',
        "l_knee": 'left',
        "r_knee": 'right',
        "l_ankle": 'left',
        "r_ankle": 'right'
    }

POSE_LABELS_TO_SIDE = {POSE_LABELS[point_name]: side for point_name, side in POSE_LABELS_TO_SIDE_FROM_NAMES.items()}

POSE_LABELS_TOP_BOTTOM_FROM_NAMES = {
        "nose": "top",
        "l_eye": 'top',
        "r_eye": 'top',
        "l_ear": 'top',
        "r_ear": 'top',
        "l_shoulder": 'top',
        "r_shoulder": 'top',
        "l_elbow": 'top',
        "r_elbow": 'top',
        "l_wrist": None,
        "r_wrist": None,
        "l_hip": 'bottom',
        "r_hip": 'bottom',
        "l_knee": 'bottom',
        "r_knee": 'bottom',
        "l_ankle": 'bottom',
        "r_ankle": 'bottom'
    }

POSE_LABELS_TOP_BOTTOM = {POSE_LABELS[point_name]: side for point_name, side in POSE_LABELS_TOP_BOTTOM_FROM_NAMES.items()}


def angle_between(v1, v2):
    dot_product = np.dot(v1, v2)
    magnitudes = np.linalg.norm(v1) * np.linalg.norm(v2)
    angle = np.arccos(dot_product / magnitudes)
    return np.degrees(angle)


class PoseClassifier:
    def __init__(self):
        pass

    def _check_prone(self, person: Person):
        if not person.has_pose:
            return False
        ar = (person.pose[:, 0].max() - person.pose[:, 0].min()) / (person.pose[:, 1].max() - person.pose[:, 1].min())

        if ar > 1.5:
            return True

        return False

    def _get_facing_side(self, person: Person):
        if not person.has_pose:
            return

        sideL = np.linalg.norm(person.pose[6] - person.pose[12])
        sideR = np.linalg.norm(person.pose[5] - person.pose[11])

        shoulders = np.linalg.norm(person.pose[5] - person.pose[6])
        hips = np.linalg.norm(person.pose[11] - person.pose[12])

        ratio = (shoulders + hips) / (sideL + sideR)  # ~0.09when sidewise, 0.54 when right
        if ratio < 0.31:
            if person.pose[:, 1].std() > person.pose[:, 0].std():
                return 'left'
            else:
                return 'right'

    def _check_knees_bent(self, person: Person):
        if not person.has_pose:
            return False

        l_knee_point = person.pose[POSE_LABELS["l_knee"]]
        l_ankle_point = person.pose[POSE_LABELS["l_ankle"]]
        l_hip_point = person.pose[POSE_LABELS["l_hip"]]

        left_leg_bent = self._check_leg_bent(l_knee_point, l_ankle_point,
                                             l_hip_point)

        r_knee_point = person.pose[POSE_LABELS["r_knee"]]
        r_ankle_point = person.pose[POSE_LABELS["r_ankle"]]
        r_hip_point = person.pose[POSE_LABELS["r_hip"]]

        right_leg_bent = self._check_leg_bent(r_knee_point, r_ankle_point,
                                              r_hip_point)

        return any([left_leg_bent, right_leg_bent])

    def _check_standing_sideways(self, person):
        if not person.has_pose:
            return False

        hip_dist = np.linalg.norm(person.pose[POSE_LABELS["l_hip"]] - person.pose[POSE_LABELS["r_hip"]])
        shoulder_dist = np.linalg.norm(person.pose[POSE_LABELS["l_shoulder"]] - person.pose[POSE_LABELS["r_shoulder"]])

        l_hip_shoulder_dist = np.linalg.norm(person.pose[POSE_LABELS["l_shoulder"]] - person.pose[POSE_LABELS["l_hip"]])
        r_hip_shoulder_dist = np.linalg.norm(person.pose[POSE_LABELS["r_shoulder"]] - person.pose[POSE_LABELS["r_hip"]])

        ratio = np.mean([hip_dist, shoulder_dist]) / np.mean([l_hip_shoulder_dist, r_hip_shoulder_dist])

        # if ratio < 0.35:
        if ratio < 0.235:
            return True

        return False

    def _check_leg_bent(self, knee_point: np.ndarray,
                        ankle_point: np.ndarray, hip_point: np.ndarray):
        ankle_point = ankle_point - knee_point
        hip_point = hip_point - knee_point
        angle = angle_between(ankle_point, hip_point)

        if angle < 100:
            return True

        return False

    def _check_hands_ups(self, person: Person):
        if not person.has_pose:
            return False

        pose_thresh = 0.5

        wrists_ids = [POSE_LABELS["l_wrist"], POSE_LABELS["r_wrist"],
                      POSE_LABELS["l_elbow"], POSE_LABELS["r_elbow"]
                      ]
        wrists_filtered = [person.pose[wrist_id] for wrist_id in wrists_ids if person.pose_conf[wrist_id] > pose_thresh]

        if not wrists_filtered:
            return False

        height_threshold_points_ids = [POSE_LABELS["l_ear"], POSE_LABELS["r_ear"],
                                       POSE_LABELS["l_eye"], POSE_LABELS["r_eye"],
                                       POSE_LABELS["l_shoulder"], POSE_LABELS["r_shoulder"],
                                       POSE_LABELS["nose"]
                                       ]

        height_threshold_points_filtered = [person.pose[bone_id] for bone_id in height_threshold_points_ids if person.pose_conf[bone_id] > pose_thresh]
        if not height_threshold_points_filtered:
            return False

        highest_wrist_point = min(wrists_filtered, key=lambda x: x[1])
        highest_threshold_point = min(height_threshold_points_filtered, key=lambda x: x[1])

        if highest_wrist_point[1] < highest_threshold_point[1]:
            return True

        return False

    def _check_facing_view(self, person,
                           face_thresh: float = 0.5,
                           body_thresh: float = 0.5) -> Optional[str]:
        """
        Determines if the person is facing the camera ("front") or away ("back")
        based on COCO pose keypoints and their confidences.

        Parameters:
        pose: np.ndarray of shape (17, 2) with the (x, y) coordinates of keypoints.
        confidences: np.ndarray of shape (17,) with the confidence for each keypoint.
        face_thresh: float, the confidence threshold for considering a keypoint detected.
        body_thresh: float, the confidence threshold for body keypoints.

        Returns:
        "front" if the face is visible (i.e. the person is looking toward the camera),
        "back" if enough body keypoints are seen but no facial keypoints are detected,
        None if there is not enough information to decide.
        """
        if not person.has_pose:
            return None

        # Check that the inputs have the expected shapes.
        if person.pose.shape != (17, 2) or person.pose_conf.shape != (17,):
            raise ValueError("Expected pose of shape (17,2) and confidences of shape (17,)")

        # --- Check face keypoints ---
        # In COCO the face keypoints are: nose (0), left_eye (1), right_eye (2), left_ear (3), right_ear (4).
        face_ids = [0, 1, 2, 3, 4]
        face_confidences = person.pose_conf[face_ids]
        # Count how many facial keypoints are detected with confidence above face_thresh.
        face_detected = face_confidences >= face_thresh
        face_count = np.sum(face_detected)

        # --- Decision Logic ---
        # 1. If the nose is clearly detected, we assume the person is facing the camera.
        if person.pose_conf[0] >= face_thresh:
            return "front"

        # 2. Alternatively, if at least two facial keypoints (eyes/ears) are detected,
        #    we also assume a front view.
        if face_count >= 2:
            return "front"

        left_right_pairs = [("l_shoulder", "r_shoulder"),
                            ("l_hip", "r_hip")]

        front_facing = 0
        back_facing = 0
        for left_point_name, right_point_name in left_right_pairs:
            left_point_idx, right_point_idx = POSE_LABELS[left_point_name], POSE_LABELS[right_point_name]

            left_point_exists = person.pose_conf[left_point_idx] > body_thresh
            right_points_exists = person.pose_conf[right_point_idx] > body_thresh

            if left_point_exists and right_points_exists:
                left_point = person.pose[left_point_idx]
                right_point = person.pose[right_point_idx]

                if left_point[0] > right_point[0]:
                    front_facing += 1
                else:
                    back_facing += 1

        if max(front_facing, back_facing) > 0:
            if front_facing > back_facing:
                return "front"
            elif front_facing < back_facing:
                return "back"

        # Otherwise, there is not enough information (e.g. due to occlusion) to decide.
        return None

    def classify_one(self, person: Person):
        poses = list()
        if self._check_hands_ups(person):
            poses.append("hands_up")

        if self._check_knees_bent(person):
            poses.append("crouching")

        if self._check_standing_sideways(person):
            poses.append('sideways')

        if self._check_prone(person):
            poses.append('prone')

        side = self._get_facing_side(person)
        if side is not None:
            poses.append(side)

        facing_view = self._check_facing_view(person)
        if facing_view is not None:
            poses.append(facing_view)

        return poses

    def classify(self, people: List[Person]):
        for person in people:
            pose_classes = self.classify_one(person)
            person.pose_info = pose_classes

        return people
