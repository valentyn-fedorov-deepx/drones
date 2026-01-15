import numpy as np
from collections import deque
from typing import Tuple, List
from loguru import logger

from src.cv_module.people.person import Person
from .sloth import EMAFilter, MedianFilter
from src.cv_module.detected_object import DetectedObject
from src.cv_module.distance_measurers.measurement import Measurement

EMA_K = 0.05
MEDIAN_WINDOW_SIZE = 6

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


def angle_between(v1, v2):
    dot_product = np.dot(v1, v2)
    magnitudes = np.linalg.norm(v1) * np.linalg.norm(v2)
    angle = np.arccos(dot_product / magnitudes)
    return np.degrees(angle)


class TrackedObjectForDistance:
    def __init__(self, ema_k: float = EMA_K, history_len: int = 120):
        self.history = deque(maxlen=history_len)
        self.timestamp = deque(maxlen=history_len)
        self.filtered_meas = deque(maxlen=history_len)
        self.time_inactive = 0
        self.height_filter = EMAFilter(ema_k)
        self.dist_filter = MedianFilter(MEDIAN_WINDOW_SIZE)


class HeightDistanceMeasurer:
    def __init__(self, focal_length: int, im_size: Tuple[int, int],
                 base_height_in_meters: float, auto_correct: bool = False,
                 use_tracker: bool = True):
        self._use_tracker = use_tracker
        self._focal_length = focal_length
        self.im_size = im_size
        self._auto_correct = auto_correct

        im_center_x = im_size[1] // 2
        im_center_y = im_size[0] // 2

        self.K = np.array([[self._focal_length, 0, im_center_x],
                           [0, self._focal_length, im_center_y],
                           [0, 0, 1]])

        self.K_inv = np.linalg.inv(self.K)
        self.height_in_meters = base_height_in_meters
        self.tracks = dict()
        self.forget = 20

    def check_fit(self, obj: TrackedObjectForDistance):
        if isinstance(obj, Person):
            x1, y1, x2, y2 = obj.bbox
            width = x2 - x1
            height = y2 - y1
            ratio = width / height

            if 0.22 > ratio or ratio > 0.55:
                return False

        return True

    def _get_filtered_meas(self, id: int):
        track = self.tracks.get(id)
        if track is None:
            raise ValueError(f"Track with the id {id} wasn't found")

        new_dist = track.dist_filter.process(track.history[-1]["dist"])

        latest_meas = track.history[-1].copy()
        latest_meas['dist'] = new_dist

        return latest_meas

    def _get_filtered_point(self, tracked_object: TrackedObjectForDistance, points: np.ndarray):
        center_point = points.mean((0, 1))
        heights = list()
        for point in points:
            height = np.mean(point[1, 1] - point[0, 1])
            heights.append(height)

        height = np.mean(heights)

        filtered_height = tracked_object.height_filter.process(height)

        top_left_point_h = (center_point[0], center_point[1] - filtered_height / 2, 1)
        bottom_left_point_h = (center_point[0], center_point[1] + filtered_height / 2, 1)

        return top_left_point_h, bottom_left_point_h

    def _get_person_points(self, obj: Person):
        ankle_indices = np.array([15, 16])
        face_indices = np.array([0, 1, 2, 3, 4])
        person_points = dict()
        x1, y1, x2, y2 = obj.bbox

        if self._auto_correct:
            hands_up = "hands_up" in obj.pose_info
        else:
            hands_up = False

        if obj.has_pose:
            pose_thresh = 0.5

            ankle_conf = obj.pose_conf[ankle_indices]
            looked_ankle_indices = ankle_indices[ankle_conf > pose_thresh]
            if looked_ankle_indices.size > 0:
                ankle_points = obj.pose[looked_ankle_indices]

                lowest_ankle_point = np.r_[ankle_points.max(axis=0), 1]

                face_conf = obj.pose_conf[face_indices]
                looked_face_indices = face_indices[face_conf > pose_thresh]
                if looked_face_indices.size > 0:
                    face_points = obj.pose[looked_face_indices]
                    highest_face_point = np.r_[face_points.min(axis=0), 1]
                    if hands_up:
                        current_height = abs(highest_face_point[1] - lowest_ankle_point[1])
                        height_increase = current_height * 0.1
                        highest_face_point[1] -= height_increase

                    person_points["pose"] = (highest_face_point, lowest_ankle_point)

        if obj.mask is not None and obj.mask.any() and not hands_up:
            non_zero_rows = np.any(obj.mask, axis=1)
            first_row = np.argmax(non_zero_rows)
            last_row = obj.mask.shape[0] - 1 - np.argmax(non_zero_rows[::-1])

            top_point = (x1, first_row, 1)
            low_point = (x2, last_row, 1)

            person_points["mask"] = (top_point, low_point)

        if not hands_up or len(person_points) == 0:
            top_left_point_h = (x1, y1, 1)
            bottom_left_point_h = (x2, y2, 1)
            person_points["bbox"] = (top_left_point_h, bottom_left_point_h)

        if self._auto_correct:
            crouching = "crouching" in obj.pose_info
        else:
            crouching = False

        if crouching or (hands_up and len(person_points) == 0):
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            ideal_ratio_front = 0.365
            ideal_ratio_sideways = 0.365

            if "sideways" in obj.pose_info:
                ideal_ratio = ideal_ratio_sideways
            else:
                ideal_ratio = ideal_ratio_front

            height_change = (bbox_width / ideal_ratio) - bbox_height
            logger.info(f"Adjusted height by {height_change} px")

            updated_person_points = dict()
            for point_source, (top_point, bottom_point) in person_points.items():
                new_top_point = np.array([top_point[0], top_point[1] - height_change, top_point[2]])
                updated_person_points[point_source] = ((new_top_point, bottom_point))

            person_points = updated_person_points

        return person_points

    def process(self, objs: List[DetectedObject]):
        measurements = dict()
        for obj in objs:
            meas = self.process_one(obj)
            measurements[obj.id] = meas
            self.tracks[obj.id].time_inactive = 0

        self.purge_old_mtracks()
        return measurements, {}

    def get_dist(self, height_in_meters, top_point, bottom_point):
        top_left_inv = self.K_inv.dot(top_point)
        bottom_left_inv = self.K_inv.dot(bottom_point)

        height_inv = bottom_left_inv[1] - top_left_inv[1]
        distance = height_in_meters / height_inv

        return distance

    def get_velocity(self, obj):
        try:
            time = self.tracks[obj].timestamp[-1] - self.tracks[obj].timestamp[0]
            # time = 1 / 24

            dist = self.tracks[obj].filtered_meas["dist"][0] - self.tracks[obj].filtered_meas["dist"][-1]

            if dist < 0:
                dist = self.tracks[obj].filtered_meas["dist"][-1] - self.tracks[obj].filtered_meas["dist"][0]

            mps = dist / time
            mph = mps * 2.23694
        except:
            mph = 0

        return mph

    def process_one(self, obj: DetectedObject):
        if obj.id not in self.tracks:
            self.tracks[obj.id] = TrackedObjectForDistance()

        if obj.real_height is None:
            height_in_meters = self.height_in_meters
        else:
            height_in_meters = obj.real_height

        x1, y1, x2, y2 = obj.bbox

        top_left_point_h = (x1, y1, 1)
        bottom_left_point_h = (x1, y2, 1)

        if isinstance(obj, Person):
            person_points = self._get_person_points(obj)
            person_points = np.array(list(person_points.values()))
            (top_point, bottom_point) = self._get_filtered_point(self.tracks.get(obj.id), person_points)

            distance = self.get_dist(height_in_meters, top_point, bottom_point)
        else:
            distance = self.get_dist(height_in_meters, top_left_point_h, bottom_left_point_h)

        cx, cy = self.im_size[0] // 2, self.im_size[1] // 2
        x_pix = (x1 + x2) / 2 - cx
        y_pix = (y1 + y2) / 2 - cy
        z = distance * self._focal_length / np.sqrt(x_pix ** 2 + y_pix ** 2 + self._focal_length ** 2)
        x = z * x_pix / self._focal_length
        y = z * y_pix / self._focal_length

        meas = dict(dist=distance, X=x, Y=y, Z=z, timestamps=obj.timestamp)

        self.tracks[obj.id].history.append(meas)
        # print(self.tracks[obj.id].history)

        if self._use_tracker:
            final_meas = self._get_filtered_meas(obj.id)

            self.tracks[obj.id].timestamp.append(obj.timestamp)
            velocity = self.get_velocity(obj.id)
        else:
            final_meas = meas

        measurement = Measurement(final_meas["X"], final_meas["Y"],
                                  final_meas["Z"], velocity)

        return measurement

    def purge_old_mtracks(self):
        keys = list(self.tracks.keys())
        for k in keys:
            self.tracks[k].time_inactive += 1
            if self.tracks[k].time_inactive > self.forget:
                del self.tracks[k]
