import numpy as np
from typing import Tuple, List

from src.cv_module.basic_object import BasicObjectWithDistance
from src.cv_module.detected_object import DetectedObject
from src.cv_module.distance_measurers.height_measurer import TrackedObjectForDistance
from src.cv_module.distance_measurers.measurement import Measurement


class WidthDistanceMeasurer:
    def __init__(self, focal_length: int, im_size: Tuple[int, int],
                 base_width_in_meters: float, auto_correct: bool = False,
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
        self.width_in_meters = base_width_in_meters
        self.tracks = dict()
        self.forget = 20

    def process(self, objs: List[DetectedObject]):
        measurements = dict()
        for obj in objs:
            meas = self.process_one(obj)
            measurements[obj.id] = meas
            self.tracks[obj.id].time_inactive = 0

        self.purge_old_mtracks()
        return measurements

    def _get_xyz(self, distance, x_pix, y_pix):
        z = distance * self._focal_length / np.sqrt(x_pix ** 2 + y_pix ** 2 + self._focal_length ** 2)
        x = z * x_pix / self._focal_length
        y = z * y_pix / self._focal_length
        return x, y, z

    def _get_filtered_meas(self, id: int, x_pix: int, y_pix: int):
        track = self.tracks.get(id)
        if track is None:
            raise ValueError(f"Track with the id {id} wasn't found")

        new_dist = track.dist_filter.process(track.history[-1]["dist"])
        x, y, z = self._get_xyz(new_dist, x_pix, y_pix)

        latest_meas = track.history[-1].copy()
        latest_meas["dist"] = new_dist
        latest_meas["X"] = x
        latest_meas["Y"] = y
        latest_meas["Z"] = z

        return latest_meas

    def get_velocity(self, obj_id):
        try:
            time = self.tracks[obj_id].timestamp[-1] - self.tracks[obj_id].timestamp[0]
            dist = self.tracks[obj_id].filtered_meas["dist"][0] - self.tracks[obj_id].filtered_meas["dist"][-1]

            if dist < 0:
                dist = self.tracks[obj_id].filtered_meas["dist"][-1] - self.tracks[obj_id].filtered_meas["dist"][0]

            mps = dist / time if time != 0 else 0
            mph = mps * 2.23694
        except:
            mph = 0

        return mph

    def _get_last_delta_meas(self, obj_id, idx=-1):
        track = self.tracks.get(obj_id)
        x_old, y_old, z_old = (track.filtered_meas[idx-1][k] for k in ("X", "Y", "Z"))
        x, y, z = (track.filtered_meas[idx][k] for k in ("X", "Y", "Z"))
        time_ms = track.timestamp[idx] - track.timestamp[idx-1]
        return x - x_old, y - y_old, z - z_old, time_ms

    def get_velocity_vector(self, obj_id):
        # Averaging two last changes in three last states
        track = self.tracks.get(obj_id)
        velocity_vector = np.array((0, 0, 0), dtype=float)
        if track is None:
            raise ValueError(f"Track with the id {obj_id} wasn't found")

        try:
            dx, dy, dz, dt = self._get_last_delta_meas(obj_id, -1)
            shift_1 = np.array((dx, dy, dz), dtype=float)
            if dt == 0:
                return velocity_vector

            v1 = shift_1 / dt  * 1000

            dx, dy, dz, dt = self._get_last_delta_meas(obj_id, -2)
            shift_2 = np.array((dx, dy, dz), dtype=float)
            if dt == 0:
                return velocity_vector

            v2 = shift_2 / dt  * 1000
            velocity_vector = np.mean([v1, v2], axis=0)
        except:
            pass

        return velocity_vector

    def get_dist(self, width_in_meters, left_point, right_point):
        top_left_inv = self.K_inv.dot(left_point)
        top_right_inv = self.K_inv.dot(right_point)

        width_inv = top_right_inv[0] - top_left_inv[0]
        distance = width_in_meters / width_inv

        return distance

    def process_one(self, obj: BasicObjectWithDistance):
        if obj.id not in self.tracks:
            self.tracks[obj.id] = TrackedObjectForDistance()

        if obj.real_width is None:
            width_in_meters = self.width_in_meters
        else:
            width_in_meters = obj.real_width

        x1, y1, x2, y2 = obj.bbox

        top_left_point_h = (x1, y1, 1)
        top_right_point_h = (x2, y1, 1)

        distance = self.get_dist(width_in_meters, top_left_point_h,
                                 top_right_point_h)

        cx, cy = self.im_size[0] // 2, self.im_size[1] // 2
        x_pix = (x1 + x2) / 2 - cx
        y_pix = (y1 + y2) / 2 - cy
        x, y, z = self._get_xyz(distance, x_pix, y_pix)
        meas = dict(dist=distance, X=x, Y=y, Z=z, timestamps=obj.timestamp)
        self.tracks[obj.id].history.append(meas)

        if self._use_tracker:
            final_meas = self._get_filtered_meas(obj.id, x_pix, y_pix)

            self.tracks[obj.id].filtered_meas.append(final_meas)
            self.tracks[obj.id].timestamp.append(obj.timestamp)
            #velocity = self.get_velocity(obj.id)
        else:
            final_meas = meas

        measurement = Measurement(
            final_meas["X"], final_meas["Y"], final_meas["Z"],
            self.get_velocity_vector(obj.id)
        )

        return measurement

    def purge_old_mtracks(self):
        keys = list(self.tracks.keys())
        for k in keys:
            self.tracks[k].time_inactive += 1
            if self.tracks[k].time_inactive > self.forget:
                del self.tracks[k]
