import numpy as np
from omegaconf import OmegaConf
import os


class PeopleRangeEstimator:
    """
    This is the OLD ranging code from Volodymyr, don't use this, use Sloth !
    """

    def __init__(self, config_dir, model_dir, imsize):
        self.distance_history = {}
        self.distance_history_filtered = {}
        self.config = OmegaConf.load(os.path.join(config_dir, "distance.yaml"))
        self.H, self.W = imsize

    @staticmethod
    def euqlidian(a, b):
        return ((a-b)**2).sum()**0.5

    def get_distance_hip2shoulder(self, detect):
        if detect.keypoints is None:
            return np.nan

        kpts = detect.keypoints.cpu().numpy().data[0]
        visible = kpts[:,:2].sum(axis=1) > 0
        kpts = kpts[:,:2]

        if visible[5] and visible[6] and visible[11] and visible[12]:
            d1 = PeopleRangeEstimator.euqlidian(kpts[5], kpts[11])
            d2 = PeopleRangeEstimator.euqlidian(kpts[6], kpts[12])
            d = max(d1, d2)
        elif visible[5] and visible[11]:
            d = PeopleRangeEstimator.euqlidian(kpts[5], kpts[11])
        elif visible[6] and visible[12]:
            d = PeopleRangeEstimator.euqlidian(kpts[6], kpts[12])
        else:
            d = np.nan

        dist = self.config.hip_to_shoulder_reference / d
        dist = dist * self.config.F

        return dist

    def get_distance_ear2ear(self, detect):
        if detect.keypoints is None:
            return np.nan

        kpts = detect.keypoints.cpu().numpy().data[0]
        visible = kpts[:,:2].sum(axis=1) > 0
        kpts = kpts[:,:2]

        if visible[3] and visible[4]:
            d = PeopleRangeEstimator.euqlidian(kpts[3], kpts[4])
        else:
            d = np.nan

        dist = self.config.ear_to_ear_reference / d
        dist = dist * self.config.F
        
        return dist
    
    def get_distance_elbow2shoulder(self, detect):
        if detect.keypoints is None:
            return np.nan

        kpts = detect.keypoints.cpu().numpy().data[0]
        visible = kpts[:,:2].sum(axis=1) > 0
        kpts = kpts[:,:2]

        if visible[5] and visible[7] and visible[6] and visible[8]:
            d1 = PeopleRangeEstimator.euqlidian(kpts[5], kpts[7])
            d2 = PeopleRangeEstimator.euqlidian(kpts[6], kpts[8])
            d = max(d1, d2)
        elif visible[5] and visible[7]:
            d = PeopleRangeEstimator.euqlidian(kpts[5], kpts[7])
        elif visible[6] and visible[8]:
            d = PeopleRangeEstimator.euqlidian(kpts[6], kpts[8])
        else:
            d = np.nan

        dist = self.config.elbow_to_shoulder_reference / d
        dist = dist * self.config.F
        
        return dist

    def update(self, results):
        measurements = {}
        if results.boxes is not None and results.boxes.id is not None:
            for detect in results:
                track_id = int(detect.boxes.id[0])
                if track_id not in self.distance_history:
                    self.distance_history[track_id] = []
                    self.distance_history_filtered[track_id] = []

                dist = np.nan
                if np.isnan(dist):
                    dist = self.get_distance_hip2shoulder(detect)
                if np.isnan(dist):
                    dist = self.get_distance_ear2ear(detect)
                if np.isnan(dist):
                    dist = self.get_distance_elbow2shoulder(detect)

                self.distance_history[track_id].append(dist)

                last = self.distance_history[track_id][-self.config.median_window:]
                last = np.array(last)
                last = last[~np.isnan(last)]

                if len(last) > 0:
                    smoothed = np.quantile(last, 0.5)
                else:
                    smoothed = np.nan

                self.distance_history_filtered[track_id].append(smoothed)

                X, Y, Z = self.get_xyz(detect, smoothed)

                measurements[track_id] = {
                    "X": X,
                    "Y": Y,
                    "Z": Z,
                    "dist": smoothed,
                    "raw": dist,
                }

        return measurements

    def get_xyz(self, detect, dist):
        x, y, w, h = detect.boxes.xywh[0]

        k = dist / ((x-self.W/2)**2 + (y-self.H/2)**2 + self.config.F**2)**0.5
        X = k * (x-self.W/2)
        Y = -k * (y-self.H/2)
        Z = k * self.config.F

        return X, Y, Z
