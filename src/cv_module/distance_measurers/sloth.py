# By Oleksiy Grechnyev, 4/1/24
# This is Sloth (the current measurer of the distance to humans)
#
# TODO: Remove dependency on Volodymyr's distance.yaml, do something about the focal length, create a yaml config

from omegaconf import OmegaConf
import collections
import numpy as np
import torch
from typing import List

from src.cv_module.distance_measurers.measurement import Measurement
from src.cv_module.people.person import Person


########################################################################################################################
def print_it(a, name: str = ''):
    m = a.float().mean() if isinstance(a, torch.Tensor) else a.mean()
    # m = a.mean()
    print(name, a.shape, a.dtype, a.min(), m, a.max())


########################################################################################################################
class UpperEnvelopeFilter:
    def __init__(self, skip_frames):
        self.follow = 0.1
        self.chi = min(0.07 * skip_frames, 0.5)
        self.x_prev = None
        self.h_prev = None

    def process(self, h, visible=True):
        if not visible and self.x_prev is None:
            return None
        elif not visible:
            x = self.x_prev
        elif self.x_prev is None or h > self.x_prev:
            x = h
        elif h <= self.h_prev:  # Going down
            x = self.chi * h + (1 - self.chi) * self.x_prev
        else:  # Going up
            x = self.x_prev + self.follow * (h - self.h_prev)

        self.h_prev = h
        self.x_prev = x
        return x


########################################################################################################################
class KalmanFilter():
    """Standard Kalman filter, var names like in wikipedia, verified results identical to filterpy.kalman """

    def __init__(self, x_init, p_init, f, q, h, r):
        self.x = x_init
        self.p = p_init
        self.dim_x = len(x_init)
        self.eye = np.eye(self.dim_x)

        self.f = f  # Transition matrix
        self.q = q  # Transition covariance
        self.h = h  # Measurement matrix
        self.r = r  # Measurement covariance

        self.h_t = h.T.copy()
        self.f_t = f.T.copy()

    def predict(self):
        self.x = self.f @ self.x
        self.p = self.f @ self.p @ self.f_t + self.q

    def update(self, z, r=None):
        if r is None:
            r = self.r
        y = z - self.h @ self.x
        s = self.h @ self.p @ self.h_t + r
        s_inv = np.linalg.inv(s)
        k = self.p @ self.h_t @ s_inv

        self.x += k @ y
        self.p = (self.eye - k @ self.h) @ self.p

    def process(self, z, r=None):
        self.predict()
        self.update(z, r)
        return self.x


########################################################################################################################
class MedianFilter:
    def __init__(self, win):
        self.q = collections.deque(maxlen=win)

    def process(self, x):
        self.q.append(x)
        return np.median(self.q)


########################################################################################################################
class EMAFilter:
    def __init__(self, k):
        self.k = k
        self.y = None

    def process(self, x):
        if self.y is None:
            self.y = x
        else:
            self.y = self.k * x + (1 - self.k) * self.y
        return self.y


########################################################################################################################
MEASURER_SLOTH_DEFAULT_CONFIG = OmegaConf.create({
    'bbox_score_thresh': 0.5,  # Bounding box threshold, used only if NOT tracking
    'kpt_score_thresh': 0.3,  # Keypoint threshold, to be declared visible
    'length_unit_side': 0.51,  # Length unit in meters, for side aka hip-shoulder, our only reference

    'use_envelope': True,  # Smooth every bone with the "upper envelope" filter, requires use_tracking

    'use_weighted_algo': True,  # Use the weighted algo to average over bones, rather than simple max
    'weighted_algo_xi': 0.03,  # Parameter in the exponent
    'adapt_to_pose': True,
    "filter_outliers": True,
    # 'max_step': -1, # If value is real number we will limit max step from previous frame 

    'postprocessing_algo': 'ema',  # none, kalman or median
    'median_window': 6,
    'ema_k': 0.01,  # Parameter for the ema filtering

    'forget_tracks': 100,  # Forget tracks inactive for a set number of timestamps

    'use_autocalibrate': False,  # Allow auto-calibration, experimental, use with care !
    'autocalib_ema_k': 0.01,  # Parameter k of the auto-calibration
})


CONSTANT_PERSON_HEIGHT_METERS = 1.75

# Only those we currently use, change if needed
NAMED_BONES = [
    ('ears', (3, 4)),
    ('shoulders', (5, 6)),
    ('l_upperarm', (5, 7)),
    ('r_upperarm', (6, 8)),
    ('l_side', (5, 11)),
    ('r_side', (6, 12)),
    ('hips', (11, 12)),
    ('l_thigh', (11, 13)),
    ('r_thigh', (12, 14)),
]

BONE_PRIOR_WEIGHTS_STANDING = {
    'l_side': 1,
    'r_side': 1,
    'ears': 0.5,
    'shoulders': 0.5,
    'l_upperarm': 0.5,
    'r_upperarm': 0.5,
    'hips': 0.5,
    'l_thigh': 0.5,
    'r_thigh': 0.5,
}

BONE_PRIOR_WEIGHTS_SITTING = {
    'l_side': 0,
    'r_side': 0,
    'ears': 0.25,
    'shoulders': 1,
    'l_upperarm': 0,
    'r_upperarm': 0,
    'hips': 1,
    'l_thigh': 0,
    'r_thigh': 0,
}

BONE_PRIOR_WEIGHTS_SIDEWAYS = {
    'l_side': 1,
    'r_side': 1,
    'ears': 0,
    'shoulders': 0,
    'l_upperarm': 0.5,
    'r_upperarm': 0.5,
    'hips': 0,
    'l_thigh': 0.5,
    'r_thigh': 0.5,
}

BONE_PRIOR_WEIGHTS_SIDEWAYS_SITTING = {
    'l_side': 1,
    'r_side': 1,
    'ears': 0,
    'shoulders': 0,
    'l_upperarm': 0.1,
    'r_upperarm': 0.1,
    'hips': 0,
    'l_thigh': 0.5,
    'r_thigh': 0.5,
}

BONE_PRIOR_WEIGHTS = dict(
    standing=BONE_PRIOR_WEIGHTS_STANDING,
    sitting=BONE_PRIOR_WEIGHTS_SITTING,
    sideways=BONE_PRIOR_WEIGHTS_SIDEWAYS
)

DEFAULT_CALIBRATION = {
    'l_side': 1,
    'r_side': 1,
    'ears': 0.31,
    'shoulders': 0.65,
    'l_upperarm': 0.54,
    'r_upperarm': 0.54,
    'hips': 0.42,
    'l_thigh': 0.8,
    'r_thigh': 0.8,
}

# For auto-calibration only
CALIB_LIMITS = {
    'ears': (0.25, 0.33),
    'shoulders': (0.55, 0.70),
    'upperarm': (0.47, 0.58),
    'hips': (0.37, 0.44),
    'thigh': (0.63, 0.82),
}

KALMAN_PARAMS = {
    'p_init': np.array([[1.0, 0.05], [0.05, 0.05]]),
    'q': np.array([[0.003, 0.01], [0.01, 0.01]]),
    'f': np.array([[1, 1], [0, 1]]),
    'h': np.array([[1, 0]]),
    'r': None,
}

# BONES_PACK = ['l_side', 'r_side', 'ears', 'shoulders', 'l_upperarm', 'r_upperarm', 'hips', 'l_thigh', 'r_thigh']

BONES_PACK = ['l_side', 'r_side', 'shoulders', 'l_upperarm', 'r_upperarm', 'hips', 'l_thigh', 'r_thigh']


def filter_outliers(data, method='zscore', threshold=2.5):
    """
    Filter outliers from a numpy array using various methods.

    Parameters:
    -----------
    data : numpy.ndarray
        Input array to filter outliers from
    method : str, optional
        Method to use for outlier detection:
        - 'zscore': Use Z-score method
        - 'iqr': Use Interquartile Range method
        - 'modified_zscore': Use modified Z-score method
        Default is 'zscore'
    threshold : float, optional
        Threshold for outlier detection:
        - For zscore/modified_zscore: number of standard deviations (default: 3)
        - For IQR: multiplier for IQR range (default: 3)

    Returns:
    --------
    numpy.ndarray
        Array with outliers removed
    numpy.ndarray
        Boolean mask indicating non-outlier values
    """

    if method == 'zscore':
        # Z-score method
        z_scores = np.abs((data - np.mean(data)) / np.std(data))
        mask = z_scores < threshold

    elif method == 'iqr':
        # IQR method
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        mask = (data >= lower_bound) & (data <= upper_bound)

    elif method == 'modified_zscore':
        # Modified Z-score method (more robust to outliers)
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        modified_z_scores = 0.6745 * np.abs(data - median) / mad
        mask = modified_z_scores < threshold

    else:
        raise ValueError("Method must be one of: 'zscore', 'iqr', 'modified_zscore'")

    return data[mask], mask


class PeopleRangeEstimatorSloth:
    """The main measurer class by Oleksiy Grechnyev"""

    def __init__(self, focal_length, skip_frames, use_tracker, im_size,
                 sloth_config=MEASURER_SLOTH_DEFAULT_CONFIG):
        self.focal_length = focal_length
        self.skip_frames = skip_frames
        self.use_tracker = use_tracker
        self.im_size = im_size

        self.config = sloth_config
        self.calibration_initial = DEFAULT_CALIBRATION.copy()
        # self.bones_to_consider = ['l_side', 'r_side']
        self.bones_to_consider = BONES_PACK
        self.calib_limits = CALIB_LIMITS.copy()
        self.kalman_params = KALMAN_PARAMS.copy()
        self.kalman_params['q'] *= min(10, skip_frames)

        self.forget = max(5, self.config.forget_tracks // skip_frames)
        self.autocalib_ema_k = min(0.05, self.config.autocalib_ema_k * skip_frames)

        self.measurer_tracks = {}

    def calc_bones(self, kpts, kpts_visible):
        bones_dists, bones_visible = {}, {}
        for k, (i, j) in NAMED_BONES:
            d = np.linalg.norm(kpts[i, :] - kpts[j, :])
            v = kpts_visible[i] and kpts_visible[j]
            bones_dists[k] = d
            bones_visible[k] = v

        return bones_dists, bones_visible

    def filter_bones(self, bones_dists, bones_visible, mtrack):
        if self.use_tracker and self.config.use_envelope:
            # Filter every bone with the "upper envelope" filter
            if 'filter_envelopes' not in mtrack:
                mtrack['filter_envelopes'] = {k: UpperEnvelopeFilter(self.skip_frames) for k in bones_dists.keys()}

            bones_filtered_dists = {}
            for k, d in bones_dists.items():
                v = bones_visible[k]
                bones_filtered_dists[k] = mtrack['filter_envelopes'][k].process(d, v)
        else:
            bones_filtered_dists = bones_dists

        return bones_filtered_dists

    def weighted_algo(self, bones: dict, track_id, weights) -> float:
        """Smart average over bones"""
        hh = np.array([h for k, h in bones])
        if self.config.filter_outliers:
            hh, hh_mask = filter_outliers(hh, method='iqr')

            bones = [bones[idx] for idx in np.where(hh_mask)[0]]
        w0 = np.array([weights[k] for k, h in bones])  # Prior weights

        if False:
            names = [k for k, h in bones]
            print(f'DOMINATING({track_id}): ', names[hh.argmax()])

        if w0.sum() == 0:
            w0 = np.ones_like(w0)

        h_max = hh.max()
        xi = self.config.weighted_algo_xi
        w = w0 * np.exp((hh - h_max) / (xi * h_max))  # Weights

        w = w / w.sum()
        return np.sum(w * hh)

    def dist_from_bones(self, bones_visible, bones_filtered_dists,
                        calibration, track_id, person) -> float:
        # Select bones to use for
        bones_actually_used = [(k, bones_filtered_dists[k] / calibration[k]) for k in self.bones_to_consider if
                               bones_visible[k]]
        if len(bones_actually_used) == 0:
            return np.nan

        # The actual distance calculation
        if self.config.use_weighted_algo:
            if self.config.adapt_to_pose:
                if 'sideways' in person.pose_info and "crouching" in person.pose_info:
                    bone_weights = BONE_PRIOR_WEIGHTS_SIDEWAYS_SITTING
                elif 'sideways' in person.pose_info:
                    bone_weights = BONE_PRIOR_WEIGHTS_SIDEWAYS
                elif "crouching" in person.pose_info:
                    bone_weights = BONE_PRIOR_WEIGHTS_SITTING
                else:
                    bone_weights = BONE_PRIOR_WEIGHTS_STANDING

                bones_actually_used_names = {bone_name for bone_name, bone_dist_px in bones_actually_used}
                bones_with_weight = {bone_name for bone_name, bone_weight in bone_weights.items() if bone_weight > 0}
                if not bones_actually_used_names.intersection(bones_with_weight):
                    bone_weights = BONE_PRIOR_WEIGHTS_STANDING
            else:
                bone_weights = BONE_PRIOR_WEIGHTS_STANDING

            h = self.weighted_algo(bones_actually_used, track_id, bone_weights)
        else:  # A trivial max over bones
            hh = [h for k, h in bones_actually_used]
            h = np.max(hh)

        dist = self.config.length_unit_side * self.focal_length / h
        return dist

    def post_process(self, dist: float, mtrack) -> float:
        algo = self.config.postprocessing_algo
        assert algo in ['none', 'kalman', 'median', 'ema']

        if algo == 'none' or not self.use_tracker or not np.isfinite(dist):
            return dist

        elif algo == 'kalman':
            if 'filter_postprocess' not in mtrack:
                # Init kalman
                params = self.kalman_params.copy()
                params['x_init'] = np.array([dist, 0])
                mtrack['filter_postprocess'] = KalmanFilter(**params)
            else:
                # Filter with dist-dependent r
                r = max(0.1, 0.02 * dist)
                dist = mtrack['filter_postprocess'].process(dist, r=r)[0]

        elif algo == 'median':
            if 'filter_postprocess' not in mtrack:
                # Init median
                mtrack['filter_postprocess'] = MedianFilter(win=self.config.median_window)
            dist = mtrack['filter_postprocess'].process(dist)

        elif algo == 'ema':
            if 'filter_postprocess' not in mtrack:
                # Init median
                mtrack['filter_postprocess'] = EMAFilter(self.config.ema_k)
            dist = mtrack['filter_postprocess'].process(dist)

        return dist

    def auto_calibrate(self, bones_filtered_dists, bones_visible, mtrack):
        if not self.use_tracker or not self.config.use_autocalibrate:
            return

        if (not bones_visible['l_side']) and (not bones_visible['r_side']):
            return

        calib = mtrack['calibration']

        # All bones in group share the correlation
        bone_groups = {
            'upperarm': ['l_upperarm', 'r_upperarm'],
            'thigh': ['l_thigh', 'r_thigh'],
            'ears': ['ears'],
            'shoulders': ['shoulders'],
            'hips': ['hips'],
        }

        # Init filters on the first run
        if not 'calib_filter' in mtrack:
            mtrack['calib_filter'] = {k: EMAFilter(self.autocalib_ema_k) for k in bone_groups.keys()}
            for k in bone_groups.keys():
                k0 = bone_groups[k][0]
                if k0 in bones_filtered_dists:
                    mtrack['calib_filter'][k].process(calib[k0])

        # Calculate the reference length
        if bones_visible['l_side'] and bones_visible['r_side']:
            ref = max(bones_filtered_dists['l_side'], bones_filtered_dists['r_side'])
        elif bones_visible['l_side']:
            ref = bones_filtered_dists['l_side']
        else:
            ref = bones_filtered_dists['r_side']

        for k, filter in mtrack['calib_filter'].items():
            # Distance in pixels for the bones in group
            dd = [bones_filtered_dists[k0] for k0 in bone_groups[k] if k0 in bones_filtered_dists and bones_visible[k0]]
            if len(dd) == 0:
                continue
            d = np.max(dd) / ref
            # Check limits
            if self.calib_limits[k][0] <= d <= self.calib_limits[k][1]:
                # Success, update calib with the EMA filter
                c = filter.process(d)
                for k0 in bone_groups[k]:
                    calib[k0] = c

    def calc_distance(self, person):
        kpts, kpt_scores, track_id = person.pose, person.pose_conf, person.id
        kpts_visible = (kpts[:, 0] > 0) & (kpts[:, 1] > 0) & (kpt_scores > self.config.kpt_score_thresh)

        # Calibration
        if self.use_tracker:
            if not track_id in self.measurer_tracks:
                self.measurer_tracks[track_id] = {
                    'calibration': self.calibration_initial.copy(),
                    'time_inactive': 0,
                    'timestamps': collections.deque(maxlen=2),  # person.timestamp,
                    'meas_history': collections.deque(maxlen=2)
                }
            self.measurer_tracks[track_id]['timestamps'].append(person.timestamp)
            mtrack = self.measurer_tracks[track_id]
            calibration = mtrack['calibration']
            mtrack['time_inactive'] = 0
        else:
            mtrack = None
            calibration = self.calibration_initial

        bones_dists, bones_visible = self.calc_bones(kpts, kpts_visible)
        bones_filtered_dists = self.filter_bones(bones_dists, bones_visible, mtrack)

        dist = self.dist_from_bones(bones_visible, bones_filtered_dists, calibration,
                                    track_id, person)

        dist = self.post_process(dist, mtrack)

        self.auto_calibrate(bones_filtered_dists, bones_visible, mtrack)

        return dist

    def process_one(self, person):
        # keypoints, keypoint_scores, box, t_id = person.pose, person.pose_conf, person.bbox, person.id
        dist = self.calc_distance(person)

        # Calculate xyz in camera coordinates
        cx, cy = self.im_size[0] // 2, self.im_size[1] // 2
        x_pix = (person.bbox[0] + person.bbox[2]) / 2 - cx
        y_pix = (person.bbox[1] + person.bbox[3]) / 2 - cy
        f = self.focal_length
        z = dist * f / np.sqrt(x_pix ** 2 + y_pix ** 2 + f ** 2)
        x = z * x_pix / f
        y = z * y_pix / f

        meas = {'dist': dist, 'X': x, 'Y': y, 'Z': z}
        self.measurer_tracks[person.id]['meas_history'].append(meas)
        velocity = self._get_velocity(person)
        return meas, velocity

    def purge_old_mtracks(self):
        if not self.use_tracker:
            return
        keys = list(self.measurer_tracks.keys())
        for k in keys:
            self.measurer_tracks[k]['time_inactive'] += 1
            if self.measurer_tracks[k]['time_inactive'] > self.forget:
                del self.measurer_tracks[k]

    def _get_velocity(self, person):
        current_meas = person.meas
        velocity = dict(dist=0, X=0, Y=0, Z=0)

        timestamps = self.measurer_tracks[person.id]['timestamps']
        if len(timestamps) > 1:
            time_delta = timestamps[-1] - timestamps[-2]
            current_meas, new_meas = self.measurer_tracks[person.id]['meas_history']
            meas_names = new_meas.keys()
            for meas_name in meas_names:
                dist_delta = np.abs(new_meas[meas_name] - current_meas[meas_name])
                speed = dist_delta / time_delta
                velocity[meas_name] = speed
        else:
            return velocity

        return velocity

    def process(self, people: List[Person]):
        measurements = dict()
        velocities = dict()
        for person in people:
            if not person.has_pose:
                continue
            meas, velocity = self.process_one(person)
            meas, velocity = meas.copy(), velocity.copy()
            person_measurement = Measurement(meas['X'], meas["Y"], meas["Z"],
                                             velocity)
            measurements[person.id] = person_measurement
            velocities[person.id] = velocity

        self.purge_old_mtracks()

        return measurements, velocities

    def print_calib(self, track_id):
        print('===================================')
        print('CALIBRATION for track', track_id)
        if track_id in self.measurer_tracks:
            for k, v in self.measurer_tracks[track_id]['calibration'].items():
                print(k, v)
        print('===================================')

    def check_fit(self, person: "Person"):
        if not person.has_pose:
            return False

        return True

########################################################################################################################
