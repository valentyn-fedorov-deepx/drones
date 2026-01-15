# By Oleksiy Grechnyev
# Run through the algo and record the data

import sys
import os
import argparse
import pathlib
import time
import datetime
import omegaconf
import pickle
from pathlib import Path
from loguru import logger
import numpy as np
import cv2 as cv

from src.offline_utils import obstacle_cache, frame_source

import numpy as np
import cv2 as cv


########################################################################################################################
def print_it(a, name: str = ''):
    # m = a.float().mean() if isinstance(a, torch.Tensor) else a.mean()
    m = a.mean()
    print(name, a.shape, a.dtype, a.min(), m, a.max())


########################################################################################################################
def downsize(img):
    img = cv.resize(img, None, None, 0.5, 0.5)
    return img


def auto_downsize(img):
    while max(img.shape[0], img.shape[1]) > 1920:
        img = downsize(img)
    return img


class MiniManager:

    def __init__(self, crop_fov=False):
        from src.cv_module.people.people_pose_estimation import PeoplePoseEstimator
        from src.cv_module.people.people_tracking import PeopleTracker
        from src.cv_module.people.people_range_estimation_v2 import PeopleRangeEstimator
        from src.cv_module.people.people_pose_classifier import PoseClassifier

        self.distance_scale_factor = 1.0
        self.crop_fov = crop_fov
        if crop_fov:
            self.H, self.W = 1024, 1224
        else:
            self.H, self.W = 2048, 2448

        config_dir = 'configs'
        model_dir = 'models'

        self.config = omegaconf.OmegaConf.load(os.path.join(config_dir, 'manager.yaml'))
        self.focal_length_mm = self.config.focal_length_mm
        self.focal_length_px = self.focal_length_mm / self.config.sensor_ratio
        print('MANAGER CONFIG=', self.config)
        bigger_side = 320 if crop_fov else 640

        # Create all modules
        if self.config.detector == "yolo_slicing":
            from src.cv_module.people.people_detection import YoloDetectorSlicing
            self.people_detector = YoloDetectorSlicing(config_dir, model_dir, (self.H, self.W), kpts_model=False,
                                                       crop_fov=crop_fov, use_trt=self.config.use_trt,
                                                       device=self.config.device)
        elif self.config.detector == "hf":
            from src.cv_module.detectors import HFDetector
            self.people_detector = HFDetector(config_dir, self.config.device)
        elif self.config.detector == "detectron":
            from src.cv_module.detectors import DetectronDetector
            self.people_detector = DetectronDetector(config_dir, self.config.device)
        elif self.config.detector == 'yolo':
            from src.cv_module.detectors import YoloDetector
            self.people_detector = YoloDetector(config_dir, model_dir, self.config.device)

        self.people_pose_estimator = PeoplePoseEstimator(config_dir, model_dir, bigger_side=bigger_side,
                                                         use_trt=self.config.use_trt, device=self.config.device)
        self.people_tracker = PeopleTracker(config_dir)

        self.people_range_estimator = PeopleRangeEstimator(self.focal_length_px,
                                                           self.config.skip_frames_sloth,
                                                           (self.H, self.W), active_method="sloth")
        self.people_pose_classifier = PoseClassifier()

    def process(self, frame, frame_idx):
        frame, nz, nxy, nxyz = frame

        detected_people = self.people_detector.predict(frame)
        detected_people = self.people_pose_estimator.process(frame, detected_people)
        results_people = self.people_tracker.track(detected_people, frame)
        self.people_pose_classifier.classify(results_people)
        self.people_range_estimator.set_distance(results_people, self.distance_scale_factor)
        # Convert to a list of dicts for a clean serialization
        result = []
        # import ipdb; ipdb.set_trace()
        logger.info(f"Saving information about {len(results_people)} people")
        for person in results_people:
            res = {}
            saved_attributes = ['id', 'bbox', 'has_pose',
                                'conf', 'height', 'meas',
                                'pose', 'pose_conf', 'pose_info']
            for k in saved_attributes:
                res[k] = getattr(person, k)
            res["frame_idx"] = frame_idx

            result.append(res)
        return result


def process(video_path, pos, max_frames, crop_fov, save_path):
    """Process video source"""
    source = frame_source.FrameSource(video_path, pos, max_frames)

    # The main algorith engine of this project
    manager = MiniManager(crop_fov=crop_fov)
    data_log = []
    frame_shape = None
    for img_i, img_nz, img_nxy, img_nxyz in source:
        if frame_shape is None:
            frame_shape = img_i.shape
        print('==================================================')
        print(f'{source.i_frame - 1} : pos_frame={source.pos - 1}, pos_sec={source.pos_sec} s')

        result = manager.process((img_i, img_nz, img_nxy, img_nxyz), source.pos)
        data_log.append({
            'pos_frame': source.pos - 1,
            'pos_sec': source.pos_sec,
            'res': result,
        })

    print('DATA_LOG_SIZE=', len(data_log))
    full_log = {
        'video_file': video_path,
        'focal_length': manager.focal_length_px,
        'skip_frames_sloth': manager.config.skip_frames_sloth,
        'distance_scale_factor': manager.distance_scale_factor,
        'resolution': (frame_shape[1], frame_shape[0]),
        'created_by': 'main_oleksiy_recorder.py by Oleksiy Grechnyev',
        'algo_detect': 'ultralytics, custom human detect + yolov8m-pose',
        'datetime': datetime.datetime.ctime(datetime.datetime.now()),
        'data_log': data_log,
    }
    save_path.mkdir(exist_ok=True, parents=True)

    with open(save_path / f'{Path(video_path).stem}.pkl', 'wb') as f:
        pickle.dump(full_log, f)


def main():
    parser = argparse.ArgumentParser(
        description='Run one video or PXI directory through the people track pipeline and write results',
    )
    parser.add_argument('video_path', help='Video file or directory with PXI files', type=pathlib.Path)
    parser.add_argument('-p', '--pos', type=int, help='Starting position')
    parser.add_argument('-m', '--max-frames', type=int, help='Number of frames to process if using --record')
    parser.add_argument('--crop-fov', action="store_true", default=False,
                        help='Specify that we are using cropped video')
    parser.add_argument('--save-path', type=Path, default='../output/pkl_records/', help="Data will be saved to this directory")
    args = parser.parse_args()
    print('args=', args)
    process(args.video_path, args.pos, args.max_frames,
            args.crop_fov, args.save_path)


########################################################################################################################
if __name__ == '__main__':
    main()
