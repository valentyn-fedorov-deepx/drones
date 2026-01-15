from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List
from loguru import logger
import sys
from collections import defaultdict
import pickle as pkl
import numpy as np
from tqdm import tqdm
import json

from src.cv_module.people.person import Person
from src.cv_module.distance_measurers import HeightDistanceMeasurer, PeopleRangeEstimatorSloth

PERSON_HEIGHT = 1.78


class PersonLogs:
    def __init__(self, id: int, logs: List[Dict]):
        self.logs = sorted(logs, key=lambda x: x["frame_idx"])
        self.id = id
        self.frame_ids = [person["frame_idx"] for person in self.logs]
        self.bbox_height_history = [person['bbox'][-1] - person['bbox'][1] for person in self.logs]
        self.bbox_width_history = [person['bbox'][-2] - person['bbox'][0] for person in self.logs]
        self.bbox_ratio_history = [width / height for height, width in zip(self.bbox_height_history, self.bbox_width_history)]

        self.x_center_history = [(person["bbox"][0] + person["bbox"][2]) / 2 for person in self.logs]
        self.y_center_history = [(person["bbox"][1] + person["bbox"][3]) / 2 for person in self.logs]
        self.center_history = list(zip(self.x_center_history, self.y_center_history))
        self.bbox_history = [person['bbox'] for person in self.logs]
        self.bbox_distances = list()

    def __add__(self, other):
        if isinstance(other, int):
            return PersonLogs(self.id, self.logs.copy())
        elif isinstance(other, PersonLogs):
            return PersonLogs(self.id, self.logs + other.logs)
        else:
            raise ValueError("Trying to add different types")

    def __radd__(self, other):
        return self.__add__(other)


def process_logs(logs: dict):
    person_data = defaultdict(list)
    for frame_data in logs['data_log']:
        for item in frame_data["res"]:
            person_data[item["id"]].append(item)

    people = list()
    for person_id, data in person_data.items():
        people.append(PersonLogs(person_id, data))

    return people


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--pkl-path", type=Path)
    parser.add_argument("--save-path", default="output", type=Path)
    parser.add_argument("--focal-length", type=int, default=75)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    focal_length_mm = args.focal_length
    sensor_ratio = 0.00345
    focal_length_px = focal_length_mm / sensor_ratio
    imgsize = (2048, 2488)

    height_measurer_tracking = HeightDistanceMeasurer(focal_length_px, imgsize,
                                                      PERSON_HEIGHT)
    height_measurer = HeightDistanceMeasurer(focal_length_px, imgsize,
                                             PERSON_HEIGHT, use_tracker=False)

    height_measurer_correct = HeightDistanceMeasurer(focal_length_px, imgsize,
                                                     PERSON_HEIGHT, auto_correct=True,
                                                     use_tracker=False)

    height_measurer_correct_tracking = HeightDistanceMeasurer(focal_length_px, imgsize,
                                                              PERSON_HEIGHT, auto_correct=True,
                                                              use_tracker=True)

    sloth_measurer = PeopleRangeEstimatorSloth(focal_length_px, 3, False,
                                               imgsize)
    sloth_measurer_tracking = PeopleRangeEstimatorSloth(focal_length_px, 3,
                                                        True, imgsize)

    with open(args.pkl_path, "rb") as f:
        data = pkl.load(f)

    people = process_logs(data)
    len(people)

    i = 0

    all_measurements = dict()
    for person in tqdm(people):
        sloth_measurements = dict()
        sloth_measurements_tracking = dict()

        height_measurements = dict()
        height_measurements_tracking = dict()
        height_measurements_correct = dict()
        height_measurements_correct_tracking = dict()

        orig_measurements = dict()
        for person_log in person.logs:
            if person_log['has_pose']:
                kpts = np.concatenate([person_log["pose"],
                                      person_log["pose_conf"][:, None]],
                                      axis=1)
            else:
                kpts = np.zeros((17, 3))

            person_on_frame = Person(id=person_log['id'], bbox=person_log["bbox"],
                                     height_in_m=PERSON_HEIGHT,
                                     conf=person_log["conf"],
                                     kpts=kpts)
            fake_id_person_on_frame = Person(id=i, bbox=person_log["bbox"],
                                             height_in_m=PERSON_HEIGHT,
                                             conf=person_log["conf"],
                                             kpts=kpts)

            if person_on_frame.has_pose:
                sloth_meas = sloth_measurer.process_one(person_on_frame)
                sloth_measurements[person_log['frame_idx']] = (sloth_meas['dist'])

                sloth_meas_tracking = sloth_measurer_tracking.process_one(person_on_frame)
                sloth_measurements_tracking[person_log['frame_idx']] = (sloth_meas_tracking['dist'])

            height_meas = height_measurer.process_one(fake_id_person_on_frame)
            height_meas_correct = height_measurer_correct.process_one(fake_id_person_on_frame)
            height_meas_tracking = height_measurer_tracking.process_one(person_on_frame)
            height_meas_correct_tracking = height_measurer_correct_tracking.process_one(person_on_frame)

            height_measurements[person_log['frame_idx']] = height_meas["dist"]
            height_measurements_correct[person_log['frame_idx']] = height_meas_correct["dist"]
            height_measurements_tracking[person_log['frame_idx']] = height_meas_tracking["dist"]
            height_measurements_correct_tracking[person_log['frame_idx']] = height_meas_correct_tracking["dist"]

            tracking_meas = person_log['meas']
            if tracking_meas is not None:
                orig_measurements[person_log['frame_idx']] = tracking_meas['dist']
            i += 1

        all_measurements[person.id] = dict(
                                           height=height_measurements,
                                           # orig=orig_measurements,
                                           sloth=sloth_measurements,
                                           sloth_tracking=sloth_measurements_tracking,
                                           height_tracking=height_measurements_tracking,
                                           height_correct=height_measurements_correct,
                                           height_correct_tracking=height_measurements_correct_tracking,
                                           )

    args.save_path.mkdir(parents=True, exist_ok=True)
    save_path = args.save_path / f"distances_{args.pkl_path.stem}.json"
    with open(save_path, "w") as f:
        json.dump(all_measurements, f)
