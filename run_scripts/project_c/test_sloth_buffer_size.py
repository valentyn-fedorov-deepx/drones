from argparse import ArgumentParser
import pickle as pkl
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np
from collections import defaultdict

from src.cv_module.people.person import Person
from src.cv_module.distance_measurers import PeopleRangeEstimatorSloth
from .single_frame_error import process_logs


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--pkl-path", type=Path)
    parser.add_argument("--save-path", default="output", type=Path)
    parser.add_argument("--focal-length", type=int, default=50)

    return parser.parse_args()


SLOTH_CONFIG = {
        "config": {
            'bbox_score_thresh': 0.5,  # Bounding box threshold, used only if NOT tracking
            'kpt_score_thresh': 0.3,  # Keypoint threshold, to be declared visible
            'length_unit_side': 0.51,  # Length unit in meters, for side aka hip-shoulder, our only reference

            'use_envelope': True,  # Smooth every bone with the "upper envelope" filter, requires use_tracking

            'use_weighted_algo': True,  # Use the weighted algo to average over bones, rather than simple max
            'weighted_algo_xi': 0.03,  # Parameter in the exponent
            'adapt_to_pose': True,
            # 'max_step': -1, # If value is real number we will limit max step from previous frame 

            'postprocessing_algo': 'ema',  # none, kalman or median
            'median_window': 6,
            'ema_k': 0.01,  # Parameter for the ema filtering

            'forget_tracks': 100,  # Forget tracks inactive for a set number of timestamps

            'use_autocalibrate': False,  # Allow auto-calibration, experimental, use with care !
            'autocalib_ema_k': 0.01,  # Parameter k of the auto-calibration
            'filter_outliers': True
        },
        "args": {
            "skip_frames": 4,
            "use_tracker": True
        },

}


WINDOW_SIZES = (5, 10, 20, 30, 40, 50, 100, 120, 140, 160, 1502000)


PERSON_HEIGHT = 1.75

if __name__ == "__main__":
    args = parse_args()

    focal_length_mm = args.focal_length
    sensor_ratio = 0.00345
    focal_length_px = focal_length_mm / sensor_ratio
    imgsize = (2048, 2488)

    sloth_config = SLOTH_CONFIG['config']
    sloth_args = SLOTH_CONFIG["args"]

    algos = dict()
    for window_size in WINDOW_SIZES:
        algos[f"window: {window_size}"] = PeopleRangeEstimatorSloth(focal_length=focal_length_px, 
                                                                    im_size=imgsize,
                                                                    sloth_config=sloth_config,
                                                                    **sloth_args)

    with open(args.pkl_path, "rb") as f:
        data = pkl.load(f)

    people = process_logs(data)

    all_measurements = dict()
    for person in tqdm(people):
        algos_measurements = defaultdict(dict)

        max_frame_idx = person.logs[-1]['frame_idx']
        update_sloth_counter = {f"window: {window_size}": np.arange(0, max_frame_idx, window_size) for window_size in WINDOW_SIZES}
        for person_log in person.logs:
            # import ipdb; ipdb.set_trace()
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
            person_on_frame.pose_info = person_log['pose_info']

            for algo_window_size, algo in algos.items():
                if person_on_frame.has_pose:
                    if update_sloth_counter[algo_window_size].size > 0 and (update_sloth_counter[algo_window_size][0] <= person_log['frame_idx']):
                        # print(f"{person_log['frame_idx']} - {algo_window_size}")
                        algo = PeopleRangeEstimatorSloth(focal_length=focal_length_px, 
                                                         im_size=imgsize,
                                                         sloth_config=sloth_config,
                                                         **sloth_args)
                        algos[algo_window_size] = algo
                        # import ipdb; ipdb.set_trace()
                        indices = update_sloth_counter[algo_window_size] > person_log['frame_idx']

                        update_sloth_counter[algo_window_size] = update_sloth_counter[algo_window_size][indices]
                        
                    sloth_meas = algo.process_one(person_on_frame)
                    if np.isnan(sloth_meas['dist']):
                        continue
                    algos_measurements[algo_window_size][person_log['frame_idx']] = sloth_meas['dist']

        all_measurements[person.id] = algos_measurements

    args.save_path.mkdir(parents=True, exist_ok=True)
    save_path = args.save_path / f"distances_sloth_{args.pkl_path.stem}.json"
    with open(save_path, "w") as f:
        json.dump(all_measurements, f)