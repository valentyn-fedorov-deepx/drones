from argparse import ArgumentParser
import cv2
import pickle as pkl
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
from typing import Dict
import numpy as np

from src.cv_module.visualization import ResultsAnnotator
from src.cv_module.people.person import Person


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--video-path", type=Path)
    parser.add_argument("--pkl-path", type=Path)
    parser.add_argument("--save-path", type=Path)

    return parser.parse_args()


def create_person_from_logs(person_logs: Dict) -> Person:
    if person_logs['has_pose']:
        kpts = np.concatenate([person_logs['pose'], person_logs['pose_conf'][:, None]], axis=1)
    else:
        kpts = None

    person = Person(person_logs['id'],
                    person_logs['bbox'],
                    person_logs['conf'],
                    None,
                    kpts)

    person.set_measurement(person_logs['meas'])
    person.pose_info = person_logs['pose_info']
    return person


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
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


if __name__ == "__main__":
    args = parse_args()

    annotator = ResultsAnnotator()

    with open(args.pkl_path, "rb") as f:
        data = pkl.load(f)

    video = cv2.VideoCapture(str(args.video_path))
    fps = 12
    save_path = args.save_path / args.video_path.name
    args.save_path.mkdir(parents=True, exist_ok=True)
    video_out = cv2.VideoWriter(str(save_path),
                                cv2.VideoWriter.fourcc(*'mp4v'), fps,
                                (2448, 2048), True)

    frame_idx_to_data = defaultdict(list)
    for single_log in data['data_log']:
        frame_idx_to_data[single_log["pos_frame"]] += single_log['res']

    sorted_frame_ids = sorted(set(frame_idx_to_data.keys()))

    # annotator.draw_human_pose()

    for frame_number in tqdm(sorted_frame_ids):
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_number-1)
        res, frame = video.read()

        frame_data = frame_idx_to_data[frame_number]
        people = list()
        for item in frame_data:
            meas = item['meas']
            if meas is not None:
                meas = f"{meas['dist']:.2f}"

            person = create_person_from_logs(item)

            people.append(person)
            plot_one_box(item['bbox'], frame, (212, 212, 212),
                         "", 5)

            if item['has_pose']:
                for point, point_conf in zip(item['pose'], item['pose_conf']):
                    if point_conf < 0.5:
                        continue

                    x_p, y_p = map(int, point)
                    cv2.circle(frame, (x_p, y_p), 4, (255, 0, 0), -1)

        frame = annotator.process(frame, frame_number, people, list(), list(), frame, frame, False)

        video_out.write(frame)

    video.release()
    video_out.release()
