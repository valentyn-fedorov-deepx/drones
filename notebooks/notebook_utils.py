from pathlib import Path
import pickle as pkl
from typing import Dict, Optional

from run_scripts.gen_video_from_record import create_person_from_logs


def process_pkl_logs(data_path: Path, vid_to_ids: Optional[Dict]):
    video_people_data = dict()
    for vid_name, person_idxs in vid_to_ids.items():
        vid_people = dict()
        pkl_path = data_path / f"{vid_name}.pkl"

        with open(pkl_path, 'rb') as f:
            logs_data = pkl.load(f)

        for frame_data in logs_data['data_log']:
            frame_idx = frame_data['pos_frame']
            people_logs = frame_data['res']
            logged_person = [person_log for person_log in people_logs if person_log['id'] in person_idxs]
            if not logged_person:
                continue
            else:
                vid_people[frame_idx] = create_person_from_logs(logged_person[0])

        video_people_data[vid_name] = vid_people

    return video_people_data
