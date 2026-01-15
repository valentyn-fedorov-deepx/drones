from pathlib import Path
import subprocess
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm

from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--save-path", type=Path)
    parser.add_argument("--record-data", action="store_true", default=False)
    parser.add_argument("--record-video", action="store_true", default=False)
    parser.add_argument("--record-video-animation", action="store_true", default=False)
    parser.add_argument("--record-single-frame-error", action="store_true", default=False)
    parser.add_argument("--record-sloth-error", action="store_true", default=False)
    parser.add_argument("--focal-length", type=float, default=50)
    parser.add_argument("--n-workers", type=int, default=5)
    parser.add_argument("--pos", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--track-ids", type=int, nargs='+')

    return parser.parse_args()


def process(save_path, focal_length, record_video,
            record_data, record_single_frame_error,
            record_video_animation, pos, max_frames,
            track_ids, record_sloth_error, video_path):
    saved_pkl_file_path = save_path / f'{Path(video_path).stem}.pkl'
    saved_json_file_path = save_path / f'distances_sloth_{Path(video_path).stem}.json'

    if record_data:
        if max_frames is not None:
            record_process = subprocess.Popen(["python", "-m", "run_scripts.main_oleksiy_recorder",
                                               str(video_path), "--save-path", save_path,
                                               "-p", str(pos), "-m", str(max_frames)])
        else:
            record_process = subprocess.Popen(["python", "-m", "run_scripts.main_oleksiy_recorder",
                                               str(video_path), "--save-path", save_path,
                                               "-p", str(pos)])

        if record_video or record_single_frame_error:
            record_process.wait()

    if record_video:
        subprocess.call(["python", "-m", "run_scripts.gen_video_from_record",
                         "--video-path", str(video_path),
                         "--pkl-path", str(saved_pkl_file_path),
                         "--save-path", str(save_path)])

    if record_sloth_error:
        subprocess.call(["python", "-m", "run_scripts.testing_sloth_error",
                         "--pkl-path", str(saved_pkl_file_path), 
                         "--save-path", str(save_path)])

    if record_single_frame_error:
        subprocess.call(["python", "-m", "run_scripts.single_frame_error",
                         "--pkl-path", str(saved_pkl_file_path),
                         "--save-path", str(save_path),
                         "--focal-length", str(focal_length)])

    if record_video_animation:
        subprocess.call(["python", "-m", "run_scripts.gen_track_animation",
                         "--measurements-path", str(saved_json_file_path),
                         "--track-ids", " ".join(track_ids),
                         "--pkl-path", str(saved_pkl_file_path),
                         "--video-path", str(video_path),
                         "--save-path", str(save_path)])


if __name__ == "__main__":
    args = parse_args()
    print(args)

    if args.data_path.is_file():
        videos = [args.data_path]
    else:
        videos = list(args.data_path.glob("*.mp4"))

    process_pool_f = partial(process, args.save_path, args.focal_length,
                             args.record_video, args.record_data,
                             args.record_single_frame_error,
                             args.record_video_animation, args.pos,
                             args.max_frames, args.track_ids,
                             args.record_sloth_error)

    n_workers = min(args.n_workers, len(videos))
    if n_workers <= 1:
        for video in videos:
            process(args.save_path, args.focal_length,
                    args.record_video, args.record_data,
                    args.record_single_frame_error,
                    args.record_video_animation,
                    args.pos, args.max_frames, args.track_ids,
                    args.record_sloth_error,
                    video)
    else:
        with Pool(n_workers) as p:
            tqdm(p.imap_unordered(process_pool_f, videos), total=len(videos))
            # p.map(process_pool_f, videos)
