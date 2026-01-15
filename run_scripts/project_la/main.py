# By Oleksiy Grechnyev, Apr 2024
# Example of using manager, run either PXI data or a video file offline


import sys
import argparse
import pathlib
import time
from datetime import datetime
from loguru import logger
import cv2 as cv

from src.project_managers.project_la_manager import ProjectLAManager
from src.offline_utils import frame_source

# DATA_DIR = '/home/seymour/w/big_data/360m/data_raw'
# VIDEO_FILE = '/home/seymour/w/1/vyzai_sniper/data_from_client/april_fool/video4/Raw/1_2024-04-24__13-06-16-707.mp4'
# START_POS = 1300


########################################################################################################################
def print_it(a, name: str = ''):
    # m = a.float().mean() if isinstance(a, torch.Tensor) else a.mean()
    m = a.mean()
    logger.info(name, a.shape, a.dtype, a.min(), m, a.max())


########################################################################################################################
def downsize(img):
    img = cv.resize(img, None, None, 0.5, 0.5)
    return img


def auto_downsize(img):
    while max(img.shape[0], img.shape[1]) > 1920:
        img = downsize(img)
    return img


########################################################################################################################
def process(video_path, pos, max_frames, record,
            save_path, dump_data):
    """Process video source"""
    logger.add(f"logs/{video_path.stem}_{datetime.now().strftime('%Y-%m-%d__%H-%M-%S-%f')[:-3]}.log", level="INFO")
    source = frame_source.FrameSource(video_path, pos, max_frames)

    im_size = source.im_size

    save_path = pathlib.Path(save_path)
    save_path.mkdir(exist_ok=True)

    if dump_data:
        (save_path / "dump").mkdir(exist_ok=True)

    # The main algorith engine of this project
    manager = ProjectLAManager(config_path="configs/project_la/manager.yaml",
                                 device='cuda')

    if record:
        save_video_path = save_path / f"{video_path.stem}_{datetime.now().strftime('%Y-%m-%d__%H-%M-%S-%f')[:-3]}.mp4" 
        fps = source.fps
        fps = 24
        video_out = cv.VideoWriter(str(save_video_path),
                                   cv.VideoWriter.fourcc(*'mp4v'), fps,
                                   (im_size[1], im_size[0]), True)
        assert video_out.isOpened()

    for data_tensor in source:
        logger.info('==================================================')
        logger.info(f'{source.i_frame} : pos_frame={source.pos}, pos_sec={source.pos_sec} s')

        start_time = time.time()
        data_tensor.calculate_all()
        data_tensor.convert_to_numpy()
        manager.process(data_tensor)
        logger.info(f"Total processing time: {time.time() - start_time:.3f} sec")

        if dump_data:
            dump_path = save_path / "dump" / f'dump_{source.i_frame}.pkl'
            manager.dump_data(dump_path)
            logger.info(f"Dumped data to {dump_path}")

        if record:
            start_time = time.time()
            frame_out = manager.generate_vizualization_for_latest_data()
            frame_out = cv.cvtColor(frame_out, cv.COLOR_RGB2BGR)
            cv.imwrite('frame_test.png', frame_out)
            # import ipdb; ipdb.set_trace()

            video_out.write(frame_out)
            logger.info(f"Recording time: {time.time() - start_time:.3f} sec")


########################################################################################################################
def main():
    parser = argparse.ArgumentParser(
        description='Run one video or PXI directory through the pipeline',
    )
    parser.add_argument('video_path', help='Video file or directory with PXI files', type=pathlib.Path)
    parser.add_argument("--save-path", type=pathlib.Path, default="output")
    parser.add_argument('-p', '--pos', type=int, help='Starting position')
    parser.add_argument('-r', '--record', action='store_true', help='Record video instead of visualization')
    parser.add_argument('-m', '--max-frames', type=int, help='Number of frames to process if using --record')
    parser.add_argument("--dump-data", action="store_true", default=False,
                        help="Dump data tensors and detected objects to a pickle file")
    args = parser.parse_args()
    logger.info('args=', args)
    max_frames = None if not args.record else args.max_frames
    process(args.video_path, args.pos, max_frames, args.record,
            args.save_path, args.dump_data)


########################################################################################################################
if __name__ == '__main__':
    main()
