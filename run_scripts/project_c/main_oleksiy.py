# By Oleksiy Grechnyev, Apr 2024
# Example of using manager, run either PXI data or a video file offline


import sys
import argparse
import pathlib
import time
from datetime import datetime
from loguru import logger
import numpy as np
import cv2 as cv

from src.project_managers.project_c_manager import ProcessingManager
from src.offline_utils import obstacle_cache, frame_source

import numpy as np
import cv2 as cv

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


def process(video_path, pos, max_frames, record, use_obstacle_cache,
            crop_fov, use_cpp_api):
    """Process video source"""
    logger.add(f"logs/{video_path.stem}_{datetime.now().strftime('%Y-%m-%d__%H-%M-%S-%f')[:-3]}.log", level="INFO")
    if use_cpp_api:
        try:
            import shapeos.vyz as vyz
            from shapeos.shapeos_sources import PXISource

            api = vyz.vyzAPI()

            source = PXISource(video_path, max_frames, pos, vyz_api=api)
        except ImportError as err:
            logger.warning("Error improting vyzapi, using people-track FrameSource.", err)
            source = frame_source.FrameSource(video_path, pos, max_frames)
    else:
        source = frame_source.FrameSource(video_path, pos, max_frames)

    # The main algorith engine of this project
    manager = ProcessingManager(crop_fov=crop_fov, obstacles_off=False)

    for img_i, img_nz, img_nxy, img_nxyz in source:
        print('==================================================')
        print(f'{source.i_frame} : pos_frame={source.pos}, pos_sec={source.pos_sec} s')

        frame_out, tms = manager.process((img_i, img_nz, img_nxy, img_nxyz), source.pos)
        # print_it(frame_out, 'frame_out')
        print('tms=', tms)
        cv.imshow('frame_out', downsize(frame_out))
        if 27 == cv.waitKey(0):
            break


def main():
    parser = argparse.ArgumentParser(
        description='Run one video or PXI directory through the people track pipeline',
    )
    parser.add_argument('video_path', help='Video file or directory with PXI files', type=pathlib.Path)
    parser.add_argument('-p', '--pos', type=int, help='Starting position')
    parser.add_argument('-r', '--record', action='store_true', help='Record video instead of visualization')
    parser.add_argument('-m', '--max-frames', type=int, help='Number of frames to process if using --record')
    parser.add_argument('-c', '--obstacle-cache', action='store_true',
                        help='Cache static obstacles to avoid running DINO+SAM every time')
    parser.add_argument('--crop-fov', action="store_true", default=False,
                        help='Specify that we are using cropped video')
    parser.add_argument('--use-cpp-api', action="store_true", default=False,
                        help="If true, we will try to use the shapeos api")
    args = parser.parse_args()
    logger.info('args=', args)
    max_frames = None if not args.record else args.max_frames
    process(args.video_path, args.pos, max_frames, args.record,
            args.obstacle_cache, args.crop_fov, args.use_cpp_api)

########################################################################################################################
if __name__ == '__main__':
    main()
