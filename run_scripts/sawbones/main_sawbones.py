import argparse
import pathlib
from datetime import datetime
from loguru import logger
import cv2 as cv

from src.offline_utils import frame_source
from src.project_managers.sawbones_manager import SawbonesManager

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
def process(video_path, pos, max_frames, record, use_cpp_api):
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
    manager = SawbonesManager()

    if record:
        pathlib.Path('./output').mkdir(exist_ok=True)
        fps = 10
        video_out = cv.VideoWriter(f"./output/{video_path.stem}_{datetime.now().strftime('%Y-%m-%d__%H-%M-%S-%f')[:-3]}.mp4",
                                   cv.VideoWriter.fourcc(*'mp4v'), fps,
                                   (2448, 2048), True)
        assert video_out.isOpened()

    for img_i, img_nz, img_nxy, img_nxyz, timestamp in source:
        logger.info('==================================================')
        logger.info(f'{source.i_frame} : pos_frame={source.pos}, pos_sec={source.pos_sec} s')

        # if source.pos == 2:
        #     import ipdb; ipdb.set_trace()
        frame_out, _ = manager.process(img_i)

        if record:
            video_out.write(frame_out)
        else:
            cv.imshow('frame_out', downsize(frame_out))
            if 27 == cv.waitKey(0):
                break


########################################################################################################################
def main():
    parser = argparse.ArgumentParser(
        description='Run one video or PXI directory through the people track pipeline',
    )
    parser.add_argument('video_path', help='Video file or directory with PXI files', type=pathlib.Path)
    parser.add_argument('-p', '--pos', type=int, help='Starting position')
    parser.add_argument('-r', '--record', action='store_true', help='Record video instead of visualization')
    parser.add_argument('-m', '--max-frames', type=int, help='Number of frames to process if using --record')
    parser.add_argument('--use-cpp-api', action="store_true", default=False,
                        help="If true, we will try to use the shapeos api")
    args = parser.parse_args()
    logger.info('args=', args)
    max_frames = None if not args.record else args.max_frames
    process(args.video_path, args.pos, max_frames, args.record,
            args.use_cpp_api)


########################################################################################################################
if __name__ == '__main__':
    main()
