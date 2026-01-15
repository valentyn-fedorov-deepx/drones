# By Oleksiy Grechnyev, 5/3/24
# Extract one frame from a video

import sys
import pathlib
import numpy as np
import cv2 as cv
import pylab as p

# VIDEO_FILE = '/home/seymour/w/1/vyzai_sniper/data_from_client/2024.03.08-batch2-500m/500mRoadside.mp4'
VIDEO_FILE = '/home/seymour/w/1/vyzai_sniper/data_from_client/april_fool/video4/Raw/10_2024-04-24__10-44-58-771.mp4'
POS_FRAMES = 0


########################################################################################################################
def print_it(a, name: str = ''):
    # m = a.float().mean() if isinstance(a, torch.Tensor) else a.mean()
    m = a.mean()
    print(name, a.shape, a.dtype, a.min(), m, a.max())


########################################################################################################################
def main1():
    p_video = pathlib.Path(VIDEO_FILE)
    name = p_video.with_suffix('.png').name
    video = cv.VideoCapture(str(p_video))
    assert video.isOpened()
    video.set(cv.CAP_PROP_POS_FRAMES, POS_FRAMES)

    ret, frame = video.read()
    assert ret and frame is not None
    cv.imwrite(f'./output/{name}', frame)


########################################################################################################################
if __name__ == '__main__':
    main1()
