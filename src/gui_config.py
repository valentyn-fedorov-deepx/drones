from logging import INFO, DEBUG
from os import path, makedirs


class Config:
    log_path: str = 'logs'
    if not path.exists(log_path):
        makedirs(log_path)
    photo_path: str = 'photos'
    if not path.exists(photo_path):
        makedirs(photo_path)
    log_str: str = 'INFO'
    log_level: int = INFO if log_str == 'INFO' else DEBUG

    not_ret_frames: int = 5

    cam_min_exposure: int = 20  # us
    cam_default_exposure: int = 5000
    cam_max_exposure: int = 1000000

    cam_min_gain: int = 0  # dB
    cam_default_gain: int = 10
    cam_max_gain: int = 24

    overlay_images = [
        '',
        path.join('src', 'threads', 'synth_obstacles', 'data', 'car'),
        path.join('src', 'threads', 'synth_obstacles', 'data', 'bush'),
        path.join('src', 'threads', 'synth_obstacles', 'data', 'post')
    ]

