# By Oleksiy Grechnyev, 5/6/24

import pathlib
import cv2 as cv
from loguru import logger

from src.data_pixel_tensor import DataPixelTensor


SUPPORTED_EXTENSIONS = ['.pxi', '.npy', '.png', '.tif']


########################################################################################################################
class FrameSource:
    """
    Read a video source (video file or directory with files) and provide frames with
    so-called normals (real or fake)
    """

    def __init__(self, video_path: str, pos: int = None,
                 max_frames: int = None, loop: bool = False):
        self.video_path = video_path
        p_video_path = pathlib.Path(video_path)

        self.files_mode = p_video_path.is_dir()
        self.pos = 0 if pos is None else pos
        self.max_frames = max_frames
        self._initial_pos = pos

        self.nproc = None
        self.video_in = None
        self.files_list = None

        self.i_frame = 0
        self.timestamp = 0

        self.loop = loop

        if not self.files_mode:
            self.video_in = cv.VideoCapture(str(video_path))
            assert self.video_in.isOpened()

            total_frames = int(self.video_in.get(cv.CAP_PROP_FRAME_COUNT))

            if self.max_frames is None:
                self.max_frames = total_frames - self.pos
            else:
                self.max_frames = min(self.max_frames, total_frames - self.pos)

            if self.pos > 0:
                self.video_in.set(cv.CAP_PROP_POS_FRAMES, self.pos)
            self.fps = self.video_in.get(cv.CAP_PROP_FPS)
        else:
            # Get PXI file list
            self.files_list = sorted([p for p in p_video_path.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS], key=lambda x: int(x.stem.split("-")[-1]))
            self.fps = 25

            if self.max_frames is None:
                self.max_frames = len(self.files_list) - self.pos
            else:
                self.max_frames = min(self.max_frames, len(self.files_list) - self.pos)

    def __len__(self):
        return self.max_frames

    def _reset(self):
        self.i_frame = 0
        self.pos = self._initial_pos
        logger.debug("Starting new loop in the frame source")
        if not self.files_mode:
            self.video_in.set(cv.CAP_PROP_POS_FRAMES, self.pos)

    def __iter__(self):
        return self

    def __next__(self):
        if self.max_frames is not None and self.i_frame >= self.max_frames:
            if self.loop:
                self._reset()
            else:
                raise StopIteration

        frame_data = self.__getitem__(self.pos)

        self.pos += 1
        self.i_frame += 1

        return frame_data

    def __getitem__(self, idx):
        if self.files_mode:
            frame_data = DataPixelTensor.from_file(str(self.files_list[idx]))
        else:
            self.video_in.set(cv.CAP_PROP_POS_FRAMES, idx)
            ret, frame = self.video_in.read()
            frame_data = DataPixelTensor(frame, fake_normals=True)
            self.video_in.set(cv.CAP_PROP_POS_FRAMES, self.pos)

        return frame_data

    def estimate_fps(self):
        import numpy as np
        first_file_path = str(self.files_list[0])
        last_file_path = str(self.files_list[-1])
        with open(first_file_path, "rb") as f:
            first_data = f.read()

        with open(last_file_path, "rb") as f:
            last_data = f.read()

        first_data = np.frombuffer(first_data, dtype=np.uint8)
        last_data = np.frombuffer(last_data, dtype=np.uint8)

        dpt_first_header = DataPixelTensor.parse_header_from_buffer(first_data)
        dpt_last_header = DataPixelTensor.parse_header_from_buffer(last_data)
        estimated_fps = len(self) / (dpt_last_header["created_at"] - dpt_first_header["created_at"])
        return int(estimated_fps)

    @property
    def im_size(self):
        frame = self.__getitem__(1)
        return frame.raw.shape

    @property
    def pos_sec(self):
        """Position in seconds """
        if not self.files_mode:
            return self.video_in.get(cv.CAP_PROP_POS_MSEC) / 1000
        else:
            return self.pos / self.fps


########################################################################################################################

if __name__ == "__main__":
    from tqdm import tqdm
    import cv2
    # source_path = "/sdb-disk/vyzai/data/pxi_source/2024.12.02_Cityscape/2024-12-02_17-22-12/pxi/"
    source_path = "/mnt/larger_disk/recording_12bit"
    frame_source = FrameSource(source_path, 3, 10)

    fps = 4
    video = cv2.VideoWriter("output1.mp4", cv.VideoWriter.fourcc(*'h264'),
                            fps, (2448, 2048), True)
    for frame_pxi in tqdm(frame_source):
        video.write(frame_pxi.view_img)

    video.release()
