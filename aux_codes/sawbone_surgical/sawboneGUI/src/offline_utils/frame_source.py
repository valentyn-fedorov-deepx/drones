import pathlib
import cv2 as cv
from loguru import logger

########################################################################################################################
class FrameSource:
    def __init__(self, video_path, pos: int = None,
                 max_frames: int = None, loop: bool = False):
        self.video_path = video_path
 
        self.pos = 0 if pos is None else pos
        self.max_frames = max_frames
        self._initial_pos = pos

        self.nproc = None
        self.video_in = None
        self.files_list = None

        self.i_frame = 0
        self.timestamp = 0

        self.loop = loop

        self.video_in = cv.VideoCapture(video_path)
        assert self.video_in.isOpened()

        total_frames = int(self.video_in.get(cv.CAP_PROP_FRAME_COUNT))

        if self.max_frames is None:
            self.max_frames = total_frames - self.pos
        else:
            self.max_frames = min(self.max_frames, total_frames - self.pos)

        if self.pos > 0:
            self.video_in.set(cv.CAP_PROP_POS_FRAMES, self.pos)
        self.fps = self.video_in.get(cv.CAP_PROP_FPS)
        

    def _reset(self):
        self.i_frame = 0
        self.pos = self._initial_pos
        logger.debug("Starting new loop in the frame source")

        self.video_in.set(cv.CAP_PROP_POS_FRAMES, self.pos)

    def __iter__(self):
        return self

    def __next__(self):
        ret, frame = self.video_in.read()
        if not ret:
            raise StopIteration

        self.pos += 1

        return frame
    
    def release(self):
        self.video_in.release()

    @property
    def pos_sec(self):
        """Position in seconds """
        return self.pos / self.fps


########################################################################################################################

if __name__ == "__main__":
    from tqdm import tqdm
    import cv2
    # source_path = "/sdb-disk/vyzai/data/pxi_source/2024.12.02_Cityscape/2024-12-02_17-22-12/pxi/"
    source_path = 0
    frame_source = FrameSource(source_path)
    # import pdb
    # pdb.set_trace()
    fps = 30
    video = cv2.VideoWriter("output1.mp4", cv.VideoWriter.fourcc(*'mp4v'),
                            fps, (2448, 2048), True)
    for frame_pxi in frame_source:
        cv2.imwrite('test1.jpg', frame_pxi)

    video.release()
