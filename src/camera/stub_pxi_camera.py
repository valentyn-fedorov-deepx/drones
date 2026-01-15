import os

from deployment.webrtc_streaming.video_track import NumpyVideoTrack
from src.data_pixel_tensor import DataPixelTensor


PXI_FOLDER_PATH = "/mnt/larger_disk/8mm_ShantyChange16mm8bit"
FILENAME_TEMPLATE = "8mm_ShantyChange16mm8bit-{:03}.pxi"
FPS = 30


class StubPxiCamera(NumpyVideoTrack):
    def __init__(
            self,
            color_data=True,
            pxi_path=PXI_FOLDER_PATH,
            lazy_calculations = True,
            return_data_tensor = None
    ):
        self.cur_frame = 0
        pxi_frame = self.__next__()
        if pxi_frame is None:
            raise IOError(f"Cannot open pxi file: {self._get_pxi_path()}")
        super().__init__(pxi_frame.width, pxi_frame.height, FPS)


    def __next__(self):
        pxi_path = self._get_pxi_path()
        if not os.path.exists(pxi_path):
            print(">>!! Can't find: " + pxi_path)
            return None

        self.cur_frame += 1
        return DataPixelTensor.from_pxi_file(pxi_path)

    def _get_pxi_path(self):
        file_name = FILENAME_TEMPLATE.format(self.cur_frame)
        return PXI_FOLDER_PATH + "/" + file_name


