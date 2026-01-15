from pathlib import Path
from typing import Union, Literal, Optional
import numpy as np
import cv2

import shapeos.vyz as vyz


class Shapeos_Source:
    def __init__(self, vyz_api,
                 process_for: Literal["display", "measurements"] = "display"):
        self.vyz_api = vyz_api

        if process_for not in ["display", "measurements"]:
            raise ValueError(f"Incorrect process_for value: {process_for}")
        self._process_for = process_for

    @property
    def process_for(self):
        return self.process_for

    @process_for.setter
    def process_for(self, processing_type: str):
        if processing_type not in ["display", "measurements"]:
            raise ValueError(f"Incorrect process_for value: {processing_type}")

        self._process_for = processing_type

    def processForDisplay(self):
        self.vyz_api.processForDisplay()
        xyz = np.asarray(self.vyz_api.getData(vyz.vyzDataType.xyz), dtype=np.uint8)
        xy = np.asarray(self.vyz_api.getData(vyz.vyzDataType.xy), dtype=np.uint8)
        z = np.asarray(self.vyz_api.getData(vyz.vyzDataType.z), dtype=np.uint8)
        z = cv2.cvtColor(z, cv2.COLOR_GRAY2BGR)
        img = np.asarray(self.vyz_api.getData(vyz.vyzDataType.raw), dtype=np.uint8)

        img = np.stack([img, img, img], axis=-1)

        return img, z, xy, xyz

    def processForMeasurements(self):
        self.vyz_api.processForMeasurement()
        xyz = np.asarray(self.vyz_api.getData(vyz.vyzDataType.xyz), dtype=np.float32)
        xy = np.asarray(self.vyz_api.getData(vyz.vyzDataType.xy), dtype=np.float32)
        z = np.asarray(self.vyz_api.getData(vyz.vyzDataType.z), dtype=np.float32)
        img = np.asarray(self.vyz_api.getData(vyz.vyzDataType.raw), dtype=np.uint8)

        img = np.stack([img, img, img], axis=-1)

        return img, z, xy, xyz

    def read_normals(self):
        if self._process_for == "display":
            return self.processForDisplay()
        elif self._process_for == "measurements":
            return self.processForMeasurements()

    def set_data(self):
        raise NotImplementedError()

    def __iter__(self):
        return self

    def __next__(self):
        self.set_data()

        img, xyz, xy, z = self.read_normals()

        return img, xyz, xy, z


class PXISource(Shapeos_Source):
    def __init__(self, data_path: Union[str, Path],
                 max_frames: Optional[int] = None,
                 pos: Optional[int] = None,
                 fps: int = 25,
                 **kwargs):
        super().__init__(**kwargs)
        self.fps = fps
        self._data_path = Path(data_path)
        self._max_frames = max_frames
        self.pos = 0 if pos is None else pos
        self.i_frame = 0
        self._file_list_pxi = sorted([p for p in self._data_path.iterdir() if p.suffix.lower() == '.pxi'],
                                      key=lambda x: int(x.stem.split("-")[-1]))

    def set_data(self):
        if self.pos >= len(self._file_list_pxi):
            raise StopIteration

        if self._max_frames is not None and self.i_frame >= self._max_frames:
            raise StopIteration

        self.vyz_api.grabRawFrame(str(self._file_list_pxi[self.pos]))

        self.i_frame += 1
        self.pos += 1

    @property
    def pos_sec(self):
        """Position in seconds """
        return self.pos / self.fps


class LiveSource(Shapeos_Source):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.i_frame = 0

    def set_data(self):
        self.vyz_api.grabRawFrame()
