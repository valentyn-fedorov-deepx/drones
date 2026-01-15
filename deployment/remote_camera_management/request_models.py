from pydantic import BaseModel

from typing import Optional


class StartRecordingRequest(BaseModel):
    record_name: str = "record"  # Name for the saved data

class ResumeRecordingRequest(StartRecordingRequest):
    begin_frame_idx: int = 0    # Suffix of the first frame


class SetCameraParamsRequest(BaseModel):
    exposure: Optional[float] = None
    gain: Optional[float] = None
    pixel_format: Optional[str] = None
    exposure_mode: Optional[str] = None
    custom_exposure_mode: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
