CAMERA_VENDOR = "daheng"
# CAMERA_VENDOR = "ids"
#CAMERA_VENDOR = "stub_np"
#CAMERA_VENDOR = "stub_pxi"

if CAMERA_VENDOR == "daheng":
    from .daheng_camera import DahengCamera as LiveCamera
elif CAMERA_VENDOR == "ids":
    from .ids_camera import IdsPeakCamera as LiveCamera
elif CAMERA_VENDOR == "stub_np":
    from .stub_np_camera import StubNpCamera as LiveCamera
elif CAMERA_VENDOR == "stub_pxi":
    from .stub_pxi_camera import StubPxiCamera as LiveCamera
else:
    pass


from src.camera.autoexposure.base_autoexposure import BaseAutoExposure

CUSTOM_AUTOEXPOSURE_METHODS = {
    "None": None,
    "Base": BaseAutoExposure,
}