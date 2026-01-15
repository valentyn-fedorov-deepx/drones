try:
    from .hf_detector import HFDetector
except (ModuleNotFoundError, ImportError) as f:
    print(f)
try:
    from .detectron_detector import DetectronDetector
except ModuleNotFoundError as f:
    print(f)
try:
    from .yolo_detector import YoloDetector
except ModuleNotFoundError as f:
    print(f)
