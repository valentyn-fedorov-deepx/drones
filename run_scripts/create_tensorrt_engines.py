# By Oleksiy Grechnyev, 4/30/24
# Create TensorRT engines usable by manager.py
# Note: This is NOT the code Volodymyr used, I am trying to recreate the logic
#
# Hopefully you know this already, but I'll remind just in case:
# A TensorRT engine depends on the particular hardware (+software versions)
# It can NEVER be transferred to another machine
# It is always generated on the machine where it will be used

import sys
import pathlib

import ultralytics


########################################################################################################################
def print_it(a, name: str = ''):
    # m = a.float().mean() if isinstance(a, torch.Tensor) else a.mean()
    m = a.mean()
    print(name, a.shape, a.dtype, a.min(), m, a.max())


########################################################################################################################
def create_one(p_root, model_name, engine_name, batch_size, imgsz):
    p_model = p_root / model_name
    p_engine0 = p_model.with_suffix('.engine')  # Output File name as created by export
    p_engine = p_root / engine_name  # Output File name that we want

    print('==========================================')
    print(f'Creating {engine_name} ...')

    if p_engine.exists():
        # We don't want to create engine if exists already, long operation!
        print(f'{engine_name} already exists, nothing to do !')
        return

    model = ultralytics.YOLO(p_model)

    model.export(format='tensorrt', half=True, batch=batch_size, imgsz=imgsz)
    p_engine0.rename(p_engine)


########################################################################################################################
def main():
    p_root = pathlib.Path('./models')
    p_root.mkdir(exist_ok=True)


    detection_model_name = "city_km_wm"
    create_one(p_root, f'{detection_model_name}.pt', f'{detection_model_name}-640-640-fp16-bs1.engine', 1, (640, 640))
    create_one(p_root, f'{detection_model_name}.pt', f'{detection_model_name}-640-640-fp16-bs4.engine', 4, (640, 640))

    create_one(p_root, 'yolov8m-pose.pt', 'yolov8m-pose-640-640-fp16-bs1.engine', 1, (640, 640))
    create_one(p_root, 'yolov8m-pose.pt', 'yolov8m-pose-320-320-fp16-bs1.engine', 1, (320, 320))

    create_one(p_root, 'yolov8m-seg.pt', 'yolov8m-seg-640-640-fp16-bs1.engine', 1, (640, 640))
    create_one(p_root, 'yolov8m-seg.pt', 'yolov8m-seg-320-320-fp16-bs1.engine', 1, (320, 320))

    if (p_root / 'obst-yolov8m-seg.pt').exists():
        create_one(p_root, 'obst-yolov8m-seg.pt', 'obst-yolov8m-seg-608-512-fp16-bs1.engine', 1, (512, 608))


########################################################################################################################
if __name__ == '__main__':
    main()
