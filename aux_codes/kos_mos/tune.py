# By Oleksiy Grechnyev, 5/3/24
# Train VOLOv8-seg on a split

import sys
import pathlib
import ultralytics

MODEL_NAME = 'yolov8m-seg.pt'
ROOT_SPLITS = '/media/sviatoslav/MainVolume/Projects/deepxhub/data/train'
SPLIT_NAME = 'final_dataset'
EPOCHS = 1000
IMGSZ = (608, 512)


space = {
    "lr0": (1e-5, 1e-1),  # initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
    "lrf": (0.0001, 0.1),  # final OneCycleLR learning rate (lr0 * lrf)
    "momentum": (0.7, 0.98, 0.3),  # SGD momentum/Adam beta1
    "weight_decay": (0.0, 0.001),  # optimizer weight decay 5e-4
    "warmup_epochs": (0.0, 100),  # warmup epochs (fractions ok)
    "warmup_momentum": (0.0, 0.95),  # warmup initial momentum
    "box": (1.0, 20.0),  # box loss gain
    "cls": (0.2, 4.0),  # cls loss gain (scale with pixels)
    "dfl": (0.4, 6.0),  # dfl loss gain
    "hsv_h": (0.0, 0.1),  # image HSV-Hue augmentation (fraction)
    "hsv_s": (0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
    "hsv_v": (0.0, 0.9),  # image HSV-Value augmentation (fraction)
    "degrees": (0.0, 0.0),  # image rotation (+/- deg)
    "translate": (0.0, 0.0),  # image translation (+/- fraction)
    "scale": (0.0, 0.95),  # image scale (+/- gain)
    "shear": (0.0, 10.0),  # image shear (+/- deg)
    "perspective": (0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
    "flipud": (0.0, 1.0),  # image flip up-down (probability)
    "fliplr": (0.0, 1.0),  # image flip left-right (probability)
    "bgr": (0.0, 0.0),  # image channel bgr (probability)
    "mosaic": (0.0, 1.0),  # image mixup (probability)
    "mixup": (0.0, 1.0),  # image mixup (probability)
    "copy_paste": (0.0, 1.0),  # segment copy-paste (probability)
}


########################################################################################################################
def print_it(a, name: str = ''):
    # m = a.float().mean() if isinstance(a, torch.Tensor) else a.mean()
    # m = np.array(a, dtype='float64').mean()
    m = a.mean()
    print(name, a.shape, a.dtype, a.min(), m, a.max())


########################################################################################################################
def main1():
    model = ultralytics.YOLO('output/' + MODEL_NAME)

    p_yaml = pathlib.Path(ROOT_SPLITS) / SPLIT_NAME / 'data.yaml'
    # import ipdb; ipdb.set_trace()
    assert p_yaml.exists()

    results = model.tune(data=str(p_yaml), epochs=300, iterations=10,
                         optimizer='AdamW', plots=False, save=False, val=False,
                         space=space)

    # results = model.train(resume=True)
    print('results=', type(results))


########################################################################################################################
if __name__ == '__main__':
    main1()
