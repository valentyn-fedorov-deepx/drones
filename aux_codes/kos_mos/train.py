# By Oleksiy Grechnyev, 5/3/24
# Train VOLOv8-seg on a split

import sys
import pathlib
import ultralytics

MODEL_NAME = 'yolov8m-seg.pt'
ROOT_SPLITS = '/media/sviatoslav/MainVolume/Projects/deepxhub/data/client_data/limited_sets/2024.11.26-without_obstacles_with_client_bush_filtered/'
SPLIT_NAME = 'fun1'
EPOCHS = 500
IMGSZ = (640, 640)


########################################################################################################################
def print_it(a, name: str = ''):
    # m = a.float().mean() if isinstance(a, torch.Tensor) else a.mean()
    # m = np.array(a, dtype='float64').mean()
    m = a.mean()
    print(name, a.shape, a.dtype, a.min(), m, a.max())


########################################################################################################################
def main1():
    model = ultralytics.YOLO('output/' + MODEL_NAME)

    p_yaml = pathlib.Path(ROOT_SPLITS) / 'data.yaml'
    assert p_yaml.exists()

    results = model.train(data=str(p_yaml), imgsz=IMGSZ, epochs=EPOCHS, amp=True,
                          scale=0.0, patience=0,
                          hsv_v=0.7, hsv_s=0.8)
    print('results=', type(results))


########################################################################################################################
if __name__ == '__main__':
    main1()
