import pyvirtualcam
from argparse import ArgumentParser

from src.camera.camera import LiveCamera

if __name__ == "__main__":

    camera = LiveCamera(exposure=50_000, depth=8)
    with pyvirtualcam.Camera(width=2448, height=2048, fps=20) as cam:
        print(f'Using virtual camera: {cam.device}')
        # frame = np.zeros((cam.height, cam.width, 3), np.uint8)  # RGB
        while True:
            data_tensor = next(camera)
            # cam.send(data_tensor.view_img)
            cam.send(data_tensor.n_xyz)
            # print('Sending frame')
            cam.sleep_until_next_frame()
