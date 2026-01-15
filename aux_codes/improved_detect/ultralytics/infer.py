from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
import cv2

from src.cv_module.people.people_detection import YoloDetectorSlicing
from src.offline_utils.frame_source import FrameSource
from src.cv_module.visualization import ResultsAnnotator, plot_one_box


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--input-path", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--save-path", type=Path)
    parser.add_argument("--device", default="cuda")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    config_dir = 'configs'

    H, W = 2048, 2448

    detector = YoloDetectorSlicing(config_dir, args.weights.parent, (H, W),
                                   crop_fov=False, use_trt=False, device=args.device)

    annotator = ResultsAnnotator()

    source = FrameSource(args.input_path,
                         None,
                         None)

    args.save_path.mkdir(parents=True, exist_ok=True)

    video_out = cv2.VideoWriter(str(args.save_path / f"{args.input_path.stem}.mp4"),
                                cv2.VideoWriter.fourcc(*'mp4v'), 4,
                                (2448, 2048), True)

    for data in tqdm(source):
        img_i, img_nz, img_nxy, img_nxyz = data
        detected_people = detector.predict(img_i)
        img_to_draw = img_i.copy()
        for detected_person in detected_people:
            x1, y1, x2, y2 = bbox = detected_person.bbox.round().astype(int)

            if detected_person.conf > 0.3:
                plot_one_box(bbox, img_to_draw, (0, 255, 0),
                             f"conf: {detected_person.conf:.1f}", 4)
            # cv2.rectangle(img_to_draw, (x1, y1), (x2, y2),
            #               (0, 255, 0), 4)

        video_out.write(img_to_draw)

    video_out.release()
