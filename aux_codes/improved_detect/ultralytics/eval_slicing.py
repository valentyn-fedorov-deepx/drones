from argparse import ArgumentParser
from tqdm import tqdm
from pathlib import Path
import cv2
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch
import json

from src.cv_module.people.people_detection import YoloDetectorSlicing
from aux_codes.improved_detect.data import UltralyticsFormatDataset


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--weights-path", type=Path,
                        default=Path('models'))
    parser.add_argument("--detector-config-path", type=Path,
                        default=Path("configs"))
    parser.add_argument("--save-path", type=Path,
                        default=Path('output/eval_detector'))
    parser.add_argument("--device", default='cpu')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    H, W = 2048, 2448
    detector = YoloDetectorSlicing(args.detector_config_path,
                                   args.weights_path, (H, W),
                                   device=args.device)

    dataset = UltralyticsFormatDataset(args.dataset_path)

    map_metric = MeanAveragePrecision("xyxy", iou_type='bbox',
                                      extended_summary=True,
                                      iou_thresholds=torch.linspace(0.3, 1.0, steps=15).tolist())

    for label, name in tqdm(zip(dataset._labels, dataset._names), total=len(dataset)):
        image_path = dataset._images_path / f"{name[:-4]}.jpg"
        image = cv2.imread(str(image_path))
        image = cv2.resize(image, (W, H),
                           interpolation=cv2.INTER_LINEAR)
        im_h, im_w = image.shape[:2]

        detected_people = detector.predict(image)
        detected_bboxes = [detected_person.bbox.tolist() for detected_person in detected_people]
        detected_bboxes_score = [detected_person.conf for detected_person in detected_people]
        detected_bboxes_class = [0] * len(detected_bboxes)

        detected_data = dict(boxes=torch.tensor(detected_bboxes),
                             labels=torch.tensor(detected_bboxes_class, dtype=torch.int),
                             scores=torch.tensor(detected_bboxes_score))

        gt_bboxes = []
        for gt_bbox in label['bboxes']:
            xc, yc, box_w, box_h = gt_bbox.numpy()
            x1 = (xc - (box_w / 2)) * im_w
            x2 = (xc + (box_w / 2)) * im_w

            y1 = (yc - (box_h / 2)) * im_h
            y2 = (yc + (box_h / 2)) * im_h
            gt_bbox_xyxy = [x1, y1, x2, y2]
            gt_bboxes.append(gt_bbox_xyxy)

        gt_data = dict(boxes=torch.tensor(gt_bboxes),
                       labels=torch.zeros(len(gt_bboxes), dtype=torch.int))

        map_metric.update([detected_data],
                          [gt_data])
        # import ipdb; ipdb.set_trace()

    res = map_metric.compute()

    del res['recall']
    del res['precision']
    del res['scores']
    del res['ious']

    res_proper_types = dict()
    for key, value in res.items():
        if isinstance(value, torch.Tensor):
            value = value.tolist()
        if isinstance(value, dict):
            value = {" ".join([str(k_.item()) for k_ in k]): v.tolist() for k, v in value.items()}
        res_proper_types[key] = value
    # res = {key: value.tolist() for key, value in res.items()}

    args.save_path.mkdir(parents=True,
                         exist_ok=True)
    with open(args.save_path / "metrics.json", 'w') as f:
        json.dump(res_proper_types, f, indent=2)
