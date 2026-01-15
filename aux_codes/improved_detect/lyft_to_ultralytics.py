from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
import cv2
import shutil
from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.geometry_utils import view_points, box_in_image, BoxVisibility


person_class_name = 'pedestrian'
cams = ['CAM_BACK',
        'CAM_FRONT_ZOOMED',
        'CAM_FRONT',
        'CAM_FRONT_LEFT',
        'CAM_FRONT_RIGHT',
        'CAM_BACK_RIGHT',
        'CAM_BACK_LEFT',
        ]

if __name__ == "__main__":
    DATA_PATH = Path("/sdb-disk/vyzai/datasets/lyft/train")
    lyft_dataset = LyftDataset(data_path=DATA_PATH, json_path=DATA_PATH / 'train_data')

    samples = list()
    classses_count = defaultdict(int)

    pb = tqdm()

    save_path = Path("datasets/lyft_people")

    images_save_path = save_path / 'images'
    labels_save_path = save_path / 'labels'

    images_save_path.mkdir(parents=True, exist_ok=True)
    labels_save_path.mkdir(exist_ok=True)

    for scene in lyft_dataset.scene:
        sample_token = scene["first_sample_token"]
        sample = lyft_dataset.get('sample', sample_token)
        while sample['next']:
            for cam_name in cams:
                cam_id = sample['data'][cam_name]
                sample_data = lyft_dataset.get('sample_data', cam_id)
                im_path, bboxes, K = sample_labels = lyft_dataset.get_sample_data(sample_data['token'])

                image = cv2.imread(im_path)
                samples.append(sample_labels)
                yolo_labels = list()
                for bbox in bboxes:
                    classses_count[bbox.name] += 1
                    if bbox.name != person_class_name:
                        continue

                    if not box_in_image(bbox, K, image.shape[:2], BoxVisibility.ANY):
                        continue

                    bbox_corners = bbox.corners()
                    corners_from_view = view_points(bbox_corners, K, True)[:2, :].T.round().astype(int)

                    x2, y2 = corners_from_view.max(0)
                    x1, y1 = corners_from_view.min(0)

                    w = x2 - x1
                    h = y2 - y1
                    x_c = (x2 + x1) / 2
                    y_c = (y2 + y1) / 2

                    x_cn = x_c / image.shape[1]
                    y_cn = y_c / image.shape[0]
                    w_n = w / image.shape[1]
                    h_n = h / image.shape[0]

                    labels = list(map(str, [0, x_cn, y_cn, w_n, h_n]))
                    labels_str = ' '.join(labels) + '\n'

                    yolo_labels.append(labels_str)

                if yolo_labels:
                    # cv2.imwrite(im_path.name, cv2.resize(image, dsize=None,
                    #                                     fx=0.3, fy=0.3))
                    # import ipdb; ipdb.set_trace()
                    with open(labels_save_path / f"{im_path.stem}.txt", 'w') as f:
                        f.writelines(yolo_labels)

                    shutil.copyfile(im_path, images_save_path / im_path.name)
                pb.update(1)
            sample = lyft_dataset.get('sample', sample['next'])
    pb.close()
