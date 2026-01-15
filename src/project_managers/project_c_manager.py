import numpy as np
import cv2
from pathlib import Path
from loguru import logger
import omegaconf
from typing import Tuple, List, Dict
import yaml
import json
import time

from src.balistics.calculator import solve_all, get_velocity, get_moa, moa_to_rad
from src.cv_module.obstacles.dynamic_obstacle_detection import DynamicObstacleDetector
from src.cv_module.obstacles.flat_earth_vol import FlatEarthVol
from src.cv_module.occlusion_interaction import ObstacleInteractionManager
from src.cv_module.detectors import YoloDetector
from src.cv_module.people.person import Person
from src.cv_module.people.people_pose_classifier import PoseClassifier
from src.cv_module.people.people_range_estimation_v2 import PeopleRangeEstimator
from src.cv_module.people.people_pose_estimation import PeoplePoseEstimator
from src.cv_module.people.people_tracking import PeopleTracker
from src.cv_module.people.people_segmentation import PeopleSegmenter
from src.project_managers.outputs.project_c_outputs import ShotEvent

with open("configs/pose.yaml", 'r') as f:
    POSE_LABELS = yaml.safe_load(f)

HUMAN_DETECTIONS_DICT = {
        0: ['nose', 'l_eye', 'r_eye',
            'l_ear', 'r_ear'],
        1: ['nose', 'l_eye', 'r_eye',
            'l_ear', 'r_ear'],
        2: ['l_shoulder', 'r_shoulder'],
        3: ['l_shoulder', 'r_shoulder'],
        4: ['l_shoulder', 'r_shoulder'],
        5: ["l_hip", 'r_hip'],
        6: ["l_knee", 'l_ankle'],
        7: ["r_knee", 'r_ankle'],
        8: ["l_elbow", "l_wrist"],
        9: ["r_elbow", "r_wrist"]
}

AIM_POINT_FROM_POSE = {
    "nose": 'head',
    "l_eye": 'head',
    "r_eye": 'head',
    "l_ear": 'head',
    "r_ear": 'head',
    "l_shoulder": "torso",
    "r_shoulder": "torso",
    "l_elbow": "larm",
    "r_elbow": "rarm",
    "l_wrist": "larm",
    "r_wrist": "rarm",
    "l_hip": "torso",
    "r_hip": "torso",
    "l_knee": "lleg",
    "r_knee": "rleg",
    "l_ankle": "lleg",
    "r_ankle": "rleg",
}


RESPONSE_IDX_TO_BODY_PART = {
    0: "head",
    1: "neck",
    2: "chest",
    3: "back",
    4: "torso",
    5: "pelvis",
    6: "lleg",
    7: "rleg",
    8: "larm",
    9: "rarm"
}


def draw_crosshair(image: np.ndarray, x: int, y: int, r: int, g: int, b: int):
    """
    Draw a filled crosshair on an image at specified coordinates with given RGB color.

    Args:
        image: numpy.ndarray - Input image
        x: int - X coordinate of crosshair center
        y: int - Y coordinate of crosshair center
        r: int - Red component (0-255)
        g: int - Green component (0-255)
        b: int - Blue component (0-255)
    """
    # Define the color of the crosshair (OpenCV uses BGR)
    color = (b, g, r)

    # Draw vertical rectangle
    cv2.rectangle(image,
                  (x - 5, y - 20),  # top-left point
                  (x + 5, y + 20),  # bottom-right point
                  color,
                  thickness=-1)  # -1 means filled

    # Draw horizontal rectangle
    cv2.rectangle(image,
                  (x - 20, y - 5),  # top-left point
                  (x + 20, y + 5),  # bottom-right point
                  color,
                  thickness=-1)  # -1 means filled


def find_person_closest_to_the_center(people: List[Person],
                                      center_point_xy: Tuple = (1224, 1024)):
    center_point_xy = np.array(center_point_xy)
    closest_person = min(people,
                         key=lambda person: np.linalg.norm(center_point_xy - np.array((person.x_pos, person.y_pos))))
    return closest_person


def get_human_detection(person: Person):

    human_detection_value = 0
    if not person.has_pose:
        return human_detection_value

    for det_idxs, pose_labels in HUMAN_DETECTIONS_DICT.items():
        # import ipdb; ipdb.set_trace()
        pose_labels_idx = np.array([POSE_LABELS[pose_label] for pose_label in pose_labels])
        if np.any(person.pose_conf[pose_labels_idx] > 0.3):
            human_detection_value += 2 ** det_idxs

    return human_detection_value


def get_point_body_part(person: Person, point_xy: Tuple):
    body_part = None

    if person.mask[int(point_xy[1]), int(point_xy[0])]:
        with open("configs/pose.yaml", 'r') as f:
            pose_name_to_idx = yaml.safe_load(f)
        pose_idx_to_name = {idx: name for name, idx in pose_name_to_idx.items()}

        closest_distance = 1e8
        closest_idx = 5
        for bone_idx, (point, conf) in enumerate(zip(person.pose, person.pose_conf)):
            if conf < 0.2:
                continue

            distance = np.linalg.norm(point_xy - point)
            if distance < closest_distance:
                closest_distance = distance
                closest_idx = bone_idx

        closest_bone_name = pose_idx_to_name[closest_idx]
        body_part = AIM_POINT_FROM_POSE[closest_bone_name]

        # person_mask_idx_to_label = {idx: label for label, idx in person_mask_label_to_idx.items()}
        # import ipdb; ipdb.set_trace()
        # body_part = person_mask_idx_to_label[person.mask[int(point_xy[1]), int(point_xy[0])]]

    return body_part


def generate_person_data(person: Person, hit_point: Tuple,
                         aim_point: Tuple = (1224, 1024)) -> Dict:

    human_detection_value = get_human_detection(person)

    ######################################

    body_part_to_response_idx = {body_part: response_idx for response_idx, body_part in RESPONSE_IDX_TO_BODY_PART.items()}

    aim_point_body_part = get_point_body_part(person, aim_point)
    if aim_point_body_part is None:
        aim_point_body_part_idx = 0
    else:
        aim_point_body_part_idx = body_part_to_response_idx[aim_point_body_part]

    hit_point_body_part = get_point_body_part(person, hit_point)
    if hit_point_body_part is None:
        hit_point_body_part_idx = 0
    else:
        hit_point_body_part_idx = body_part_to_response_idx[hit_point_body_part]

    ######################################

    # 0 = Stand
    # 1 = Kneel
    # 2 = Prone
    # 3 = Profile_Left
    # 4 = Profile_Right

    body_pose = 0
    if 'crouching' in person.pose_info:
        body_pose = 0
    elif 'kneel' in person.pose_info:
        body_pose = 1
    elif 'prone' in person.pose_info:
        body_pose = 2
    elif 'left' in person.pose_info:
        body_pose = 3
    elif 'right' in person.pose_info:
        body_pose = 4

    ######################################

    res = dict(human_detection=human_detection_value,
               aim_point_body_part=aim_point_body_part_idx,
               hit_point_body_part=hit_point_body_part_idx,
               body_pose=body_pose,
               target_velocity_x=person.velocity['X'],
               target_velocity_y=person.velocity['Y'],
               target_velocity_z=person.velocity['Z'],
               target_range=person.dist)

    return res


def generate_polygon_impact(aim_person: Person, impact_person: Person):
    contours, hierarchy = cv2.findContours(impact_person.mask.astype(np.uint8) * 255,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    impact_polygon_string = ''
    for point in contours[0]:
        point = point.flatten()
        # impact_polygon_string += point[0] + ", " + point[1] + "; "
        impact_polygon_string += f"{point[0]:d},{point[1]:d}; "

    polygon_impact_info = dict(impact_polygon_shift_x=0,
                               impact_polygon_shift_y=0,
                               impact_polygon_size=0,
                               impact_polygon_string=impact_polygon_string
                               )
    if aim_person is None or impact_person is None:
        return polygon_impact_info

    center_shift = np.array([impact_person.x_pos, impact_person.y_pos]) - np.array([aim_person.x_pos, aim_person.y_pos])
    polygon_impact_info["impact_polygon_shift_x"] = center_shift[0]
    polygon_impact_info["impact_polygon_shift_y"] = center_shift[1]
    return polygon_impact_info


def generate_occlusion_data(occlustion_manager_output) -> Dict:
    if not occlustion_manager_output:
        return dict(occlusion=0,
                    occluding_object=0,
                    object_x=0,
                    object_y=0,
                    object_z=0,
                    object_range=0)

    occlusion_name_to_idx = {
        'tree trunk': 1,
        'obstacle': 2,
        'hedge': 3,
        'post': 4
    }

    obstacle, person_idx = occlustion_manager_output[0]
    return dict(occlusion=1,
                occluding_object=occlusion_name_to_idx[obstacle.name],
                object_x=0,
                object_y=0,
                object_z=0,
                object_range=obstacle.dist_abc)


def generate_shooter_info():
    return dict(pitch=0, roll=0, yaw=0, accel_x=0,
                accel_y=0, accel_z=0,
                shooter_lat=0, shooter_lon=0,
                shooter_alt=0)


def generate_balistics_info(person: Person,
                            aim_point_im: np.ndarray):
    # SolveAll(
    #         1, 0.25, _initialVelocity, _sightHeight, _shootingAngle,
    #         _zeroAngle, 0, 90, &sln
    #     );

    camera_focal_length = 22000.00

    aim_point_z = person.dist
    aim_point_x = aim_point_z * aim_point_im[0] / camera_focal_length
    aim_point_y = aim_point_z * aim_point_im[1] / camera_focal_length

    aim_point = np.array((aim_point_x, aim_point_y, aim_point_z))

    yardage = person.dist * 1.09361
    initial_velocity = 3000.0
    sight_height = 0
    zero_angle = 0
    shooting_angle = 0
    projectile_mass = 0.03

    solve_balistics = False
    if solve_balistics:
        solution, n = solve_all(1, 0.25, initial_velocity, sight_height, shooting_angle,
                                zero_angle, 0, 90)

        residual_velocity = get_velocity(solution, yardage)
        residual_energy = 0.5 * projectile_mass * np.power(residual_velocity, 2)
        moa = get_moa(solution, yardage)
        drop = camera_focal_length * np.tanh(moa_to_rad(moa))
    else:
        residual_velocity = 0
        residual_energy = 0
        drop = 2

    hit_point_im = aim_point_im.copy()
    hit_point_im[1] += drop

    hit_point = np.array([
        aim_point_z * hit_point_im[0] / camera_focal_length,
        aim_point_z * hit_point_im[1] / camera_focal_length,
        aim_point_z
    ])

    return dict(aim_point_x=aim_point[0],
                aim_point_y=aim_point[1],
                aim_point_z=aim_point[2],
                hit_point_x=hit_point[0],
                hit_point_y=hit_point[1],
                hit_point_z=hit_point[2],
                residual_energy=residual_energy,
                residual_velocity=residual_velocity), hit_point_im.astype(int)


def generate_vizualization(image, person, shot_point, aim_point, obstacle):
    image_to_draw = image.copy()

    person_contour, _ = cv2.findContours(person.mask.astype(np.uint8) * 255, cv2.RETR_TREE,
                                         cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image_to_draw, person_contour, -1, (0, 255, 0), 3)

    draw_crosshair(image_to_draw, shot_point[0], shot_point[1], 255, 0, 0)
    draw_crosshair(image_to_draw, aim_point[0], aim_point[1], 0, 255, 0)

    if obstacle:
        obstacle_contour, _ = cv2.findContours(obstacle.mask, cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image_to_draw, obstacle_contour, -1, (0, 255, 255), 3)

    return dict(jpeg_bit_stream=image_to_draw,
                image_size=0)


class ProjectCManager:
    def __init__(self, configs_dir: Path, models_dir: Path,
                 device: str = 'cpu'):

        configs_dir = Path(configs_dir)
        models_dir = Path(models_dir)
        logger.info("Creating ProjectCManager")
        self._configs_dir = configs_dir
        self._models_dir = models_dir
        self._device = device

        self.H, self.W = 2048, 2448

        self.config = omegaconf.OmegaConf.load(self._configs_dir / "manager.yaml")        

        self.focal_length_px = self.config.focal_length_mm / self.config.sensor_ratio

        self._people_detector = YoloDetector(configs_dir, models_dir,
                                             device=device)
        self._people_tracker = PeopleTracker(configs_dir)

        self._people_pose_estimator = PeoplePoseEstimator(configs_dir,
                                                          models_dir,
                                                          device=device)

        self._people_pose_classifier = PoseClassifier()

        self._sloth = PeopleRangeEstimator(self.focal_length_px,
                                           self.config.skip_frames_sloth,
                                           (self.H, self.W), "sloth")

        self._occlusion_interaction = ObstacleInteractionManager()
        self._obstacles_detector = DynamicObstacleDetector(configs_dir,models_dir,
                                                           detect_mode='obstacles',
                                                           imsize=(self.H, self.W),
                                                           imgsz_infer=(640, 640),
                                                           device=device)

        self._people_segmenter = PeopleSegmenter(self._configs_dir, self._models_dir,
                                                 device=device)

        self._people_from_the_last_process = list()
        self._latest_image = None
        self._flat_earth = self.flat_earth = FlatEarthVol(25000, 0, 1, 1024)
        img_rows = 2048
        img_cols = 2448
        self.aim_point = np.array((img_cols / 2,
                                   img_rows / 2), dtype=int)

        # self.aim_point = np.array((img_cols / 2 + 45,
        #                            img_rows / 2 + 135), dtype=int)

    def process(self, frame):
        detected_people = self._people_detector.predict_cropped(frame.view_img)

        detected_people = self._people_pose_estimator.process(frame.view_img,
                                                              detected_people)

        results_people = self._people_tracker.track(detected_people, frame)
        for person in results_people:
            person.timestamp = frame.created_at

        self._sloth.set_distance(results_people)

        self._people_from_the_last_process = results_people
        self._latest_image = frame.view_img
        self._flat_earth.update_from_people(self._people_from_the_last_process)

    def process_shot(self):
        closest_person = find_person_closest_to_the_center(self._people_from_the_last_process)
        self._people_segmenter.process(self._latest_image, [closest_person])
        obstacles = self._obstacles_detector.process(self._latest_image)
        for obstacle in obstacles:
            self._flat_earth.update_dist_to_obstacle(obstacle)
        occluded_people = self._occlusion_interaction.process([closest_person],
                                                              obstacles)

        occluded_data = generate_occlusion_data(occluded_people)

        balistics_info, hit_point_im = generate_balistics_info(closest_person, self.aim_point)

        person_data = generate_person_data(closest_person, hit_point_im,
                                           self.aim_point)
        shooter_data = generate_shooter_info()
        polygon_impact = generate_polygon_impact(closest_person,
                                                 closest_person)
        image_vizualization = generate_vizualization(self._latest_image,
                                                     closest_person,
                                                     hit_point_im, self.aim_point,
                                                     None)

        packet_info = dict(packet_id=0,
                           packet_size=0,
                           timestamp=time.time())

        shot_event = ShotEvent(**packet_info,
                               **occluded_data,
                               **person_data,
                               **shooter_data,
                               **polygon_impact,
                               **balistics_info,
                               **image_vizualization)

        return shot_event


if __name__ == "__main__":
    from src.offline_utils.frame_source import FrameSource
    from tqdm import tqdm

    source_path = '/sdb-disk/vyzai/data_from_a_client/test_videos/100m.mp4'
    source = FrameSource(source_path, None, None)
    manager = ProjectCManager("configs/", "models/", device='cuda')

    for i in tqdm(range(5)):
        item = next(source)
        manager.process(item)

    shot_event = manager.process_shot()

    image = shot_event.jpeg_bit_stream
    cv2.imwrite('project_c_viz.png', image)

    encoded_shot_event = shot_event.encode_to_bytes()
    shot_event_restored = ShotEvent.decode_from_bytes(encoded_shot_event)

    for key, value in shot_event.__dict__.items():
        restored_value = getattr(shot_event_restored, key)

        if isinstance(restored_value, (int, float, str)):
            if restored_value != value:
                print(f"{key}: {value}, {restored_value}")
        elif isinstance(restored_value, np.ndarray):
            if not np.allclose(restored_value, value):
                print(key)
        else:
            print(f"{key}: {type(value)}")

    cv2.imwrite('project_c_viz_restored.png', shot_event_restored.jpeg_bit_stream)
    import json
    with open("shot_event.json", 'w') as f:
        json.dump(shot_event_restored.to_dict(), f, indent=2)
