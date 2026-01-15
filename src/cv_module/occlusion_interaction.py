import numpy as np
from .obstacles.obstacle import Obstacle
from typing import List
from scipy.ndimage import binary_dilation


class ObstacleInteractionManager:
    """
    Implement the logic: check if a human is occluded by an obstacle
    A custom and somewhat obscure logic based on human keypoints

    Warning: the code contains the following logic hardcoded in MANY places:
    keypoint is visible if its coordinates are >0
    kepypoint score is ignored
    This is ultralytics-only and pretty obscure!
    TODO: clean this up
    """

    def __init__(self):
        pass

    @staticmethod
    def mask2bbox(mask):
        a = np.where(mask > 0)
        bbox = np.min(a[1]), np.min(a[0]), np.max(a[1]), np.max(a[0])
        return bbox

    @staticmethod
    def distance_to_closest_nonzero(mask, point):
        mask = np.array(mask) > 0
        non_zero_pixels = np.array(np.where(mask.T != 0)).T
        distances = np.linalg.norm(non_zero_pixels - point, axis=1)
        min_distance = np.min(distances)

        return min_distance

    @staticmethod
    def reflect_wrt_line(p, ln):
        x1, y1 = p
        a, b, c = ln
        x2 = (x1 * (b**2 - a**2) - 2*a*b*y1 - 2*a*c) / (a**2 + b**2)
        y2 = (y1 * (a**2 - b**2) - 2*a*b*x1 - 2*b*c) / (a**2 + b**2)
        return x2, y2

    def get_symm_axis(self, kpts):
        xs = kpts[:,0]
        ys = kpts[:,1]
        visible = (xs > 0) * (ys > 0)

        shoulders = visible[5] and visible[6]
        torso = visible[11] and visible[12]
        if shoulders and torso:
            x1, y1 = (xs[5]+xs[6])/2, (ys[5]+ys[6])/2
            x2, y2 = (xs[11]+xs[12])/2, (ys[11]+ys[12])/2
            a, b, c = y2 - y1, x1 - x2, x2*y1 - x1*y2
        elif shoulders:
            x1, y1 = xs[5], ys[5]
            x2, y2 = xs[6], ys[6]
            a, b, c = (x2 - x1), (y2 - y1), (x1**2 + y1**2 - x2**2 - y2**2) / 2
        elif torso:
            x1, y1 = xs[11], ys[11]
            x2, y2 = xs[12], ys[12]
            a, b, c = (x2 - x1), (y2 - y1), (x1**2 + y1**2 - x2**2 - y2**2) / 2
        else:
            a, b, c = None, None, None
        
        if a is not None and abs(a/b) < 5:
            a, b, c = None, None, None

        return a, b, c

    def anticipate_symmetrical_keypoints(self, kpts):
        """Halluciante reflections around the symmetry axis"""
        # shoulders, hips, elbows, knees
        symmetrical_keypoint_pairs = [(5, 6), (11, 12), (7, 8), (13, 14)]

        visible = (kpts[:,0] > 0) * (kpts[:,1] > 0)  # TODO BAD
        symm_axis = self.get_symm_axis(kpts)

        kpts_lst = []
        if symm_axis[0] is None:
            return kpts_lst

        for kpt1, kpt2 in symmetrical_keypoint_pairs:
            if visible[kpt1] and not visible[kpt2]:
                kpts_lst.append(self.reflect_wrt_line(kpts[kpt1], symm_axis))
            elif visible[kpt2] and not visible[kpt1]:
                kpts_lst.append(self.reflect_wrt_line(kpts[kpt2], symm_axis))

        return kpts_lst

    def get_keypoints_outside_mask(self, bbox, mask, kpts):
        """Find pose keypoints outside human mask (and add afew new ones))"""
        x1, y1, x2, y2 = map(int, bbox)
        diag = ((x2 - x1)**2 + (y2 - y1)**2)**0.5

        kpts_ignore = [9, 10]   # Ignore hands
        kpt2kpt_interior_pairs = [(5, 6), (11, 12)]  # shoulders, hips
        kpt2kpt_interior_coeffs = [0.25, 0.5, 0.75]

        points_to_check = []
        points_outside = []

        visible = (kpts[:,0] > 0) * (kpts[:,1] > 0)  # TODO Hardcoded, BAD !

        for idx in range(len(kpts)):
            if visible[idx] and (idx not in kpts_ignore):
                points_to_check.append(kpts[idx])

        # Add 3 middle point between shoulder, 3 between hips
        for kpt1, kpt2 in kpt2kpt_interior_pairs:
            if visible[kpt1] and visible[kpt2]:
                points_to_check.extend([k*kpts[kpt1]+(1-k)*kpts[kpt2] for k in kpt2kpt_interior_coeffs])

        for point_to_check in points_to_check:
            is_outside = mask[int(point_to_check[1]), int(point_to_check[0])] == 0
            if is_outside:
                points_outside.append(point_to_check)

        # We want points outside the dilated mask, i.e. not too near to the original mask edge
        if len(points_outside) > 0:
            mask_crop = mask[y1:y2+2,x1:x2+2]
            dilation = min(int(diag*0.01), 5)
            if dilation > 0:
                mask_crop = binary_dilation(mask_crop, iterations=dilation)
                # mask = binary_dilation(mask, iterations=dilation)
            points_outside_new = []
            for point_to_check in points_outside:
                # is_outside = mask[int(point_to_check[1]), int(point_to_check[0])] == 0
                is_outside_bbox = not (x1 <= point_to_check[0] <= x2 and y1 <= point_to_check[1] <= y2)
                if is_outside_bbox:
                    points_outside_new.append(point_to_check)
                    continue
                is_outside = mask_crop[int(point_to_check[1]-y1), int(point_to_check[0]-x1)] == 0
                if is_outside:
                    points_outside_new.append(point_to_check)
                    continue
            points_outside = points_outside_new
        
        return points_outside

    def get_keypoints_horizontal_occlusion(self, bbox, kpts):
        keypoints_under_bbox = []
        keypoint_pairs_to_check = {
            (11, 12): [0.3, 0.2, 0.1],    # hips
            (13, 14): [0.3, 0.2, 0.1],    # knees
            (15, 16): [0.1],              # feet
        }
        x1, y1, x2, y2 = map(int, bbox)
        visible = (kpts[:,0] > 0) * (kpts[:,1] > 0)  # TODO BAD

        # Return [] if no head AND no hips
        if not (visible[:5].any() or visible[[11,12]].all()):   # head, hips
            return keypoints_under_bbox

        # Hallucinate some points along the vertical middle instead of occluded legs
        for (kpt1, kpt2), under_bbox_ratios in keypoint_pairs_to_check.items():
            if (not visible[kpt1]) and (not visible[kpt2]):
                for ratio in under_bbox_ratios:
                    keypoints_under_bbox.append([(x1+x2)/2, y2 + (y2 - y1) * ratio])

        return keypoints_under_bbox

    def get_keypoints_vertical_occlusion(self, bbox, kpts):
        """Hallucinate points along the horizontal middle line if an arm is fully occluded"""
        keypoints_near_bbox = []
        near_bbox_ratio = 0.1

        x1, y1, x2, y2 = map(int, bbox)
        visible = (kpts[:,0] > 0) * (kpts[:,1] > 0)  # TODO BAD

        left_arm = visible[7] or visible[9]    # left elbow, wrist
        rigth_arm = visible[8] or visible[10]  # right elbow, wrist
        if (visible[11] or visible[12]) and (not left_arm or not rigth_arm):   # [11,12] == hips
            p_min = [x1-(x2-x1)*near_bbox_ratio, (y1+y2)/2]
            p_max = [x2+(x2-x1)*near_bbox_ratio, (y1+y2)/2]
            if not left_arm and not rigth_arm:
                keypoints_near_bbox.append(p_min)
                keypoints_near_bbox.append(p_max)
            else:
                if not left_arm:
                    if visible[8]:
                        visible_arm_x = kpts[8][0]
                    elif visible[10]:
                        visible_arm_x = kpts[10][0]
                if not rigth_arm:
                    if visible[7]:
                        visible_arm_x = kpts[7][0]
                    elif visible[9]:
                        visible_arm_x = kpts[9][0]
                if visible_arm_x > (x1+x2)/2:
                    keypoints_near_bbox.append(p_min)
                else:
                    keypoints_near_bbox.append(p_max)

        return keypoints_near_bbox

    def get_occlusion_candidate_points(self, bbox, mask, kpts):
        """
        Find pose keypoints (real or hallucinated), which are outside the person's mask
        Thus they are possibly occluded by an obstacle
        """
        candidate_points = []

        candidate_points.extend(self.get_keypoints_outside_mask(bbox, mask, kpts))
        candidate_points.extend(self.anticipate_symmetrical_keypoints(kpts))
        candidate_points.extend(self.get_keypoints_horizontal_occlusion(bbox, kpts))
        candidate_points.extend(self.get_keypoints_vertical_occlusion(bbox, kpts))

        # Filter points that are indeed outside the mask
        candidate_points_new = []
        for point_to_check in candidate_points:
            if not (0 <= int(point_to_check[1]) < mask.shape[0] and 0 <= int(point_to_check[0]) < mask.shape[1]):
                continue
            is_outside = mask[int(point_to_check[1]), int(point_to_check[0])] == 0
            if is_outside:
                candidate_points_new.append(point_to_check)
        candidate_points = candidate_points_new

        return candidate_points

    def check_candidates_for_occlusion(self, candidate_points, person_mask, person_bbox, person_area, person_kpts, obstacles: List[Obstacle]):
        """
        Find obstacles occluding candidate_points, with additional checks by area and bboxes
        """

        # person_area = person_mask.sum()
        # if person_area == 0:
        #     return []
        # person_bbox = self.mask2bbox(person_mask)

        phrase2overlay_threshold = {
            "hedge": 0.25,
            "post": 0.5,
            "car_d": 0.25,
            "tree trunk": 0.5,
            "obstacle": 0.25,
            "other": 0.05,
        }
        phrase2vertical_delta_threshold = {
            "car_d": -0.2,
            "hedge": -0.5,
        }

        # use lower threshold when person feets are fully visible
        visible = (person_kpts[:,0] > 0) * (person_kpts[:,1] > 0)
        if visible[15] or visible[16]:
            dy_ratio = 0.04
        else:
            dy_ratio = 0.1

        occlusion_obstacle_idxs = set()
        for point_to_check in candidate_points:

            for idx, obstacle in enumerate(obstacles):
                if idx in occlusion_obstacle_idxs:
                    continue

                x, y = map(int, point_to_check)
                if not obstacle.is_inside(x, y):
                    continue

                # filter out obstacles with range > range to person (compare ground-points)
                person_top, person_bot, obstacle_bot = person_bbox[1], person_bbox[3], obstacle.bbox[3]
                if obstacle_bot < person_bot - phrase2vertical_delta_threshold.get(obstacle.name, dy_ratio) * (person_bot-person_top):
                    continue

                # if person is in front of obstacle -> it should shadow the mask of obstacle
                # but since person mask is not always perfect (especially for thin obstacles like post) use different thresholds
                # some obstacles may have semi-transparent texture so it won't work for them (e.g. hedge)
                overlay_ratio = obstacle.intersection_area(person_mask, person_bbox) / min(person_area, obstacle.area)
                if overlay_ratio > phrase2overlay_threshold[obstacle.name]:
                    continue

                occlusion_obstacle_idxs.add(idx)

        return sorted(occlusion_obstacle_idxs)

    def process(self, people, obstacles):
        """Main method: detect occlusions from results (human detections) and obstacles"""
        occlusions = set()
        # if results.masks is None:
        #     return []

        # Loop over all humans
        # for idx in range(len(results.boxes)):
        for idx, person in enumerate(people):
            if not person.has_pose:
                continue
            bbox = person.bbox.astype(int)
            mask = person.mask
            kpts = person.pose

            area = mask[bbox[1]:bbox[3],bbox[0]:bbox[2]].sum()
            if area == 0:
                continue
            import time
            t1 = time.time()

            # Get pose keypoints (real+hallucinated) that might be occluded, no obstacles yet
            candidate_points = self.get_occlusion_candidate_points(bbox, mask, kpts)
            # print("candidates", time.time()-t1)
            t1 = time.time()

            # Check candidate_points if they are occluded by an obstacle
            obstalce_idxs = self.check_candidates_for_occlusion(candidate_points, mask, bbox, area, kpts, obstacles)
            # print("check", time.time()-t1)
            occlusions.update([(item, idx) for item in obstalce_idxs])
        return [(obstacles[o_idx], p_idx) for o_idx, p_idx in sorted(occlusions)]
