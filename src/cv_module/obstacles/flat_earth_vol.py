# By Oleksiy Grechnyev, 5/6/24
import numpy as np

from sklearn.linear_model import LinearRegression


########################################################################################################################
def print_it(a, name: str = ''):
    # m = a.float().mean() if isinstance(a, torch.Tensor) else a.mean()
    m = a.mean()
    print(name, a.shape, a.dtype, a.min(), m, a.max())


########################################################################################################################
class FlatEarthVol:
    """
    Volodymyr's Flat Earth refactored from ProcessingManager
    It's actually very bad
    TODO: Write a proper FLatEarth
    """

    def __init__(self, A, a, b, c):
        self.A = A
        self.a = a
        self.b = b
        self.c = c

        self.xs = []
        self.ys = []
        self.dists = []

    def update_from_people(self, results_people):
        """Update flat earth based on people detection"""
        for person in results_people:
            # kpts = results_people.keypoints.data[results_people.boxes.id == k][0]
            # bbox = results_people.boxes.data[results_people.boxes.id == k][0, :4]
            # mask = results_people.masks.data[results_people.boxes.id == k][0]
            # if person.mask[person.bbox[1]:person.bbox[3], person.bbox[0]:person.bbox[2]].sum() == 0:
            #     continue

            if not person.has_pose:
                continue

            visible = (person.pose[:, 0] > 0) * (person.pose[:, 1] > 0)
            if visible[15] and visible[16]:
                dist = person.meas["dist"]
                if not np.isnan(dist):
                    self.xs.append(float((person.pose[15, 0] + person.pose[16, 0]) / 2))
                    self.ys.append(float(max(person.pose[15, 1], person.pose[16, 1])))
                    self.dists.append(float(dist))

        if len(self.dists) > 10:
            model = LinearRegression()
            xs, ys, dists = map(np.array, [self.xs, self.ys, self.dists])
            points = np.stack([xs, ys], axis=-1)
            model.fit(points, 1 / dists)
            a, b, c = list(model.coef_) + [model.intercept_]
            A = 1 / (a ** 2 + b ** 2) ** 0.5
            a, b, c = [item * A for item in [a, b, c]]
            self.A, self.a, self.b, self.c = A, a, b, c

    def update_dist_to_obstacle(self, obstacle):
        x, y = (obstacle.bbox[0] + obstacle.bbox[2]) / 2, obstacle.bbox[3]
        dist_to_horizon = self.a * x + self.b * y + self.c
        if dist_to_horizon < 0:
            dist = np.nan
        else:
            dist = self.A / dist_to_horizon

        obstacle.dist_abc = dist

########################################################################################################################
