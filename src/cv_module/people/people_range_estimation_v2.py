import numpy as np
from typing import List, Tuple
from collections import defaultdict
from loguru import logger

from src.cv_module.distance_measurers import PeopleRangeEstimatorSloth, HeightDistanceMeasurer


class PeopleRangeEstimator:
    def __init__(self, focal_length: float, im_size: Tuple[int],
                 skip_frames: int = 4, active_method="bbox"):
        self.methods = dict(sloth=PeopleRangeEstimatorSloth(focal_length,
                                                            skip_frames,
                                                            use_tracker=True,
                                                            im_size=im_size),
                            bbox=HeightDistanceMeasurer(focal_length=focal_length,
                                                        im_size=im_size,
                                                        base_height_in_meters=1.75))

        self.active_method_name = active_method

    @property
    def active_method(self):
        return self.methods[self.active_method_name]

    def auto_distance_measuring(self, people, distance_scale_factor: float):
        logger.info("Starting auto distance measuring")
        for person in people:
            person_measurements = defaultdict(list)
            for method_name, method in self.methods.items():
                if method.check_fit(person):
                    measurements = method.process_one(person)
                    for meas_name, meas_value in measurements.items():
                        person_measurements[meas_name].append(meas_value)

                    logger.info(f"Person #{person.id}, {method_name}: {measurements['dist']}")

            if len(person_measurements) != 0:
                final_measurements = {meas_name: np.mean(meas_values) for meas_name, meas_values in person_measurements.items()}
                final_measurements["dist"] *= distance_scale_factor
                person.set_measurement(final_measurements)

        for method in self.methods.values():
            method.purge_old_mtracks()

    def set_distance(self, people: List["Person"], distance_scale_factor: float = 1):
        people_dict = {person.id: person for person in people}
        if self.active_method_name == "auto":
            self.auto_distance_measuring(people, distance_scale_factor)
        else:
            method = self.active_method
            measurements, velocities = method.process(people)

            for person_id, meas in measurements.items():
                meas.Z *= distance_scale_factor
                people_dict[person_id].set_measurement(meas)

        return people
