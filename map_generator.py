import numpy as np
from scipy import ndimage as nd
import math
import random

from typing import Tuple

Point = Tuple[int, int]


class Map:
    def __init__(self, agent_location: Point = None, alpha: float = 0.3, init_prob_value: float = 0.5,
                 p_ta: float = 1.0, size: int = 20, sensor_info: int = 10, targets_locations=None):
        self.agent_point = agent_location
        self.board_size = size  # assuming board is symmetrical
        self.p_fa = alpha * p_ta
        self.p_ta = p_ta
        self.probability_map = np.full((size, size), init_prob_value)
        self.sensor_info = sensor_info

        if agent_location is None:
            self.agent_location = (0, 0)

        if targets_locations is None:
            targets_locations = [(1, 1), (2, 2)]
        self.targets = targets_locations

    @staticmethod
    def _calc_distance_between_points(point_a: Point, point_b: Point) -> float:
        """
        Calculating Euclidean distance between two points
        """
        return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])

    def _iterate_map(self) -> Tuple[Tuple[int, int], np.float64]:
        for idx, value in np.ndenumerate(self.probability_map):
            yield idx, value

    def _is_agent(self, point: Point) -> bool:
        return point == self.agent_point

    def _is_target(self, point: Point) -> bool:
        return point in self.targets

    def _get_probability_for_point(self, point: Point) -> float:
        return self.p_ta if self._is_target(point) else self.p_fa

    def _calc_weight_factor_ci(self, ci: Point) -> float:
        distance = Map._calc_distance_between_points(point_a=self.agent_point, point_b=ci)
        return 1 / math.exp(distance / self.sensor_info)

    def calc_x_ci(self, point: Point, weight_factor: float) -> bool:
        probability = self._get_probability_for_point(point=point)
        return True if (random.random() < (probability * weight_factor)) else False

    def _calc_p_s_ci(self, idx: Point, ci_value: float, weight_factor_ci: float):
        x_ci = self.calc_x_ci(point=idx, weight_factor=weight_factor_ci)
        p_s_ci_prev = ci_value

        if x_ci:
            num = p_s_ci_prev * self.p_ta
            denum = num + (1 - p_s_ci_prev) * self.p_fa
        else:
            num = p_s_ci_prev * (1 - self.p_ta + self.p_ta * (1 - weight_factor_ci))
            denum = num + (1 - p_s_ci_prev) * ((1 - self.p_fa) + self.p_fa * (1 - weight_factor_ci))

        return num / denum

    def get_s_single_frame(self):
        for idx, ci_value in self._iterate_map():
            weight_factor_ci = self._calc_weight_factor_ci(ci=idx)

            self.probability_map[idx] = self._calc_p_s_ci(idx=idx, ci_value=ci_value, weight_factor_ci=weight_factor_ci)

        return self.probability_map

    def calc_center_of_mass(self) -> Point:
        center_of_mass = nd.center_of_mass(self.probability_map)
        return tuple(np.round(center_of_mass))

    def move_agent_single_step(self, probability_center_of_mass: Point):
        if self.agent_point == probability_center_of_mass:
            return

        agent_pos_x, agent_pos_y = self.agent_point
        com_pos_x, com_pos_y = probability_center_of_mass

        x_diff = com_pos_x - agent_pos_x
        y_diff = com_pos_y - agent_pos_y

        agent_pos_x += np.sign(x_diff)
        agent_pos_y += np.sign(y_diff)

        self.agent_point = int(agent_pos_x), int(agent_pos_y)

    def targets_found(self):
        return (self.probability_map[tuple(zip(*self.targets))] > 0.95).all()

    def run_algo(self, max_time_to_run: int):
        for t in range(max_time_to_run):
            self.probability_map = self.get_s_single_frame()
            probability_center_of_mass = self.calc_center_of_mass()
            self.move_agent_single_step(probability_center_of_mass)
            if self.targets_found():
                return t


if __name__ == '__main__':
    main_map = Map(agent_location=(0, 0),
                   alpha=0.5,
                   init_prob_value=0.5,
                   p_ta=1.0,
                   size=20,
                   sensor_info=10,
                   targets_locations=[(1, 0), (1,6), (8,4)])

    time_to_find_targets = main_map.run_algo(max_time_to_run=10000)
    print(f'It took {time_to_find_targets} iterations to find the targets')

