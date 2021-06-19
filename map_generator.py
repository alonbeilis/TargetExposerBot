import math
import numpy as np
import random

from copy import deepcopy
from scipy import ndimage as nd
from typing import Tuple

Point = Tuple[int, int]


class Map:
    def __init__(self,
                 agent_location: Point = None,
                 alpha: float = 0.3,
                 init_prob_value: float = 0.5,
                 p_ta: float = 1.0,
                 probability_to_match: float = 0.95,
                 size: int = 20,
                 sensor_info: int = 10,
                 targets_locations=None):

        self.agent_point = agent_location
        self.alpha = alpha
        self.board_size = size  # assuming board is symmetrical
        self.init_prob_value = init_prob_value
        self.p_fa = self.alpha * p_ta
        self.p_ta = p_ta
        self.probability_map = np.full((size, size), self.init_prob_value)
        self.probability_to_match = probability_to_match
        self.sensor_info = sensor_info

        if agent_location is None:
            self.agent_location = (0, 0)

        if targets_locations is None:
            targets_locations = [(1, 1), (2, 2)]
        self.targets = targets_locations

    @staticmethod
    def _calc_distance_between_points(point_a: Point, point_b: Point) -> float:
        """
        Calculating Euclidean distance between two points.
        :returns a float, indicating the distance between two points.
        """
        return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])

    def _iterate_map(self) -> Tuple[Tuple[int, int], np.float64]:
        """
        Iterating over the map.
        :returns tuple of the point index, value of map[index].
        """
        for idx, value in np.ndenumerate(self.probability_map):
            yield idx, value

    def _is_agent(self, point: Point) -> bool:
        """
        Checking if a point is the agent point.
        :returns a boolean.
        """
        return point == self.agent_point

    def _is_target(self, point: Point) -> bool:
        """
        Checking if a point is a target point.
        :returns a boolean.
        """
        return point in self.targets

    def _get_probability_for_point(self, point: Point) -> float:
        """
        Calculating the probability for getting a signal in a certain point.
        :returns float, indicating the probability.
        """
        return self.p_ta if self._is_target(point) else self.p_fa

    def _calc_weight_factor_ci(self, ci: Point) -> float:
        """
        Calculating the weight factor for a point
        :returns a float, indicating the weight factor.
        """
        distance = Map._calc_distance_between_points(point_a=self.agent_point, point_b=ci)
        return 1 / math.exp(distance / self.sensor_info)

    def calc_x_ci(self, point: Point, weight_factor: float) -> bool:
        """
        Calculating whether a point got a signal or not.
        :returns boolean.
        """
        probability = self._get_probability_for_point(point=point)
        return True if (random.random() < (probability * weight_factor)) else False

    def _calc_p_s_ci(self, idx: Point, ci_value: float, weight_factor_ci: float) -> float:
        """
        Calculating the probability for finding a target, per point.
        :returns a float, indicating the probability.
        """
        x_ci = self.calc_x_ci(point=idx, weight_factor=weight_factor_ci)
        p_s_ci_prev = ci_value

        if x_ci:
            num = p_s_ci_prev * self.p_ta
            denum = num + (1 - p_s_ci_prev) * self.p_fa
        else:
            num = p_s_ci_prev * (1 - self.p_ta + self.p_ta * (1 - weight_factor_ci))
            denum = num + (1 - p_s_ci_prev) * ((1 - self.p_fa) + self.p_fa * (1 - weight_factor_ci))

        return num / denum

    def update_probability_map_single_frame(self):
        """
        Iterates over the map.
        for each point, calculates the probability for finding a target.
        """
        for idx, ci_value in self._iterate_map():
            weight_factor_ci = self._calc_weight_factor_ci(ci=idx)
            self.probability_map[idx] = self._calc_p_s_ci(idx=idx, ci_value=ci_value, weight_factor_ci=weight_factor_ci)

    def calc_center_of_mass(self) -> Point:
        """
        For the probability map, calculates the center of mass of probabilities.
        :returns a tuple, indicating center of mass coordinates.
        """
        center_of_mass = nd.center_of_mass(self.probability_map)
        return tuple(np.round(center_of_mass))

    def move_agent_single_step(self, point: Point):
        """
        Moves the agent one step towards point.
        """
        if self.agent_point == point:
            return

        agent_pos_x, agent_pos_y = self.agent_point
        point_pos_x, point_pos_y = point

        x_diff = point_pos_x - agent_pos_x
        y_diff = point_pos_y - agent_pos_y

        agent_pos_x += np.sign(x_diff)
        agent_pos_y += np.sign(y_diff)

        self.agent_point = int(agent_pos_x), int(agent_pos_y)

    def all_targets_found(self) -> bool:
        """
        Checks if all of the targets found - probability is over the probability to match
        :returns boolean
        """
        for idx, val in self._iterate_map():
            if self._is_target(point=idx):
                if val > self.probability_to_match:
                    self.targets.remove(idx)

        return (self.probability_map[
                    tuple(zip(*self.targets))] > self.probability_to_match).all() if self.targets else True

    def get_time_for_finding_all_targets(self, max_time_to_run: int, move_agent: bool = False):

        for t in range(max_time_to_run):
            self.update_probability_map_single_frame()
            if move_agent:
                self.move_agent_single_step(point=self.calc_center_of_mass())
            if self.all_targets_found():
                return t


def run_algo(iterations: int = 50, max_time_to_run: int = 10000, move_agent: bool = False, **kwargs) -> int:
    time_to_find_targets = 0

    for iteration in range(iterations):
        main_map = Map(**deepcopy(kwargs))
        time_to_find_targets += main_map.get_time_for_finding_all_targets(max_time_to_run=max_time_to_run,
                                                                          move_agent=move_agent)
    return np.round(time_to_find_targets / iterations)


def main(iterations: int = 100, max_time_to_run: int = 1000, **kw):
    avg_time_with_agent_movement = run_algo(move_agent=True, **kw)
    avg_time_without_agent_movement = run_algo(move_agent=False, **kw)

    print(f'{iterations} iterations', )
    print(f'avg time for finding targets with movement of the agent: {avg_time_with_agent_movement}')
    print(f'avg time for finding targets without movement of the agent: {avg_time_without_agent_movement}')


if __name__ == '__main__':
    algo_args = dict(agent_location=(0, 0),
                     alpha=0.5,
                     init_prob_value=0.5,
                     p_ta=1.0,
                     probability_to_match=0.95,
                     size=20,
                     sensor_info=10,
                     targets_locations=[(0, 9), (4, 14), (15, 8)])

    main(iterations=50, max_time_to_run=1000, **algo_args)
