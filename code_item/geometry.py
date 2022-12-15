# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.

import numpy as np


def dist_between_points(a, b):
    """
    Return the Euclidean distance between two points
    :param a: first point
    :param b: second point
    :return: Euclidean distance between a and b
    """
    distance = np.linalg.norm(np.array(b) - np.array(a))
    return distance

def steer(start, goal, d):
    """
    Return a point in the direction of the goal, that is distance away from start
    :param start: start location
    :param goal: goal location
    :param d: distance away from start
    :return: point in the direction of the goal, distance away from start
    """
    start, end = np.array(start), np.array(goal)
    v = end - start
    if np.sum(v**2) == 0:
        steered_point = start
    else:
        u = v / (np.sqrt(np.sum(v ** 2)))
        steered_point = start + u * d
    return tuple(steered_point)
