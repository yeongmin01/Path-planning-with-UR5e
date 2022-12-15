from geometry import dist_between_points
from skimage.measure import profile_line
import numpy as np


def Obstacle_cost(x_near, x_new, map):

    line = profile_line(map, x_near, x_new, linewidth=1, mode='constant')
    num = len(line)

    cost = 1 - line

    obs_cost = np.sum(cost)/num

    return obs_cost

def cost_to_go(map, a: tuple, b: tuple) -> float:
    """
    :param a: current location
    :param b: next location
    :return: estimated segment_cost-to-go from a to b
    """

    return  (0.8*(dist_between_points(a, b)/4)) + (0.2*Obstacle_cost(a, b, map))


def path_cost(E, a, b, map):
    """
    Cost of the unique path from x_init to x
    :param E: edges, in form of E[child] = parent
    :param a: initial location
    :param b: goal location
    :return: segment_cost of unique path from x_init to x
    """

    cost = 0
    while not b == a:
        p = E[b]
        cost +=  (0.8*(dist_between_points(b, p)/4)) + (0.2*Obstacle_cost(b, p, map))
        b = p


    return cost



def segment_cost(a, b, map):
    """
    Cost function of the line between x_near and x_new
    :param a: start of line
    :param b: end of line
    :return: segment_cost function between a and b
    """

    return   (0.8*dist_between_points(a, b)/4) + (0.2*Obstacle_cost(a, b, map))


