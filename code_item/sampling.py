import numpy as np
from geometry import steer

def sampling(map) :

    row = map.shape[0]

    p = np.ravel(map) / np.sum(map)

    x_sample = np.random.choice(len(p) , p=p)

    x = x_sample // row
    y = x_sample % row

    x_rand =  np.array([x,y])

    return x_rand


def new_and_near(self, tree, q, map):
    """
    Return a new steered vertex and the vertex in tree that is nearest
    :param tree: int, tree being searched
    :param q: length of edge when steering
    :return: vertex, new steered vertex, vertex, nearest vertex in tree to new vertex
    """
    # x_rand = self.X.sample_free()
    # x_nearest = self.get_nearest(tree, x_rand)
    # x_new = self.bound_point(steer(x_nearest, x_rand, q[0]))
    while True:
        x_rand = self.X.sample_free()
        x_nearest = self.get_nearest(tree, x_rand)
        x_new = self.bound_point(steer(x_nearest, x_rand, q[0]))

        x_pob = np.around(x_new)

        if x_pob[0] >= 100:
            x_pob[0] = 99
        if x_pob[1] >= 100:
            x_pob[1] = 99
        x_pob = map[int(x_pob[0]), int(x_pob[1])]
        p = np.random.uniform(0, 1)
        # print(x_pob,p)
        if x_pob > p:
            break

    # print(dist_between_points(x_rand,x_nearest),dist_between_points(x_new,x_nearest))
    # check if new point is in X_free and not already in V
    if not self.trees[0].V.count(x_new) == 0:  # or not self.X.obstacle_free(x_new):
        return None, None
    self.samples_taken += 1
    return x_new, x_nearest
