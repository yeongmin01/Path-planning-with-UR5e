import numpy as np
from skimage.measure import profile_line
import matplotlib.pyplot as plt

class node(object):
    def __init__(self, x, y, cost = 0, parent = None ):

        self.x = x
        self.y = y
        self.arr = np.array([self.x, self.y])
        self.cost = cost
        self.parent = parent


class rrt_star():
    def __init__(self, x_init, x_goal, map, eta, w1, w2):

        self.map = map
        self.eta = eta
        self.x_init = x_init
        self.x_goal = x_goal
        self.nodes = [self.x_init]
        self.w1 = w1
        self.w2 = w2

    def Uniform_sampling(self):

        x = np.random.uniform(low=0, high=self.map.shape[0]-1)
        y = np.random.uniform(low=0, high=self.map.shape[1]-1)

        x_rand = node(x, y)

        return x_rand

    def Sampling(self):
        row = self.map.shape[1]

        p = np.ravel(self.map) / np.sum(self.map)

        x_sample = np.random.choice(len(p), p=p)

        x = x_sample // row
        y = x_sample % row

        x = np.random.uniform(low=x - 0.5, high=x + 0.5)
        y = np.random.uniform(low=y - 0.5, high=y + 0.5)

        x_rand = node(x, y)

        return x_rand

    def Distance_cost(self, start, end):

        distance_cost = np.linalg.norm(start.arr - end.arr)

        return distance_cost

    def Obstacle_cost(self, start, end):

        seg_length = 1
        seg_point = int(np.ceil(np.linalg.norm(start.arr - end.arr) / seg_length))

        value = 0
        if seg_point > 1:
            v = (end.arr - start.arr) / (seg_point)

            for i in range(seg_point + 1):
                seg = start.arr + i * v
                seg = np.around(seg)
                if 1 - self.map[int(seg[0]), int(seg[1])] == 1 :
                    cost = 1e10

                    return cost
                else:
                    value += 1 - self.map[int(seg[0]), int(seg[1])]

            cost = value / (seg_point + 1)

            return cost

        else:

            value = self.map[int(start.arr[0]), int(start.arr[1])] + self.map[int(end.arr[0]), int(end.arr[1])]
            cost = value / 2

            return cost
        # line = profile_line(self.map, start.arr, end.arr, linewidth=2, mode='constant')
        # num = len(line)
        #
        # cost = 1 - line
        #
        # obs_cost = np.sum(cost) / num
        #
        # return obs_cost

    def Line_cost(self, start, end):

        cost = self.w1*(self.Distance_cost(start, end)/(self.eta)) + self.w2*(self.Obstacle_cost(start, end))

        return cost

    def Nearest(self, x_rand):

        vertex = []
        v = []
        i = 0
        for x_near in self.nodes:

            dist = self.Distance_cost(x_near, x_rand)
            vertex.append([dist, i, x_near])
            i+=1

        vertex.sort()
        x_nearest = vertex[0][2]

        return x_nearest

    def Steer(self, x_rand, x_nearest):

        d = self.Distance_cost(x_rand, x_nearest)

        if d < self.eta :
            x_new = node(x_rand.x, x_rand.y)
        else:
            new_x = x_nearest.x + self.eta*((x_rand.x - x_nearest.x)/d)
            new_y = x_nearest.y + self.eta*((x_rand.y - x_nearest.y)/d)

            x_new  = node(new_x, new_y)

        return x_new


    def Exist_Check(self, x_new):

        for x_near in self.nodes:
            if x_new.x == x_near.x and x_new.y == x_near.y:
                return False
            else :
                return True

    def New_Check(self, x_new):
        # print(x_new.arr)
        x_pob = np.array([x_new.x, x_new.y])
        x_pob = np.around(x_pob)

        if x_pob[0] >= self.map.shape[0]:
            x_pob[0] = self.map.shape[0] - 1
        if x_pob[1] >= self.map.shape[1]:
            x_pob[1] = self.map.shape[1] - 1

        x_pob = self.map[int(x_pob[0]), int(x_pob[1])]
        p = np.random.uniform(0, 1)
        # print(x_pob,p)
        if x_pob > p and self.Exist_Check(x_new):
            return True
        else:
            return False


    # def Add_Parent(self, x_new, x_nearest):

    #     x_new.cost = x_nearest.cost + self.Line_cost(x_nearest, x_new)
    #     # x_new.parent = x_nearest
    #     for x_near in self.nodes:
    #         if self.Distance_cost(x_near, x_new) <= self.eta and self.Obstacle_cost(x_near, x_new) < 1 :
    #
    #             if x_near.cost + self.Line_cost(x_near, x_new) < x_new.cost:
    #
    #                 x_nearest = x_near
    #                 x_new.cost = x_nearest.cost + self.Line_cost(x_nearest, x_new)
    #         x_new.parent = x_nearest
    #
    #
    #     return x_new

    def Add_Parent(self, x_new, x_nearest):

        x_min = x_nearest
        c_min = x_min.cost + self.Line_cost(x_min, x_new)
        # x_new.parent = x_min
        for x_near in self.nodes:
            if self.Distance_cost(x_near, x_new) <= self.eta :

                if x_near.cost + self.Line_cost(x_near, x_new) < c_min:
                    x_min = x_near
                    c_min = x_near.cost + self.Line_cost(x_near, x_new)
            x_new.parent = x_min
            x_new.cost = c_min

        return x_new

    def Rewire(self, x_new):

        for x_near in self.nodes:
            if x_near is not x_new.parent:
                if self.Distance_cost(x_near, x_new) <= self.eta  : #and self.Obstacle_cost(x_new, x_near) < 1
                    if x_new.cost + self.Line_cost(x_new, x_near) < x_near.cost:
                        x_near.parent = x_new
                        x_near.cost = x_new.cost + self.Line_cost(x_new, x_near)

        return

    def Get_Path(self):

        temp_path = []
        path = []
        n = 0
        for i in self.nodes:
            if self.Distance_cost(i, self.x_goal) < self.eta: #5
                cost = i.cost + self.Line_cost(self.x_goal, i)
                temp_path.append([cost,n, i])
                n += 1
        temp_path.sort()
        # closest_node = temp_path[0][2]
        # i = closest_node
        #
        # self.x_goal.cost = temp_path[0][0]
        #
        # while i is not self.x_init:
        #     path.append(i)
        #     i = i.parent
        # path.append(self.x_init)
        #
        # self.x_goal.parent = path[0]
        # path.insert(0, self.x_goal)
        #
        # return path

        if temp_path == []:

            print("cannot find path")

            return None

        else:
            closest_node = temp_path[0][2]
            i = closest_node

            self.x_goal.cost = temp_path[0][0]

            while i is not self.x_init:
                path.append(i)
                i = i.parent
            path.append(self.x_init)

            self.x_goal.parent = path[0]
            path.insert(0, self.x_goal)

            return path

    def Cost_Graph(self):

        temp_path = []
        path = []
        n = 0
        for i in self.nodes:
            if self.Distance_cost(i, self.x_goal) < 3:
                cost = i.cost + self.Line_cost(self.x_goal, i)
                temp_path.append([cost, n, i])
                n += 1
        temp_path.sort()

        if temp_path == []:
            return 0
        else:
            closest_node = temp_path[0][2]
            i = closest_node

            self.x_goal.cost = temp_path[0][0]

            return self.x_goal.cost

    def Draw_Tree(self):

        for i in self.nodes:
            if i is not self.x_init:
                plt.plot([i.x, i.parent.x], [i.y, i.parent.y], "b")

    def Draw_path(self, path):

        for i in path:
            if i is not self.x_init:

                plt.plot([i.x, i.parent.x], [i.y, i.parent.y], "r", linewidth=2.5)

    def Check_can_connect_to_goal(self, path_iter):

        x_nearest = self.Nearest(self.x_goal)
        if self.Distance_cost(x_nearest, self.x_goal) < 5:

            path_iter += 1

            return path_iter

        else:
            return path_iter


