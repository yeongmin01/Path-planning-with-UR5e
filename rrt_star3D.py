import numpy as np
import plotly.graph_objs as go

class node(object): # definition node position, cost and parent
    def __init__(self, x, y, z, cost = 0, parent = None ):

        self.x = x
        self.y = y
        self.z = z
        self.arr = np.array([self.x, self.y, self.z])
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

    def Sampling(self):
        height = self.map.shape[0]
        width = self.map.shape[1]
        depth = self.map.shape[2]

        p = np.ravel(self.map) / np.sum(self.map)

        x_sample = np.random.choice(len(p), p=p)

        z = x_sample // (width * height)
        x = (x_sample - z*(width * height)) % width
        y = (x_sample - z*(width * height)) // width

        x = np.random.uniform(low=x - 0.5, high=x + 0.5)
        y = np.random.uniform(low=y - 0.5, high=y + 0.5)
        z = np.random.uniform(low=z - 0.5, high=z + 0.5)

        x_rand = node(x, y, z)

        return x_rand

    def Distance_Cost(self, start, end):

        distance_cost = np.linalg.norm(start.arr - end.arr)

        return distance_cost

    def Obstacle_Cost(self, start, end):

        seg_length = 1
        seg_point = int(np.ceil(np.linalg.norm(start.arr - end.arr) / seg_length))

        value = 0
        if seg_point > 1:
            v = (end.arr - start.arr) / (seg_point)

            for i in range(seg_point+1):
                seg = start.arr + i * v
                seg = np.around(seg)
                if 1 - self.map[int(seg[0]), int(seg[1]), int(seg[2])] == 1:
                    cost = 1e10

                    return cost
                else:
                    value += 1 - self.map[int(seg[0]), int(seg[1]), int(seg[2])]

            cost = value / (seg_point+1)

            return cost

        else:

            value = self.map[int(start.arr[0]), int(start.arr[1]), int(start.arr[2])] + self.map[int(end.arr[0]), int(end.arr[1]), int(end.arr[2])]
            cost = value / 2

            return cost

    def Line_Cost(self, start, end):

        cost = self.w1*(self.Distance_Cost(start, end)/(self.eta)) + self.w2*(self.Obstacle_Cost(start, end))

        return cost

    def Nearest(self, x_rand):

        vertex = []
        v = []
        i = 0
        for x_near in self.nodes:

            dist = self.Distance_Cost(x_near, x_rand)
            vertex.append([dist, i, x_near])
            i+=1

        vertex.sort()
        x_nearest = vertex[0][2]

        return x_nearest

    def Steer(self, x_rand, x_nearest):

        d = self.Distance_Cost(x_rand, x_nearest)

        if d < self.eta :
            x_new = node(x_rand.x, x_rand.y, x_rand.z)
        else:
            new_x = x_nearest.x + self.eta * ((x_rand.x - x_nearest.x)/d)
            new_y = x_nearest.y + self.eta * ((x_rand.y - x_nearest.y)/d)
            new_z = x_nearest.z + self.eta * ((x_rand.z - x_nearest.z)/d)

            x_new  = node(new_x, new_y, new_z)

        return x_new

    def Exist_Check(self, x_new):

        for x_near in self.nodes:
            if x_new.x == x_near.x and x_new.y == x_near.y and x_new.z == x_near.z:
                return False
            else :
                return True

    def New_Check(self, x_new): # Check the probability value of x_new to determine whether to add it to the tree

        x_pob = np.array([x_new.x, x_new.y, x_new.z])
        x_pob = np.around(x_pob)

        if x_pob[0] >= self.map.shape[0]:
            x_pob[0] = self.map.shape[0] - 1
        if x_pob[1] >= self.map.shape[1]:
            x_pob[1] = self.map.shape[1] - 1
        if x_pob[2] >= self.map.shape[2]:
            x_pob[2] = self.map.shape[2] - 1

        x_pob = self.map[int(x_pob[0]), int(x_pob[1]), int(x_pob[2])]
        p = np.random.uniform(0, 1)

        if x_pob >= p and self.Exist_Check(x_new):
            return True
        else:
            return False


    def Add_Parent(self, x_new, x_nearest):

        x_min = x_nearest
        c_min = x_min.cost + self.Line_Cost(x_min, x_new)

        for x_near in self.nodes:
            if self.Distance_Cost(x_near, x_new) <= self.eta:

                if x_near.cost + self.Line_Cost(x_near, x_new) < c_min:
                    x_min = x_near
                    c_min = x_near.cost + self.Line_Cost(x_near, x_new)
            x_new.parent = x_min
            x_new.cost = c_min

        return x_new

    def Rewire(self, x_new):

        for x_near in self.nodes:
            if x_near is not x_new.parent:
                if self.Distance_Cost(x_near, x_new) <= self.eta :
                    if x_new.cost + self.Line_Cost(x_new, x_near) < x_near.cost:

                        x_near.parent = x_new
                        x_near.cost = x_new.cost + self.Line_Cost(x_new, x_near)

        return

    def Get_Path(self):

        temp_path = []
        path = []
        n = 0
        for i in self.nodes:
            if self.Distance_Cost(i, self.x_goal) < self.eta:
                cost = i.cost + self.Line_Cost(self.x_goal, i)
                temp_path.append([cost,n, i])
                n += 1
        temp_path.sort()


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
            if self.Distance_Cost(i, self.x_goal) < self.eta:
                cost = i.cost + self.Line_Cost(self.x_goal, i)
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

    def Draw(self, path, map): # Use plotly for plotting

        goal_path = np.empty((0,3))
        data = []
        obs = []

        for i in range(len(map)): # Draw map
            if map[i][3] != 1:
                obs.append(map[i])

        print(len(obs))
        obs = np.array(obs)

        trace1 = go.Scatter3d(x=obs[:, 0], y=obs[:, 1], z=obs[:, 2], marker=dict(color=['rgb({},{},{})'.format(r, g, b) for r, g, b in zip(255 * obs[:, 3], 255 * obs[:, 3], 255 * obs[:, 3])], size=2),mode='markers')
        data.append(trace1)

        for node in self.nodes: # Draw tree
            if node is not self.x_init:
                trace2 = go.Scatter3d(x=[node.x,node.parent.x], y=[node.y,node.parent.y], z=[node.z,node.parent.z], line=dict(color="blue", width=1),
                                      mode='lines')
                data.append(trace2)

        if len(path) >= 2:
            for node in path: # Draw optimal path
                goal_path = np.append(goal_path, [node.arr], axis = 0)

            trace3 = go.Scatter3d(x=goal_path[:, 0], y=goal_path[:, 1], z=goal_path[:, 2], line=dict(color="red", width=10),
                                mode='lines')
            data.append(trace3)

        trace4 = go.Scatter3d(x=[self.x_goal.x], y=[self.x_goal.y], z=[self.x_goal.z], marker=dict(size=5, color="red"), mode="markers")
        data.append(trace4)

        trace5 = go.Scatter3d(x=[self.x_init.x], y=[self.x_init.y], z=[self.x_init.z],
                              marker=dict(size=5, color="green"), mode="markers")
        data.append(trace5)

        layout = go.Layout(title='3D Planning', showlegend=False)
        fig = go.Figure(data = data, layout = layout)
        fig.show()


