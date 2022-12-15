import numpy as np
import plotly.graph_objs as go
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

import numpy as np
import time
import math
from generate_map import map, Reshape_map
from rrt_star3D import node, rrt_star
from RobotUR3D import Robot, map, draw_map, h_map
from mesh_gen import draw_mesh
import plotly.graph_objs as go

sample = 10000
map = h_map()
base_position = [150, 100, 0] #[100,100,10]
robot = Robot(base_position, map)

c_map = np.load("config3D(73,real).npy")
# ss = time.time()
# c_map = robot.construct_config_space()
# ee = time.time()

m = c_map.shape[0] * c_map.shape[1] * c_map.shape[2]
r = (2 * (1 + 1/3)**(1/3)) * ((m*3)/(4*math.pi))**(1/3)
eta =  r * (math.log(sample) / sample)**(1/3) # eta : 10

iteration = 10000

sample_taken = 0
total_iter = 0
x_init = node(29, 1, 54) # degree : [0, -90, 0], real_sinario : [-35, -175, 90]
x_init.parent = x_init
x_goal = node(30,35,28)  # degree1  : [-160, -100, -110] , real_sinario : [-30,-5,-40, -90, -90, 180]

distance_weight = 0.3
obstacle_weight = 0.7

rrt = rrt_star(x_init, x_goal, c_map, eta, distance_weight, obstacle_weight)


def draw_map():

    xyz = np.load("xyz.npy")
    rgb = np.load("rgb.npy")

    data = []

    trace1 = go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], marker=dict(
        color=['rgb({},{},{})'.format(r, g, b) for r, g, b in
               zip(rgb[:, 0], rgb[:, 1], rgb[:, 2])], size=1), mode='markers')

    data.append(trace1)

    trace2 = go.Scatter3d(x=[0], y=[0], z=[0], marker=dict(color="black", size=0.1), mode='markers')
    data.append(trace2)

    trace3 = go.Scatter3d(x=[101], y=[201], z=[101], marker=dict(color="black", size=0.1),
                          mode='markers')
    data.append(trace3)

    # fig = go.Figure(data=data)
    # fig.show()
    return data

data = draw_map()


goal_path = np.load("path(73,real).npy")
ee_position = []
overlap = [] # degree of overlap between objects

for i in range(1, len(goal_path)):

    position, rotate, height = robot.robot_position(goal_path[i][0]*5-180, goal_path[i][1]*5-180, goal_path[i][2]*5-180)
    ee_position.append(position[-1])
#     profile = robot.profile(position, rotate, height)
#
#     for a in range(profile.shape[0]):
#         for  b in range(profile.shape[1]):
#             for c in range(profile.shape[2]):
#                 if profile[a][b][c] == 0.7:
#                     overlap.append([a,b,c])
#
# list=list(set([tuple(t) for t in overlap]))
#
# print(len(list))


ee_position = np.array(ee_position)
#
p1, r1, h1 = robot.robot_position(-35,-175,90)
data.append(draw_mesh(p1,r1,h1))

# p2, r2, h2 = robot.robot_position(goal_path[4][0]*5-180,goal_path[4][1]*5-180,goal_path[4][2]*5-180)
# data.append(draw_mesh(p2,r2,h2))
#
# p3, r3, h3 = robot.robot_position(goal_path[5][0]*5-180,goal_path[5][1]*5-180,goal_path[5][2]*5-180)
# data.append(draw_mesh(p3,r3,h3))

p5, r5, h5 = robot.robot_position(goal_path[-1][0]*5-180,goal_path[-1][1]*5-180,goal_path[-1][2]*5-180)
data.append(draw_mesh(p5,r5,h5))

trace = go.Scatter3d(x=ee_position[:, 0], y=ee_position[:, 1], z=ee_position[:, 2], line=dict(color="red", width=5), mode='lines')
data.append(trace)

trace1 = go.Scatter3d(x=[100], y=[200], z=[100], marker=dict(color="black", size=0.1), mode='markers')
data.append(trace1)

layout = go.Layout(title='3D Planning', showlegend=False)
fig = go.Figure(data=data, layout=layout)
fig.show()

# xyz = np.load("xyz.npy")
# rgb = np.load("rgb.npy")
#
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
#
# ax.scatter3D(xyz[:,0], xyz[:,1], xyz[0,2], c=rgb/255)
# plt.show()