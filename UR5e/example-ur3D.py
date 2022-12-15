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

s = time.time()
t1, t2, t3 = 0, 0, 0
np.random.seed(0)
while True:

    s1 = time.time()
    while True:

        x_rand = rrt.Sampling()
        total_iter += 1
        if total_iter % 100 == 0:
            print(total_iter, "Iteration")

        x_nearest = rrt.Nearest(x_rand)
        x_new = rrt.Steer(x_rand, x_nearest)
        b = rrt.New_Check(x_new)
        if b == True :
            break

        if total_iter == iteration:
            break
    if total_iter == iteration:
        break

    e1 = time.time()
    t1 += (e1 - s1)

    sample_taken += 1

    s2 = time.time()
    x_new  = rrt.Add_Parent(x_new, x_nearest)
    e2 = time.time()
    t2 += (e2 - s2)

    rrt.nodes.append(x_new)
    s3 = time.time()
    rrt.Rewire(x_new)
    e3 = time.time()
    t3 += (e3 - s3)


e = time.time()
path = rrt.Get_Path()

print("eta : ", eta)
print("Total time : ", e - s,"sec")
print("Sampling time : ", t1,"sec", (t1*100)/(e-s),"%")
print("Add_Parent time : ", t2,"sec", (t2*100)/(e-s),"%")
print("Rewire time : ", t3,"sec", (t3*100)/(e-s),"%")
print("Total_sample = ", sample_taken)
print("Cost = ", rrt.x_goal.cost)

rrt.Draw(path, Reshape_map(c_map))
path = np.flip(path, axis=0)

goal_path = np.array([x_init.arr])
if len(path) >= 2:
    for node in path:  # Draw optimal path
        goal_path = np.append(goal_path, [node.arr], axis=0)

np.save("path(73,real)", goal_path)
#
data = draw_map(map)
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
