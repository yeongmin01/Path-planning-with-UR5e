import numpy as np
import time
import math
import matplotlib.pyplot as plt
from rrt_star_general import node, rrt_star
from obstacle_space import Obstacle_generater, Obstacle_center, Collision_range, Obstacle_BARN_113


map_size = np.array([0,49])
iteration = 1000
m = (map_size[1]+1) * (map_size[1]+1)
r = (2 * (1 + 1/2)**(1/2)) * (m/math.pi)**(1/2)
eta =  r * (math.log(iteration) / iteration)**(1/2)

sample_taken = 0
total_iter = 0

# obstacle = [((10,10),(15,10),(15,15),(10,15),(10,10)), ((20,20),(25,20),(25,25),(20,25),(20,20)), ((5,20),(10,20),(10,15),(5,15),(5,20)),
#             ((10,25),(15,25),(15,20),(10,20),(10,25))]
# obstacle_center = Obstacle_center(obstacle)
# collision_range = Collision_range(obstacle)
# obs = Obstacle_generater(obstacle)
collision_range = (2**(1/2))/2
obstacle, obstacle_center  = Obstacle_BARN_113()
obs = Obstacle_generater(obstacle)

x_init = node(5, 27)
x_goal = node(25, 3)
x_init.parent = x_init

Graph_sample_num = 0
Graph_data = np.array([[0,0]])

map = []
rrt = rrt_star(map, x_init, x_goal, eta, obs, obstacle_center, collision_range, map_size)
# rrt.Draw_obs()
# plt.gca().invert_yaxis()
# plt.show()

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

        if rrt.Node_collision_check(x_new):
            break

        if total_iter > iteration:
            break

    if total_iter > iteration:
        break

    e1 = time.time()
    t1 += (e1 - s1)

    s2 = time.time()
    if rrt.Edge_collision_check(x_nearest, x_new) == False:
        continue

    sample_taken += 1

    x_new.cost = x_nearest.cost + rrt.Line_Cost(x_nearest, x_new)
    x_new.parent = x_nearest

    rrt.nodes.append(x_new)

#     Graph_sample_num += 1
#     if Graph_sample_num%100 == 0:
#         Graph_cost = rrt.Cost_Graph()
#         Graph_data = np.append(Graph_data, np.array([[Graph_sample_num, Graph_cost]]), axis = 0)
#
#
# np.save("data/data0", Graph_data)

e = time.time()

path = rrt.Get_Path()
print("eta : ", eta)
print("Total time : ", e - s,"초")
print("Sampling time : ", t1,"초", (t1*100)/(e-s),"%")
print("Total_sample = ", sample_taken)
print("Cost : ", rrt.x_goal.cost)

rrt.Draw_obs()
rrt.Draw_Tree()
rrt.Draw_path(path)
plt.gca().invert_yaxis()
plt.title("RRT")
plt.show()

# iter = 10
# total_time = 0
# total_sample = 0
#
#
# for i in range(iter):
#
#     np.random.seed(i)
#     sample_taken = 0
#     total_iter = 0
#     rrt.nodes = [rrt.x_init]
#
#     s = time.time()
#     while True:
#
#
#         while True:
#
#             x_rand = rrt.Sampling()
#
#             total_iter += 1
#             if total_iter % 100 == 0:
#                 print(total_iter, "Iteration")
#
#             x_nearest = rrt.Nearest(x_rand)
#
#             x_new = rrt.Steer(x_rand, x_nearest)
#
#             if rrt.Node_collision_check(x_new):
#                 break
#
#             if total_iter > iteration:
#                 break
#         if total_iter > iteration:
#             break
#
#
#
#         if rrt.Edge_collision_check(x_nearest, x_new) == False:
#             continue
#
#         sample_taken += 1
#
#         x_new.cost = x_nearest.cost + rrt.Line_Cost(x_nearest, x_new)
#         x_new.parent = x_nearest
#
#         rrt.nodes.append(x_new)
#
#     path = rrt.Get_Path()
#
#     e = time.time()
#     total_time += e-s
#     total_sample += sample_taken
#
# print("Total_time : ", total_time/iter)
# print("total_iteration : ", total_sample/iter)