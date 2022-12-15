import numpy as np
import time
import math
import matplotlib.pyplot as plt
from rrt_star_general import node, rrt_star
from obstacle_space import Obstacle_generater, Obstacle_BARN_113, bmap

map_size = np.array([0,29])
iteration = 1000
m = (map_size[1]+1) * (map_size[1]+1)
r = (2 * (1 + 1/2)**(1/2)) * (m/math.pi)**(1/2)
eta =  r * (math.log(iteration) / iteration)**(1/2)

sample_taken = 0
total_iter = 0


collision_range = (2**(1/2))/2
obstacle, obstacle_center  = Obstacle_BARN_113()#bmap()
obs = Obstacle_generater(obstacle)

x_init = node(3, 27)
x_goal = node(27, 3)

x_init.parent = x_init

Graph_sample_num = 0
Graph_data = np.array([[0,0]])

map = []
rrt = rrt_star(map, x_init, x_goal, eta, obs, obstacle_center, collision_range, map_size)
# rrt.Draw_obs()
# plt.gca().invert_yaxis()
# plt.show()

# s = time.time()
# t1, t2, t3 = 0, 0, 0
#
# np.random.seed(0)
# while True:
#
#     s1 = time.time()
#     while True:
#
#         x_rand = rrt.Sampling()
#
#         total_iter += 1
#         if total_iter % 100 == 0:
#             print(total_iter, "Iteration")
#
#         x_nearest = rrt.Nearest(x_rand)
#
#         x_new = rrt.Steer(x_rand, x_nearest)
#
#         if rrt.Node_collision_check(x_new):
#             break
#
#         if total_iter == iteration:
#             break
#     if total_iter == iteration:
#         break
#
#     e1 = time.time()
#     t1 += (e1 - s1)
#
#     s2 = time.time()
#     if rrt.Edge_collision_check(x_nearest, x_new) == False:
#         continue
#
#     sample_taken += 1
#
#     x_new , x_nearest = rrt.Add_Parent(x_new, x_nearest)
#     e2 = time.time()
#     t2 += (e2 - s2)
#
#     rrt.nodes.append(x_new)
#     s3 = time.time()
#     rrt.Rewire(x_new)
#     e3 = time.time()
#     t3 += (e3 - s3)

#     Graph_sample_num += 1
#     if Graph_sample_num%100 == 0:
#         Graph_cost = rrt.Cost_Graph()
#         Graph_data = np.append(Graph_data, np.array([[Graph_sample_num, Graph_cost]]), axis = 0)
#
#
# np.save("data/data1", Graph_data)

# e = time.time()
#
# path = rrt.Get_Path()
# print("eta : ", eta)
# print("Total time : ", e - s,"초")
# print("Sampling time : ", t1,"초", (t1*100)/(e-s),"%")
# print("Add_Parent time : ", t2,"초", (t2*100)/(e-s),"%")
# print("Rewire time : ", t3,"초", (t3*100)/(e-s),"%")
# print("Total_sample = ", sample_taken)
# print("Cost : ", rrt.x_goal.cost)
#
# rrt.Draw_obs()
# rrt.Draw_Tree()
# rrt.Draw_path(path)
# plt.xlim(-0.5, 29.5)
# plt.ylim(-0.5, 29.5)
# plt.gca().invert_yaxis()
# plt.show()

iter = 50
total_time = 0
total_sample = 0
total_cost = 0
count = 0
t1, t2, t3 = 0, 0, 0

iter_time = []
sample = []
cost = []

for i in range(iter):
    print(i,"번째")
    np.random.seed(i)
    sample_taken = 0
    total_iter = 0
    rrt.nodes = [rrt.x_init]

    s = time.time()
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

            if total_iter == iteration:
                break
        e1 = time.time()
        t1 += e1- s1

        if total_iter == iteration:
            break

        s2 = time.time()
        if rrt.Edge_collision_check(x_nearest, x_new) == False:
            continue

        sample_taken += 1

        x_new, x_nearest = rrt.Add_Parent(x_new, x_nearest)
        e2 = time.time()
        t2 += e2 - s2

        rrt.nodes.append(x_new)

        s3 = time.time()
        rrt.Rewire(x_new)
        e3 = time.time()
        t3 += e3 - s3

    e = time.time()
    total_time += e - s
    iter_time.append(e-s)
    total_sample += sample_taken
    sample.append(sample_taken)

    path = rrt.Get_Path()
    if path == None:
        count +=1
    else:

        total_cost += rrt.x_goal.cost
        cost.append(rrt.x_goal.cost)

print("cannot find path : ", count)
print("Sampling time : ", t1/(iter),"초", (t1*100)/total_time,"%")
print("Add_Parent time : ", t2/ (iter),"초", (t2*100)/total_time,"%")
print("Rewire time : ", t3/ (iter),"초", (t3*100)/total_time,"%")
print("Total_time : ", total_time / (iter) , np.std(iter_time))
print("total_sample : ", total_sample / (iter), np.std(sample))
print("total_cost : ", total_cost / (iter-count), np.std(cost))
