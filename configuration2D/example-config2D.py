import numpy as np
import time
import math
import matplotlib.pyplot as plt
from rrt_star2D import node , rrt_star
from RobotArm2D import Robot, map, pmap

iteration = 3000
map = pmap()
# map = map()
base_position = [15, 15]
link_lenths = [5, 5]
robot = Robot(base_position, link_lenths, map)
c_map = robot.construct_config_space()
m = c_map.shape[0] * c_map.shape[1]
r = (2 * (1 + 1/2)**(1/2)) * (m/math.pi)**(1/2)
eta =  r * (math.log(iteration) / iteration)**(1/2)

sample_taken = 0
total_iter = 0
x_init = node(35, 30)
x_goal = node(250, 300)
x_init.parent = x_init

distance_weight = 0.5
obstacle_weight = 0.5

Graph_sample_num = 0
Graph_data = np.array([[0,0]])

rrt = rrt_star(x_init, x_goal, c_map, eta, distance_weight, obstacle_weight)


s = time.time()
np.random.seed(0)
t1, t2, t3 = 0, 0, 0

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
    x_new = rrt.Add_Parent(x_new, x_nearest)
    e2 = time.time()
    t2 += (e2 - s2)

    rrt.nodes.append(x_new)
    s3 = time.time()
    rrt.Rewire(x_new)
    e3 = time.time()
    t3 += (e3 - s3)


e = time.time()

plt.figure(figsize=(12,10))
plt.axes().set_aspect('equal')

rrt.Draw_Tree()
path = rrt.Get_Path()
print("eta : ", eta)
print("Total time : ", e - s,"초")
print("Sampling time : ", t1,"초", (t1*100)/(e-s),"%")
print("Add_Parent time : ", t2,"초", (t2*100)/(e-s),"%")
print("Rewire time : ", t3,"초", (t3*100)/(e-s),"%")
print("Total_sample = ", sample_taken)
print("Cost : ", rrt.x_goal.cost)

plt.imshow(np.transpose(c_map), cmap = "gray", interpolation = 'nearest')
rrt.Draw_Tree()
rrt.Draw_path(path)
plt.colorbar()
plt.show()

plt.figure(figsize=(10,10))
plt.axes().set_aspect('equal')
plt.imshow(np.transpose(map),cmap = "gray", interpolation = 'nearest')

w_path = []
for i in path:

    position = robot.robot_position(i.arr[0], i.arr[1])
    w_path.append(position)
for i in range(len(w_path)):
    plt.plot([w_path[i][0][0], w_path[i][1][0]], [w_path[i][0][1], w_path[i][1][1]], "k", linewidth=2.5)
    plt.plot([w_path[i][1][0], w_path[i][2][0]], [w_path[i][1][1], w_path[i][2][1]], "k", linewidth=2.5)
for i in range(len(w_path) - 1):
    plt.plot([w_path[i][2][0], w_path[i+1][2][0]], [w_path[i][2][1], w_path[i+1][2][1]], "r", linewidth=2.5)

plt.colorbar()
plt.show()