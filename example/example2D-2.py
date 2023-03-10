import numpy as np
import time
import matplotlib.pyplot as plt
import math
from generate_map import BARN_map_113
from rrt_star2D import node , rrt_star


iteration = 10000
sample_taken = 0
total_iter = 0
x_init = node(15, 110)
x_init.parent = x_init
x_goal = node(110, 17)
filter_size = 3 # 1 = 3x3, 2 = 5x5, 3 = 7x7
classify = True
distance_weight = 0.5
obstacle_weight = 0.5

Graph_sample_num = 0
Graph_data = np.array([[0,0]])

ss = time.time()
map = BARN_map_113(filter_size,classify)
ee = time.time()
print(ee-ss)

m = map.shape[0] * map.shape[1]
r = (2 * (1 + 1/2)**(1/2)) * (m/math.pi)**(1/2)
eta =  r * (math.log(iteration) / iteration)**(1/2)
iteration = 20000
rrt = rrt_star(x_init, x_goal, map, eta, distance_weight, obstacle_weight)

plt.figure(figsize=(12,10))
plt.axes().set_aspect('equal')

s = time.time()
np.random.seed(0)
t1, t2, t3 = 0, 0, 0
p = []

while True:

    s1 = time.time()
    while True:
        x_rand = rrt.Sampling()
        total_iter += 1
        # print(x_rand.arr)
        x_nearest = rrt.Nearest(x_rand)
        # print(x_nearest.arr)
        x_new = rrt.Steer(x_rand, x_nearest)
        b = rrt.New_Check(x_new)
        if b == True :
            break

        if total_iter == iteration:
            break
    e1 = time.time()
    t1 += e1 - s1

    if total_iter == iteration:
        break

    sample_taken += 1
    if sample_taken%100 == 0:
        print(sample_taken,"Sample Taken")

    s2 = time.time()
    x_new = rrt.Add_Parent(x_new, x_nearest)
    e2 = time.time()
    t2 += (e2 - s2)

    rrt.nodes.append(x_new)

    s3 = time.time()
    rrt.Rewire(x_new)
    e3 = time.time()
    t3 += (e3 - s3)

    # if total_iter % 2000 == 0:
    #     if rrt.Get_Path() != None:
    #         p.append( rrt.Get_Path())


    Graph_sample_num += 1
    if Graph_sample_num%100 == 0:
        Graph_cost = rrt.Cost_Graph()
        Graph_data = np.append(Graph_data,np.array([[Graph_sample_num, Graph_cost]]), axis = 0)


np.save("data/data0", Graph_data)

e = time.time()
path = rrt.Get_Path()
print("ets : ", eta)
print(e-s, "???")
print("Total time : ", e - s,"???")
print("Sampling time : ", t1,"???", (t1*100)/(e-s),"%")
print("Add_Parent time : ", t2,"???", (t2*100)/(e-s),"%")
print("Rewire time : ", t3,"???", (t3*100)/(e-s),"%")
print("Total_iteration = ", total_iter)
print("Cost : ", rrt.x_goal.cost)


rrt.Draw_Tree()
# for i in range(len(p)):
#     rrt.Draw_path(path[i])
rrt.Draw_path(path)
plt.imshow(np.transpose(1-map),cmap = "jet", interpolation = 'nearest')
plt.colorbar()
plt.show()