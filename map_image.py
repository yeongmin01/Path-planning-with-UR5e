from generate_map import map, d_map, BARN_map_137, BARN_map_113
import matplotlib.pyplot as plt
import numpy as np
import cv2

map = BARN_map_113(0,False)
plt.figure(figsize=(10,10))
plt.axes().set_aspect('equal')
plt.imshow(np.transpose(1 - map), cmap="jet", interpolation="nearest")
plt.show()

map1 = BARN_map_113(3,False)
plt.figure(figsize=(10,10))
plt.axes().set_aspect('equal')
plt.imshow(np.transpose(1 - map1), cmap="jet", interpolation="nearest")

plt.show()

map2 = BARN_map_113(3,True)
plt.figure(figsize=(10,10))
plt.axes().set_aspect('equal')
plt.imshow(np.transpose(1 - map2), cmap="jet", interpolation="nearest")
plt.show()

# map1 = BARN_map_137(0)
# plt.figure(figsize=(10,10))
# plt.axes().set_aspect('equal')
# plt.imshow(map1, cmap="gray", interpolation="nearest")
# plt.show()
#
# map = 1- BARN_map_137(3)
# plt.figure(figsize=(12,10))
# plt.axes().set_aspect('equal')
# plt.imshow(map, cmap="RdYlBu_r", interpolation="nearest")
# plt.scatter(15,5)
# plt.colorbar()
# plt.show()

# plt.figure(figsize=(10,10))
# plt.axes().set_aspect('equal')
# plt.imshow(map, cmap="gray", interpolation="nearest")

# def Sampling(map):
#     row = map.shape[0]
#
#     p = np.ravel(map) / np.sum(map)
#
#     x_sample = np.random.choice(len(p), p=p)
#
#     x = x_sample // row
#     y = x_sample % row
#
#     x = np.random.uniform(low = x-0.5, high = x+0.5)
#     y = np.random.uniform(low = y-0.5, high = y+0.5)
#
#     x_rand = np.array([x,y])
#
#     return x_rand
#
# np.random.seed(3)
# for i in range(50000) :
#     x = Sampling(map)
#     x_new = x
#
#     plt.scatter(x[1], x[0], s=5, c="blue")
#
# plt.show()

