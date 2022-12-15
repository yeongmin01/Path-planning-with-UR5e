import numpy as np
import matplotlib.pyplot as plt

data0 = np.load("data/data0.npy")
data1 = np.load("data/data1.npy")
data2 = np.load("data/data2.npy")
data3 = np.load("data/data3.npy")
data4 = np.load("data/data4.npy")

plt.figure(figsize=(10,5))
# plt.xlim(0, 10000)
plt.xlabel("Iteration", fontsize = 16)
plt.ylabel("Cost", fontsize = 16)

seed0 = []
seed1 = []
seed2 = []
seed3 = []
seed4 = []
#
for i in range(len(data0)-1):
    if data0[i][1] != 0 :
        seed0.append([data0[i][0],data0[i][1]])

plt.plot(list(zip(*seed0))[0], list(zip(*seed0))[1], "r", linewidth=3, label = "seed 0")

for i in range(len(data1)-1):
    if data1[i][1] != 0 :
        seed1.append([data1[i][0], data1[i][1]])
plt.plot(list(zip(*seed1))[0], list(zip(*seed1))[1], "y", linewidth=3, label = "seed 1")

for i in range(len(data2)-1):
    if data2[i][1] != 0 :
        seed2.append([data2[i][0], data2[i][1]])
plt.plot(list(zip(*seed2))[0], list(zip(*seed2))[1], "g", linewidth=3, label = "seed 2")
#
for i in range(len(data3)-1):
    if data3[i][1] != 0 :
        seed3.append([data3[i][0], data3[i][1]])
plt.plot(list(zip(*seed3))[0], list(zip(*seed3))[1], "b", linewidth=3, label = "seed 3")

for i in range(len(data4)-1):
    if data4[i][1] != 0 :
        seed4.append([data4[i][0], data4[i][1]])
plt.plot(list(zip(*seed4))[0], list(zip(*seed4))[1], "c", linewidth=3, label = "seed 4")

plt.legend()
plt.show()
