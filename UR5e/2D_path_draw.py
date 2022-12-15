import numpy as np
import matplotlib.pyplot as plt
from RobotUR2D import Robot, pmap
from skimage.measure import profile_line

iteration = 5000
map = pmap()
print(np.sum(1-map))

base_position = [90, 0]
robot = Robot(base_position, map)

plt.figure(figsize=(10,5))
plt.axes().set_aspect('equal')
plt.imshow(np.transpose(map),cmap = "gray", interpolation = 'nearest')

path11,path21,path31,path41,path51 = np.load("2D-data/seed1/path(leaf-x)-1.npy"), np.load("2D-data/seed2/path(leaf-x)-2.npy"), np.load("2D-data/seed3/path(leaf-x)-3.npy"), np.load("2D-data/seed4/path(leaf-x)-4.npy"), np.load("2D-data/seed5/path(leaf-x)-5.npy")
path12,path22,path32,path42,path52 = np.load("2D-data/seed1/path(잎 구분x)-1.npy"), np.load("2D-data/seed2/path(잎 구분x)-2.npy"), np.load("2D-data/seed3/path(잎 구분x)-3.npy"), np.load("2D-data/seed4/path(잎 구분x)-4.npy"), np.load("2D-data/seed5/path(잎 구분x)-5.npy")
path13,path23,path33,path43,path53 = np.load("2D-data/seed1/path(잎 구분,0.5)-1.npy"), np.load("2D-data/seed2/path(잎 구분,0.5)-2.npy"), np.load("2D-data/seed3/path(잎 구분,0.5)-3.npy"), np.load("2D-data/seed4/path(잎 구분,0.5)-4.npy"), np.load("2D-data/seed5/path(잎 구분,0.5)-5.npy")
path14,path24,path34,path44,path54 = np.load("2D-data/seed1/path(잎 구분,0.75)-1.npy"), np.load("2D-data/seed2/path(잎 구분,0.75)-2.npy"), np.load("2D-data/seed3/path(잎 구분,0.75)-3.npy"), np.load("2D-data/seed4/path(잎 구분,0.75)-4.npy"), np.load("2D-data/seed5/path(잎 구분,0.75)-5.npy")

w_path11,w_path21,w_path31,w_path41,w_path51 = [], [], [], [], []
w_path12,w_path22,w_path32,w_path42,w_path52 = [], [], [], [], []
w_path13,w_path23,w_path33,w_path43,w_path53 = [], [], [], [], []
w_path14,w_path24,w_path34,w_path44,w_path54 = [], [], [], [], []

p11, p21, p31, p41, p51 = 0, 0, 0, 0, 0
p12, p22, p32, p42, p52 = 0, 0, 0, 0, 0
p13, p23, p33, p43, p53 = 0, 0, 0, 0, 0
p14, p24, p34, p44, p54 = 0, 0, 0, 0, 0

for a in path11:
    position11 = robot.robot_position(a[0] - 180, a[1] - 180)
    p111 = profile_line(1-map, position11[0], position11[1], linewidth=8, order=0, reduce_func=None)
    p112 = profile_line(1-map, position11[1], position11[2], linewidth=7, order=0, reduce_func=None)
    p11 += np.sum(p111) + np.sum(p112)
    w_path11.append(position11)
for b in path21:
    position21 = robot.robot_position(b[0] - 180, b[1] - 180)
    p211 = profile_line(1-map, position21[0], position21[1], linewidth=8, order=0, reduce_func=None)
    p212 = profile_line(1-map, position21[1], position21[2], linewidth=7, order=0, reduce_func=None)
    p21 += np.sum(p211) + np.sum(p212)
    w_path21.append(position21)
for c in path31:
    position31 = robot.robot_position(c[0] - 180, c[1] - 180)
    p311 = profile_line(1-map, position31[0], position31[1], linewidth=8, order=0, reduce_func=None)
    p312 = profile_line(1-map, position31[1], position31[2], linewidth=7, order=0, reduce_func=None)
    p31 += np.sum(p311) + np.sum(p312)
    w_path31.append(position31)
for d in path41:
    position41 = robot.robot_position(d[0] - 180, d[1] - 180)
    p411 = profile_line(1-map, position41[0], position41[1], linewidth=8, order=0, reduce_func=None)
    p412 = profile_line(1-map, position41[1], position41[2], linewidth=7, order=0, reduce_func=None)
    p41 += np.sum(p411) + np.sum(p412)
    w_path41.append(position41)
for e in path51:
    position51 = robot.robot_position(e[0] - 180, e[1] - 180)
    p511 = profile_line(1-map, position51[0], position51[1], linewidth=8, order=0, reduce_func=None)
    p512 = profile_line(1-map, position51[1], position51[2], linewidth=7, order=0, reduce_func=None)
    p51 += np.sum(p511) + np.sum(p512)
    w_path51.append(position51)

for a in path12:
    position12 = robot.robot_position(a[0] - 180, a[1] - 180)
    p121 = profile_line(1-map, position12[0], position12[1], linewidth=8, order=0, reduce_func=None)
    p122 = profile_line(1-map, position12[1], position12[2], linewidth=7, order=0, reduce_func=None)
    p12 += np.sum(p121) + np.sum(p122)
    w_path12.append(position12)
for b in path22:
    position22 = robot.robot_position(b[0] - 180, b[1] - 180)
    p221 = profile_line(1-map, position22[0], position22[1], linewidth=8, order=0, reduce_func=None)
    p222 = profile_line(1-map, position22[1], position22[2], linewidth=7, order=0, reduce_func=None)
    p22 += np.sum(p221) + np.sum(p222)
    w_path22.append(position22)
for c in path32:
    position32 = robot.robot_position(c[0] - 180, c[1] - 180)
    p321 = profile_line(1-map, position32[0], position32[1], linewidth=8, order=0, reduce_func=None)
    p322 = profile_line(1-map, position32[1], position32[2], linewidth=7, order=0, reduce_func=None)
    p32 += np.sum(p321) + np.sum(p322)
    w_path32.append(position32)
for d in path42:
    position42 = robot.robot_position(d[0] - 180, d[1] - 180)
    p421 = profile_line(1-map, position42[0], position42[1], linewidth=8, order=0, reduce_func=None)
    p422 = profile_line(1-map, position42[1], position42[2], linewidth=7, order=0, reduce_func=None)
    p42 += np.sum(p421) + np.sum(p422)
    w_path42.append(position42)
for e in path52:
    position52 = robot.robot_position(e[0] - 180, e[1] - 180)
    p521 = profile_line(1-map, position52[0], position52[1], linewidth=8, order=0, reduce_func=None)
    p522 = profile_line(1-map, position52[1], position52[2], linewidth=7, order=0, reduce_func=None)
    p52 += np.sum(p521) + np.sum(p522)
    w_path52.append(position52)

for a in path13:
    position13 = robot.robot_position(a[0] - 180, a[1] - 180)
    p131 = profile_line(1-map, position13[0], position13[1], linewidth=8, order=0, reduce_func=None)
    p132 = profile_line(1-map, position13[1], position13[2], linewidth=7, order=0, reduce_func=None)
    p13 += np.sum(p131) + np.sum(p132)
    w_path13.append(position13)
for b in path23:
    position23 = robot.robot_position(b[0] - 180, b[1] - 180)
    p231 = profile_line(1-map, position23[0], position23[1], linewidth=8, order=0, reduce_func=None)
    p232 = profile_line(1-map, position23[1], position23[2], linewidth=7, order=0, reduce_func=None)
    p23 += np.sum(p231) + np.sum(p232)
    w_path23.append(position23)
for c in path33:
    position33 = robot.robot_position(c[0] - 180, c[1] - 180)
    p331 = profile_line(1-map, position33[0], position33[1], linewidth=8, order=0, reduce_func=None)
    p332 = profile_line(1-map, position33[1], position33[2], linewidth=7, order=0, reduce_func=None)
    p33 += np.sum(p331) + np.sum(p332)
    w_path33.append(position33)
for d in path43:
    position43 = robot.robot_position(d[0] - 180, d[1] - 180)
    p431 = profile_line(1-map, position43[0], position43[1], linewidth=8, order=0, reduce_func=None)
    p432 = profile_line(1-map, position43[1], position43[2], linewidth=7, order=0, reduce_func=None)
    p43 += np.sum(p431) + np.sum(p432)
    w_path43.append(position43)
for e in path53:
    position53 = robot.robot_position(e[0] - 180, e[1] - 180)
    p531 = profile_line(1-map, position53[0], position53[1], linewidth=8, order=0, reduce_func=None)
    p532 = profile_line(1-map, position53[1], position53[2], linewidth=7, order=0, reduce_func=None)
    p53 += np.sum(p531) + np.sum(p532)
    w_path51.append(position53)

for a in path14:
    position14 = robot.robot_position(a[0] - 180, a[1] - 180)
    p141 = profile_line(1-map, position14[0], position14[1], linewidth=8, order=0, reduce_func=None)
    p142 = profile_line(1-map, position14[1], position14[2], linewidth=7, order=0, reduce_func=None)
    p14 += np.sum(p141) + np.sum(p142)
    w_path14.append(position14)
for b in path24:
    position24 = robot.robot_position(b[0] - 180, b[1] - 180)
    p241 = profile_line(1-map, position24[0], position24[1], linewidth=8, order=0, reduce_func=None)
    p242 = profile_line(1-map, position24[1], position24[2], linewidth=7, order=0, reduce_func=None)
    p24 += np.sum(p241) + np.sum(p242)
    w_path24.append(position24)
for c in path34:
    position34 = robot.robot_position(c[0] - 180, c[1] - 180)
    p341 = profile_line(1-map, position34[0], position34[1], linewidth=8, order=0, reduce_func=None)
    p342 = profile_line(1-map, position34[1], position34[2], linewidth=7, order=0, reduce_func=None)
    p34 += np.sum(p341) + np.sum(p342)
    w_path34.append(position34)
for d in path44:
    position44 = robot.robot_position(d[0] - 180, d[1] - 180)
    p441 = profile_line(1-map, position44[0], position44[1], linewidth=8, order=0, reduce_func=None)
    p442 = profile_line(1-map, position44[1], position44[2], linewidth=7, order=0, reduce_func=None)
    p44 += np.sum(p441) + np.sum(p442)
    w_path44.append(position44)
for e in path54:
    position54 = robot.robot_position(e[0] - 180, e[1] - 180)
    p541 = profile_line(1-map, position54[0], position54[1], linewidth=8, order=0, reduce_func=None)
    p542 = profile_line(1-map, position54[1], position54[2], linewidth=7, order=0, reduce_func=None)
    p54 += np.sum(p541) + np.sum(p542)
    w_path54.append(position54)

plt.plot([base_position[0], w_path11[0][0][0]], [base_position[1], w_path11[0][0][1]], "y", linewidth=10)
plt.plot([w_path11[0][0][0], w_path11[0][1][0]], [w_path11[0][0][1], w_path11[0][1][1]], "b", linewidth=8)
plt.plot([w_path11[0][1][0], w_path11[0][2][0]], [w_path11[0][1][1], w_path11[0][2][1]], "r", linewidth=7)

plt.plot([base_position[0], w_path11[-1][0][0]], [base_position[1], w_path11[-1][0][1]], "y", linewidth=10)
plt.plot([w_path11[-1][0][0], w_path11[-1][1][0]], [w_path11[-1][0][1], w_path11[-1][1][1]], "b", linewidth=8)
plt.plot([w_path11[-1][1][0], w_path11[-1][2][0]], [w_path11[-1][1][1], w_path11[-1][2][1]], "r", linewidth=7)

for i in range(len(w_path11) - 1):
    plt.plot([w_path11[i][2][0], w_path11[i+1][2][0]], [w_path11[i][2][1], w_path11[i+1][2][1]], "g", linewidth=2.5)

for i in range(len(w_path21) - 1):
    plt.plot([w_path21[i][2][0], w_path21[i+1][2][0]], [w_path21[i][2][1], w_path21[i+1][2][1]], "g", linewidth=2.5)

for i in range(len(w_path31) - 1):
    plt.plot([w_path31[i][2][0], w_path31[i+1][2][0]], [w_path31[i][2][1], w_path31[i+1][2][1]], "g", linewidth=2.5)

for i in range(len(w_path41) - 1):
    plt.plot([w_path41[i][2][0], w_path41[i+1][2][0]], [w_path41[i][2][1], w_path41[i+1][2][1]], "g", linewidth=2.5)

# for i in range(len(w_path51) - 1):
#     plt.plot([w_path51[i][2][0], w_path51[i+1][2][0]], [w_path51[i][2][1], w_path51[i+1][2][1]], "g", linewidth=2.5)
####################################################################################################################  leaf-x
# for i in range(len(w_path12) - 1):
#     plt.plot([w_path21[i][2][0], w_path21[i+1][2][0]], [w_path21[i][2][1], w_path21[i+1][2][1]], "b", linewidth=2.5)
#
# for i in range(len(w_path22) - 1):
#     plt.plot([w_path22[i][2][0], w_path22[i+1][2][0]], [w_path22[i][2][1], w_path22[i+1][2][1]], "b", linewidth=2.5)
#
# for i in range(len(w_path32) - 1):
#     plt.plot([w_path32[i][2][0], w_path32[i+1][2][0]], [w_path32[i][2][1], w_path32[i+1][2][1]], "b", linewidth=2.5)
#
# for i in range(len(w_path42) - 1):
#     plt.plot([w_path42[i][2][0], w_path42[i+1][2][0]], [w_path42[i][2][1], w_path42[i+1][2][1]], "b", linewidth=2.5)
#
# for i in range(len(w_path52) - 1):
#     plt.plot([w_path52[i][2][0], w_path52[i+1][2][0]], [w_path52[i][2][1], w_path52[i+1][2][1]], "b", linewidth=2.5)
#################################################################################################################### 잎 구분 x
# for i in range(len(w_path13) - 1):
#     plt.plot([w_path13[i][2][0], w_path13[i+1][2][0]], [w_path13[i][2][1], w_path13[i+1][2][1]], "y", linewidth=2.5)
#
# for i in range(len(w_path23) - 1):
#     plt.plot([w_path23[i][2][0], w_path23[i+1][2][0]], [w_path23[i][2][1], w_path23[i+1][2][1]], "y", linewidth=2.5)
#
# for i in range(len(w_path33) - 1):
#     plt.plot([w_path33[i][2][0], w_path33[i+1][2][0]], [w_path33[i][2][1], w_path33[i+1][2][1]], "y", linewidth=2.5)
#
# for i in range(len(w_path43) - 1):
#     plt.plot([w_path43[i][2][0], w_path43[i+1][2][0]], [w_path43[i][2][1], w_path43[i+1][2][1]], "y", linewidth=2.5)
#
# for i in range(len(w_path53) - 1):
#     plt.plot([w_path53[i][2][0], w_path53[i+1][2][0]], [w_path53[i][2][1], w_path53[i+1][2][1]], "y", linewidth=2.5)
#################################################################################################################### 잎 구분 0.5
for i in range(len(w_path14) - 1):
    plt.plot([w_path14[i][2][0], w_path14[i+1][2][0]], [w_path14[i][2][1], w_path14[i+1][2][1]], "r", linewidth=2.5)

for i in range(len(w_path24) - 1):
    plt.plot([w_path24[i][2][0], w_path24[i+1][2][0]], [w_path24[i][2][1], w_path24[i+1][2][1]], "r", linewidth=2.5)

for i in range(len(w_path34) - 1):
    plt.plot([w_path34[i][2][0], w_path34[i+1][2][0]], [w_path34[i][2][1], w_path34[i+1][2][1]], "r", linewidth=2.5)

for i in range(len(w_path44) - 1):
    plt.plot([w_path44[i][2][0], w_path44[i+1][2][0]], [w_path44[i][2][1], w_path44[i+1][2][1]], "r", linewidth=2.5)

# for i in range(len(w_path54) - 1):
#     plt.plot([w_path54[i][2][0], w_path54[i+1][2][0]], [w_path54[i][2][1], w_path54[i+1][2][1]], "r", linewidth=2.5)
#################################################################################################################### 잎 구분 0.75

plt.gca().invert_yaxis()
plt.show()

