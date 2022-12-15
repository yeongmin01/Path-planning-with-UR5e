import numpy as np
import matplotlib.pyplot as plt
from RobotUR2D import Robot, pmap
import cv2

iteration = 5000
map = pmap()

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
    w_path11.append(position11)
for b in path21:
    position21 = robot.robot_position(b[0] - 180, b[1] - 180)
    w_path21.append(position21)
for c in path31:
    position31 = robot.robot_position(c[0] - 180, c[1] - 180)
    w_path31.append(position31)
for d in path41:
    position41 = robot.robot_position(d[0] - 180, d[1] - 180)
    w_path41.append(position41)
for e in path51:
    position51 = robot.robot_position(e[0] - 180, e[1] - 180)
    w_path51.append(position51)

for a in path14:
    position14 = robot.robot_position(a[0] - 180, a[1] - 180)
    w_path14.append(position14)
for b in path24:
    position24 = robot.robot_position(b[0] - 180, b[1] - 180)
    w_path24.append(position24)
for c in path34:
    position34 = robot.robot_position(c[0] - 180, c[1] - 180)
    w_path34.append(position34)
for d in path44:
    position44 = robot.robot_position(d[0] - 180, d[1] - 180)
    w_path44.append(position44)
for e in path54:
    position54 = robot.robot_position(e[0] - 180, e[1] - 180)
    w_path54.append(position54)

def Overlap(w_path, map): # 장애물 1
    point1 = np.array([[w_path[0][0][0], w_path[0][0][1]]], np.int32)
    point2 = np.array([[w_path[-1][1][0], w_path[-1][1][1]]], np.int32)
    p = np.int32(np.array(w_path)[:, 2])
    p = np.append(point1, p, axis=0)
    p = np.append(point2, p, axis=0)

    img = np.zeros([101, 201], np.uint8)
    img = cv2.fillPoly(img, [p], [255, 255, 255])

    overlap = np.transpose(1 - map) * img
    # img = cv2.flip(img11,0)
    # cv2.imshow("fillPoly",img)
    # cv2.waitKey(0)
    # cv2.destroyWindow()

    return overlap

overlap11 = Overlap(w_path11, map)
overlap21 = Overlap(w_path21, map)
overlap31 = Overlap(w_path31, map)
overlap41 = Overlap(w_path41, map)
overlap51 = Overlap(w_path51, map)

overlap14 = Overlap(w_path14, map)
overlap24 = Overlap(w_path24, map)
overlap34 = Overlap(w_path34, map)
overlap44 = Overlap(w_path44, map)
overlap54 = Overlap(w_path54, map)

overlap1 = [np.sum(overlap11)/255, np.sum(overlap21)/255, np.sum(overlap31)/255, np.sum(overlap41)/255, np.sum(overlap51)/255]
overlap4 = [np.sum(overlap14)/255, np.sum(overlap24)/255, np.sum(overlap34)/255, np.sum(overlap44)/255, np.sum(overlap54)/255]

print(np.mean(overlap1), np.mean(overlap4))
print(np.std(overlap1), np.std(overlap4))

# point1 = np.array([[w_path11[0][0][0],w_path11[0][0][1]]], np.int32)
# point2 = np.array([[w_path11[-1][1][0],w_path11[-1][1][1]]], np.int32)
# p11 = np.int32(np.array(w_path11)[:,2])
# p11 = np.append(point1, p11, axis=0)
# p11 = np.append(point2, p11, axis=0)
# img11 = np.zeros([101,201],np.uint8)
# img11 = cv2.fillPoly(img11,[p11],[255,255,255])
#
# plt.figure(figsize=(10,5))
# plt.axes().set_aspect('equal')
# plt.imshow(img11,cmap = "gray", interpolation = 'nearest')
# plt.colorbar()
# plt.show()
#
# overlap = np.transpose(1-map) * img11
#
# plt.figure(figsize=(10,5))
# plt.axes().set_aspect('equal')
# plt.imshow(overlap,cmap = "gray", interpolation = 'nearest')
# plt.colorbar()
# plt.show()
#
# print(np.sum(overlap)/255)



