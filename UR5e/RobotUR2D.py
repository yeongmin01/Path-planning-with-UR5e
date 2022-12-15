import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import profile_line
from PIL import Image, ImageDraw
from UR5e_DH import T01, T12, T23, T34, T45, T56


# def map():
#     map = np.zeros([201, 201])
#
#     map[145:175, 50:70] = 1#0.5
#     map[50:70, 60:80] = 1
#     map[25:60, 40:60] = 1
#     map[50:70, 150:175] = 1
#     map[160:180, 160:180] = 1
#
#     return 1 - map

def map():
    map = np.zeros([201, 101])

    map[145:175, 50:70] = 0.5
    map[50:70, 60:80] = 1
    map[25:60, 40:60] = 1

    return 1 - map

def pmap():
    image = Image.open("image/map1.png").convert("RGB")
    # image = Image.open("image/map2.png").convert("RGB")
    map = np.asarray(image)

    r_map = np.zeros([201, 101])
    for i in range(np.shape(map)[0]):
        for j in range(np.shape(map)[1]):
            if map[i][j][0] >= 200 and map[i][j][1] <= 100 and map[i][j][2] <= 100:
                r_map[i][j] = 1
            elif map[i][j][0] <= 100 and map[i][j][1] <= 100 and map[i][j][2] >= 200:
                r_map[i][j] = 1#0.8
            elif map[i][j][0] <= 100 and map[i][j][1] >= 200 and map[i][j][2] <= 100:
                r_map[i][j] = 1#0.3
            # elif map[i][j][0] <= 150 and map[i][j][1] <= 150 and map[i][j][2] <= 150:
            #     r_map[i][j] = 0.8

    return 1 - r_map

class Robot(object):

    def __init__(self, base_position = None, map = None):

        self.base_position = base_position
        self.map = map

    def robot_position(self, theta2, theta3):

        position = []
        theta2 = (theta2 * np.pi) / 180
        theta3 = (theta3 * np.pi) / 180

        t01 = T01(0)
        x1 = self.base_position[0] + t01[0,3]
        z1 = self.base_position[1] + t01[2,3]
        # z1 = self.base_position[1] + 16.25
        position.append([x1, z1])

        t02 = np.dot(t01, T12(theta2))
        x2 = self.base_position[0] + t02[0,3]
        z2 = self.base_position[1] + t02[2,3]
        position.append([x2, z2])

        t03 = np.dot(t02, T23(theta3))
        t04 = np.dot(t03, T34(-np.pi/2))
        t05 = np.dot(t04, T45())

        t06 = np.dot(t05, T56())
        x3 = self.base_position[0] + t06[0,3]
        z3 = self.base_position[1] + t06[2,3]
        position.append([x3, z3])

        return position

    def construct_config_space(self, grid = 361):

        configuration_space = []
        theta1, theta2 = np.linspace(-180, 180, grid), np.linspace(-180, 180, grid)

        for i in theta1:
            for j in theta2:

                prob = 0
                robot_position = self.robot_position(i, j)

                if robot_position[1][1] <= self.base_position[1] or robot_position[2][1] <= self.base_position[1]:
                    prob = 0

                else:
                    profile_prob1 = np.ravel(profile_line(self.map, robot_position[0], robot_position[1], linewidth=8, order=0, reduce_func=None))
                    profile_prob2 = np.ravel(profile_line(self.map, robot_position[1], robot_position[2], linewidth=7, order=0, reduce_func=None))

                    profile_prob = np.concatenate((profile_prob1, profile_prob2))
                    if 0 in profile_prob:
                        prob = 0
                    else:
                        prob = np.min(profile_prob)

                # profile_prob1 = np.ravel(profile_line(self.map, robot_position[0], robot_position[1], linewidth=8, order=0, reduce_func=None))
                # profile_prob2 = np.ravel(profile_line(self.map, robot_position[1], robot_position[2], linewidth=7, order=0, reduce_func=None))
                #
                # profile_prob = np.concatenate((profile_prob1, profile_prob2))
                # if 0 in profile_prob:
                #     prob = 0
                # else:
                #     prob = np.min(profile_prob)

                configuration_space.append([i, j, prob])
            print("done")

        c_map = np.zeros((361,361))
        for i in range(361):
            for j in range(361):
                c_map[i][j] = configuration_space[i*361+j][2]

        return c_map



# map = map()
#
# map = pmap()
#
# base_position = [90, 0]
# robot = Robot(base_position, map)
#
# plt.figure(figsize=(10,5))
# plt.axes().set_aspect('equal')
# r1 = robot.robot_position(0,0)
# r2 = robot.robot_position(-85,-85)
# plt.plot([robot.base_position[0], r1[0][0]],[robot.base_position[1], r1[0][1]] , "y", linewidth=9)
# plt.plot([r1[0][0], r1[1][0]],[r1[0][1], r1[1][1]] , "b", linewidth=8)
# plt.plot([r1[1][0], r1[2][0]],[r1[1][1], r1[2][1]] , "r", linewidth=8)
#
# plt.plot([robot.base_position[0], r2[0][0]],[robot.base_position[1], r2[0][1]] , "y", linewidth=9)
# plt.plot([r2[0][0], r2[1][0]],[r2[0][1], r2[1][1]] , "b", linewidth=8)
# plt.plot([r2[1][0], r2[2][0]],[r2[1][1], r2[2][1]] , "r", linewidth=8)
#
# plt.imshow(np.transpose(map),cmap = "gray", interpolation = 'nearest')
# plt.gca().invert_yaxis()
# plt.show()
# #
# configuration = robot.construct_config_space()
# np.save("2D-data/config2D(잎구분x)", configuration)
#
# plt.figure(figsize=(12,10))
# plt.axes().set_aspect('equal')
# plt.imshow(np.transpose(configuration),cmap = "gray", interpolation = 'nearest')
# plt.scatter(180,180,s=50,c="blue")
# plt.scatter(95,95,s=50,c="red")
# plt.gca().invert_yaxis()
# plt.show()

