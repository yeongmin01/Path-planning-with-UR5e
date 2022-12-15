import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import profile_line
from PIL import Image, ImageDraw

def map():
    map = np.zeros([31, 31])
    map[22:28, 10:21] = 1
    map[4:11, 10:21] = 1



    return 1 - map

def pmap():
    image_size = 30
    box_size = 4
    no_box = 10
    image = Image.new('RGB', (image_size, image_size))
    d = ImageDraw.Draw(image)

    np.random.seed(9)  # 9
    for i in range(no_box):
        xy = np.random.randint(image_size, size=2)
        rgb = np.random.randint(255, size=3)
        d.rectangle([xy[0], xy[1], xy[0] + box_size, xy[1] + box_size], fill=(rgb[0], rgb[1], rgb[2]))
        # d.rectangle([xy[0], xy[1], xy[0] + box_size, xy[1] + box_size], fill=(255,255,255))
    d.rectangle([17, 0, 22, 15], fill=(255, 255, 255))
    # d.rectangle([11, 50, 18, 20], fill=(255, 255, 255))
    # d.rectangle([11, 10, 18, 0], fill=(255, 255, 255))
    # d.rectangle([32, 50, 39, 40], fill=(255, 255, 255))
    # d.rectangle([32, 30, 39, 0], fill=(255, 255, 255))

    imgArray = np.array(image)

    map = np.zeros((imgArray.shape[1], imgArray.shape[0]))
    for i in range(imgArray.shape[0]):
        for j in range(imgArray.shape[1]):
            map[i][j] = (int(imgArray[i][j][0]) + int(imgArray[i][j][1]) + int(imgArray[i][j][2])) / (255 * 3)

    map = np.transpose(map)
    map = 1 - map

    return map

class Robot(object):

    def __init__(self, base_position = None, link_lenths = None, map = None):

        self.base_position = base_position
        self.link_lenths = np.array(link_lenths)
        self.map = map

    def robot_position(self, theta1, theta2):

        position = []
        position.append(self.base_position)

        theta1 = (theta1 * np.pi) / 180
        theta2 = (theta2 * np.pi) / 180

        x1 = self.base_position[0] + self.link_lenths[0] * np.cos(theta1)
        y1 = self.base_position[1] + self.link_lenths[0] * np.sin(theta1)

        position.append([x1, y1])

        x2 = self.base_position[0] + self.link_lenths[0] * np.cos(theta1) + self.link_lenths[1] * np.cos(theta1 + theta2)
        y2 = self.base_position[1] + self.link_lenths[0] * np.sin(theta1) + self.link_lenths[1] * np.sin(theta1 + theta2)

        position.append([x2, y2])

        return position
    def construct_config_space(self, grid = 361):

        configuration_space = []
        theta1, theta2 = np.linspace(0, 360, grid), np.linspace(0, 360, grid)

        for i in theta1:
            for j in theta2:

                robot_position = self.robot_position(i, j)

                prob = 0
                profile_prob1 = profile_line(self.map, robot_position[0], robot_position[1], linewidth=2, order=0, reduce_func=None)
                profile_prob2 = profile_line(self.map, robot_position[1], robot_position[2], linewidth=2, order=0, reduce_func=None)
                profile_prob = np.concatenate((profile_prob1, profile_prob2))
                if 0 in profile_prob:
                    prob = 0
                else:
                    prob = np.min(profile_prob)
                # for k in range(2):
                #     profile_prob = profile_line(self.map, robot_position[k], robot_position[k+1], linewidth=2, order = 0, reduce_func = None)
                #
                #     if 0 in profile_prob:
                #         prob = 0
                #         break
                #
                #     #prob += profile_prob.sum() / (np.shape(profile_prob)[0] * np.shape(profile_prob)[1] * 2)
                #     prob = np.min(profile_prob) / 2

                configuration_space.append([i, j, prob])

                if i*j %1000 == 0:
                    print("done")

        c_map = np.zeros((361,361))
        for i in range(361):
            for j in range(361):
                c_map[i][j] = configuration_space[i*361+j][2]

        return c_map

map = map()
# plt.figure(figsize=(10,10))
# plt.axes().set_aspect('equal')
# plt.imshow(np.transpose(map),cmap = "gray", interpolation = 'nearest')
# plt.show()
#
# map = pmap()
# plt.figure(figsize=(10,10))
# plt.axes().set_aspect('equal')
# plt.imshow(np.transpose(map),cmap = "gray", interpolation = 'nearest')
# plt.show()

base_position = [15, 15]
link_lenths = [5, 5]
robot = Robot(base_position, link_lenths, map)
configuration = robot.construct_config_space()

plt.figure(figsize=(10,10))
plt.axes().set_aspect('equal')
plt.imshow(np.transpose(configuration),cmap = "gray", interpolation = 'nearest')
plt.show()

plt.figure(figsize=(10,10))
plt.axes().set_aspect('equal')
r1 = robot.robot_position(90,0)
plt.plot([robot.base_position[0], r1[0][0]],[robot.base_position[1], r1[0][1]] , "b", linewidth=8)
plt.plot([r1[0][0], r1[1][0]],[r1[0][1], r1[1][1]] , "b", linewidth=8)
plt.plot([r1[1][0], r1[2][0]],[r1[1][1], r1[2][1]] , "r", linewidth=8)


plt.imshow(np.transpose(map),cmap = "gray", interpolation = 'nearest')
plt.gca().invert_yaxis()
plt.show()

# x = np.array([[1,1,1],[3,3,3],[2,2,2]])
# print(np.min(x))