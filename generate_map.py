import numpy as np
import cv2
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# def pmap():
#
#     size = 3
#
#     img = Image.open('map/map01.png')
#     imgArray = np.array(img)
#
#     map = np.zeros((imgArray.shape[1],imgArray.shape[0]))
#     for i in range(imgArray.shape[0]):
#         for j in range(imgArray.shape[1]):
#             if imgArray[i][j][3] == 255:
#                 map[i][j] = 1
#
#     map = np.transpose(map)
#     map = 1 - map
#
#     kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]).astype(np.uint8)
#     map = cv2.dilate(map, kernel, iterations=1)
#
#     f1 = np.zeros((map.shape[0], 1))
#     for i in range(size):
#         map = np.hstack((map, f1))
#         map = np.hstack((f1, map))
#
#     f2 = np.zeros((1, map.shape[1]))
#     for i in range(size):
#         map = np.vstack((map, f2))
#         map = np.vstack((f2, map))
#
#     kernel_map = np.array([])
#     for i in range(size, map.shape[0] - size):
#         for j in range(size, map.shape[1] - size):
#             kernel_map = np.append(kernel_map,
#                                    np.sum(map[i - size:i + size + 1, j - size:j + size + 1]) / ((2 * size + 1) ** 2))
#
#     kernel_map = np.reshape(kernel_map, (map.shape[0] - 2 * size, map.shape[1] - 2 * size))
#
#     return kernel_map
def map():

    map = np. ones((51,51,51))

    map[0:51, 25:51, 0:51] = 0

    return map

def pmap():
    image_size = 50
    box_size = 3
    no_box = 20
    image = Image.new('RGB', (image_size, image_size))
    d = ImageDraw.Draw(image)

    np.random.seed(6)  # 6
    for i in range(no_box):
        xy = np.random.randint(image_size, size=2)
        rgb = np.random.randint(155, size=3)
        # print(rgb)
        # print(xy)
        d.rectangle([xy[0], xy[1], xy[0] + box_size, xy[1] + box_size], fill=(rgb[0], rgb[1], rgb[2]))

    d.rectangle([11, 50, 18, 20], fill=(255, 255, 255))
    d.rectangle([11, 10, 18, 0], fill=(255, 255, 255))
    d.rectangle([32, 50, 39, 40], fill=(255, 255, 255))
    d.rectangle([32, 30, 39, 0], fill=(255, 255, 255))
    # image.show()
    imgArray = np.array(image)

    map = np.zeros((imgArray.shape[1], imgArray.shape[0]))
    for i in range(imgArray.shape[0]):
        for j in range(imgArray.shape[1]):
            map[i][j] = (int(imgArray[i][j][0]) + int(imgArray[i][j][1]) + int(imgArray[i][j][2])) / (255 * 3)

    map = np.transpose(map)
    map = 1 - map

    return map

def Reshape_map(map):

    r_map = []
    for i in range(map.shape[0]):
        for j in range(map.shape[1]):
            for k in range(map.shape[2]):
                r_map.append([i, j, k, map[i,j,k]])

    return r_map

def BARN113():

    map = np.load("map/grid_113.npy").astype(np.uint8)
    map = 1 - map

    return map

def BARN_map_113(size, classify):

    map = np.load("map/grid_113.npy").astype(np.uint8)
    map = np.repeat(map, 4, axis=1)
    map = np.repeat(map, 4, axis=0)

    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]).astype(np.uint8)
    map = cv2.dilate(map, kernel, iterations=1)
    map = (map).astype(np.float)

    if classify == True:
        # for i in range(80, 95):
        #     for j in range(15, 30):
        #         map[i, j] = 0.3 * map[i, j]
        #
        # for i in range(60, 80):
        #     for j in range(40, 60):
        #         map[i, j] = 0.3 * map[i, j]


        for i in range(30, 80):
            for j in range(60, 100):
                map[i, j] = 0.2 * map[i, j]

        for i in range(80,100):
            for j in range(40, 60):
                map[i, j] = 0.2 * map[i, j]
        #
        for i in range(90,110):
            for j in range(17, 40):
                map[i, j] = 0.2 * map[i, j]


    map = 1 - map

    f1 = np.zeros((map.shape[0], 1))
    for i in range(size):
        map = np.hstack((map, f1))
        map = np.hstack((f1, map))

    f2 = np.zeros((1, map.shape[1]))
    for i in range(size):
        map = np.vstack((map, f2))
        map = np.vstack((f2, map))

    kernel_map = np.array([])
    for i in range(size, map.shape[0] - size):
        for j in range(size, map.shape[1] - size):
            kernel_map = np.append(kernel_map, np.sum(map[i - size:i + size + 1, j - size:j + size + 1]) / ((2 * size + 1) ** 2))

    kernel_map = np.reshape(kernel_map, (map.shape[0] - 2 * size, map.shape[1] - 2 * size))

    return kernel_map


# plt.figure(figsize=(10,10))
# plt.axes().set_aspect('equal')
# map = pmap()
#
# plt.imshow(np.transpose(map),cmap = "gray", interpolation = 'nearest')
# plt.show()