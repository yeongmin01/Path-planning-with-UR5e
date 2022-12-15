import numpy as np
from shapely.geometry import Polygon
from PIL import Image, ImageDraw

def Obstacle_generater(obstacle):

    obs = []
    for o in obstacle:
        obs.append(Polygon(o))

    return  obs

def Obstacle_center(obstacle):

    center = []
    for o in obstacle:
        c_x = (o[0][0] + o[2][0]) / 2
        c_y = (o[0][1] + o[2][1]) / 2

        c = [c_x, c_y]
        center.append(c)

    return center

def Collision_range(obstacle):

    range = []
    for o in obstacle:

        d = np.linalg.norm(np.array(o[2]) - np.array(o[0]))
        range.append(d/2)

    return range

def Obstacle_BARN_113():

    map = np.load("map/grid_113.npy").astype(np.uint8)

    obs_center = []
    obs = []
    for i in range(map.shape[0]):
        for j in range(map.shape[1]):
            if map[i][j] == 1:
                obs_center.append([i,j])
                obs.append(((i-0.5, j-0.5),(i+0.5, j-0.5), (i+0.5, j+0.5), (i-0.5, j+0.5), (i-0.5, j-0.5)))

    return obs, obs_center

def bmap():

    # img = Image.open('map/map01.png')
    # imgArray = np.array(img)
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

    obs_center = []
    obs = []
    for i in range(imgArray.shape[0]):
        for j in range(imgArray.shape[1]):
            if imgArray[i][j][0] != 0 or imgArray[i][j][1] != 0 or imgArray[i][j][2] != 0:
                obs_center.append([j, i])
                obs.append(((j - 0.5, i - 0.5), (j + 0.5, i - 0.5), (j + 0.5, i + 0.5), (j - 0.5, i + 0.5),
                            (j - 0.5, i - 0.5)))

    return obs, obs_center