import numpy as np
import plotly.graph_objs as go
import math
import time
from UR5e_DH import T01, T12, T23, T34, T45, T56
from mesh_gen import draw_mesh

def map():
    map = np.zeros([201, 101, 101])

    map[80:120, 60:80, 0:100] = 1

    map[10:30, 80:100, 0:80] = 1
    map[10:30, 0:20, 0:80] = 1
    map[10:30, 0:100, 80:100] = 1

    map[170:190, 80:100, 0:80] = 1
    map[170:190, 0:20, 0:80] = 1
    map[170:190, 0:100, 80:100] = 1

    return 1 - map

def draw_map(map):
    data = []

    r_map = []
    for i in range(map.shape[0]):
        for j in range(map.shape[1]):
            for k in range(map.shape[2]):
                r_map.append([i, j, k, map[i, j, k]])

    obstacle = []
    for i in range(len(r_map)):  # Draw map
        if r_map[i][3] != 1:
            obstacle.append(r_map[i])

    print(len(obstacle))
    obstacle = np.array(obstacle)
    rgb = np.load("3D-data/rgb.npy")

    trace1 = go.Scatter3d(x=obstacle[:, 0], y=obstacle[:, 1], z=obstacle[:, 2], marker=dict(
        color=['rgb({},{},{})'.format(r, g, b) for r, g, b in
               zip(255 * obstacle[:, 3], 255 * obstacle[:, 3], 255 * obstacle[:, 3])], size=1), mode='markers')

    data.append(trace1)

    trace2 = go.Scatter3d(x=[0], y=[0], z=[0], marker=dict(color="black", size=0.1), mode='markers')
    data.append(trace2)

    trace3 = go.Scatter3d(x=[map.shape[0]], y=[map.shape[1]], z=[map.shape[2]], marker=dict(color="black", size=0.1),
                          mode='markers')
    data.append(trace3)

    # fig = go.Figure(data=data)
    # fig.show()
    return data

def h_map():

    # whd = np.load("xyz.npy") * 100  # z axis 0.25m, y axis 1.75cm

    # x = np.array([np.round(-whd[:, 2] + 100).astype("int32")])
    # y = np.array([np.round(whd[:, 0] + (100 + 1.75)).astype("int32")])
    # z = np.array([np.round(-whd[:, 1] + (50-10)).astype("int32")])
    # xyz = np.concatenate((np.concatenate((x.T, y.T), axis=1), z.T), axis=1)

    xyz = np.load("xyz.npy").astype("int32")
    rgb = np.load("rgb.npy")

    map = np.zeros([201, 201, 101])

    for i in range(len(xyz)):
        x, y, z = xyz[i,0], xyz[i,1], xyz[i,2]
        if x >= 0 and x < 201 and y >= 0 and y < 201 and z >= 0 and z < 101:
            #(rgb[i,0] + rgb[i,1] + rgb[i,2])/(255*3)  #xx = 1, xxx = rgb

            if rgb[i,0] >= rgb[i,1] and rgb[i,0] >= rgb[i,2] and rgb[i,0] >= 120 :
                map[x][y][z] = 1 # 1
            else:
                map[x][y][z] = 0.5


            # elif rgb[i,0] < 100  and rgb[i,1] >= 100 and rgb[i,1] <= 255:
            #     map[x][y][z] = 0.3 # 0.3
            # elif rgb[i,2] == 255 :
            #     map[x][y][z] = 0.6 # 0.6

    # map[0:37, 0:201, 0:101] = 0 # Tomato back wall
    # map[0:50, 0:200, 70:101] = 1 # tomato top barrier

    return 1-map

class Robot(object):

    def __init__(self, base_position = None, map = None):

        self.base_position = np.array(base_position)
        self.map = map
        self.volume = [10,8,7,7,7,7]

    def robot_position(self, theta1, theta2, theta3): # Calculate joint position, rotation, and distance between joints through kinematics

        joint_rotate = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]])
        joint_position = np.array([self.base_position])
        height = np.array([])

        theta1 = (theta1 * np.pi) / 180
        theta2 = (theta2 * np.pi) / 180
        theta3 = (theta3 * np.pi) / 180

        t01 = T01(theta1)
        joint_rotate = np.append(joint_rotate, [t01[0:3, 0:3]], axis=0)
        joint_position = np.append(joint_position, [t01[0:3, 3] + self.base_position], axis=0)
        height = np.append(height, np.linalg.norm(joint_position[1] - joint_position[0]))

        t02 = np.dot(t01, T12(theta2))
        joint_rotate = np.append(joint_rotate, [t02[0:3, 0:3]], axis=0)
        joint_position = np.append(joint_position, [t02[0:3, 3] + self.base_position], axis=0)
        height = np.append(height, np.linalg.norm(joint_position[2] - joint_position[1]))

        t03 = np.dot(t02, T23(theta3))
        joint_rotate = np.append(joint_rotate, [t03[0:3, 0:3]], axis=0)
        joint_position = np.append(joint_position, [t03[0:3, 3]+ self.base_position], axis=0)
        height = np.append(height, np.linalg.norm(joint_position[3] - joint_position[2]))

        t04 = np.dot(t03, T34(-np.pi / 2))
        joint_rotate = np.append(joint_rotate, [t04[0:3, 0:3]], axis=0)
        joint_position = np.append(joint_position, [t04[0:3, 3]+ self.base_position], axis=0)
        height = np.append(height, np.linalg.norm(joint_position[4] - joint_position[3]))

        t05 = np.dot(t04, T45(-np.pi/2))
        joint_rotate = np.append(joint_rotate, [t05[0:3, 0:3]], axis=0)
        joint_position = np.append(joint_position, [t05[0:3, 3]+ self.base_position], axis=0)
        height = np.append(height, np.linalg.norm(joint_position[5] - joint_position[4]))

        t06 = np.dot(t05, T56())
        joint_rotate = np.append(joint_rotate, [t06[0:3, 0:3]], axis=0)
        joint_position = np.append(joint_position, [t06[0:3, 3]+ self.base_position], axis=0)
        height = np.append(height, np.linalg.norm(joint_position[6] - joint_position[5]))

        return joint_position, joint_rotate, height

    def profile(self, joint_position, joint_rotate, height): # Profiling the robot's posture on the obstacle map

        for i in range(len(joint_position)):
            if joint_position[i,2] < 0 :
                return 0

        vertex = np.array([[[0, 0, 0], [0, 0, 0]]]) # Create an exterior point of a cuboid centered on the joint location

        vertex01 = np.array([[-self.volume[0] / 2, -height[0], -self.volume[0] / 2], [self.volume[0] / 2, 0, self.volume[0] / 2]])
        vertex = np.append(vertex, [vertex01], axis=0)
        vertex12 = np.array([[0, -self.volume[1] / 2, 14 - self.volume[1] / 2], [height[1], self.volume[1] / 2, 14 + self.volume[1] / 2]])
        vertex = np.append(vertex, [vertex12], axis=0)
        vertex23 = np.array([[0, -self.volume[2] / 2, -self.volume[2] / 2], [height[2], self.volume[2] / 2, self.volume[2] / 2]])
        vertex = np.append(vertex, [vertex23], axis=0)
        vertex34 = np.array([[-self.volume[3] / 2, -height[3], -self.volume[3] / 2], [self.volume[3] / 2, 0, self.volume[3] / 2]])
        vertex = np.append(vertex, [vertex34], axis=0)
        vertex45 = np.array([[-self.volume[4] / 2, 0, -self.volume[4] / 2], [self.volume[4] / 2, height[4], self.volume[4] / 2]])
        vertex = np.append(vertex, [vertex45], axis=0)
        vertex56 = np.array([[-self.volume[5] / 2, -self.volume[5] / 2, -height[5]], [self.volume[5] / 2, self.volume[5] / 2, 0]])
        vertex = np.append(vertex, [vertex56], axis=0)

        for i in range(1, len(vertex)): # Create a point inside the cuboid

            vertex_x = np.linspace(vertex[i][0][0], vertex[i][1][0],
                                   abs(int(np.ceil(vertex[i][1][0] - vertex[i][0][0]))) + 1)
            vertex_y = np.linspace(vertex[i][0][1], vertex[i][1][1],
                                   abs(int(np.ceil(vertex[i][1][1] - vertex[i][0][1]))) + 1)
            vertex_z = np.linspace(vertex[i][0][2], vertex[i][1][2],
                                   abs(int(np.ceil(vertex[i][1][2] - vertex[i][0][2]))) + 1)

            mesh = []

            for c in vertex_z:
                for b in vertex_y:
                    for a in vertex_x:
                        mesh.append(np.dot(joint_rotate[i], [a, b, c]) + joint_position[i])

        mesh = np.round(mesh).astype("int32")

        mesh_map = np.zeros([201, 201, 101])

        for i in mesh:
            x, y, z = i[0], i[1], i[2]
            if x >= 0 and x < 201 and y >= 0 and y < 201 and z >= 0 and z < 101:
                mesh_map[x][y][z] = 1

        profile_map = mesh_map * (1-self.map) # The product of a map representing a robot mesh and an obstacle map element

        return 1-profile_map #, 1-mesh_map

    def construct_config_space(self, grid = 361): # Generate configuration space

        configuration_space = []
        theta1, theta2, theta3 = np.linspace(-180, 180, grid), np.linspace(-180, 180, grid), np.linspace(-180, 180, grid)

        for i in theta1:
            print("theta1 : ", i)
            for j in theta2:
                for k in theta3:

                    position, rotate, height = self.robot_position(i, j, k)

                    profile = self.profile(position,rotate,height)
                    prob = np.min(profile)

                    configuration_space.append([i, j, k, prob])

                print("done")


        c_map = np.zeros((grid, grid, grid))
        for a in range(grid):
            for b in range(grid):
                for c in range(grid):
                    c_map[a][b][c] = configuration_space[a * grid * grid + b * grid + c][3]


        return c_map


# map= h_map()
# data =  draw_map(map)
# fig = go.Figure(data=data)
# fig.show()
#
# base_position = [150, 100, 0]
# robot = Robot(base_position, map)
# grid = 73
#
# s = time.time()
# configuration = robot.construct_config_space(grid)
# np.save("config3D(73,real)", configuration)
# e = time.time()
# print(e-s)
# #
# configuration = np.load("config3D(73,real).npy")
# data = draw_map(configuration)
# fig = go.Figure(data=data)
# fig.show()
#
# data =  draw_map(map)
#
# p1, r1, h1 = robot.robot_position(0,-90,0)
# data.append(draw_mesh(p1,r1,h1))
#
# p2, r2, h2 = robot.robot_position(-25,-20,-15)
# data.append(draw_mesh(p2,r2,h2))

# fig = go.Figure(data=data)
# fig.show()
#
# data = draw_map(np.load("config3D(13,real).npy"))
#
# trace0 = go.Scatter3d(x=[36], y=[18], z=[36], marker=dict(color="blue", size=5), mode='markers')
# data.append(trace0)
#
# trace1 = go.Scatter3d(x=[31], y=[32], z=[33], marker=dict(color="red", size=5), mode='markers')
# data.append(trace1)
#
# fig = go.Figure(data=data)
# fig.show()

