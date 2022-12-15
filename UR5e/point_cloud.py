import numpy as np
from UR5e_DH import T01, T12, T23, T34, T45, T56

x = np.array([1,2,3]) * 2

T6e = np.array([[-1, 0, 0, 3.3],
                [0, -1, 0, 10],
                [0, 0, 1, -3],
                [0, 0, 0, 1]])

theta = np.array([-233, -6, -37, -137, 98, -180]) * (np.pi/180)
t01 = T01(theta[0])
t02 = np.dot(t01,T12(theta[1]))
t03 = np.dot(t02,T23(theta[2]))
t04 = np.dot(t03,T34(theta[3]))
t05 = np.dot(t04,T45(theta[4]))
t06 = np.dot(t05,T56(theta[5]))
t0e = np.dot(t06,T6e)
#
#
# rgb1 = np.load("3D-data/real_camera_data/rgb1.npy")
rgb2 = np.load("3D-data/real_camera_data/rgb2.npy")
#
# whd1 = np.load("3D-data/real_camera_data/whd1.npy") * 100
# add1 = np.ones(len(whd1))
# whd1 = np.c_[whd1, add1]
#
whd2 = np.load("3D-data/real_camera_data/whd2.npy") * 100
add2 = np.ones(len(whd2))
whd2 = np.c_[whd2, add2]
#
# xyz1 = np.dot(t0e,whd1.T).T
xyz2 = np.dot(t0e,whd2.T).T
#
base_position = np.array([150,100,0])
xyz = xyz2[:,0:3] + base_position

np.save("xyz", xyz)
np.save("rgb", rgb2)
# xyz = np.concatenate((xyz1, xyz2), axis=0)
# xyz = xyz[:,0:3] + base_position
# rgb = np.concatenate((rgb1, rgb2), axis=0)
# print(np.min(xyz[:,0]))
#
# np.save("xyz", xyz)
# np.save("rgb", rgb)


