import numpy as np
from UR5e_DH import T01, T12, T23, T34, T45, T56
import plotly.graph_objs as go

def draw_mesh(joint_position, joint_rotate, height): #joint_position, joint_rotate, height
    # joint_rotate = np.array([[[1,0,0],[0,1,0],[0,0,1]]])
    # joint_position = np.array([[100,0,0]])
    # height = np.array([])
    #
    # t01 = T01()
    # joint_rotate = np.append(joint_rotate, [t01[0:3,0:3]], axis = 0)
    # joint_position = np.append(joint_position, [t01[0:3,3]+[100,0,0]], axis = 0)
    # height = np.append(height, np.linalg.norm(joint_position[1] - joint_position[0]))
    #
    #
    # t02 = np.dot(t01, T12())
    # joint_rotate = np.concatenate((joint_rotate, [t02[0:3,0:3]]), axis = 0)
    # joint_position = np.append(joint_position, [t02[0:3,3]+[100,0,0]], axis = 0)
    # height = np.append(height, np.linalg.norm(joint_position[2] - joint_position[1]))
    #
    # t03 = np.dot(t02, T23())
    # joint_rotate = np.concatenate((joint_rotate, [t03[0:3,0:3]]), axis = 0)
    # joint_position = np.append(joint_position, [t03[0:3,3]+[100,0,0]], axis = 0)
    # height = np.append(height, np.linalg.norm(joint_position[3] - joint_position[2]))
    #
    # t04 = np.dot(t03, T34(-np.pi/2))
    # joint_rotate = np.concatenate((joint_rotate, [t04[0:3,0:3]]), axis = 0)
    # joint_position = np.append(joint_position, [t04[0:3,3]+[100,0,0]], axis = 0)
    # height = np.append(height, np.linalg.norm(joint_position[4] - joint_position[3]))
    #
    # t05 = np.dot(t04, T45())
    # joint_rotate = np.concatenate((joint_rotate, [t05[0:3,0:3]]), axis = 0)
    # joint_position = np.append(joint_position, [t05[0:3,3]+[100,0,0]], axis = 0)
    # height = np.append(height, np.linalg.norm(joint_position[5] - joint_position[4]))
    #
    # t06 = np.dot(t05, T56())
    # joint_rotate = np.concatenate((joint_rotate, [t06[0:3,0:3]]), axis = 0)
    # joint_position = np.append(joint_position, [t06[0:3,3]+[100,0,0]], axis = 0)
    # height = np.append(height, np.linalg.norm(joint_position[6] - joint_position[5]))

    volume = [10,8,7,7,7,7]

    vertex = np.array([[[0,0,0],[0,0,0]]])

    vertex01 = np.array([[-volume[0]/2, -height[0], -volume[0]/2],[volume[0]/2, 0, volume[0]/2]])
    vertex = np.append(vertex, [vertex01], axis = 0)
    vertex12 = np.array([[0, -volume[1]/2, 14-volume[1]/2],[height[1], volume[1]/2, 14+volume[1]/2]])
    vertex = np.append(vertex, [vertex12], axis = 0)
    vertex23 = np.array([[0, -volume[2]/2, -volume[2]/2],[height[2], volume[2]/2, volume[2]/2]])
    vertex = np.append(vertex, [vertex23], axis = 0)
    vertex34 = np.array([[-volume[3]/2, -height[3], -volume[3]/2],[volume[3]/2, 0, volume[3]/2]])
    vertex = np.append(vertex, [vertex34], axis = 0)
    vertex45 = np.array([[-volume[4]/2, 0, -volume[4]/2],[volume[4]/2, height[4], volume[4]/2]])
    vertex = np.append(vertex, [vertex45], axis = 0)
    vertex56 = np.array([[-volume[5]/2, -volume[5]/2, -height[5]],[volume[5]/2, volume[5]/2, 0]])
    vertex = np.append(vertex, [vertex56], axis = 0)


    data = []
    mesh = []
    cc = []
    color = ["white","black","red","blue","green","green","green"]
    for i in range(1,len(vertex)):

        vertex_x = np.linspace(vertex[i][0][0], vertex[i][1][0], abs(int(np.ceil(vertex[i][1][0] - vertex[i][0][0]))) + 1)
        vertex_y = np.linspace(vertex[i][0][1], vertex[i][1][1], abs(int(np.ceil(vertex[i][1][1] - vertex[i][0][1]))) + 1)
        vertex_z = np.linspace(vertex[i][0][2], vertex[i][1][2], abs(int(np.ceil(vertex[i][1][2] - vertex[i][0][2]))) + 1)


        for c in vertex_z:
            for b in vertex_y:
                for a in vertex_x:

                    mesh.append(np.dot(joint_rotate[i],[a, b, c]) + joint_position[i])
                    cc.append(color[i])

    mesh = np.array(mesh)
    mesh = np.round(mesh).astype("int32")

    trace0 = go.Scatter3d(x=mesh[:, 0], y=mesh[:, 1], z=mesh[:, 2], marker=dict(color=cc, size=2), mode='markers')

    return trace0

