import numpy as np
import cv2
import plotly.graph_objs as go

# whd = np.load("xyz.npy")*100 # z축으로 0.25m, y축 으로 1.75cm
#
# x = np.array([np.round(whd[:,2]).astype("int32")])
# y = np.array([np.round(-whd[:,0] + (100-1.75)).astype("int32")])
# z = np.array([np.round(-whd[:,1] + 50).astype("int32")])
# xyz = np.concatenate((np.concatenate((x.T, y.T), axis = 1),z.T),axis = 1)
# print(z)
rgb = np.load("rgbc.npy")
np.savetxt("rgb", rgb, fmt="%f")
# map = np.zeros([101, 201, 101])
#
# for i in xyz:
#     x, y, z = i[0], i[1], i[2]
#     if x >= 0 and x < 101 and y >= 0 and y < 201 and z >= 0 and z < 101:
#         map[x][y][z] = 1

# data = []
# trace0 = go.Scatter3d(x=xyz[:,0], y=xyz[:,1], z=xyz[:,2], marker=dict(color=['rgb({},{},{})'.format(r, g, b) for r, g, b in zip(rgb[:,0], rgb[:,1], rgb[:,2])], size=2), mode='markers')
# data.append(trace0)
#
# trace1 = go.Scatter3d(x=[min(x)], y=[min(y)], z=[min(z)], marker=dict(color="black", size=1), mode='markers')
# data.append(trace1)
#
# trace2 = go.Scatter3d(x=[max(x)], y=[max(y)], z=[max(z)], marker=dict(color="black", size=1), mode='markers')
# data.append(trace2)
#
# fig = go.Figure(data=data)
# fig.show()