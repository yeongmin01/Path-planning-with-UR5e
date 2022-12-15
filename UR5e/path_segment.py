import numpy as np

def segment(p,seg_point):

    path = []
    for i in range(len(p)-1):

        v = (p[i+1] - p[i]) / seg_point

        for j in range(seg_point):
            path.append(p[i] + j*v)

    path.append(p[-1])

    return path

# x = ((np.load("path(73,leaf-x).npy") * 5 - 180) * np.pi)/180
# np.save("joint_value(73,leaf-x)", x)
# np.savetxt("joint_value_rad", x, fmt='%f')

x = np.load("path(73,real).npy") * 5 - 180
joint_value = []
for i in range(1, len(x)-1):
    d = np.linalg.norm(x[i] - x[i+1])

    seg_point = int(d)

    if seg_point ==0:
        joint_value.append(x[i]*np.pi/180)
        joint_value.append(x[i+1]*np.pi/180)

    else :
        v = (x[i+1] - x[i]) / seg_point

        for j in range(seg_point):
            joint_value.append((x[i] + j*v)*np.pi/180 )

joint_value.append(x[-1]*np.pi/180)

np.savetxt("joint_value_rad(73,real)", joint_value, fmt='%f')