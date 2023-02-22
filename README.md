# Path planning Algorithm
- Uniform sampling -> Bias sampling
- Geometrical collision checking process -> Collision avoidance through cost function comparison

## rrt_star2D, rrt_star3D, rrt_star_general
[rrt_star code] https://github.com/Ytodi31/Motion_planning_RRT_star

### node()
- Class include node position, cost,  parent 

### Sampling()
- Bias sampling 
- Generate the random node x_rand depending on the probability of the map

### Steer()
- Get x_new from x_rand and x_nearest

### New_Check()
- Check whether or not to add to the tree of X-new using the probability

### Distance cost(), Obstacle cost(), Line cost()
- Distance cost : Euclidean distance of x_new and x_nearest
- Obstacle cost : Average of Profilng the edge of x_new and x_nearest
- Line cost : Distance cost + Obstacle cost 
  - **If the path crosses an obstacle, it has a high cost**

### Add_parent()
- Create the edge with the lowest cost

### Rewire()
- Tree reconstruction

# UR5e
## RobotUR2D, RobotUR3D

### profile()

- vertex : Create an exterior point of a cuboid link
- Make robot body mesh using forward kinematics

```python
for c in vertex_z:
                for b in vertex_y:
                    for a in vertex_x:
                        mesh.append(np.dot(joint_rotate[i], [a, b, c]) + joint_position[i])
```

- Profiling the robot's posture on the obstacle map

```python
profile_map = mesh_map * (1-self.map)
```

<img src="https://user-images.githubusercontent.com/88310751/220538762-78fa6d55-1048-4143-b476-aa38c443eee4.jpg" width="400" height="300">

### construct_config_space()

- Generate configuration space

```python
for i in theta1:
  for j in theta2:
    for k in theta3:

      position, rotate, height = self.robot_position(i, j, k)

      profile = self.profile(position,rotate,height)
      prob = np.min(profile) # get minimum value from profile_map 

      configuration_space.append([i, j, k, prob])
```
