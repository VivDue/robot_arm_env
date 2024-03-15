import matplotlib.pyplot as plt
import numpy as np

import matplotlib.animation as animation

# Fixing random state for reproducibility
np.random.seed(19680801)


def update_links(ax, joints):
    ax.plot(joints[:,0], joints[:,1], joints[:,2])
    ax.legend()
    # use circles to represent joints
    # use euler coordinates to represent joint orientation


def update_lines(num, walks, lines):
    for line, walk in zip(lines, walks):
        # NOTE: there is no .set_data() for 3 dim data...
        line.set_data(walk[num-5:num, :2].T)
        line.set_3d_properties(walk[num-5:num, 2])
    return lines

T = np.zeros(shape=(6, 4, 4))

def transform(n):
    a = np.array([0, -0.24355, -0.2132, 0, 0, 0])
    d = np.array([0.15185, 0, 0, 0.13105, 0.08535, 0.0921])
    alpha = np.array([90, 0, 0, 90, -90, 0])
    tetha = np.array([0, 0, 0, 0, 0, 0])
    alpha = np.deg2rad(alpha)
    tetha = np.deg2rad(tetha)
    T[n, 0, 0] = np.cos(tetha[n])
    T[n, 0, 1] = -1 * np.sin(tetha[n]) * np.cos(alpha[n])
    T[n, 0, 2] = np.sin(tetha[n]) * np.sin(alpha[n])
    T[n, 0, 3] = np.cos(tetha[n]) * a[n]

    T[n, 1, 0] = np.sin(tetha[n])
    T[n, 1, 1] = np.cos(tetha[n]) * np.cos(alpha[n])
    T[n, 1, 2] = -1 * np.cos(tetha[n]) * np.sin(alpha[n])
    T[n, 1, 3] = np.sin(tetha[n]) * a[n]

    T[n, 2, 0] = 0
    T[n, 2, 1] = np.sin(alpha[n])
    T[n, 2, 2] = np.cos(alpha[n])
    T[n, 2, 3] = d[n]

    T[n, 3, 0] = 0
    T[n, 3, 1] = 0
    T[n, 3, 2] = 0
    T[n, 3, 3] = 1
# Data: 40 random walks as (num_steps, 3) arrays
for n in range(6):
    transform(n)
joints = np.zeros(shape=(7, 4, 4))

joints[1] = T[0]
joints[2] = T[0] @ T[1]
joints[3] = T[0] @ T[1] @ T[2]
joints[4] = T[0] @ T[1] @ T[2] @ T[3]
joints[5] = T[0] @ T[1] @ T[2] @ T[3] @ T[4]
joints[6] = T[0] @ T[1] @ T[2] @ T[3] @ T[4] @ T[5]

print(joints)

# Attaching 3D axis to the figure
fig = plt.figure()
ax = fig.add_subplot(projection="3d")

# Extract x, y, and z data for the first 5 joints
x_data = joints[0:6,1, 3]  # first row (x)
y_data = joints[0:6,2, 3]  # second row (y)
z_data = joints[0:6,3, 3]  # third row (z)

# Plot the lines with markers (replace markersize with desired size)
lines = [ax.plot(x_data[0:7], y_data[0:7], z_data[0:7], marker='o',markerfacecolor='k', markeredgecolor='k', markersize=4)]


# Setting the axes properties
ax.set(xlim3d=(-1, 1), xlabel='X')
ax.set(ylim3d=(-1, 1), ylabel='Y')
ax.set(zlim3d=(0, 1), zlabel='Z')

plt.show()