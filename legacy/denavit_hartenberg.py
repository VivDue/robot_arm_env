import numpy as np
import matplotlib.pyplot as plt

# DH Parameters (replace with your actual values)
a = np.array([0, -0.24355, -0.2132, 0, 0, 0])
d = np.array([0.15185, 0, 0, 0.13105, 0.08535, 0.0921])
alpha = np.array([90, 0, 0, 90, -90, 0])
alpha = np.deg2rad(alpha)
tetha = np.array([45, -45, 0, 180, 0, 45])  # Initial joint angles (replace with desired values)
tetha = np.deg2rad(tetha)

T = np.zeros(shape=(7, 4, 4))


def transform(n):
  global tetha, alpha, a, d
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




# Calculate joint poses
joints = np.zeros(shape=(7, 4, 4))
for n in range(6):
  transform(n)
  if n == 0:
    joints[n] = T[n]
  else:
    joints[n] = joints[n-1] @ T[n]  # Chain multiplication for cumulative transformation


# Extract and plot joint positions (all joints)
x = np.zeros(shape=(7))
y = np.zeros(shape=(7))
z = np.zeros(shape=(7))
for n in range(6):
  x[n+1] = joints[n, 0, 3]
  y[n+1] = joints[n, 1, 3]
  z[n+1] = joints[n, 2, 3]

#print(joints)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, markerfacecolor='k', markeredgecolor='b', marker='o', markersize=5, alpha=0.7, linewidth = 4)

tcp_x = np.zeros(shape=(4))
tcp_y = np.zeros(shape=(4))
tcp_z = np.zeros(shape=(4))

scale = 0.05
for m in range(7):
    for n in range(4):
        tcp_x[n] = joints[m, 0, n]*scale  # End effector pose
        tcp_y[n] = joints[m, 1, n]*scale  # End effector pose
        tcp_z[n] = joints[m, 2, n]*scale  # End effector pose

    # iterate through the tcp points and colors
    for n,c in zip(range(3),['r','g','b']):
        ax.plot([x[m],tcp_x[n]+x[m]], [y[m],tcp_y[n]+y[m]], [z[m],tcp_z[n]+z[m]], color = c, markersize=5, alpha=0.7, linewidth = 1)
   
    
# Setting the axes properties
ax.set(xlim3d=(-0.75, 0.75), xlabel='X')
ax.set(ylim3d=(-0.75, 0.75), ylabel='Y')
ax.set(zlim3d=(0, 0.5), zlabel='Z')
plt.show()
