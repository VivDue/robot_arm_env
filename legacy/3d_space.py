import matplotlib.pyplot as plt
import numpy as np

import matplotlib.colors

start = np.array([0.1,0.3,0.2])
end =  np.array([2.2,1.6,3.9])
t = np.linspace(0,1,10)
b = np.array([2,2,-3])
print(t)
x = b[0]*t + start[0]
y = b[1]*t + start[1]
z = b[2]*t + start[2]

ax = plt.figure().add_subplot(projection='3d')



ax.plot(x, y, z, label='trajectory')
ax.legend()


# prepare some coordinates
x, y, z = np.indices((20, 20, 20))



start = np.array([1,1,1])
end =  np.array([6,3,7])
t = np.linspace(0,10,40)
b = np.array([2,2,3])
print(t)
x1 = b[0]*t + start[0]
y1 = b[1]*t + start[1]
z1 = b[2]*t + start[2]

# draw cuboids in the top left and bottom right corners, and a link between
# them
old_link = (x < 0) & (y < 0) & ( z < 0)  
for x1,y1,z1 in zip(x1,y1,z1):
    r = 1
    link = (x - x1)**2 + (y - y1)**2 + (z - z1)**2 < r**2  
    link = link | old_link 
    old_link = link
print(link)
# combine the objects into a single boolean array

# set the colors of each object
colors = np.empty(link.shape, dtype=object)
colors[link] = 'green'

# and plot everything

ax.voxels(link, facecolors=colors, edgecolor='k')


# draw cuboids in the top left and bottom right corners, and a link between
# them
x2 = b[0]*t + start[0]
y2 = b[1]*t + start[1]
z2 = b[2]*t + start[2]
old_link = (x < 0) & (y < 0) & ( z < 0)  
for x2,y2,z2 in zip(x2,y2,z2):
    r = 2
    link = (x - x2)**2 + (y - y2)**2 + (z - z2)**2 < r**2  
    link = link | old_link 
    old_link = link
print(link)
# combine the objects into a single boolean array

# set the colors of each object
colors = np.empty(link.shape, dtype=object)
colors[link] = 'yellow'

# and plot everything

ax.voxels(link, facecolors=colors, edgecolor='k',alpha=0.15)

# draw cuboids in the top left and bottom right corners, and a link between
# them
x3 = b[0]*t + start[0]
y3 = b[1]*t + start[1]
z3 = b[2]*t + start[2]
old_link = (x < 0) & (y < 0) & ( z < 0)  
for x3,y3,z3 in zip(x3,y3,z3):
    r = 3
    link = (x - x3)**2 + (y - y3)**2 + (z - z3)**2 < r**2  
    link = link | old_link 
    old_link = link
print(link)
# combine the objects into a single boolean array

# set the colors of each object
colors = np.empty(link.shape, dtype=object)
colors[link] = 'red'

# and plot everything

ax.voxels(link, facecolors=colors, edgecolor='k',alpha=0.15)

plt.show()
