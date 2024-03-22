import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.cm import plasma
# create a 4d array of random values
data = np.random.rand(4,500)
data = np.amax(data, axis=0)
#data = np.linspace(0, 1, 500)
data = np.reshape(data, (10, 10, 5), order = 'F')
print(data)

# create a 3d voxel plot of the data where the values are represented by the color of the voxels
# Create the 3D voxel plot (consider using colormaps for better visualization)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Apply the colormap
cmap = plasma  # Replace with your chosen colormap
norm = plt.Normalize(data.min(), data.max())  # Normalize data for colormap
colors = cmap(norm(data))  # Map Q-values to colormap

# Create voxels with color
data_bool = data > 0.5  # Define a boolean mask for data
ax.voxels(data_bool, facecolors=colors, shade = False, alpha = 0.5)

# Customize the plot (labels, title, etc.)
ax.set_xlabel("Value 1")
ax.set_ylabel("Value 2")
ax.set_zlabel("Value 3")
ax.set_title("Maximum Action Value (Reward) Landscape")

plt.show()
