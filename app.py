# Install opensimplex if you haven't already
# pip install opensimplex

from opensimplex import OpenSimplex
import matplotlib

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from scipy.ndimage import laplace, median_filter

# Initialize OpenSimplex with a seed
seed = 42
noise_gen = OpenSimplex(seed)

# Create a grid of points to generate noise for
width, height = 300, 300
noise = np.zeros((width, height))

for x in range(width):
    for y in range(height):
        noise[x][y] = noise_gen.noise2(x * 0.1, y * 0.1)

# Threshold the noise to create black lines
threshold = 0.2
binary_noise = np.where(np.abs(noise) < threshold, 0, 1)
filtered_noise = median_filter(binary_noise, size=4)

# Plot the generated noise
plt.imshow(binary_noise, cmap=mcolors.ListedColormap(["#ba8c5d", "green"]))
plt.colorbar()
plt.show()
