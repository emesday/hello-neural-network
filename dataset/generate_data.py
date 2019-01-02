import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

np.random.seed(2019)
num_points = 1000
W = np.array([0.8, 1.6])
b = np.array([2.17])

x = np.random.normal(0, 1, (num_points, 2))
y = np.dot(x, W) + b + np.random.normal(0, 0.05, num_points)

dataset = np.hstack((y[:, np.newaxis], x))
np.savetxt('dataset.txt', dataset, fmt='%.2f')
