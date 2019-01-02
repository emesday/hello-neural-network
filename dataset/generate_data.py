import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2019)
num_points = 1000
x = np.random.normal(0, 1, num_points)
y = 0.8 * x + 1.6 + np.random.normal(0, 0.05, num_points)

with open('dataset.txt', 'w') as f:
    f.write('%s\n' % num_points)
    x.tofile(f, sep='\n')
    y.tofile(f, sep='\n')

plt.plot(x, y, 'ro', ms=1)
plt.savefig('dataset.png')

