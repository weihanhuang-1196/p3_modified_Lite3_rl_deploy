import numpy as np
x1, y1, z1 = 0.693815, 0.000000, 0.500000
x2, y2, z2 = 0, 0, 1

p1 = np.array([x1, y1, z1])
p2 = np.array([x2, y2, z2])

dist = np.linalg.norm(p1 - p2)
print(dist)