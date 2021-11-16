import numpy as np

L = np.linspace(0, 10, 100)
dx1 = L[1] - L[0]

x, y = np.meshgrid(L, L, indexing="ij")

dx2 = x[0, 0] - x[1, 0]

print(dx1, dx2)