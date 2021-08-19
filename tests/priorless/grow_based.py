import numpy as np
import matplotlib.pyplot as plt

def get_neighbor_idc(seed, shape):
    neighbors = []
    options = [-1, 0, 1]
    for i in options:
        for j in options:
            if i==0 and j==0:
                continue
            neighbors.append([seed[0]+i, seed[1]+j])
    return neighbors

shape = (100, 100)
n_dim = len(shape)

n_seeds = 1
seeds = [np.random.randint(0, high=shape[0], size=n_dim) for i in range(n_seeds)]

seed = seeds[0]
neighbor_idc = get_neighbor_idc(seed, shape)

img = np.zeros((shape))
img[seed[0], seed[1]] = 1

for n in neighbor_idc:
    img[n[0], n[1]] = 0.5

plt.figure()
plt.imshow(img)
plt.show()

# while True:
