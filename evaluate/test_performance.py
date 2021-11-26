import sys; sys.path.insert(0, r'C:\Users\lukas\Dokumente\projects\esinet')
import numpy as np
from esinet import evaluate
from scipy.spatial.distance import cdist
import time

n_dip = 1284
pos = np.random.randn(n_dip, 3)
true_source = np.random.randn(n_dip)
predicted_source = np.random.randn(n_dip)
argsorted_distance_matrix = np.argsort(cdist(pos, pos), axis=1)

n = 10
start = time.time()
for _ in range(n):
    evaluate.eval_auc(true_source, predicted_source, pos)
stop = time.time()
print(f'took {(1000*(stop-start))/n:.1f} ms per AUC')


# y = np.random.randn(n_dip)
# distance_matrix = np.random.randn(n_dip, n_dip)
# k_neighbors = 5
# threshold = 0.2
# argsorted_distance_matrix = np.argsort(distance_matrix, axis=1)

# start = time.time()

# close_idc = argsorted_distance_matrix[:, 1:k_neighbors+1]
# # close_idc = np.argpartition(distance_matrix, k_neighbors, axis=1)[:, 1:k_neighbors+1]

# t_sort = time.time()
# print(f'sorting takes {1000*(t_sort-start):.1f} ms')

# mask = ((y >= np.max(y[close_idc], axis=1)) & (y > threshold)).astype(int)
# stop1 = time.time()
# print(f'took {(1000*(stop1-start)):.1f} ms')

# print(mask.mean())
# for i, _ in enumerate(y):
#     distances = distance_matrix[i]
#     close_idc = np.argsort(distances)[1:k_neighbors+1]
#     if y[i] > np.max(y[close_idc]) and y[i] > threshold:
#         mask[i] = 1
# print(mask.mean())


# stop2 = time.time()
# print(f'took {(1000*(stop2-stop1)):.1f} ms')
