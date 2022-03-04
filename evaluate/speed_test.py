from time import time
import numpy as np
import tensorflow as tf

n_samples = 1000
t_low, t_high = [1, 1000]
n_dip = 1284
sources = [np.random.randn(np.random.randint(t_low, t_high), n_dip) for _ in range(n_samples)]

# start = time()

# for i, source in enumerate(sources):
# 	for j, time_slice in enumerate(source):
# 		sources[i][j, :] = time_slice / np.max(np.abs(time_slice))



# end = time()
# print(f'time elapsed: {(end-start):.1f} seconds')


sources = tf.convert_to_tensor(sources)

start = time()

for i, source in enumerate(sources):
	for j, time_slice in enumerate(source):
		sources[i][j, :] = time_slice / np.max(np.abs(time_slice))



end = time()
print(f'time elapsed: {(end-start):.1f} seconds')
