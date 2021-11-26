from esinet.minimum_norm import mne_eloreta
from esinet.simulation import Simulation
from esinet.forward import create_forward_model, get_info
from esinet.net import Net
from esinet.util import wrap_mne_inverse, unpack_fwd
import time

info = get_info()
info['sfreq'] = 100
fwd = create_forward_model(info=info, sampling='ico5')
pos = unpack_fwd(fwd)[2]
print(pos.shape)
settings = dict(duration_of_trial=0.2, number_of_sources=20)
sim = Simulation(fwd, info, settings=settings, parallel=False)

n_samples = 10000

start = time.time()
sim.simulate(n_samples=n_samples)
end = time.time()

print(f'Simulation time: {((end-start)*1000):.1f} ms ({((end-start)):.1f} s)')

# import numpy as np
# print('\n\n')
# n = 10000*10
# k = 1284
# start_lc = time.time()
# a = np.stack([np.random.randn(k) for _ in range(n)], axis=0)
# end_lc = time.time()

# start_pa = time.time()
# a = np.zeros((n,k))
# for i in range(n):
#     a[i] = np.random.randn(k)
# end_pa = time.time()

# print(f'List comprehension: {1000*(end_lc-start_lc):.2f} ms\nPre-allocation: {1000*(end_pa-start_pa):.2f} ms')

