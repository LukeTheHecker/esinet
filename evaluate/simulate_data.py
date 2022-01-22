import sys; sys.path.insert(0, '..\\')
import pickle as pkl
import numpy as np
from copy import deepcopy
import mne
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from esinet import util
from esinet import Simulation
from esinet import Net
from esinet import forward
from esinet.losses import combi as loss

plot_params = dict(surface='white', hemi='both', verbose=0)


info = forward.get_info()
info['sfreq'] = 100
fwd = forward.create_forward_model(info=info)
fwd_free = forward.create_forward_model(info=info, fixed_ori=False)

########################################################################
# Create Data set
n_samples = 10000
duration_of_trial = (0.01, 2)
method = 'standard'
exponent = 3

settings = dict(duration_of_trial=duration_of_trial, method=method)
sim_short = Simulation(fwd, info, verbose=True, settings=settings).simulate(n_samples=n_samples)

n_samples = 200
duration_of_trial = (2, 10)
settings = dict(duration_of_trial=duration_of_trial, method=method)
sim_long = Simulation(fwd, info, verbose=True, settings=settings).simulate(n_samples=n_samples)

print("Adding:")
sim = sim_short + sim_long
del sim_short, sim_long
sim.shuffle()

sim.save(f'simulations/sim_{sim.n_samples}_1-1000points_standard.pkl')

########################################################################