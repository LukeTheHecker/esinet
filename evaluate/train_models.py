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

# Load Data Set
with open(r'simulations/sim_10200_200-1000points.pkl', 'rb') as f:
    sim_lstm = pkl.load(f)

########################################################################
# # Create Data set
# n_samples = 10000
# duration_of_trial = (0.01, 2)
# settings = dict(duration_of_trial=duration_of_trial, method='standard')
# sim_lstm_short = Simulation(fwd, info, verbose=True, settings=settings).simulate(n_samples=n_samples)

# n_samples = 200
# duration_of_trial = (2, 10)
# settings = dict(duration_of_trial=duration_of_trial, method='standard')
# sim_lstm_long = Simulation(fwd, info, verbose=True, settings=settings).simulate(n_samples=n_samples)

# print("Adding:")
# sim_lstm = sim_lstm_short + sim_lstm_long
# del sim_lstm_short, sim_lstm_long
# sim_lstm.shuffle()

if type(duration_of_trial) == tuple:
    sim_lstm.save(f'simulations/sim_{sim_lstm.n_samples}_{int(duration_of_trial[0]*100)}-{int(duration_of_trial[1]*100)}points.pkl')
else:
    sim_lstm.save(f'simulations/sim_{sim_lstm.n_samples}_{int(duration_of_trial*100)}points.pkl')
########################################################################


########################################################################
epochs = 30
patience = 3
dropout = 0.2
batch_size = 8
validation_split = 0.05
validation_freq = 2 
optimizer = tf.keras.optimizers.Adam() 
device = '/GPU:0'
########################################################################


########################################################################
# Dense net
model_params = dict(n_dense_layers=2, n_dense_units=200, 
    n_lstm_layers=0)
train_params = dict(epochs=epochs, patience=patience, loss=loss, 
    optimizer=optimizer, return_history=True, 
    metrics=[tf.keras.losses.mean_squared_error], batch_size=batch_size,
    validation_freq=validation_freq, validation_split=validation_split,
    device=device)
# Train
net_dense = Net(fwd, **model_params)
_, history_dense = net_dense.fit(sim_lstm, **train_params)
net_dense.model.compile(optimizer='adam', loss='mean_squared_error')
########################################################################


########################################################################
# LSTM 
device = '/CPU:0'

model_params = dict(n_lstm_layers=2, 
    n_lstm_units=100, n_dense_layers=0, 
    model_type='v2')
train_params = dict(epochs=epochs, patience=patience, loss=loss, 
    optimizer=optimizer, return_history=True, 
    metrics=[tf.keras.losses.mean_squared_error], batch_size=batch_size, 
    device=device, validation_freq=validation_freq, 
    validation_split=validation_split)
# Train
net_lstm = Net(fwd, **model_params)
net_lstm.fit(sim_lstm, **train_params)
net_lstm.model.compile(optimizer='adam', loss='mean_squared_error')
########################################################################

########################################################################
# ConvDip
model_params = dict(n_lstm_layers=2, 
    n_lstm_units=100, n_dense_layers=0, 
    model_type='v2')
train_params = dict(epochs=epochs, patience=patience, loss=loss, 
    optimizer=optimizer, return_history=True, 
    metrics=[tf.keras.losses.mean_squared_error], batch_size=batch_size, 
    device=device, validation_freq=validation_freq, 
    validation_split=validation_split)
# Train
net_lstm = Net(fwd, **model_params)
net_lstm.fit(sim_lstm, **train_params)
net_lstm.model.compile(optimizer='adam', loss='mean_squared_error')
########################################################################


########################################################################
# Save
models = [net_dense, net_lstm]
model_names = ['Dense', 'LSTM']
net_dense.save(r'models', name='dense-net_1-1000points_standard-cosine-mse')
net_lstm.save(r'models', name='lstm-net_1-1000points_standard_cosine-mse')
########################################################################