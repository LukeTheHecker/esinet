import sys; sys.path.insert(0, r'C:\Users\Lukas\Documents\projects\esinet')
import pickle as pkl
import numpy as np
from copy import deepcopy
import mne
import seaborn as sns
import matplotlib.pyplot as plt
from esinet import util
from esinet import Simulation
from esinet import Net
from esinet import forward

plot_params = dict(surface='white', hemi='both', verbose=0)


info = forward.get_info()
info['sfreq'] = 100
fwd = forward.create_forward_model(info=info)
fwd_free = forward.create_forward_model(info=info, fixed_ori=False)


with open(r'simulations/sim_10000_100points.pkl', 'rb') as f:
    sim_lstm = pkl.load(f)


import tensorflow as tf
epochs = 150
patience = 5
activation_function = 'relu'
def combi(y_true, y_pred):
    error_1 = tf.keras.losses.CosineSimilarity()(y_true, y_pred)
    error_2 = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
    return error_1 + error_2
loss = combi # 'mean_squared_error'
dropout = 0.2
optimizer = tf.keras.optimizers.Adam(clipvalue=0.5) #  'adam'  # tf.keras.optimizers.RMSprop(learning_rate=0.0001)

# Dense net
# model_params = dict(activation_function=activation_function, n_dense_layers=3, 
#     n_dense_units=300, n_lstm_layers=0, model_type='v2')
# train_params = dict(epochs=epochs, patience=patience, tensorboard=True, 
#     dropout=dropout, loss=loss, optimizer=optimizer, return_history=True,
#     metrics=[tf.keras.losses.mean_squared_error], batch_size=8)
# # Train
# net_dense = Net(fwd, **model_params)
# _, history_dense = net_dense.fit(sim_lstm, **train_params)
# net_dense.model.compile(optimizer='adam', loss='mean_squared_error')
# net_dense.save(r'models', name='dense-net-100points-noise-cosine-largemodels')

# LSTM v2
model_params = dict(activation_function=activation_function, n_lstm_layers=3, 
    n_lstm_units=300, n_dense_layers=0, 
    model_type='v2')
train_params = dict(epochs=epochs, patience=patience, tensorboard=True, 
    dropout=0.2, loss=loss, optimizer=optimizer, return_history=True,
    metrics=[tf.keras.losses.mean_squared_error], batch_size=8, device='/GPU:0')

# Train
net_lstm = Net(fwd, **model_params)
_, history_lstm = net_lstm.fit(sim_lstm, **train_params)
net_lstm.model.compile(optimizer='adam', loss='mean_squared_error')
net_lstm.save(r'models', name='lstm-net-100points-noise-cosine-largemodels')

