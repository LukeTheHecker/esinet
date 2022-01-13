import sys; sys.path.insert(0, '..\\')
import pickle as pkl
import tensorflow as tf
from esinet import Net, forward
from esinet.losses import combi as loss

plot_params = dict(surface='white', hemi='both', verbose=0)


info = forward.get_info()
info['sfreq'] = 100
fwd = forward.create_forward_model(info=info)
fwd_free = forward.create_forward_model(info=info, fixed_ori=False)

# Load Data Set
pth = r'simulations/sim_10200_1-1000points.pkl'
# pth = r'simulations/sim_10200_1-1000points_noise.pkl'

with open(pth, 'rb') as f:
    sim = pkl.load(f)




########################################################################
epochs = 150
patience = 3
dropout = 0.2
batch_size = 8
validation_split = 0.05
validation_freq = 2 
optimizer = tf.keras.optimizers.Adam() 
device = '/GPU:0'

train_params = dict(epochs=epochs, patience=patience, loss=loss, 
    optimizer=optimizer, return_history=True, 
    metrics=[tf.keras.losses.mean_squared_error], batch_size=batch_size,
    validation_freq=validation_freq, validation_split=validation_split,
    device=device)
########################################################################

########################################################################
# Specifications of the models
model_params_dict = {
    # "Dense Small": dict(n_dense_layers=2, n_dense_units=70, n_lstm_layers=0),
    # "Dense Medium": dict(n_dense_layers=2, n_dense_units=300, n_lstm_layers=0),
    "Dense Large": dict(n_dense_layers=4, n_dense_units=400, n_lstm_layers=0),

    # "LSTM Small": dict(n_lstm_layers=2, n_lstm_units=25, n_dense_layers=0,),
    # "LSTM Medium": dict(n_lstm_layers=2, n_lstm_units=85, n_dense_layers=0,),
    "LSTM Large": dict(n_lstm_layers=3, n_lstm_units=110, n_dense_layers=0,),
    
    # "ConvDip Small": dict(n_lstm_layers=1, n_dense_layers=1, n_dense_units=70, model_type='convdip'),
    # "ConvDip Medium": dict(n_lstm_layers=2, n_dense_layers=3, n_dense_units=250, model_type='convdip'),
    # "ConvDip Large": dict(n_lstm_layers=2, n_dense_layers=4, n_dense_units=400, model_type='convdip'),

}

for model_name, model_params in model_params_dict.items():
    net = Net(fwd, **model_params)
    net.fit(sim, **train_params)
    net.model.compile(optimizer='adam', loss='mean_squared_error')
    net.save(r'models', name=f'{model_name}_1-1000points_standard-cosine-mse')
    del net


del sim
plot_params = dict(surface='white', hemi='both', verbose=0)


info = forward.get_info()
info['sfreq'] = 100
fwd = forward.create_forward_model(info=info)
fwd_free = forward.create_forward_model(info=info, fixed_ori=False)

# Load Data Set
pth = r'simulations/sim_10200_1-1000points_noise.pkl'
# pth = r'simulations/sim_10200_1-1000points.pkl'

with open(pth, 'rb') as f:
    sim = pkl.load(f)




########################################################################
epochs = 150
patience = 3
dropout = 0.2
batch_size = 8
validation_split = 0.05
validation_freq = 2 
optimizer = tf.keras.optimizers.Adam() 
device = '/GPU:0'

train_params = dict(epochs=epochs, patience=patience, loss=loss, 
    optimizer=optimizer, return_history=True, 
    metrics=[tf.keras.losses.mean_squared_error], batch_size=batch_size,
    validation_freq=validation_freq, validation_split=validation_split,
    device=device)
########################################################################

########################################################################
# Specifications of the models
model_params_dict = {
    # "Dense Small": dict(n_dense_layers=2, n_dense_units=70, n_lstm_layers=0),
    # "Dense Medium": dict(n_dense_layers=2, n_dense_units=300, n_lstm_layers=0),
    "Dense Large": dict(n_dense_layers=4, n_dense_units=400, n_lstm_layers=0),

    # "LSTM Small": dict(n_lstm_layers=2, n_lstm_units=25, n_dense_layers=0,),
    # "LSTM Medium": dict(n_lstm_layers=2, n_lstm_units=85, n_dense_layers=0,),
    "LSTM Large": dict(n_lstm_layers=3, n_lstm_units=110, n_dense_layers=0,),
    
    # "ConvDip Small": dict(n_lstm_layers=1, n_dense_layers=1, n_dense_units=70, model_type='convdip'),
    # "ConvDip Medium": dict(n_lstm_layers=2, n_dense_layers=3, n_dense_units=250, model_type='convdip'),
    # "ConvDip Large": dict(n_lstm_layers=2, n_dense_layers=4, n_dense_units=400, model_type='convdip'),

}

for model_name, model_params in model_params_dict.items():
    net = Net(fwd, **model_params)
    net.fit(sim, **train_params)
    net.model.compile(optimizer='adam', loss='mean_squared_error')
    net.save(r'models', name=f'{model_name}_1-1000points_noise-cosine-mse')
    del net