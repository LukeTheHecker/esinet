import sys; sys.path.insert(0, '../')
from esinet import util
from esinet import Simulation
from esinet import Net
from esinet.forward import create_forward_model, get_info
from scipy.stats import pearsonr
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import mne

plot_params = dict(surface='white', hemi='both', verbose=0)

info = get_info(sfreq=100)
fwd = create_forward_model(sampling="ico3", info=info)


def prep_data(sim):
    X = np.squeeze(np.stack([eeg.average().data for eeg in sim.eeg_data]))
    X = np.stack([(x - np.mean(x)) / np.std(x) for x in X], axis=0)
    y = np.squeeze(np.stack([src.data for src in sim.source_data]))
    y = np.stack([(x / np.max(abs(x))) for x in y], axis=0)

    X = np.swapaxes(X, 1,2)
    y = np.swapaxes(y, 1,2)
    return X, y

def sparsity(y_true, y_pred):
    return K.mean(K.square(y_pred)) / K.max(K.square(y_pred))
def custom_loss():
    def loss(y_true, y_pred):
        loss1 = tf.keras.losses.CosineSimilarity()(y_true, y_pred)
        loss2 = sparsity(None, y_pred)
        return loss1 + loss2 * 1e-3
    return loss

from esinet.evaluate import auc_metric, eval_auc, eval_nmse, eval_mean_localization_error

def eval(y_true, y_hat):
    n_samples = y_true.shape[0]
    n_time = y_true.shape[1]
    aucs = np.zeros((n_samples, n_time))
    mles = np.zeros((n_samples, n_time))
    nmses = np.zeros((n_samples, n_time))
    for i in range(n_samples):
        for j in range(n_time):
            aucs[i,j] = np.mean(eval_auc(y_true[i,j], y_hat[i,j], pos))
            nmses[i,j] = eval_nmse(y_true[i,j], y_hat[i,j])
            mles[i,j] = eval_mean_localization_error(y_true[i,j], y_hat[i,j], pos)

    return aucs, nmses, mles

def threshold_activation(x):
    return tf.cast(x > 0.5, dtype=tf.float32)

class Compressor:
    ''' Compression using Graph Fourier Transform
    '''
    def __init__(self):
        pass
    def fit(self, fwd, k=600):
        A = mne.spatial_src_adjacency(fwd["src"], verbose=0).toarray()
        D = np.diag(A.sum(axis=0))
        L = D-A
        U, s, V = np.linalg.svd(L)

        self.U = U[:, -k:]
        self.s = s[-k:]
        self.V = V[:, -k:]
        return self
        
    def encode(self, X):
        ''' Encodes a true signal X
        Parameters
        ----------
        X : numpy.ndarray
            True signal
        
        Return
        ------
        X_comp : numpy.ndarray
            Compressed signal
        '''
        X_comp = self.U.T @ X

        return X_comp

    def decode(self, X_comp):
        ''' Decodes a compressed signal X

        Parameters
        ----------
        X : numpy.ndarray
            Compressed signal
        
        Return
        ------
        X_unfold : numpy.ndarray
            Decoded signal
        '''
        X_unfold = self.U @ X_comp
        return X_unfold




n_samples = 10000
settings = dict(duration_of_trial=0.25, extents=(1,40), number_of_sources=(1,10), target_snr=99999)
sim = Simulation(fwd, info, settings=settings).simulate(n_samples=n_samples)
X, y = prep_data(sim)

# WITHOUT GFT:
print("\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
print("\nWITHOUT GFT\n")
print("\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")

# comp = Compressor()
# comp.fit(fwd)
# y_comp = np.stack([comp.encode(yy.T).T for yy in y], axis=0)
y_comp = y

import tensorflow as tf
from tensorflow.keras.layers import Dense, TimeDistributed, Bidirectional, LSTM, GRU, multiply, Activation, Dropout
from tensorflow.keras.regularizers import l1
from esinet.losses import nmse_loss, nmae_loss

leadfield, pos = util.unpack_fwd(fwd)[1:3]
n_channels, _ = leadfield.shape
n_dipoles = y_comp.shape[-1]
input_shape = (None, None, n_channels)
tf.keras.backend.set_image_data_format('channels_last')

n_dense_units = 200
n_lstm_units = 64
activation_function = "tanh"
batch_size = 32
epochs = 30
dropout = 0.2


inputs = tf.keras.Input(shape=(None, n_channels), name='Input')
fc1 = TimeDistributed(Dense(n_dense_units, 
            activation=activation_function), 
            name='FC1')(inputs)
fc1 = Dropout(dropout)(fc1)
direct_out = TimeDistributed(Dense(n_dipoles, 
            activation="linear"),
            name='FC2')(fc1)
lstm1 = Bidirectional(GRU(n_lstm_units, return_sequences=True, 
            input_shape=(None, n_dense_units), dropout=dropout), 
            name='LSTM1')(fc1)
mask = TimeDistributed(Dense(n_dipoles, 
            activation="sigmoid"), 
            name='Mask')(lstm1)
multi = multiply([direct_out, mask], name="multiply")
model = tf.keras.Model(inputs=inputs, outputs=multi, name='Contextualizer')
model.compile(loss=tf.keras.losses.CosineSimilarity(), optimizer="adam")
model.summary()
model.fit(X, y_comp, epochs=epochs, batch_size=batch_size, validation_split=0.15)

import tensorflow as tf
from tensorflow.keras.layers import Dense, TimeDistributed, Bidirectional, LSTM, GRU, multiply, Activation
from tensorflow.keras.regularizers import l1
from esinet.losses import nmae_loss
leadfield, pos = util.unpack_fwd(fwd)[1:3]
n_channels, _ = leadfield.shape
n_dipoles = y_comp.shape[-1]
input_shape = (None, None, n_channels)
tf.keras.backend.set_image_data_format('channels_last')

n_dense_units = 300
n_lstm_units = 128
activation_function = "tanh"
batch_size = 32
dropout = 0.2

inputs = tf.keras.Input(shape=(None, n_channels), name='Input')
fc1 = TimeDistributed(Dense(n_dense_units, 
            activation=activation_function), 
            name='FC1')(inputs)
fc1 = Dropout(dropout)(fc1)
lstm1 = Bidirectional(GRU(n_lstm_units, return_sequences=True, name='LSTM1'))(fc1)

direct_out = TimeDistributed(Dense(n_dipoles, 
            activation="linear"),
            name='FC2')(lstm1)


model2 = tf.keras.Model(inputs=inputs, outputs=direct_out, name='LSTM_Old')


model2.compile(loss=tf.keras.losses.CosineSimilarity(), optimizer="adam")
model2.summary()
model2.fit(X, y_comp, epochs=epochs, batch_size=batch_size, validation_split=0.15)

import tensorflow as tf
from tensorflow.keras.layers import Dense, TimeDistributed, Bidirectional, LSTM, GRU, multiply, Activation
from tensorflow.keras.regularizers import l1
from esinet.losses import nmae_loss
leadfield, pos = util.unpack_fwd(fwd)[1:3]
n_channels, _ = leadfield.shape
n_dipoles = y_comp.shape[-1]
input_shape = (None, None, n_channels)
tf.keras.backend.set_image_data_format('channels_last')

n_dense_units = 600
n_lstm_units = 30
activation_function = "tanh"
batch_size = 32
dropout = 0.1

inputs = tf.keras.Input(shape=(None, n_channels), name='Input')
fc1 = TimeDistributed(Dense(n_dense_units, 
            activation=activation_function), 
            name='FC1')(inputs)
direct_out = TimeDistributed(Dense(n_dipoles, 
            activation="linear"),
            name='FC2')(fc1)


model3 = tf.keras.Model(inputs=inputs, outputs=direct_out, name='FC')


model3.compile(loss=tf.keras.losses.CosineSimilarity(), optimizer="adam")

model3.summary()
model3.fit(X, y_comp, epochs=epochs, batch_size=batch_size, validation_split=0.15)

# WITH GFT:
print("\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
print("\nWITH GFT\n")
print("\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")

comp = Compressor()
comp.fit(fwd)
y_comp = np.stack([comp.encode(yy.T).T for yy in y], axis=0)

import tensorflow as tf
from tensorflow.keras.layers import Dense, TimeDistributed, Bidirectional, LSTM, GRU, multiply, Activation, Dropout
from tensorflow.keras.regularizers import l1
from esinet.losses import nmse_loss, nmae_loss

leadfield, pos = util.unpack_fwd(fwd)[1:3]
n_channels, _ = leadfield.shape
n_dipoles = y_comp.shape[-1]
input_shape = (None, None, n_channels)
tf.keras.backend.set_image_data_format('channels_last')

n_dense_units = 200
n_lstm_units = 64
activation_function = "tanh"
batch_size = 32
dropout = 0.2


inputs = tf.keras.Input(shape=(None, n_channels), name='Input')
fc1 = TimeDistributed(Dense(n_dense_units, 
            activation=activation_function), 
            name='FC1')(inputs)
fc1 = Dropout(dropout)(fc1)
direct_out = TimeDistributed(Dense(n_dipoles, 
            activation="linear"),
            name='FC2')(fc1)
lstm1 = Bidirectional(GRU(n_lstm_units, return_sequences=True, 
            input_shape=(None, n_dense_units), dropout=dropout), 
            name='LSTM1')(fc1)
mask = TimeDistributed(Dense(n_dipoles, 
            activation="sigmoid"), 
            name='Mask')(lstm1)
multi = multiply([direct_out, mask], name="multiply")
model = tf.keras.Model(inputs=inputs, outputs=multi, name='Contextualizer')
model.compile(loss=tf.keras.losses.CosineSimilarity(), optimizer="adam")
model.summary()
model.fit(X, y_comp, epochs=epochs, batch_size=batch_size, validation_split=0.15)

import tensorflow as tf
from tensorflow.keras.layers import Dense, TimeDistributed, Bidirectional, LSTM, GRU, multiply, Activation
from tensorflow.keras.regularizers import l1
from esinet.losses import nmae_loss
leadfield, pos = util.unpack_fwd(fwd)[1:3]
n_channels, _ = leadfield.shape
n_dipoles = y_comp.shape[-1]
input_shape = (None, None, n_channels)
tf.keras.backend.set_image_data_format('channels_last')

n_dense_units = 300
n_lstm_units = 128
activation_function = "tanh"
batch_size = 32
dropout = 0.2

inputs = tf.keras.Input(shape=(None, n_channels), name='Input')
fc1 = TimeDistributed(Dense(n_dense_units, 
            activation=activation_function), 
            name='FC1')(inputs)
fc1 = Dropout(dropout)(fc1)
lstm1 = Bidirectional(GRU(n_lstm_units, return_sequences=True, name='LSTM1'))(fc1)

direct_out = TimeDistributed(Dense(n_dipoles, 
            activation="linear"),
            name='FC2')(lstm1)


model2 = tf.keras.Model(inputs=inputs, outputs=direct_out, name='LSTM_Old')


model2.compile(loss=tf.keras.losses.CosineSimilarity(), optimizer="adam")
model2.summary()
model2.fit(X, y_comp, epochs=epochs, batch_size=batch_size, validation_split=0.15)

import tensorflow as tf
from tensorflow.keras.layers import Dense, TimeDistributed, Bidirectional, LSTM, GRU, multiply, Activation
from tensorflow.keras.regularizers import l1
from esinet.losses import nmae_loss
leadfield, pos = util.unpack_fwd(fwd)[1:3]
n_channels, _ = leadfield.shape
n_dipoles = y_comp.shape[-1]
input_shape = (None, None, n_channels)
tf.keras.backend.set_image_data_format('channels_last')

n_dense_units = 600
n_lstm_units = 30
activation_function = "tanh"
batch_size = 32
dropout = 0.1

inputs = tf.keras.Input(shape=(None, n_channels), name='Input')
fc1 = TimeDistributed(Dense(n_dense_units, 
            activation=activation_function), 
            name='FC1')(inputs)
direct_out = TimeDistributed(Dense(n_dipoles, 
            activation="linear"),
            name='FC2')(fc1)


model3 = tf.keras.Model(inputs=inputs, outputs=direct_out, name='FC')


model3.compile(loss=tf.keras.losses.CosineSimilarity(), optimizer="adam")

model3.summary()
model3.fit(X, y_comp, epochs=epochs, batch_size=batch_size, validation_split=0.15)


