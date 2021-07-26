import mne
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import pearsonr
from copy import deepcopy
import time
from .. import util

class Net(keras.Sequential):
    ''' The neural network class that creates and trains the model. 
    Inherits the keras.Sequential class
    
    Attributes
    ----------
    fwd : mne.Forward
        the mne.Forward forward model class.
    n_layers : int
        Number of hidden layers in the neural network.
    n_neurons : int
        Number of neurons per hidden layer.
    activation_function : str
        The activation function used for each fully connected layer.

    Methods
    -------
    fit : trains the neural network with the EEG and source data
    train : trains the neural network with the EEG and source data
    predict : perform prediciton on EEG data
    evaluate : evaluate the performance of the model
    

    '''
    def __init__(self, fwd, n_layers=1, n_neurons=128, 
        activation_function='swish', verbose=False):

        super().__init__()
        _, leadfield, _, _ = util.unpack_fwd(fwd)
        self.fwd = fwd
        self.leadfield = leadfield
        self.n_channels = leadfield.shape[0]
        self.n_dipoles = leadfield.shape[1]
        
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.activation_function = activation_function

        self.verbose = verbose

        self._build_model()
    
    def fit(self, eeg, sources, optimizer=None, learning_rate=0.005, 
        validation_split=0.1, epochs=100, metrics=None, device=None, delta=1, 
        batch_size=128, loss=None):
        ''' Train the neural network using training data (eeg) and labels (sources).
        
        Parameters
        ----------
        eeg : mne.Epochs/ numpy.ndarray
            The simulated EEG data
        sources : mne.SourceEstimates/ list of mne.SourceEstimates
            The simulated EEG data

        Return
        ------

        '''
        # Handle EEG input
        # if type(eeg) == mne.epochs.EpochsArray:
        eeg = np.squeeze(eeg.get_data())
        # elif type(eeg) == list:
        #     if type(eeg[0]) == mne.epochs.EpochsArray:
        #         eeg = np.stack([ep.average().data for ep in eeg], axis=0)
        #     elif type(eeg[0]) == mne.epochs.EvokedArray:
        #         eeg = np.stack([ep.data for ep in eeg], axis=0)
        # if len(eeg.shape) == 4 and eeg.shape[-1] > 1:
        #     print(f'Simulations have a temporal dimension (i.e. more than a single time point). Please simulate data without a temporal dimension!\n Solution: When using the function <run_simulations> set durOftrial=0.')
        #     raise ValueError('eeg must contain data without temporal dimension. ')
    
        # Handle source input
        if type(sources) == mne.source_estimate.SourceEstimate:
            sources = sources.data.T
        elif type(sources) == list:
            if type(sources[0]) == mne.source_estimate.SourceEstimate:
                sources = np.stack([source.data for source in sources], axis=0)
        
        if len(sources.shape) == 4 and eeg.shape[-1] > 1:
            print(f'Simulations have a temporal dimension (i.e. more than a single time point). Please simulate data without a temporal dimension!\n Solution: When using the function <run_simulations> set durOftrial=0.')
            raise ValueError('eeg must contain data without temporal dimension. ')
        
        
        # Extract data
        y = np.squeeze(sources)
        x = np.squeeze(eeg)
        # Prepare data
        # Scale sources
        y_scaled = np.stack([sample / np.max(sample) for sample in y])
        # Common average referencing for eeg
        x = np.stack([sample - np.mean(sample) for sample in x])
        # Scale EEG
        x_scaled = np.stack([sample / np.max(np.abs(sample)) for sample in x])
        
        # Early stopping
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', \
            mode='min', verbose=self.verbose, patience=25, restore_best_weights=True)
            
        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam(lr=0.001)
        if loss is None:
            loss = tf.keras.losses.Huber(delta=delta)
        elif type(loss) == list:
            loss = loss[0](*loss[1])
        if metrics is None:
            metrics = tf.keras.losses.Huber(delta=delta)

        self.compile(optimizer, loss, metrics=metrics)
        if device is None:
            history = super().fit(x_scaled, y_scaled, epochs=epochs, batch_size=batch_size, shuffle=False, \
                    validation_split=validation_split, verbose=self.verbose, callbacks=[es])
        else:
            with tf.device(device):
                history = super().fit(x_scaled, y_scaled, epochs=epochs, batch_size=batch_size, shuffle=False, \
                    validation_split=validation_split, verbose = self.verbose, callbacks=[es])
    
    def train(self, eeg, sources, optimizer=None, learning_rate=0.005, 
        validation_split=0.1, epochs=100, metrics=None, device=None, delta=1, 
        batch_size=128, loss=None):
        ''' Train the neural network using training data (eeg) and labels (sources).
        '''
        self.fit(eeg, sources, optimizer=optimizer, learning_rate=learning_rate, 
        validation_split=validation_split, epochs=epochs, metrics=metrics, device=device, delta=delta, 
        batch_size=batch_size, loss=loss)
    
    def predict(self, EEG):
        ''' Predict sources from EEG data.

        Parameters
        ----------
        EEG : mne.Epochs
            shape (timepoints, electrodes), EEG data to infer sources from 
        dtype : str
            either of:
                'raw' : will return the source as a raw numpy array 
                'SourceEstimate' or 'SE' : will return a mne.SourceEstimate object
        
        Return
        ------
        outsource : either numpy.ndarray (if dtype='raw') or mne.SourceEstimate instance
        '''
    

        if isinstance(EEG, mne.epochs.EvokedArray):
            print("a")
            sfreq = EEG.info['sfreq']
            tmin = EEG.tmin
            EEG = np.squeeze(EEG.data)
        elif isinstance(EEG, (mne.epochs.EpochsFIF, mne.epochs.EpochsArray, mne.Epochs)):
            print("b")
            sfreq = EEG.info['sfreq']
            tmin = EEG.tmin
            EEG = EEG._data  # np.squeeze(EEG.average().data)
            print(EEG.shape)
        elif isinstance(EEG, np.ndarray):
            print("c")
            sfreq = 1
            tmin = 0
            EEG = np.squeeze(np.array(EEG))
        else:
            print("d")
            msg = f'EEG must be of type <numpy.ndarray> or <mne.epochs.EpochsArray>; got {type(EEG)} instead.'
            raise ValueError(msg)

        if len(EEG.shape) == 1:
            EEG = np.expand_dims(EEG, axis=0)
        
        if EEG.shape[1] != self.n_channels:
            EEG = EEG.T

        # Prepare EEG to ensure common average reference and appropriate scaling
        EEG_prepd = deepcopy(EEG)
        for i in range(EEG.shape[0]):
            # Common average reference
            EEG_prepd[i, :] -= np.mean(EEG_prepd[i, :])
            # Scaling
            EEG_prepd[i, :] /= np.max(np.abs(EEG_prepd[i, :]))
        
        # Predict using the model
        if len(np.squeeze(EEG_prepd).shape) == 3:
            # predict per trial
            source_predicted = [super().predict(trial) for trial in EEG_prepd]
            # Scale ConvDips prediction
            source_predicted_scaled = []
            predicted_source_estimate = []
            for trial in range(EEG_prepd.shape[0]):
                source_predicted_scaled.append( np.squeeze(np.stack([self.solve_p(source_frame, EEG_frame) for source_frame, EEG_frame in zip(source_predicted[trial], EEG[trial])], axis=0)) )
                predicted_source_estimate.append( util.source_to_sourceEstimate(np.squeeze(source_predicted_scaled[trial]), self.fwd, sfreq=sfreq, tmin=tmin) )
        else:
            source_predicted = super().predict(np.squeeze(EEG_prepd))
            # Scale ConvDips prediction
            source_predicted_scaled = np.squeeze(np.stack([self.solve_p(source_frame, EEG_frame) for source_frame, EEG_frame in zip(source_predicted, EEG)], axis=0))   
            predicted_source_estimate = util.source_to_sourceEstimate(np.squeeze(source_predicted_scaled), self.fwd, sfreq=sfreq, tmin=tmin)

        return predicted_source_estimate

    def _build_model(self):
        ''' Build the neural network architecture using the 
        tensorflow.keras.Sequential() API.'''
        for i in range(self.n_layers):
            self.add(layers.Dense(units=self.n_neurons,
                                activation=self.activation_function))

        self.add(layers.Dense(self.n_dipoles, 
            activation=keras.layers.ReLU(max_value=1)))
        # model.add()
        self.build(input_shape=(None, self.n_channels))
        self.summary()

    
    def solve_p(self, y_est, x_true):
        # Check if y_est is just zeros:
        if np.max(y_est) == 0:
            return y_est
        y_est = np.squeeze(np.array(y_est))
        x_true = np.squeeze(np.array(x_true))
        # Get EEG from predicted source using leadfield
        x_est = np.matmul(self.leadfield, y_est)

        # optimize forward solution
        tol = 1e-10
        options = dict(maxiter=1000, disp=False)

        # base scaling
        rms_est = np.mean(np.abs(x_est))
        rms_true = np.mean(np.abs(x_true))
        base_scaler = rms_true / rms_est

        
        opt = minimize_scalar(self.mse_opt, args=(self.leadfield, y_est* base_scaler, x_true), \
            bounds=(0, 1), method='bounded', options=options, tol=tol)
        
        scaler = opt.x
        y_scaled = y_est * scaler * base_scaler
        return y_scaled

    @staticmethod
    def mse_opt(scaler, leadfield, y_est, x_true):
        x_est = np.matmul(leadfield, y_est) 
        error = np.abs(pearsonr(x_true-x_est, x_true)[0])
        return error
    