import mne
from mne.viz.topomap import (_setup_interp, _make_head_outlines, _check_sphere, 
    _check_extrapolate)
from mne.channels.layout import _find_topomap_coords
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import (LSTM, GRU, Dense, Flatten, Bidirectional, 
    TimeDistributed, InputLayer, Activation, Reshape, concatenate, Concatenate, 
    Dropout, Conv1D, Conv2D, multiply)
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda
from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.utils import pad_sequences

from scipy.optimize import minimize_scalar
# import pickle as pkl
import dill as pkl
import datetime
# from sklearn import linear_model
import numpy as np
from scipy.stats import pearsonr
from copy import deepcopy
from time import time
from tqdm import tqdm

from . import util
from . import evaluate
from . import losses
from .custom_layers import BahdanauAttention, Attention

# Fix from: https://github.com/tensorflow/tensorflow/issues/35100
# devices = tf.config.experimental.list_physical_devices('GPU')
# if len(devices) > 0:
#     print(devices)
#     tf.config.experimental.set_memory_growth(devices, True)

class Net:
    ''' The neural network class that creates and trains the model. 
    
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
    n_jobs : int
        Number of jobs/ cores to use during parallel processing
    model : str
        Determines the neural network architecture.
            'auto' : automated selection for fully connected if training data 
                contains single time instances (non-temporal data)
            'single' : The single time instance model that does not learn 
                temporal relations.
            'temporal' : The LSTM model which estimates multiples inverse 
                solutions in one go.
    Methods
    -------
    fit : trains the neural network with the EEG and source data
    train : trains the neural network with the EEG and source data
    predict : perform prediciton on EEG data
    evaluate : evaluate the performance of the model
    '''
    
    def __init__(self, fwd, n_dense_layers=1, n_lstm_layers=2, 
        n_dense_units=200, n_lstm_units=32, activation_function='tanh', 
        n_filters=64, kernel_size=(3,3), l1_reg=None, n_jobs=-1, model_type='auto', 
        scale_individually=True, rescale_sources='brent', 
        verbose=0):

        self._embed_fwd(fwd)
        
        self.n_dense_layers = n_dense_layers
        self.n_lstm_layers = n_lstm_layers
        self.n_dense_units = n_dense_units
        self.n_lstm_units = n_lstm_units
        self.l1_reg = l1_reg
        self.activation_function = activation_function
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        # self.default_loss = tf.keras.losses.Huber(delta=delta)
        self.default_loss = 'mean_squared_error'  # losses.weighted_huber_loss
        # self.parallel = parallel
        self.n_jobs = n_jobs
        self.model_type = model_type
        self.compiled = False
        self.scale_individually = scale_individually
        self.rescale_sources = rescale_sources
        self.verbose = verbose

    def _embed_fwd(self, fwd):
        ''' Saves crucial attributes from the Forward model.
        
        Parameters
        ----------
        fwd : mne.Forward
            The forward model object.
        '''
        _, leadfield, _, _ = util.unpack_fwd(fwd)
        self.fwd = deepcopy(fwd)
        self.leadfield = leadfield
        self.n_channels = leadfield.shape[0]
        self.n_dipoles = leadfield.shape[1]
        self.interp_channel_shape = (9,9)
    
    @staticmethod
    def _handle_data_input(arguments):
        ''' Handles data input to the functions fit() and predict().
        
        Parameters
        ----------
        arguments : tuple
            The input arguments to fit and predict which contain data.
        
        Return
        ------
        eeg : mne.Epochs
            The M/EEG data.
        sources : mne.SourceEstimates/list
            The source data.

        '''
        if len(arguments) == 1:
            if isinstance(arguments[0], (mne.Epochs, mne.Evoked, mne.io.Raw, mne.EpochsArray, mne.EvokedArray, mne.epochs.EpochsFIF)):
                eeg = arguments[0]
                sources = None
            else:
                simulation = arguments[0]
                eeg = simulation.eeg_data
                sources = simulation.source_data
                # msg = f'First input should be of type simulation or Epochs, but {arguments[1]} is {type(arguments[1])}'
                # raise AttributeError(msg)

        elif len(arguments) == 2:
            eeg = arguments[0]
            sources = arguments[1]
        else:
            msg = f'Input is {type()} must be either the EEG data and Source data or the Simulation object.'
            raise AttributeError(msg)

        return eeg, sources

    def fit(self, *args, optimizer=None, learning_rate=0.001, 
        validation_split=0.05, epochs=50, metrics=None, device=None, 
        false_positive_penalty=2, delta=1., batch_size=8, loss=None, 
        sample_weight=None, return_history=False, dropout=0.2, patience=7, 
        tensorboard=False, validation_freq=1, revert_order=True):
        ''' Train the neural network using training data (eeg) and labels (sources).
        
        Parameters
        ----------
        *args : 
            Can be either two objects: 
                eeg : mne.Epochs/ numpy.ndarray
                    The simulated EEG data
                sources : mne.SourceEstimates/ list of mne.SourceEstimates
                    The simulated EEG data
                or only one:
                simulation : esinet.simulation.Simulation
                    The Simulation object

            - two objects: EEG object (e.g. mne.Epochs) and Source object (e.g. mne.SourceEstimate)
        
        optimizer : tf.keras.optimizers
            The optimizer that for backpropagation.
        learning_rate : float
            The learning rate for training the neural network
        validation_split : float
            Proportion of data to keep as validation set.
        delta : int/float
            The delta parameter of the huber loss function
        epochs : int
            Number of epochs to train. In one epoch all training samples 
            are used once for training.
        metrics : list/str
            The metrics to be used for performance monitoring during training.
        device : str
            The device to use, e.g. a graphics card.
        false_positive_penalty : float
            Defines weighting of false-positive predictions. Increase for conservative 
            inverse solutions, decrease for liberal prediction.
        batch_size : int
            The number of samples to simultaneously calculate the error 
            during backpropagation.
        loss : tf.keras.losses
            The loss function.
        sample_weight : numpy.ndarray
            Optional numpy array of sample weights.

        Return
        ------
        self : esinet.Net
            Method returns the object itself.

        '''
        self.loss = loss
        self.dropout = dropout
    
        print("preprocess data")
        x_scaled, y_scaled = self.prep_data(args)
        
        # Early stopping
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', \
            mode='min', verbose=self.verbose, patience=patience, restore_best_weights=True)
        if tensorboard:
            log_dir = "logs/fit/" + self.model.name + '_' + datetime.datetime.now().strftime("%m%d-%H%M")
            tensorboard_callback = tf.keras.callbacks.TensorBoard(
                log_dir=log_dir, histogram_freq=1)
            callbacks = [es, tensorboard_callback]
        else:
            callbacks = []#[es]
        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        if self.loss is None:
            self.loss = tf.keras.losses.CosineSimilarity()


        elif type(loss) == list:
            self.loss = self.loss[0](*self.loss[1])
        
        # Compile if it wasnt compiled before
        if not self.compiled:
            self.model.compile(optimizer, self.loss, metrics=metrics)
            self.compiled = True
        
        if self.model_type.lower() == 'convdip':
            # print("interpolating for convdip...")
            elec_pos = _find_topomap_coords(self.info, self.info.ch_names)
            interpolator = self.make_interpolator(elec_pos, res=self.interp_channel_shape[0])
            x_scaled_interp = deepcopy(x_scaled)
            for i, sample in enumerate(x_scaled):
                list_of_time_slices = []
                for time_slice in sample:
                    time_slice_interp = interpolator.set_values(time_slice)()[::-1]
                    time_slice_interp = time_slice_interp[:, :, np.newaxis]
                    list_of_time_slices.append(time_slice_interp)
                x_scaled_interp[i] = np.stack(list_of_time_slices, axis=0)
                x_scaled_interp[i][np.isnan(x_scaled_interp[i])] = 0
            x_scaled = x_scaled_interp
            del x_scaled_interp
            print("\t...done")
            
        print("fit model")
        n_samples = len(x_scaled)
        stop_idx = int(round(n_samples * (1-validation_split)))
        gen = self.generate_batches(x_scaled[:stop_idx], y_scaled[:stop_idx], batch_size, revert_order=revert_order)
        steps_per_epoch = stop_idx // batch_size
        validation_data = (pad_sequences(x_scaled[stop_idx:], dtype='float32'), pad_sequences(y_scaled[stop_idx:], dtype='float32'))

        
        if device is None:
            history = self.model.fit(x=gen, 
                    epochs=epochs, batch_size=batch_size, 
                    steps_per_epoch=steps_per_epoch, verbose=self.verbose, callbacks=callbacks, 
                    sample_weight=sample_weight, validation_data=validation_data, 
                    validation_freq=validation_freq)
        else:
            with tf.device(device):
                history = self.model.fit(x=gen, 
                    epochs=epochs, batch_size=batch_size, 
                    steps_per_epoch=steps_per_epoch, verbose=self.verbose, callbacks=callbacks, 
                    sample_weight=sample_weight, validation_data=validation_data, 
                    validation_freq=validation_freq)
                

        del x_scaled, y_scaled
        if return_history:
            return self, history
        else:
            return self
    @staticmethod
    def generate_batches(x, y, batch_size, revert_order=True):
            n_batches = int(len(x) / batch_size)
            x = x[:int(n_batches*batch_size)]
            y = y[:int(n_batches*batch_size)]
            
            time_lengths = [x_let.shape[0] for x_let in x]
            idc = list(np.argsort(time_lengths).astype(int))
            
            x = [x[i] for i in idc]
            y = [y[i] for i in idc]
            while True:
                x_pad = []
                y_pad = []
                for batch in range(n_batches):
                    x_batch = x[batch*batch_size:(batch+1)*batch_size]
                    y_batch = y[batch*batch_size:(batch+1)*batch_size]
                    

                    if revert_order:
                        if np.random.randn()>0:
                            # x_batch = np.flip(x_batch, axis=1)
                            # y_batch = np.flip(y_batch, axis=1)
                            x_batch = [np.flip(xx, axis=1) for xx in x_batch]
                            y_batch = [np.flip(yy, axis=1) for yy in y_batch]
                    
                    
                    
                    x_padlet = pad_sequences(x_batch , dtype='float32' )
                    y_padlet = pad_sequences(y_batch , dtype='float32' )
                    
                        
                    x_pad.append( x_padlet )
                    y_pad.append( y_padlet )
                
                new_order = np.arange(len(x_pad))
                np.random.shuffle(new_order)
                x_pad = [x_pad[i] for i in new_order]
                y_pad = [y_pad[i] for i in new_order]
                for x_padlet, y_padlet in zip(x_pad, y_pad):
                    yield (x_padlet, y_padlet)


    def prep_data(self, args):
        ''' Train the neural network using training data (eeg) and labels (sources).
        
        Parameters
        ----------
        *args : 
            Can be either two objects: 
                eeg : mne.Epochs/ numpy.ndarray
                    The simulated EEG data
                sources : mne.SourceEstimates/ list of mne.SourceEstimates
                    The simulated EEG data
                or only one:
                simulation : esinet.simulation.Simulation
                    The Simulation object

            - two objects: EEG object (e.g. mne.Epochs) and Source object (e.g. mne.SourceEstimate)
        
        optimizer : tf.keras.optimizers
            The optimizer that for backpropagation.
        learning_rate : float
            The learning rate for training the neural network
        validation_split : float
            Proportion of data to keep as validation set.
        delta : int/float
            The delta parameter of the huber loss function
        epochs : int
            Number of epochs to train. In one epoch all training samples 
            are used once for training.
        metrics : list/str
            The metrics to be used for performance monitoring during training.
        device : str
            The device to use, e.g. a graphics card.
        false_positive_penalty : float
            Defines weighting of false-positive predictions. Increase for conservative 
            inverse solutions, decrease for liberal prediction.
        batch_size : int
            The number of samples to simultaneously calculate the error 
            during backpropagation.
        loss : tf.keras.losses
            The loss function.
        sample_weight : numpy.ndarray
            Optional numpy array of sample weights.

        Return
        ------
        self : esinet.Net
            Method returns the object itself.

        '''

        
        eeg, sources = self._handle_data_input(args)
        self.info = eeg[0].info
        self.subject = sources.subject if type(sources) == mne.SourceEstimate \
            else sources[0].subject

        # Ensure that the forward model has the same 
        # channels as the eeg object
        self._check_model(eeg)

        # Handle EEG input
        if (type(eeg) == list and isinstance(eeg[0], util.EPOCH_INSTANCES)) or isinstance(eeg, util.EPOCH_INSTANCES):
            eeg = [eeg[i].get_data(copy=True) for i, _ in enumerate(eeg)]
        else:
            eeg = [sample_eeg[0] for sample_eeg in eeg]

        for i, eeg_sample in enumerate(eeg):
            if len(eeg_sample.shape) == 1:
                eeg[i] = eeg_sample[:, np.newaxis]
            if len(eeg_sample.shape) == 3:
                eeg[i] = eeg_sample[0]
        
        # check if temporal dimension has all-equal entries
        self.equal_temporal = np.all( np.array([sample_eeg.shape[-1] for sample_eeg in eeg]) == eeg[0].shape[-1])
        
        sources = [source.data for source in sources]

        # enforce shape: list of samples, samples of shape (channels/dipoles, time)
        assert len(sources[0].shape) == 2, "sources samples must be two-dimensional"
        assert len(eeg[0].shape) == 2, "eeg samples must be two-dimensional"
        assert type(sources) == list, "sources must be a list of samples"
        assert type(eeg) == list, "eeg must be a list of samples"
        assert type(sources[0]) == np.ndarray, "sources must be a list of numpy.ndarrays"
        assert type(eeg[0]) == np.ndarray, "eeg must be a list of numpy.ndarrays"
        

        # Scale sources
        y_scaled = self.scale_source(sources)
        # Scale EEG
        x_scaled = self.scale_eeg(eeg)

        # LSTM net expects dimensions to be: (samples, time, channels)
        x_scaled = [np.swapaxes(x,0,1) for x in x_scaled]
        y_scaled = [np.swapaxes(y,0,1) for y in y_scaled]
        
        # if self.model_type.lower() == 'convdip':
        #     x_scaled = [interp(x) for x in x_scaled]

        return x_scaled, y_scaled

    def scale_eeg(self, eeg):
        ''' Scales the EEG prior to training/ predicting with the neural 
        network.

        Parameters
        ----------
        eeg : numpy.ndarray
            A 3D matrix of the EEG data (samples, channels, time_points)
        
        Return
        ------
        eeg : numpy.ndarray
            Scaled EEG
        '''
        eeg_out = deepcopy(eeg)
        
        if self.scale_individually:
            for sample, eeg_sample in enumerate(eeg):
                # Common average ref:
                for time in range(eeg_sample.shape[-1]):
                    eeg_out[sample][:, time] -= np.mean(eeg_sample[:, time])
                    # eeg_out[sample][:, time] /= np.max(np.abs(eeg_sample[:, time]))
                    eeg_out[sample][:, time] /= eeg_out[sample][:, time].std()
                    
                    
        else:
            for sample, eeg_sample in enumerate(eeg):
                eeg_out[sample] = self.robust_minmax_scaler(eeg_sample)
                # Common average ref:
                for time in range(eeg_sample.shape[-1]):
                    eeg_out[sample][:, time] -= np.mean(eeg_sample[:, time])
        return eeg_out
    

    def scale_source(self, source):
        ''' Scales the sources prior to training the neural network.

        Parameters
        ----------
        source : numpy.ndarray
            A 3D matrix of the source data (samples, dipoles, time_points)
        
        Return
        ------
        source : numpy.ndarray
            Scaled sources
        '''
        source_out = deepcopy(source)
        # for sample in range(source.shape[0]):
        #     for time in range(source.shape[2]):
        #         # source_out[sample, :, time] /= source_out[sample, :, time].std()
        #         source_out[sample, :, time] /= np.max(np.abs(source_out[sample, :, time]))
        for sample, _ in enumerate(source):
            # source_out[sample, :, time] /= source_out[sample, :, time].std()
            source_out[sample] /= np.max(np.abs(source_out[sample]))

        return source_out
            
    @staticmethod
    def robust_minmax_scaler(eeg):
        lower, upper = [np.percentile(eeg, 25), np.percentile(eeg, 75)]
        return (eeg-lower) / (upper-lower)

    def predict(self, *args, verbose=1):
        ''' Predict sources from EEG data.

        Parameters
        ----------
        *args : 
            Can be either 
                eeg : mne.Epochs/ numpy.ndarray
                    The simulated EEG data
                sources : mne.SourceEstimates/ list of mne.SourceEstimates
                    The simulated EEG data
                or
                simulation : esinet.simulation.Simulation
                    The Simulation object
        
        Return
        ------
        outsource : either numpy.ndarray (if dtype='raw') or mne.SourceEstimate instance
        '''
        
        eeg, _ = self._handle_data_input(args)

        if isinstance(eeg, util.EVOKED_INSTANCES):
            # Ensure there are no extra channels in our EEG
            eeg = eeg.pick(self.fwd.ch_names)    

            sfreq = eeg.info['sfreq']
            tmin = eeg.tmin
            eeg = eeg.data
            # add empty trial dimension
            eeg = np.expand_dims(eeg, axis=0)
            if len(eeg.shape) == 2:
                # add empty time dimension
                eeg = np.expand_dims(eeg, axis=2)
        elif isinstance(eeg, util.EPOCH_INSTANCES):
            # Ensure there are no extra channels in our EEG
            eeg = eeg.pick(self.fwd.ch_names)
            eeg.load_data()

            sfreq = eeg.info['sfreq']
            tmin = eeg.tmin
            eeg = eeg._data
        elif isinstance(eeg, list) and isinstance(eeg[0], util.EPOCH_INSTANCES):
            sfreq = eeg[0].info['sfreq']
            tmin = eeg[0].tmin
            eeg = [e.get_data(copy=True)[0] for e in eeg]
            
        # else:
        #     msg = f'eeg must be of type <mne.EvokedArray> or <mne.epochs.EpochsArray>; got {type(eeg)} instead.'
        #     raise ValueError(msg)
        # Prepare EEG to ensure common average reference and appropriate scaling
        # eeg_prep =  self._prep_eeg(eeg)
        eeg_prep = self.scale_eeg(deepcopy(eeg))
        
        # Reshape to (samples, time, channels)
        eeg_prep = [np.swapaxes(e, 0, 1) for e in eeg_prep]
        
        if self.model_type.lower() == 'convdip':
            print("interpolating for convdip...")
            elec_pos = _find_topomap_coords(self.info, self.info.ch_names)
            interpolator = self.make_interpolator(elec_pos, res=self.interp_channel_shape[0])
            eeg_prep_interp = deepcopy(eeg_prep)
            for i, sample in tqdm(enumerate(eeg_prep)):
                list_of_time_slices = []
                for time_slice in sample:
                    time_slice_interp = interpolator.set_values(time_slice)()[::-1]
                    time_slice_interp = time_slice_interp[:, :, np.newaxis]
                    list_of_time_slices.append(time_slice_interp)
                eeg_prep_interp[i] = np.stack(list_of_time_slices, axis=0)
                eeg_prep_interp[i][np.isnan(eeg_prep_interp[i])] = 0
            eeg_prep = eeg_prep_interp
            del eeg_prep_interp

            predicted_sources = self.predict_sources_interp(eeg_prep)
        else:
            # Predicted sources all in one go
            predicted_sources = self.predict_sources(eeg_prep)

        # Rescale Predicitons
        if self.rescale_sources.lower() == 'brent':
            predicted_sources_scaled = self._solve_p_wrap(predicted_sources, eeg)
        elif self.rescale_sources.lower() == 'rms':
            predicted_sources_scaled = self._scale_p_wrap(predicted_sources, eeg)
        else:
            print("Warning: <rescale_sources> is set to {self.rescale_sources}, but needs to be brent or rms. Setting to default (brent)")
            predicted_sources_scaled = self._solve_p_wrap(predicted_sources, eeg)



        # Convert sources (numpy.ndarrays) to mne.SourceEstimates objects
        if verbose>0:
            eeg_hat = list()
            for predicted_source in predicted_sources_scaled:
                eeg_hat.append( self.leadfield @ predicted_source )
            
            residual_variances = [round(self.calc_residual_variance(M_hat, M), 2) for M_hat, M in zip(eeg_hat, eeg)]
            print(f"Residual Variance(s): {residual_variances} [%]")

        predicted_source_estimate = [
            util.source_to_sourceEstimate(predicted_source_scaled, self.fwd, \
                sfreq=sfreq, tmin=tmin, subject=self.subject) \
                for predicted_source_scaled in predicted_sources_scaled]
        
        return predicted_source_estimate

    def calc_residual_variance(self, M_hat, M):
        return 100 *  np.sum( (M-M_hat)**2 ) / np.sum(M**2)

    def predict_sources(self, eeg):
        ''' Predict sources of 3D EEG (samples, channels, time) by reshaping 
        to speed up the process.
        
        Parameters
        ----------
        eeg : numpy.ndarray
            3D numpy array of EEG data (samples, channels, time)
        '''
        assert len(eeg[0].shape)==2, 'eeg must be a list of 2D numpy array of dim (channels, time)'
        predicted_sources = [self.model.predict(e[:, np.newaxis], verbose=self.verbose)[:,0].T for e in eeg]
        # predicted_sources = np.swapaxes(predicted_sources,1,2)
        # predicted_sources = [np.swapaxes(src, 0, 1) for src in predicted_sources]
        # predicted_sources = [np.swapaxes(src, 0, 1) for src in predicted_sources]
        

        return predicted_sources

    def predict_sources_interp(self, eeg):
        ''' Predict sources of 3D EEG (samples, channels, time) by reshaping 
        to speed up the process.
        
        Parameters
        ----------
        eeg : numpy.ndarray
            3D numpy array of EEG data (samples, channels, time)
        '''
        assert len(eeg[0].shape)==4, 'eeg must be a list of 4D numpy array of dim (time, height, width, 1)'

        predicted_sources = [self.model.predict(e[np.newaxis, :, :], verbose=self.verbose)[0] for e in eeg]
            
        # predicted_sources = np.swapaxes(predicted_sources,1,2)
        predicted_sources = [np.swapaxes(src, 0, 1) for src in predicted_sources]
        # print("shape of predicted sources: ", predicted_sources[0].shape)

        return predicted_sources

    def _scale_p_wrap(self, y_est, x_true):
        ''' Wrapper for parallel (or, alternatively, serial) scaling of 
        predicted sources.
        '''

        # assert len(y_est[0].shape) == 3, 'Sources must be 3-Dimensional'
        # assert len(x_true.shape) == 3, 'EEG must be 3-Dimensional'
        y_est_scaled = deepcopy(y_est)

        for trial, _ in enumerate(x_true):
            for time in range(x_true[trial].shape[-1]):
                scaled = self.scale_p(y_est[trial][:, time], x_true[trial][:, time])
                y_est_scaled[trial][:, time] = scaled

        return y_est_scaled

    def _solve_p_wrap(self, y_est, x_true):
        ''' Wrapper for parallel (or, alternatively, serial) scaling of 
        predicted sources.
        '''
        # assert len(y_est.shape) == 3, 'Sources must be 3-Dimensional'
        # assert len(x_true.shape) == 3, 'EEG must be 3-Dimensional'

        y_est_scaled = deepcopy(y_est)

        for trial, _ in enumerate(x_true):
            for time in range(x_true[trial].shape[-1]):
                scaled = self.solve_p(y_est[trial][:, time], x_true[trial][:, time])
                y_est_scaled[trial][:, time] = scaled

        return y_est_scaled

    # @staticmethod
    # def _prep_eeg(eeg):
    #     ''' Takes a 3D EEG array and re-references to common average and scales 
    #     individual scalp maps to max(abs(scalp_map) == 1
    #     '''
    #     assert len(eeg.shape) == 3, 'Input array <eeg> has wrong shape.'

    #     eeg_prep = deepcopy(eeg)
    #     for trial in range(eeg_prep.shape[0]):
    #         for time in range(eeg_prep.shape[2]):
    #             # Common average reference
    #             eeg_prep[trial, :, time] -= np.mean(eeg_prep[trial, :, time])
    #             # Scaling
    #             eeg_prep[trial, :, time] /= np.max(np.abs(eeg_prep[trial, :, time]))
    #     return eeg_prep

    def evaluate_mse(self, *args):
        ''' Evaluate the model regarding mean squared error
        
        Parameters
        ----------
        *args : 
            Can be either 
                eeg : mne.Epochs/ numpy.ndarray
                    The simulated EEG data
                sources : mne.SourceEstimates/ list of mne.SourceEstimates
                    The simulated EEG data
                or
                simulation : esinet.simulation.Simulation
                    The Simulation object

        Return
        ------
        mean_squared_errors : numpy.ndarray
            The mean squared error of each sample

        Example
        -------
        net = Net()
        net.fit(simulation)
        mean_squared_errors = net.evaluate_mse(simulation)
        print(mean_squared_errors.mean())
        '''

        eeg, sources = self._handle_data_input(args)
        
        y_hat = self.predict(eeg)
        
        if type(y_hat) == list:
            y_hat = np.stack([y.data for y in y_hat], axis=0)
        else:
            y_hat = y_hat.data

        if type(sources) == list:
            y = np.stack([y.data for y in sources], axis=0)
        else:
            y = sources.data
        
        if len(y_hat.shape) == 2:
            y = np.expand_dims(y, axis=0)
            y_hat = np.expand_dims(y_hat, axis=0)

        mean_squared_errors = np.mean((y_hat - y)**2, axis=1)
        return mean_squared_errors


    def evaluate_nmse(self, *args):
        ''' Evaluate the model regarding normalized mean squared error
        
        Parameters
        ----------
        *args : 
            Can be either 
                eeg : mne.Epochs/ numpy.ndarray
                    The simulated EEG data
                sources : mne.SourceEstimates/ list of mne.SourceEstimates
                    The simulated EEG data
                or
                simulation : esinet.simulation.Simulation
                    The Simulation object

        Return
        ------
        normalized_mean_squared_errors : numpy.ndarray
            The normalized mean squared error of each sample

        Example
        -------
        net = Net()
        net.fit(simulation)
        normalized_mean_squared_errors = net.evaluate_nmse(simulation)
        print(normalized_mean_squared_errors.mean())
        '''

        eeg, sources = self._handle_data_input(args)
        
        y_hat = self.predict(eeg)
        
        if type(y_hat) == list:
            y_hat = np.stack([y.data for y in y_hat], axis=0)
        else:
            y_hat = y_hat.data

        if type(sources) == list:
            y = np.stack([y.data for y in sources], axis=0)
        else:
            y = sources.data
        
        if len(y_hat.shape) == 2:
            y = np.expand_dims(y, axis=0)
            y_hat = np.expand_dims(y_hat, axis=0)

        for s in range(y_hat.shape[0]):
            for t in range(y_hat.shape[2]):
                y_hat[s, :, t] /= np.max(np.abs(y_hat[s, :, t]))
                y[s, :, t] /= np.max(np.abs(y[s, :, t]))
        
        normalized_mean_squared_errors = np.mean((y_hat - y)**2, axis=1)
        
        return normalized_mean_squared_errors

    def _build_model(self):
        ''' Build the neural network architecture using the 
        tensorflow.keras.Sequential() API. Depending on the input data this 
        function will either build:

        (1) A simple single hidden layer fully connected ANN for single time instance data
        (2) A LSTM network for spatio-temporal prediction
        '''
        if self.model_type.lower() == 'convdip':
            self._build_convdip_model()
        elif self.model_type.lower() == "cnn":
            self._build_cnn_model()
        elif self.model_type.lower() == 'fc':
            self._build_fc_model()
        elif self.model_type.lower() == 'lstm':
            self._build_temporal_model()
        else:
            self._build_temporal_model()

        if self.verbose:
            self.model.summary()
    
    
    def _build_temporal_model(self):
        ''' Build the temporal artificial neural network model using LSTM layers.
        '''
        name = "LSTM Model"
        self.model = keras.Sequential(name=name)
        tf.keras.backend.set_image_data_format('channels_last')
        input_shape = (None, self.n_channels)
        
        # LSTM layers
        if isinstance(self.n_lstm_units, (tuple, list)):
            self.n_lstm_units = self.n_lstm_units[0]
        # Dropout
        if isinstance(self.dropout, (tuple, list)):
            self.dropout = self.dropout[0]

        # Model Architecture
        inputs = tf.keras.Input(shape=input_shape, name='Input')
        ## FC-Path
        fc1 = TimeDistributed(Dense(self.n_dense_units, 
                    activation=self.activation_function), 
                    name='FC1')(inputs)
        fc1 = Dropout(self.dropout)(fc1)
        direct_out = TimeDistributed(Dense(self.n_dipoles, 
            activation="linear"),
            name='FC2')(fc1)
        # LSTM Path
        lstm1 = Bidirectional(LSTM(self.n_lstm_units, return_sequences=True, 
            input_shape=(None, self.n_dense_units), dropout=self.dropout), 
            name='LSTM1')(fc1)
        mask = TimeDistributed(Dense(self.n_dipoles, 
                    activation="sigmoid"), 
                    name='Mask')(lstm1)
        
        # Combination
        multi = multiply([direct_out, mask], name="multiply")
        self.model = tf.keras.Model(inputs=inputs, outputs=multi, name='Contextualizer')
        if self.l1_reg is not None:
            self.model.add_loss(self.l1_reg * self.l1_sparsity(multi))
        
    def _build_fc_model(self):
        ''' Build the temporal artificial neural network model using LSTM layers.
        '''
        # self.model = keras.Sequential(name=name)
        tf.keras.backend.set_image_data_format('channels_last')
        input_shape = (None, self.n_channels)
        # self.model.add(InputLayer(input_shape=input_shape, name='Input'))
        inputs = tf.keras.Input(shape=input_shape, name='Input_FC')
        
  
        if not isinstance(self.dropout, (tuple, list)):
            dropout = [self.dropout]*self.n_lstm_layers
        else:
            dropout = self.dropout
        
    
        # Hidden Dense layer(s):
        if not isinstance(self.n_dense_units, (tuple, list)):
            self.n_dense_units = [self.n_dense_units] * self.n_dense_layers
        
        if not isinstance(self.dropout, (tuple, list)):
            dropout = [self.dropout]*self.n_dense_layers
        else:
            dropout = self.dropout
        
        add_to = inputs
        for i in range(self.n_dense_layers):
            dense = TimeDistributed(Dense(self.n_dense_units[i], 
                activation=self.activation_function), name=f'FC_{i}')(add_to)
            dense = Dropout(dropout[i], name=f'Drop_{i}')(dense)
            add_to = dense

        # Final For-each layer:
        out = TimeDistributed(Dense(self.n_dipoles, activation='linear'), name='FC_Out')(dense)
        self.model = tf.keras.Model(inputs=inputs, outputs=out, name='FC_Model')
        if self.l1_reg is not None:
            self.model.add_loss(self.l1_reg * self.l1_sparsity(out))        


        # self.model.build(input_shape=input_shape)

    def _build_cnn_model(self):
        tf.keras.backend.image_data_format() == 'channels_last'
        input_shape = (None, self.n_channels, 1)

        inputs = tf.keras.Input(shape=input_shape, name='Input_CNN')
        fc = TimeDistributed(Conv1D(self.n_filters, self.n_channels, activation=self.activation_function, name="HL_D1"))(inputs)
        fc = TimeDistributed(Flatten())(fc)
            
        # LSTM path
        lstm1 = Bidirectional(GRU(self.n_lstm_units, return_sequences=True), name='GRU')(fc)
        mask = TimeDistributed(Dense(self.n_dipoles, activation="sigmoid"), name='Mask')(lstm1)

        direct_out = TimeDistributed(Dense(self.n_dipoles, activation="tanh", name="Output_Final"))(fc)
        multi = multiply([direct_out, mask], name="multiply")

        self.model = tf.keras.Model(inputs=inputs, outputs=multi, name='Contextual_CNN_Model')
        if self.l1_reg is not None:
            self.model.add_loss(self.l1_reg * self.l1_sparsity(multi))
        # model.compile(loss=tf.keras.losses.CosineSimilarity(), optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

    def _build_convdip_model(self):
        # self.model = keras.Sequential(name='ConvDip-model')
        tf.keras.backend.set_image_data_format('channels_last')
        input_shape = (None, *self.interp_channel_shape, 1)
        inputs = tf.keras.Input(shape=input_shape, name='Input_ConvDip')
        # Some definitions
              

        # Hidden Dense layer(s):
        if not isinstance(self.n_dense_units, (tuple, list)):
            self.n_dense_units = [self.n_dense_units] * self.n_dense_layers
        
        if not isinstance(self.dropout, (tuple, list)):
            dropout = [self.dropout]*(self.n_dense_layers+self.n_lstm_layers)
        else:
            dropout = self.dropout

        # self.model.add(InputLayer(input_shape=input_shape, name='Input'))
        add_to = inputs
        for i in range(self.n_lstm_layers):
            conv = TimeDistributed(Conv2D(self.n_filters, self.kernel_size, activation=self.activation_function, name=f"Conv2D_{i}"))(add_to)
            conv = Dropout(dropout[i], name=f'Drop_conv2d_{i}')(conv)
            add_to = conv



        flat = TimeDistributed(Flatten())(conv)
        add_to = flat
        for i in range(self.n_dense_layers):
            dense = TimeDistributed(Dense(self.n_dense_units[i], activation=self.activation_function, name=f'FC_{i}'))(add_to)
            dense = Dropout(dropout[i], name=f'Drop_FC_{i}')(dense)
            add_to = dense

        # Outout Layer
        out = TimeDistributed(Dense(self.n_dipoles, activation='linear'), name='FC_Out')(dense)
        self.model = tf.keras.Model(inputs=inputs, outputs=out, name='ConvDip_Model')
        if self.l1_reg is not None:
            self.model.add_loss(self.l1_reg * self.l1_sparsity(out))
        

    @staticmethod
    def l1_sparsity(x):
        new_x = tf.math.l2_normalize(x)
        return K.mean(K.abs(new_x))
        
  

    def _freeze_lstm(self):
        for i, layer in enumerate(self.model.layers):
            if 'LSTM' in layer.name or 'RNN' in layer.name:
                print(f'freezing {layer.name}')
                self.model.layers[i].trainable = False
    
    def _unfreeze_lstm(self):
        for i, layer in enumerate(self.model.layers):
            if 'LSTM' in layer.name or 'RNN' in layer.name:
                print(f'unfreezing {layer.name}')
                self.model.layers[i].trainable = True
    
    def _freeze_fc(self):
        for i, layer in enumerate(self.model.layers):
            if 'FC' in layer.name and not 'Out' in layer.name:
                print(f'freezing {layer.name}')
                self.model.layers[i].trainable = False

    def _unfreeze_fc(self):
        for i, layer in enumerate(self.model.layers):
            if 'FC' in layer.name:
                print(f'unfreezing {layer.name}')
                self.model.layers[i].trainable = True

    def _build_perceptron_model(self):
        ''' Build the artificial neural network model using Dense layers.
        '''
        input_shape = (None, None, self.n_channels)
        tf.keras.backend.set_image_data_format('channels_last')

        self.model = keras.Sequential()
        # Add hidden layers
        for _ in range(self.n_dense_layers):
            self.model.add(TimeDistributed(Dense(units=self.n_dense_units,
                                activation=self.activation_function)))
        # Add output layer
        self.model.add(TimeDistributed(Dense(self.n_dipoles, activation='linear')))
        
        # Build model with input layer
        self.model.build(input_shape=input_shape)

    


    def _check_model(self, eeg):
        ''' Check whether the current forward model has the same 
        channels as the eeg. Rebuild model if thats not the case.
        
        Parameters
        ----------
        eeg : mne.Epochs or equivalent
            The EEG instance.

        '''
        # Dont do anything if model is already built.
        if self.compiled:
            return
        
        # Else assure that channels are appropriate
        if eeg[0].ch_names != self.fwd.ch_names:
            self.fwd = self.fwd.pick_channels(eeg[0].ch_names)
            # Write all changes to the attributes
            self._embed_fwd(self.fwd)
        
        self.n_timepoints = len(eeg[0].times)
        # Finally, build model
        self._build_model()
            
    def scale_p(self, y_est, x_true):
        ''' Scale the prediction to yield same estimated GFP as true GFP

        Parameters
        ---------
        y_est : numpy.ndarray
            The estimated source vector.
        x_true : numpy.ndarray
            The original input EEG vector.
        
        Return
        ------
        y_est_scaled : numpy.ndarray
            The scaled estimated source vector.
        
        '''
        # Check if y_est is just zeros:
        if np.max(y_est) == 0:
            return y_est
        y_est = np.squeeze(np.array(y_est))
        x_true = np.squeeze(np.array(x_true))
        # Get EEG from predicted source using leadfield
        x_est = np.matmul(self.leadfield, y_est)

        gfp_true = np.std(x_true)
        gfp_est = np.std(x_est)
        scaler = gfp_true / gfp_est
        y_est_scaled = y_est * scaler
        return y_est_scaled
        
    def solve_p(self, y_est, x_true):
        '''
        Parameters
        ---------
        y_est : numpy.ndarray
            The estimated source vector.
        x_true : numpy.ndarray
            The original input EEG vector.
        
        Return
        ------
        y_scaled : numpy.ndarray
            The scaled estimated source vector.
        
        '''
        # Check if y_est is just zeros:
        if np.max(y_est) == 0:
            return y_est
        y_est = np.squeeze(np.array(y_est))
        x_true = np.squeeze(np.array(x_true))
        # Get EEG from predicted source using leadfield
        x_est = np.matmul(self.leadfield, y_est)

        # optimize forward solution
        tol = 1e-9
        options = dict(maxiter=1000, disp=False)

        # base scaling
        rms_est = np.mean(np.abs(x_est))
        rms_true = np.mean(np.abs(x_true))
        base_scaler = rms_true / rms_est

        
        opt = minimize_scalar(self.correlation_criterion, args=(self.leadfield, y_est* base_scaler, x_true), \
            bounds=(0, 1), method='bounded', options=options, tol=tol)
        
        # opt = minimize_scalar(self.correlation_criterion, args=(self.leadfield, y_est* base_scaler, x_true), \
        #     bounds=(0, 1), method='L-BFGS-B', options=options, tol=tol)

        scaler = opt.x
        y_scaled = y_est * scaler * base_scaler
        return y_scaled

    @staticmethod
    def correlation_criterion(scaler, leadfield, y_est, x_true):
        ''' Perform forward projections of a source using the leadfield.
        This is the objective function which is minimized in Net::solve_p().
        
        Parameters
        ----------
        scaler : float
            scales the source y_est
        leadfield : numpy.ndarray
            The leadfield (or sometimes called gain matrix).
        y_est : numpy.ndarray
            Estimated/predicted source.
        x_true : numpy.ndarray
            True, unscaled EEG.
        '''

        x_est = np.matmul(leadfield, y_est) 
        error = np.abs(pearsonr(x_true-x_est, x_true)[0])
        return error
    
    def save(self, path, name='model'):
        # get list of folders in path
        list_of_folders = os.listdir(path)
        model_ints = []
        for folder in list_of_folders:
            full_path = os.path.join(path, folder)
            if not os.path.isdir(full_path):
                continue
            if folder.startswith(name):
                new_integer = int(folder.split('_')[-1])
                model_ints.append(new_integer)
        if len(model_ints) == 0:
            model_name = f'\\{name}_0'
        else:
            model_name = f'\\{name}_{max(model_ints)+1}'
        new_path = path+model_name
        os.mkdir(new_path)

        # Save model only
        self.model.save(new_path)
        # self.model.save_weights(new_path)

        # copy_model = tf.keras.models.clone_model(self.model)
        # copy_model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001, momentum=0.35), loss='huber')
        # copy_model.set_weights(self.model.get_weights())


        
        # Save rest
        # Delete model since it is not serializable
        self.model = None

        with open(new_path + '\\instance.pkl', 'wb') as f:
            pkl.dump(self, f)
        
        # Attach model again now that everything is saved
        try:
            self.model = tf.keras.models.load_model(new_path, custom_objects={'loss': self.loss})
        except:
            print("Load model did not work using custom_objects. Now trying it without...")
            self.model = tf.keras.models.load_model(new_path)
        
        return self

    @staticmethod
    def make_interpolator(elec_pos, res=9, ch_type='eeg', image_interp="linear"):
        extrapolate = _check_extrapolate('auto', ch_type)
        sphere = sphere = _check_sphere(None)
        outlines = 'head'
        outlines = _make_head_outlines(sphere, elec_pos, outlines, (0., 0.))
        border = 'mean'
        extent, Xi, Yi, interpolator = _setup_interp(
            elec_pos, res, image_interp, extrapolate, outlines, border)
        interpolator.set_locations(Xi, Yi)

        return interpolator

    

class CovNet:
    ''' Class for the Covariance-based Convolutional Neural Network (CovCNN) for EEG inverse solutions.
    
    Attributes
    ----------
    forward : mne.Forward
        The mne-python Forward model instance.
    '''

    def __init__(self, forward, name="Cov-CNN", n_filters="auto", 
                activation_function="tanh", batch_size="auto", 
                n_timepoints=20, batch_repetitions=10,
                learning_rate=1e-3, loss="cosine_similarity",
                n_sources=10, n_orders=2, epsilon=0.5, 
                snr_range=(1,100), alpha="auto", verbose=0, **kwargs):
        ''' Calculate inverse operator.

        Parameters
        ----------
        forward : mne.Forward
            The mne-python Forward model instance.
        alpha : float
            The regularization parameter.
        
        Return
        ------
        self : object returns itself for convenience
        '''
        # Leadfield
        self.forward = forward
        self.leadfield = deepcopy(forward["sol"]["data"])
        self.leadfield -= self.leadfield.mean(axis=0)

        n_channels, n_dipoles = self.leadfield.shape
        if batch_size == "auto":
            batch_size = n_dipoles
        if n_filters == "auto":
            n_filters = n_channels
            
        # Store Parameters
        
        
        # Architecture
        self.name = name
        self.n_filters = n_filters
        self.activation_function = activation_function
        # Training
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.loss = loss
        # Training Data
        self.n_timepoints = n_timepoints
        self.n_sources = n_sources
        self.n_orders = n_orders
        self.batch_repetitions = batch_repetitions
        self.snr_range = snr_range
        # Inference
        self.epsilon = epsilon
        # Other
        self.verbose = verbose
        print("Build Model:..")
        self.build_model()
        

    def predict(self, evoked) -> mne.SourceEstimate:
        source_mat = self.apply_model(evoked)
        stc = self.source_to_object(source_mat, evoked)

        return stc

    def apply_model(self, evoked) -> np.ndarray:
        y = deepcopy(evoked.data)
        y -= y.mean(axis=0)
        print("werks")
        # y /= np.linalg.norm(y, axis=0)

        n_channels, n_times = y.shape

        # Compute Data Covariance Matrix
        C = y@y.T
        # Scale
        C /= abs(C).max()
        

        # Add empty batch and (color-) channel dimension
        C = C[np.newaxis, :, :, np.newaxis]
        gammas = self.model.predict(C, verbose=self.verbose)[0]
        gammas /= gammas.max()

        
        


        # Select dipole indices
        gammas[gammas<self.epsilon] = 0
        dipole_idc = np.where(gammas!=0)[0]
        print("Active dipoles: ", len(dipole_idc))

        # 1) Calculate weighted minimum norm solution at active dipoles
        n_dipoles = len(gammas)
        y = deepcopy(evoked.data)
        y -= y.mean(axis=0)
        x_hat = np.zeros((n_dipoles, n_times))
        L = self.leadfield[:, dipole_idc]
        W = np.diag(np.linalg.norm(L, axis=0))
        x_hat[dipole_idc, :] = np.linalg.inv(L.T @ L + W.T@W) @ L.T @ y

        
        return x_hat        
        
        
    def fit(self, sim, patience=7, validation_split=0.05, epochs=300, return_history=True):
        callbacks = [tf.keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True),]
        
        x_train = np.stack(
            self.prep_x([ep.average().data for ep in sim.eeg_data])
            , axis=0)
        y_train = np.stack(
            self.prep_y([stc.data for stc in sim.source_data])
            , axis=0)
        # print(x_train.shape, y_train.shape)
        
        history = self.model.fit(x_train, y_train, epochs=epochs, 
            validation_split=validation_split, callbacks=callbacks)

        if return_history:
            return self, history
        return self
    def prep_y(self, y):
        n_samples = len(y)
        y_scaled = []

        for i in range(n_samples):
            y_sample = y[i]

            y_sample = np.mean(abs(y_sample), axis=1)
            thr = y_sample.max()*1e-3
            y_sample = (y_sample>thr).astype(float)
            y_scaled.append(y_sample)
        return y_scaled


    def prep_x(self, x):
        n_samples = len(x)
        C_scaled = []
        for i in range(n_samples):
            x_sample = x[i]
            # Common Average Reference
            x_sample -= x_sample.mean(axis=0)
            x_sample /= np.linalg.norm(x_sample, axis=0)
            C = x_sample @ x_sample.T
            C /= abs(C).max()
            C_scaled.append(C)
        return C_scaled
            

    def build_model(self,):
        n_channels, n_dipoles = self.leadfield.shape

        inputs = tf.keras.Input(shape=(n_channels, n_channels, 1), name='Input')

        cnn1 = Conv2D(self.n_filters, (1, n_channels),
                    activation=self.activation_function, padding="valid",
                    name='CNN1')(inputs)

        flat = Flatten()(cnn1)
        
        fc1 = Dense(200, 
            activation=self.activation_function, 
            name='FC1')(flat)
        out = Dense(n_dipoles, 
            activation="sigmoid", 
            name='Output')(fc1)

        model = tf.keras.Model(inputs=inputs, outputs=out, name='CovCNN')
        model.compile(loss=self.loss, optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        if self.verbose > 0:
            model.summary()
        
        self.model = model
 
    def source_to_object(self, source_mat, evoked):
        ''' Converts the source_mat matrix to an mne.SourceEstimate object '''
        # Convert source to mne.SourceEstimate object
        source_model = self.forward['src']
        vertices = [source_model[0]['vertno'], source_model[1]['vertno']]
        tmin = evoked.tmin
        sfreq = evoked.info["sfreq"]
        tstep = 1/sfreq
        subject = evoked.info["subject_info"]

        if type(subject) == dict:
            subject = "bst_raw"

        if subject is None:
            subject = "fsaverage"
        
        stc = mne.SourceEstimate(source_mat, vertices, tmin=tmin, tstep=tstep, subject=subject, verbose=self.verbose)
        return stc

    def save(self, path, name='model'):
        # get list of folders in path
        list_of_folders = os.listdir(path)
        model_ints = []

        for folder in list_of_folders:
            full_path = os.path.join(path, folder)
            if not os.path.isdir(full_path):
                continue
            if folder.startswith(name):
                new_integer = int(folder.split('_')[-1])
                model_ints.append(new_integer)
        if len(model_ints) == 0:
            model_name = f'\\{name}_0'
        else:
            model_name = f'\\{name}_{max(model_ints)+1}'

        new_path = path+model_name
        os.mkdir(new_path)

        # Save model only
        self.model.save(new_path)

        
        # Save rest
        # Delete model since it is not serializable
        self.model = None

        with open(new_path + '\\instance.pkl', 'wb') as f:
            pkl.dump(self, f)
        
        # Attach model again now that everything is saved
        try:
            self.model = tf.keras.models.load_model(new_path, custom_objects={'loss': self.loss})
        except:
            print("Load model did not work using custom_objects. Now trying it without...")
            self.model = tf.keras.models.load_model(new_path)
        
        return self

def build_nas_lstm(hp):
    ''' Find optimal model using keras tuner.
    '''
    n_dipoles = 1284
    n_channels = 61
    n_lstm_layers = hp.Int("lstm_layers", min_value=0, max_value=3, step=1)
    n_dense_layers = hp.Int("dense_layers", min_value=0, max_value=3, step=1)
    activation_out = 'linear'  # hp.Choice(f"activation_out", ["tanh", 'sigmoid', 'linear'])
    activation = 'relu'  # hp.Choice('actvation_all', all_acts)

    model = keras.Sequential(name='LSTM_NAS')
    tf.keras.backend.set_image_data_format('channels_last')
    input_shape = (None, n_channels)
    model.add(InputLayer(input_shape=input_shape, name='Input'))

    # LSTM layers
    for i in range(n_lstm_layers):
        n_lstm_units = hp.Int(f"lstm_units_l-{i}", min_value=25, max_value=500, step=1)
        dropout = hp.Float(f"dropout_lstm_l-{i}", min_value=0, max_value=0.5)
        model.add(Bidirectional(LSTM(n_lstm_units, 
            return_sequences=True, input_shape=input_shape, 
            dropout=dropout, activation=activation), 
            name=f'LSTM{i}'))
    # Hidden Dense layer(s):
    for i in range(n_dense_layers):
        n_dense_units = hp.Int(f"dense_units_l-{i}", min_value=50, max_value=1000, step=1)
        dropout = hp.Float(f"dropout_dense_l-{i}", min_value=0, max_value=0.5)

        model.add(TimeDistributed(Dense(n_dense_units, 
            activation=activation), name=f'FC_{i}'))
        model.add(Dropout(dropout, name=f'DropoutLayer_dense_{i}'))

    # Final For-each layer:
    model.add(TimeDistributed(
        Dense(n_dipoles, activation=activation_out), name='FC_Out')
    )
    model.build(input_shape=input_shape)
    momentum = hp.Float('Momentum', min_value=0, max_value=0.9)
    nesterov = hp.Choice('Nesterov', [False, True])
    learning_rate = hp.Choice('learning_rate', [0.01, 0.001])
    optimizer = hp.Choice("Optimizer", [0,1,2])
    optimizers = [keras.optimizers.RMSprop(learning_rate=learning_rate, momentum=momentum), keras.optimizers.Adam(learning_rate=learning_rate), keras.optimizers.SGD(learning_rate=learning_rate, nesterov=nesterov)]
    model.compile(
        optimizer=optimizers[optimizer],
        loss="huber",
        # metrics=[tf.keras.metrics.AUC()],
        # metrics=[evaluate.modified_auc_metric()],
        metrics=[evaluate.auc],
    )
    return model

    