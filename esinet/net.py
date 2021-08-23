import mne
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import (LSTM, Dense, Flatten, Bidirectional, 
    TimeDistributed, InputLayer, Activation, Reshape, concatenate, Dropout)
from tensorflow.keras import backend as K
from keras.layers.core import Lambda
import datetime
# from sklearn import linear_model
import numpy as np
from scipy.stats import pearsonr
from copy import deepcopy
from time import time

from tensorflow.python.keras.backend import dropout
from . import util
from . import losses

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
    
    def __init__(self, fwd, n_dense_layers=1, n_lstm_layers=1, 
        n_dense_units=100, n_lstm_units=100, activation_function='swish', 
        n_jobs=-1, model_type='auto', verbose=True):

        self._embed_fwd(fwd)
        
        self.n_dense_layers = n_dense_layers
        self.n_lstm_layers = n_lstm_layers
        self.n_dense_units = n_dense_units
        self.n_lstm_units = n_lstm_units
        self.activation_function = activation_function
        # self.default_loss = tf.keras.losses.Huber(delta=delta)
        self.default_loss = losses.weighted_huber_loss
        # self.parallel = parallel
        self.n_jobs = n_jobs
        self.model_type = model_type
        self.compiled = False
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
        
    def _handle_data_input(self, arguments):
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
        validation_split=0.1, epochs=50, metrics=None, device=None, 
        false_positive_penalty=2, delta=1., batch_size=128, loss=None, 
        sample_weight=None, return_history=False, dropout=0.2, patience=7, 
        tensorboard=False):
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

        self.dropout = dropout
        eeg, sources = self._handle_data_input(args)
        self.subject = sources.subject if type(sources) == mne.SourceEstimate \
            else sources[0].subject
        # Decide gross model architecture
        if self.model_type == 'single' or (self.model_type=='auto' and type(sources) != list):
            self.temporal = False
        else:
            self.temporal = True
        
        # Ensure that the forward model has the same 
        # channels as the eeg object
        self._check_model(eeg)

        # Handle EEG input
        eeg = eeg.get_data()

        # Handle source input
        if type(sources) == mne.source_estimate.SourceEstimate:
            sources = sources.data.T
            # add empty temporal dimension
            sources = np.expand_dims(sources, axis=2)
        elif type(sources) == list:
            if type(sources[0]) == mne.source_estimate.SourceEstimate:
                sources = np.stack([source.data for source in sources], axis=0)
        
        
        # Extract data
        y = sources
        x = eeg

        # Prepare data
        # Scale sources
        y_scaled = self.scale_source(y)
        # Scale EEG
        x_scaled = self.scale_eeg(x)

        # Early stopping
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', \
            mode='min', verbose=self.verbose, patience=patience, restore_best_weights=True)
        if tensorboard:
            log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            tensorboard_callback = tf.keras.callbacks.TensorBoard(
                log_dir=log_dir, histogram_freq=1)
            callbacks = [es, tensorboard_callback]
        else:
            callbacks = [es]
        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        if loss is None:
            loss = self.default_loss(weight=false_positive_penalty, delta=delta)

        elif type(loss) == list:
            loss = loss[0](*loss[1])
        if metrics is None:
            metrics = [self.default_loss(weight=false_positive_penalty, delta=delta)]
        
        # Compile if it wasnt compiled before
        if not self.compiled:
            self.model.compile(optimizer, loss, metrics=metrics)
            self.compiled = True

        if self.temporal:
            # LSTM net expects dimensions to be: (samples, time, channels)
            x_scaled = np.swapaxes(x_scaled,1,2)
            y_scaled = np.swapaxes(y_scaled,1,2)
        else:
            # Squeeze to remove empty time dimension
            x_scaled = np.squeeze(x_scaled)
            y_scaled = np.squeeze(y_scaled)

        if device is None:
            history = self.model.fit(x_scaled, y_scaled, 
                epochs=epochs, batch_size=batch_size, shuffle=True, 
                validation_split=validation_split, verbose=self.verbose, 
                callbacks=callbacks, sample_weight=sample_weight)
        else:
            with tf.device(device):
                history = self.model.fit(x_scaled, y_scaled, 
                    epochs=epochs, batch_size=batch_size, shuffle=True, 
                    validation_split=validation_split, verbose=self.verbose,
                    callbacks=callbacks, sample_weight=sample_weight)
                
        if return_history:
            return self, history
        else:
            return self

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

        # Common average ref
        for sample in range(eeg.shape[0]):
            for time in range(eeg.shape[2]):
                eeg[sample, :, time] -= np.mean(eeg[sample, :, time])
                eeg[sample, :, time] /= eeg[sample, :, time].std()
        
        # Normalize
        # for sample in range(eeg.shape[0]):
        #     eeg[sample] /= eeg[sample].std()

        return eeg
            

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
        for sample in range(source.shape[0]):
            for time in range(source.shape[2]):
                source[sample, :, time] /= np.max(np.abs(source[sample, :, time]))

        return source
            

    def predict(self, *args):
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
            eeg = eeg.pick_channels(self.fwd.ch_names)    

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
            eeg = eeg.pick_channels(self.fwd.ch_names)
            eeg.load_data()

            sfreq = eeg.info['sfreq']
            tmin = eeg.tmin
            eeg = eeg._data
        else:
            msg = f'eeg must be of type <mne.EvokedArray> or <mne.epochs.EpochsArray>; got {type(eeg)} instead.'
            raise ValueError(msg)
        
        # Prepare EEG to ensure common average reference and appropriate scaling
        # eeg_prep =  self._prep_eeg(eeg)
        eeg_prep = self.scale_eeg(eeg)
        
        # Predicted sources all in one go
        if self.temporal:
            eeg_prep = np.swapaxes(eeg_prep, 1,2)
        predicted_sources = self.predict_sources(eeg_prep)       
        
        # Rescale Predicitons
        # predicted_sources_scaled = self._solve_p_wrap(predicted_sources, eeg)
        predicted_sources_scaled = self._scale_p_wrap(predicted_sources, eeg)

        # Convert sources (numpy.ndarrays) to mne.SourceEstimates objects
        if predicted_sources.shape[-1] == 1:
            predicted_source_estimate = [util.source_to_sourceEstimate(predicted_sources_scaled[:, :, 0], self.fwd, sfreq=sfreq, tmin=tmin, subject=self.subject)]
        else:    
            predicted_source_estimate = [util.source_to_sourceEstimate(predicted_sources_scaled[k], self.fwd, sfreq=sfreq, tmin=tmin, subject=self.subject)
                    for k in range(predicted_sources_scaled.shape[0])]

        if len(predicted_source_estimate) == 1:
            predicted_source_estimate = predicted_source_estimate[0]

        return predicted_source_estimate

    def predict_sources(self, eeg):
        ''' Predict sources of 3D EEG (samples, channels, time) by reshaping 
        to speed up the process.
        
        Parameters
        ----------
        eeg : numpy.ndarray
            3D numpy array of EEG data (samples, channels, time)
        '''
        assert len(eeg.shape)==3, 'eeg must be a 3D numpy array of dim (samples, channels, time)'
        if not self.temporal:
            # Predict sources all at once
            n_samples, n_elec, n_time = eeg.shape
            ## swap electrode and time axis
            eeg_tmp = np.swapaxes(eeg, 1, 2)
            ## reshape axis
            new_shape = (n_samples*n_time, n_elec)
            eeg_tmp = eeg_tmp.reshape(new_shape)
            ## predict
            
            predicted_sources = self.model.predict(eeg_tmp)
            
            ## Get to old shape
            predicted_sources = predicted_sources.reshape(n_samples, n_time, self.n_dipoles)
            predicted_sources = np.swapaxes(predicted_sources, 1,2)
        else:
            
            predicted_sources = self.model.predict(eeg)
            
            predicted_sources = np.swapaxes(predicted_sources,1,2)

        return predicted_sources

    def _scale_p_wrap(self, y_est, x_true):
        ''' Wrapper for parallel (or, alternatively, serial) scaling of 
        predicted sources.
        '''
        assert len(y_est.shape) == 3, 'Sources must be 3-Dimensional'
        assert len(x_true.shape) == 3, 'EEG must be 3-Dimensional'

        y_est_scaled = deepcopy(y_est)

        for trial in range(x_true.shape[0]):
            for time in range(x_true.shape[2]):
                y_est_scaled[trial, :, time] = self.scale_p(y_est[trial, :, time], x_true[trial, :, time])

        return y_est_scaled

    def _solve_p_wrap(self, y_est, x_true):
        ''' Wrapper for parallel (or, alternatively, serial) scaling of 
        predicted sources.
        '''
        assert len(y_est.shape) == 3, 'Sources must be 3-Dimensional'
        assert len(x_true.shape) == 3, 'EEG must be 3-Dimensional'

        y_est_scaled = deepcopy(y_est)

        for trial in range(x_true.shape[0]):
            for time in range(x_true.shape[2]):
                y_est_scaled[trial, :, time] = self.solve_p(y_est[trial, :, time], x_true[trial, :, time])

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
        mean_squared_errors = net.evaluate(simulation)
        print(mean_squared_errors.mean())
        '''

        eeg, sources = self._handle_data_input(args)
        
        y_hat = self.mode.predict(eeg).data
        y = sources.data
        mean_squared_errors = np.mean((y_hat - y)**2, axis=0)

        return mean_squared_errors


    def _build_model(self):
        ''' Build the neural network architecture using the tensorflow.keras.Sequential() API. 
        Depending on the input data this function will either build:

        (1) A simple single hidden layer fully connected ANN for single time instance data
        (2) A LSTM network for spatio-temporal prediction
        '''
        if self.temporal:
            # self._build_temporal_model()
            self._build_temporal_model_v2()
            # self._build_temporal_model_v3()
            # self._build_temporal_model_v4()
        else:
            self._build_perceptron_model()
        

        if self.verbose:
            self.model.summary()
    
    def _build_temporal_model(self):
        ''' Build the temporal artificial neural network model using LSTM layers.
        '''
        print('yep, the oldest')
        self.model = keras.Sequential()
        tf.keras.backend.set_image_data_format('channels_last')
        input_shape = (self.n_timepoints, self.n_channels)
        self.model.add(InputLayer(input_shape=input_shape))
        
        for _ in range(self.n_lstm_layers):
            self.model.add(Bidirectional(LSTM(self.n_lstm_units, return_sequences=True, 
                input_shape=(self.n_timepoints, self.n_channels), 
                dropout=self.dropout, activation=self.activation_function)))
        self.model.add(Flatten())

        self.model.add(Dense(int(self.n_timepoints*self.n_dipoles), 
            activation='linear'))

        self.model.add(Dense(int(self.n_timepoints*self.n_dipoles), 
            activation='linear'))
        self.model.add(Reshape((self.n_timepoints, self.n_dipoles)))
        self.model.add(Activation('linear'))
        
        self.model.build(input_shape=input_shape)
        
    def _build_temporal_model_v2(self):
        ''' Build the temporal artificial neural network model using LSTM layers.
        '''
        self.model = keras.Sequential()
        tf.keras.backend.set_image_data_format('channels_last')
        input_shape = (None, self.n_channels)
        self.model.add(InputLayer(input_shape=input_shape))
        
        # LSTM layers
        for _ in range(self.n_lstm_layers):
            self.model.add(Bidirectional(LSTM(self.n_lstm_units, 
                return_sequences=True, input_shape=input_shape, 
                dropout=self.dropout, activation=self.activation_function)))

        # Hidden Dense layer(s):
        for _ in range(self.n_dense_layers-1):
            self.model.add(Dense(self.n_dense_units, 
                activation=self.activation_function))

        # Final For-each layer:
        self.model.add(TimeDistributed(
            Dense(self.n_dipoles, activation='linear'))
        )

        self.model.build(input_shape=input_shape)


    def _build_temporal_model_v3(self):
        ''' A mixed dense / LSTM network, inspired by:
        "Deep Burst Denoising" (Godarg et al., 2018)
        '''
        inputs = keras.Input(shape=(None, self.n_channels), name='Input')
        # SINGLE TIME FRAME PATH
        fc1 = TimeDistributed(Dense(self.n_dense_units, 
            activation=self.activation_function), 
            name='FC1')(inputs)
        fc1 = Dropout(self.dropout, name='Dropout1')(fc1)

        fc2 = TimeDistributed(Dense(self.n_dipoles,
            activation=self.activation_function), 
            name='FC2')(fc1)
        fc2 = Dropout(self.dropout, name='Dropout2')(fc2)

        model_s = keras.Model(inputs=inputs, outputs=fc2, 
            name='single_time_ frame_model')

        # MULTI TIME FRAME PATH
        lstm1 = Bidirectional(LSTM(self.n_lstm_units, return_sequences=True, 
            input_shape=(None, self.n_dense_units), dropout=self.dropout, 
            activation=self.activation_function), name='LSTM1')(fc1)

        concat = concatenate([lstm1, fc2], name='Concat')

        lstm2 = Bidirectional(LSTM(self.n_lstm_units, return_sequences=True, 
            input_shape=(None, self.n_dense_units), dropout=self.dropout, 
            activation=self.activation_function), name='LSTM2')(concat)

        output = TimeDistributed(Dense(self.n_dipoles), name='FC_Out')(lstm2)
        model_m = keras.Model(inputs=inputs, outputs=output, name='multi_time_frame_model')

        self.model = model_m
    

    def _build_temporal_model_v4(self):
        ''' A large and stupid model for testing
        '''
        n_timepoints = 3
        inputs = keras.Input(shape=(n_timepoints, self.n_channels), name='Input')
        # SINGLE TIME FRAME PATH
        fc1 = TimeDistributed(Dense(self.n_dense_units, 
            activation=self.activation_function), 
            name='FC1')(inputs)
        
        fc2 = TimeDistributed(Dense(self.n_dense_units,
            activation=self.activation_function), 
            name='FC2')(fc1)

        flat = Flatten(name='Flatten')(fc2)

        last_fc = Dense(int(self.n_dipoles*n_timepoints), name='FC_Out')(flat)
        out = Reshape((n_timepoints, self.n_dipoles))(last_fc)

        model_s = keras.Model(inputs=inputs, outputs=out)

        self.model = model_s
    
  

    def _freeze_lstm(self):
        for i, layer in enumerate(self.model.layers):
            if 'LSTM' in layer.name:
                print(f'freezing {layer.name}')

                self.model.layers[i].trainable = False
    
    def _unfreeze_lstm(self):
        for i, layer in enumerate(self.model.layers):
            if 'LSTM' in layer.name:
                self.model.layers[i].trainable = True
    
    def _freeze_fc(self):
        for i, layer in enumerate(self.model.layers):
            if 'FC' in layer.name and not 'Out' in layer.name:
                print(f'freezing {layer.name}')
                self.model.layers[i].trainable = False

    def _unfreeze_fc(self):
        for i, layer in enumerate(self.model.layers):
            if 'FC' in layer.name:
                self.model.layers[i].trainable = True

    def _build_perceptron_model(self):
        ''' Build the artificial neural network model using Dense layers.
        '''
        self.model = keras.Sequential()
        # Add hidden layers
        for _ in range(self.n_dense_layers):
            self.model.add(Dense(units=self.n_dense_units,
                                activation=self.activation_function))
        # Add output layer
        self.model.add(Dense(self.n_dipoles, activation='linear'))
        
        # Build model with input layer
        self.model.build(input_shape=(None, self.n_channels))

    


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
        if eeg.ch_names != self.fwd.ch_names:
            self.fwd = self.fwd.pick_channels(eeg.ch_names)
            # Write all changes to the attributes
            self._embed_fwd(self.fwd)
        
        self.n_timepoints = len(eeg.times)
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
        
    # def solve_p(self, y_est, x_true):
    #     '''
    #     Parameters
    #     ---------
    #     y_est : numpy.ndarray
    #         The estimated source vector.
    #     x_true : numpy.ndarray
    #         The original input EEG vector.
        
    #     Return
    #     ------
    #     y_scaled : numpy.ndarray
    #         The scaled estimated source vector.
        
    #     '''
    #     # Check if y_est is just zeros:
    #     if np.max(y_est) == 0:
    #         return y_est
    #     y_est = np.squeeze(np.array(y_est))
    #     x_true = np.squeeze(np.array(x_true))
    #     # Get EEG from predicted source using leadfield
    #     x_est = np.matmul(self.leadfield, y_est)

    #     # optimize forward solution
    #     tol = 1e-3
    #     options = dict(maxiter=1000, disp=False)

    #     # base scaling
    #     rms_est = np.mean(np.abs(x_est))
    #     rms_true = np.mean(np.abs(x_true))
    #     base_scaler = rms_true / rms_est

        
    #     opt = minimize_scalar(self.correlation_criterion, args=(self.leadfield, y_est* base_scaler, x_true), \
    #         bounds=(0, 1), method='bounded', options=options, tol=tol)
        
    #     scaler = opt.x
    #     y_scaled = y_est * scaler * base_scaler
    #     return y_scaled

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

# class EnsembleNet:
#     ''' Uses ensemble of neural networks to perform predictions
#     Attributes
#     ----------
#     nets : list
#         a list of instances of the Net class
#     ensemble_mode : str
#         Decides how the various predictions will be combined.
#         'average' : average all predictions with equal weight
    
#     Methods
#     -------
#     predict : performs predictions with each Net instance and combines them.
#     vote_average : the implementation of the ensemble_mode 'average'

#     Examples
#     --------
#     ### Build two Nets nad train them
#     k = 2  # number of models
#     nets = [Net(fwd).fit(simulation.eeg_data, simulation.source_data) for _ in range(k)]
#     ### Combine them into an EnsembleNet
#     ens_net = EnsembleNet(nets)
#     ### Perform prediction
#     y_hat = nets[0].predict(simulation_test)
#     y_hat_ens = ens_net.predict(simulation_test.eeg_data)
#     ### Plot result
#     a = simulation_test.source_data.plot(**plot_params)  # Ground truth
#     b = y_hat.plot(**plot_params)  # single-model prediction
#     c = y_hat_ens.plot(**plot_params)  # ensemble predicion



#     '''
#     def __init__(self, nets, ensemble_mode='average'):
#         self.nets = nets
#         self.ensemble_mode = ensemble_mode
        
#         if ensemble_mode == 'average':
#             self.vote = self.vote_average
#         # if ensemble_mode == 'stack':
#         #     self.vote = self.vote_stack
#         else:
#             msg = f'ensemble_mode {ensemble_mode} not supported'
#             raise AttributeError(msg)
        

#     def predict(self, *args):
#         predictions = [net.predict(args[1]) for net in self.nets]
#         predictions_data = np.stack([prediction.data for prediction in predictions], axis=0)
        
#         ensemble_prediction = predictions[0]
#         ensemble_prediction.data = self.vote(predictions_data)

#         return ensemble_prediction

#     def vote_average(self, predictions_data):
#         return np.mean(predictions_data, axis=0)

# class BoostNet:
#     ''' The Boosted neural network class that creates and trains the boosted model. 
        
#     Attributes
#     ----------
#     fwd : mne.Forward
#         the mne.Forward forward model class.
#     n_nets : int
#         The number of neural networks to use.
#     n_layers : int
#         Number of hidden layers in the neural network.
#     n_neurons : int
#         Number of neurons per hidden layer.
#     activation_function : str
#         The activation function used for each fully connected layer.

#     Methods
#     -------
#     fit : trains the neural network with the EEG and source data
#     train : trains the neural network with the EEG and source data
#     predict : perform prediciton on EEG data
#     evaluate : evaluate the performance of the model
#     '''

#     def __init__(self, fwd, n_nets=5, n_layers=1, n_neurons=128, 
#         activation_function='swish', verbose=False):

#         self.nets = [Net(fwd, n_layers=n_layers, n_neurons=n_neurons, 
#             activation_function=activation_function, verbose=verbose) 
#             for _ in range(n_nets)]

#         self.linear_regressor = linear_model.LinearRegression()

#         self.verbose=verbose
#         self.n_nets = n_nets

#     def fit(self, *args, **kwargs):
#         ''' Train the boost model.

#         Parameters
#         ----------
#         *args : esinet.simulation.Simulation
#             Can be either 
#                 eeg : mne.Epochs/ numpy.ndarray
#                     The simulated EEG data
#                 sources : mne.SourceEstimates/ list of mne.SourceEstimates
#                     The simulated EEG data
#                 or
#                 simulation : esinet.simulation.Simulation
#                     The Simulation object

#         **kwargs
#             Arbitrary keyword arguments.

#         Return
#         ------
#         self : BoostNet()
#         '''

#         eeg, sources = self._handle_data_input(args)
#         self.subject = sources.subject if type(sources) == mne.SourceEstimate else sources[0].subject

#         if self.verbose:
#             print("Fit neural networks")
#         self._fit_nets(eeg, sources, **kwargs)

#         ensemble_predictions, _ = self._get_ensemble_predictions(eeg, sources)
           
#         if self.verbose:
#             print("Fit regressor")
#         # Train linear regressor to combine predictions
#         self.linear_regressor.fit(ensemble_predictions, sources.data.T)

#         return self
    
#     def predict(self, *args):
#         ''' Perform prediction of sources based on EEG data using the Boosted Model.
        
#         Parameters
#         ----------
#         *args : 
#             Can be either 
#                 eeg : mne.Epochs/ numpy.ndarray
#                     The simulated EEG data
#                 sources : mne.SourceEstimates/ list of mne.SourceEstimates
#                     The simulated EEG data
#                 or
#                 simulation : esinet.simulation.Simulation
#                     The Simulation object
#         **kwargs
#             Arbitrary keyword arguments.
        
#         Return
#         ------
#         '''

#         eeg, sources = self._handle_data_input(args)

#         ensemble_predictions, y_hats = self._get_ensemble_predictions(eeg, sources)
#         prediction = np.clip(self.linear_regressor.predict(ensemble_predictions), a_min=0, a_max=np.inf)
        
#         y_hat = y_hats[0]
#         y_hat.data = prediction.T
#         return y_hat

#     def evaluate_mse(self, *args):
#         ''' Evaluate the model regarding mean squared error
        
#         Parameters
#         ----------
#         *args : 
#             Can be either 
#                 eeg : mne.Epochs/ numpy.ndarray
#                     The simulated EEG data
#                 sources : mne.SourceEstimates/ list of mne.SourceEstimates
#                     The simulated EEG data
#                 or
#                 simulation : esinet.simulation.Simulation
#                     The Simulation object

#         Return
#         ------
#         mean_squared_errors : numpy.ndarray
#             The mean squared error of each sample

#         Example
#         -------
#         net = BoostNet()
#         net.fit(simulation)
#         mean_squared_errors = net.evaluate(simulation)
#         print(mean_squared_errors.mean())
#         '''

#         eeg, sources = self._handle_data_input(args)
#         y_hat = self.predict(eeg, sources).data
#         y_true = sources.data
#         mean_squared_errors = np.mean((y_hat - y_true)**2, axis=0)
#         return mean_squared_errors


#     def _get_ensemble_predictions(self, *args):

#         eeg, sources = self._handle_data_input(args)

#         y_hats = [subnet.predict(eeg, sources) for subnet in self.nets]
#         ensemble_predictions = np.stack([y_hat[0].data for y_hat in y_hats], axis=0).T
#         ensemble_predictions = ensemble_predictions.reshape(ensemble_predictions.shape[0], np.prod((ensemble_predictions.shape[1], ensemble_predictions.shape[2])))
#         return ensemble_predictions, y_hats

#     def _fit_nets(self, *args, **kwargs):

#         eeg, sources = self._handle_data_input(args)
#         n_samples = eeg.get_data().shape[0]
#         # sample_weight = np.ones((sources._data.shape[1]))
        
#         for net in self.nets:
#             sample_idc = np.random.choice(np.arange(n_samples), 
#                 int(0.8*n_samples), replace=True)
#             eeg_bootstrap = eeg.copy()[sample_idc]
#             sources_bootstrap = sources.copy()
#             sources_bootstrap.data = sources_bootstrap.data[:, sample_idc]
#             net.fit(eeg_bootstrap, sources_bootstrap, **kwargs)#, sample_weight=sample_weight)
#             # sample_weight = net.evaluate_mse(eeg, sources)
#             # print(f'new sample weights: mean={sample_weight.mean()} +- {sample_weight.std()}')

        
#     def _handle_data_input(self, arguments):
#         ''' Handles data input to the functions fit() and predict().
        
#         Parameters
#         ----------
#         arguments : tuple
#             The input arguments to fit and predict which contain data.
        
#         Return
#         ------
#         eeg : mne.Epochs
#             The M/EEG data.
#         sources : mne.SourceEstimates/list
#             The source data.

#         '''
#         if len(arguments) == 1:
#             if isinstance(arguments[0], (mne.Epochs, mne.Evoked, mne.io.Raw, mne.EpochsArray, mne.EvokedArray, mne.epochs.EpochsFIF)):
#                 eeg = arguments[0]
#                 sources = None
#             else:
#                 simulation = arguments[0]
#                 eeg = simulation.eeg_data
#                 sources = simulation.source_data
#                 # msg = f'First input should be of type simulation or Epochs, but {arguments[1]} is {type(arguments[1])}'
#                 # raise AttributeError(msg)

#         elif len(arguments) == 2:
#             eeg = arguments[0]
#             sources = arguments[1]
#         else:
#             msg = f'Input is {type()} must be either the EEG data and Source data or the Simulation object.'
#             raise AttributeError(msg)

#         return eeg, sources
