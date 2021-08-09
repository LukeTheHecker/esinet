from random import sample

from tensorflow.python.keras.losses import mean_squared_error
from esinet.simulation.simulation import Simulation
import mne
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn import linear_model
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import pearsonr
from joblib import Parallel, delayed
from tqdm.notebook import tqdm
from copy import deepcopy
import time
from .. import util
from ..simulation.simulation import Simulation
from . import losses
# import .losses

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
        activation_function='swish', parallel=True, 
        n_jobs=-1, verbose=False):

        super().__init__()
        self._embed_fwd(fwd)
        
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.activation_function = activation_function
        # self.default_loss = tf.keras.losses.Huber(delta=delta)
        self.default_loss = losses.weighted_huber_loss
        self.parallel = parallel
        self.n_jobs = n_jobs
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
        if len(arguments) == 2:
            if isinstance(arguments[1], (mne.Epochs, mne.Evoked, mne.io.Raw, mne.EpochsArray, mne.EvokedArray, mne.epochs.EpochsFIF)):
                eeg = arguments[1]
                sources = None
            else:
                simulation = arguments[1]
                eeg = simulation.eeg_data
                sources = simulation.source_data
                # msg = f'First input should be of type simulation or Epochs, but {arguments[1]} is {type(arguments[1])}'
                # raise AttributeError(msg)

        elif len(arguments) == 3:
            eeg = arguments[1]
            sources = arguments[2]
        else:
            msg = f'Input is {type()} must be either the EEG data and Source data or the Simulation object.'
            raise AttributeError(msg)

        return eeg, sources

    def fit(*args, optimizer=None, learning_rate=0.001, 
        validation_split=0.1, epochs=100, metrics=None, device=None, false_positive_penalty=2, 
        delta=1., batch_size=128, loss=None, sample_weight=None):
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
        false_positive_penalty : int
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

        '''

        self = args[0]

        eeg, sources = self._handle_data_input(args)
        self.subject = sources.subject if type(sources) == mne.SourceEstimate else sources[0].subject

        if type(sources) == list:
            self.temporal = True
        else:
            self.temporal = False
        # Ensure that the forward model has the same 
        # channels as the eeg object
        self._check_model(eeg)

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
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        if loss is None:
            loss = self.default_loss(w=false_positive_penalty, delta=delta)

        elif type(loss) == list:
            loss = loss[0](*loss[1])
        if metrics is None:
            metrics = [self.default_loss(w=false_positive_penalty, delta=delta), 'mean_squared_error']

        self.compile(optimizer, loss, metrics=metrics)
        if self.temporal:
            y_scaled = np.reshape(y_scaled, (y_scaled.shape[0], int(np.prod((y_scaled.shape[1], y_scaled.shape[2])))))


        if device is None:
            try:
                super(Net, self).fit(x_scaled, y_scaled, epochs=epochs, batch_size=batch_size, shuffle=False, \
                    validation_split=validation_split, verbose=self.verbose, callbacks=[es],
                    sample_weight=sample_weight)
            except:
                super().fit(x_scaled, y_scaled, epochs=epochs, batch_size=batch_size, shuffle=False, \
                    validation_split=validation_split, verbose=self.verbose, callbacks=[es],
                    sample_weight=sample_weight)
        else:
            with tf.device(device):
                try:
                    super(Net, self).fit(x_scaled, y_scaled, epochs=epochs, batch_size=batch_size, shuffle=False, \
                        validation_split=validation_split, verbose = self.verbose, callbacks=[es],
                        sample_weight=sample_weight)
                except:
                    super().fit(x_scaled, y_scaled, epochs=epochs, batch_size=batch_size, shuffle=False, \
                        validation_split=validation_split, verbose = self.verbose, callbacks=[es],
                        sample_weight=sample_weight)
        return self


    def predict(*args):
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
        
        self = args[0]
        
        eeg, _ = self._handle_data_input(args)
        if isinstance(eeg, util.EVOKED_INSTANCES):
            sfreq = eeg.info['sfreq']
            tmin = eeg.tmin
            eeg = eeg.data
            # add empty trial dimension
            eeg = np.expand_dims(eeg, axis=0)
            if len(eeg.shape) == 2:
                # add empty time dimension
                eeg = np.expand_dims(eeg, axis=2)
        elif isinstance(eeg, util.EPOCH_INSTANCES):
            sfreq = eeg.info['sfreq']
            tmin = eeg.tmin
            eeg = eeg._data
        else:
            msg = f'eeg must be of type <mne.EvokedArray> or <mne.epochs.EpochsArray>; got {type(eeg)} instead.'
            raise ValueError(msg)

        # Prepare EEG to ensure common average reference and appropriate scaling
        eeg_prep =  self._prep_eeg(eeg)
        predicted_sources = np.zeros((eeg_prep.shape[0], self.n_dipoles, eeg_prep.shape[2]))
   
        # Predict sources in batches of trials
        for time in range(eeg.shape[2]):
            
            predicted_sources[:, :, time] = super(Net, self).predict(eeg_prep[:, :, time])
        
        # Rescale Predicitons
        predicted_sources_scaled = self._solve_p_wrap(predicted_sources, eeg)
        
        # Convert sources (numpy.ndarrays) to mne.SourceEstimates objects
        predicted_source_estimate = [util.source_to_sourceEstimate(predicted_sources_scaled[k], self.fwd, sfreq=sfreq, tmin=tmin, subject=self.subject)
                for k in range(predicted_sources_scaled.shape[0])]

        if len(predicted_source_estimate) == 1:
            predicted_source_estimate = predicted_source_estimate[0]

        return predicted_source_estimate

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

    @staticmethod
    def _prep_eeg(eeg):
        ''' Takes a 3D EEG array and re-references to common average and scales 
        individual scalp maps to max(abs(scalp_map) == 1
        '''
        assert len(eeg.shape) == 3, 'Input array <eeg> has wrong shape.'

        eeg_prep = deepcopy(eeg)
        for trial in range(eeg_prep.shape[0]):
            for time in range(eeg_prep.shape[2]):
                # Common average reference
                eeg_prep[trial, :, time] -= np.mean(eeg_prep[trial, :, time])
                # Scaling
                eeg_prep[trial, :, time] /= np.max(np.abs(eeg_prep[trial, :, time]))
        return eeg_prep

    def evaluate_mse(*args):
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

        self = args[0]
        eeg, sources = self._handle_data_input(args)
        
        y_hat = self.predict(eeg).data
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
            self._build_temporal_model()
        else:
            self._build_perceptron_model()
        

        if self.verbose:
            self.summary()
    
    def _build_temporal_model(self):
        ''' Build the temporal artificial neural network model using LSTM layers.
        '''
        input_shape = (self.n_channels, self.n_timepoints)
        print(input_shape)
        self.add(layers.InputLayer(input_shape=input_shape))
        self.add(layers.LSTM(4, return_sequences=True))
        for _ in range(self.n_layers-1):
            self.add(layers.LSTM(4, return_sequences=False))

        self.add(layers.Flatten())
        self.add(layers.Dense(int(self.n_dipoles*self.n_timepoints), 
            activation=keras.layers.ReLU(max_value=1)))
        self.build(input_shape=input_shape)
        

    def _build_perceptron_model(self):
        ''' Build the artificial neural network model using Dense layers.
        '''
        # Add hidden layers
        for i in range(self.n_layers):
            self.add(layers.Dense(units=self.n_neurons,
                                activation=self.activation_function))
        # Add output layer
        self.add(layers.Dense(self.n_dipoles, 
            activation=keras.layers.ReLU(max_value=1)))
        
        # Build model with input layer
        self.build(input_shape=(None, self.n_channels))

    def _check_model(self, eeg):
        ''' Check whether the current forward model has the same 
        channels as the eeg. Rebuild model if thats not the case.
        
        Parameters
        ----------
        eeg : mne.Epochs or equivalent
            The EEG instance.

        '''
        # Dont do anything if model is already built.
        if self.built:
            return
        
        # Else assure that channels are appropriate
        if eeg.ch_names != self.fwd.ch_names:
            self.fwd = self.fwd.pick_channels(eeg.ch_names)
            # Write all changes to the attributes
            self._embed_fwd(self.fwd)
        
        self.n_timepoints = len(eeg.times)
        # Finally, build model
        self._build_model()
            
            
        
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
        tol = 1e-3
        options = dict(maxiter=1000, disp=False)

        # base scaling
        rms_est = np.mean(np.abs(x_est))
        rms_true = np.mean(np.abs(x_true))
        base_scaler = rms_true / rms_est

        
        opt = minimize_scalar(self.correlation_criterion, args=(self.leadfield, y_est* base_scaler, x_true), \
            bounds=(0, 1), method='bounded', options=options, tol=tol)
        
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

class EnsembleNet:
    ''' Uses ensemble of neural networks to perform predictions
    Attributes
    ----------
    nets : list
        a list of instances of the Net class
    ensemble_mode : str
        Decides how the various predictions will be combined.
        'average' : average all predictions with equal weight
    
    Methods
    -------
    predict : performs predictions with each Net instance and combines them.
    vote_average : the implementation of the ensemble_mode 'average'

    Examples
    --------
    ### Build two Nets nad train them
    k = 2  # number of models
    nets = [Net(fwd).fit(simulation.eeg_data, simulation.source_data) for _ in range(k)]
    ### Combine them into an EnsembleNet
    ens_net = EnsembleNet(nets)
    ### Perform prediction
    y_hat = nets[0].predict(simulation_test)
    y_hat_ens = ens_net.predict(simulation_test.eeg_data)
    ### Plot result
    a = simulation_test.source_data.plot(**plot_params)  # Ground truth
    b = y_hat.plot(**plot_params)  # single-model prediction
    c = y_hat_ens.plot(**plot_params)  # ensemble predicion



    '''
    def __init__(self, nets, ensemble_mode='average'):
        self.nets = nets
        self.ensemble_mode = ensemble_mode
        
        if ensemble_mode == 'average':
            self.vote = self.vote_average
        # if ensemble_mode == 'stack':
        #     self.vote = self.vote_stack
        else:
            msg = f'ensemble_mode {ensemble_mode} not supported'
            raise AttributeError(msg)
        

    def predict(*args):
        self = args[0]
        predictions = [net.predict(args[1]) for net in self.nets]
        predictions_data = np.stack([prediction.data for prediction in predictions], axis=0)
        
        ensemble_prediction = predictions[0]
        ensemble_prediction.data = self.vote(predictions_data)

        return ensemble_prediction

    def vote_average(self, predictions_data):
        return np.mean(predictions_data, axis=0)

class BoostNet:
    ''' The Boosted neural network class that creates and trains the boosted model. 
        
    Attributes
    ----------
    fwd : mne.Forward
        the mne.Forward forward model class.
    n_nets : int
        The number of neural networks to use.
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

    def __init__(self, fwd, n_nets=5, n_layers=1, n_neurons=128, 
        activation_function='swish', verbose=False):

        self.nets = [Net(fwd, n_layers=n_layers, n_neurons=n_neurons, 
            activation_function=activation_function, verbose=verbose) 
            for _ in range(n_nets)]

        self.linear_regressor = linear_model.LinearRegression()

        self.verbose=verbose
        self.n_nets = n_nets

    def fit(*args, **kwargs):
        ''' Train the boost model.

        Parameters
        ----------
        *args : esinet.simulation.Simulation
            Can be either 
                eeg : mne.Epochs/ numpy.ndarray
                    The simulated EEG data
                sources : mne.SourceEstimates/ list of mne.SourceEstimates
                    The simulated EEG data
                or
                simulation : esinet.simulation.Simulation
                    The Simulation object

        **kwargs
            Arbitrary keyword arguments.

        Return
        ------
        self : BoostNet()
        '''

        self = args[0]
        eeg, sources = self._handle_data_input(args)
        self.subject = sources.subject if type(sources) == mne.SourceEstimate else sources[0].subject

        if self.verbose:
            print("Fit neural networks")
        self._fit_nets(eeg, sources, **kwargs)

        ensemble_predictions, _ = self._get_ensemble_predictions(eeg, sources)
           
        if self.verbose:
            print("Fit regressor")
        # Train linear regressor to combine predictions
        self.linear_regressor.fit(ensemble_predictions, sources.data.T)

        return self
    
    def predict(*args):
        ''' Perform prediction of sources based on EEG data using the Boosted Model.
        
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
        **kwargs
            Arbitrary keyword arguments.
        
        Return
        ------
        '''

        self = args[0]
        eeg, sources = self._handle_data_input(args)

        ensemble_predictions, y_hats = self._get_ensemble_predictions(eeg, sources)
        prediction = np.clip(self.linear_regressor.predict(ensemble_predictions), a_min=0, a_max=np.inf)
        
        y_hat = y_hats[0]
        y_hat.data = prediction.T
        return y_hat

    def evaluate_mse(*args):
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
        net = BoostNet()
        net.fit(simulation)
        mean_squared_errors = net.evaluate(simulation)
        print(mean_squared_errors.mean())
        '''

        self = args[0]
        eeg, sources = self._handle_data_input(args)
        y_hat = self.predict(eeg, sources).data
        y_true = sources.data
        mean_squared_errors = np.mean((y_hat - y_true)**2, axis=0)
        return mean_squared_errors


    def _get_ensemble_predictions(*args):
        self = args[0]
        eeg, sources = self._handle_data_input(args)

        y_hats = [subnet.predict(eeg, sources) for subnet in self.nets]
        
        ensemble_predictions = np.stack([y_hat.data for y_hat in y_hats], axis=0).T
        ensemble_predictions = ensemble_predictions.reshape(ensemble_predictions.shape[0], np.prod((ensemble_predictions.shape[1], ensemble_predictions.shape[2])))
     
        return ensemble_predictions, y_hats

    def _fit_nets(*args, **kwargs):
        self = args[0]
        eeg, sources = self._handle_data_input(args)

        sample_weight = np.ones((sources._data.shape[1]))

        for net in self.nets:
            net.fit(eeg, sources, **kwargs, sample_weight=sample_weight)
            sample_weight = net.evaluate_mse(eeg, sources)
            print(f'new sample weights: mean={sample_weight.mean()} +- {sample_weight.std()}')

        
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
        if len(arguments) == 2:
            if isinstance(arguments[1], (mne.Epochs, mne.Evoked, mne.io.Raw, mne.EpochsArray, mne.EvokedArray, mne.epochs.EpochsFIF)):
                eeg = arguments[1]
                sources = None
            else:
                simulation = arguments[1]
                eeg = simulation.eeg_data
                sources = simulation.source_data
                # msg = f'First input should be of type simulation or Epochs, but {arguments[1]} is {type(arguments[1])}'
                # raise AttributeError(msg)

        elif len(arguments) == 3:
            eeg = arguments[1]
            sources = arguments[2]
        else:
            msg = f'Input is {type()} must be either the EEG data and Source data or the Simulation object.'
            raise AttributeError(msg)

        return eeg, sources

