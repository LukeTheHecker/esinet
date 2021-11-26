import pickle as pkl
import mne
import numpy as np
import os
from copy import deepcopy
import logging
from time import time
from .. import simulation
from .. import net

# from .. import Simulation
# from .. import Net

EPOCH_INSTANCES = (mne.epochs.EpochsArray, mne.Epochs, mne.EpochsArray, mne.epochs.EpochsFIF)
EVOKED_INSTANCES = (mne.Evoked, mne.EvokedArray)
RAW_INSTANCES = (mne.io.Raw, mne.io.RawArray)

def load_info(pth_fwd):
    with open(pth_fwd + '/info.pkl', 'rb') as file:  
        info = pkl.load(file)
    return info
    
def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def load_leadfield(pth_fwd):
    ''' Load the leadfield matrix from the path of the forward model.'''

    if os.path.isfile(pth_fwd + '/leadfield.pkl'):
        with open(pth_fwd + '/leadfield.pkl', 'rb') as file:  
            leadfield = pkl.load(file)
    else:
        fwd = load_fwd(pth_fwd)
        fwd_fixed = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True,
                                                use_cps=True, verbose=0)
        leadfield = fwd_fixed['sol']['data']
    return leadfield[0]

def load_fwd(pth_fwd):
    fwd = mne.read_forward_solution(pth_fwd + '/fsaverage-fwd.fif', verbose=0)
    return fwd

def get_neighbors(fwd):
    """Retreive the list of direct neighbors for each dipole in the source model
    Parameters:
    -----------
    fwd : mne.Forward, the mne Forward object 
    Return:
    -------
    neighbors : numpy.ndarray, a matrix containing the neighbor 
        indices for each dipole in the source model

    """
    tris_lr = [fwd['src'][0]['use_tris'], fwd['src'][1]['use_tris']]
    neighbors = simulations.get_triangle_neighbors(tris_lr)
    neighbors = np.array([np.array(d) for d in neighbors], dtype='object')
    return neighbors

def source_to_sourceEstimate(data, fwd, sfreq=1, subject='fsaverage', 
    simulationInfo=None, tmin=0):
    ''' Takes source data and creates mne.SourceEstimate object
    https://mne.tools/stable/generated/mne.SourceEstimate.html

    Parameters:
    -----------
    data : numpy.ndarray, shape (number of dipoles x number of timepoints)
    pth_fwd : path to the forward model files sfreq : sample frequency, needed
        if data is time-resolved (i.e. if last dim of data > 1)

    Return:
    -------
    src : mne.SourceEstimate, instance of SourceEstimate.

    '''
    
    data = np.squeeze(np.array(data))
    if len(data.shape) == 1:
        data = np.expand_dims(data, axis=1)
    
    source_model = fwd['src']
    number_of_dipoles = int(fwd['src'][0]['nuse']+fwd['src'][1]['nuse'])
    
    if data.shape[0] != number_of_dipoles:
        data = np.transpose(data)

    vertices = [source_model[0]['vertno'], source_model[1]['vertno']]
    src = mne.SourceEstimate(data, vertices, tmin=tmin, tstep=1/sfreq, 
        subject=subject)
    
    if simulationInfo is not None:
        setattr(src, 'simulationInfo', simulationInfo)


    return src

def eeg_to_Epochs(data, pth_fwd, info=None):
    if info is None:
        info = load_info(pth_fwd)

    epochs = mne.EpochsArray(data, info, verbose=0)
    
    # Rereference to common average if its not the case
    if int(epochs.info['custom_ref_applied']) != 0:
        epochs.set_eeg_reference('average', projection=True, verbose=0)
    
    return epochs

def rms(x):
    ''' Calculate the root mean square of some signal x.
    Parameters
    ----------
    x : numpy.ndarray, list
        The signal/data.

    Return
    ------
    rms : float
    '''
    return np.sqrt(np.mean(np.square(x)))

def unpack_fwd(fwd):
    """ Helper function that extract the most important data structures from the 
    mne.Forward object

    Parameters
    ----------
    fwd : mne.Forward
        The forward model object

    Return
    ------
    fwd_fixed : mne.Forward
        Forward model for fixed dipole orientations
    leadfield : numpy.ndarray
        The leadfield (gain matrix)
    pos : numpy.ndarray
        The positions of dipoles in the source model
    tris : numpy.ndarray
        The triangles that describe the source mmodel
    neighbors : numpy.ndarray
        the neighbors of each dipole in the source model

    """
    if fwd['surf_ori']:
        fwd_fixed = fwd
    else:
        fwd_fixed = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True,
                                                    use_cps=True, verbose=0)
    tris = fwd['src'][0]['use_tris']
    leadfield = fwd_fixed['sol']['data']

    source = fwd['src']
    try:
        subject_his_id = source[0]['subject_his_id']
        pos_left = mne.vertex_to_mni(source[0]['vertno'], 0, subject_his_id, verbose=0)
        pos_right = mne.vertex_to_mni(source[1]['vertno'],  1, subject_his_id, verbose=0)
    except:
        subject_his_id = 'fsaverage'
        pos_left = mne.vertex_to_mni(source[0]['vertno'], 0, subject_his_id, verbose=0)
        pos_right = mne.vertex_to_mni(source[1]['vertno'],  1, subject_his_id, verbose=0)

    pos = np.concatenate([pos_left, pos_right], axis=0)

    return fwd_fixed, leadfield, pos, tris#


def calc_snr_range(mne_obj, baseline_span=(-0.2, 0.0), data_span=(0.0, 0.5)):
    """ Calculate the signal to noise ratio (SNR) range of your mne object.
    
    Parameters
    ----------
    mne_obj : mne.Epochs, mne.Evoked
        The mne object that contains your m/eeg data.
    baseline_span : tuple, list
        The range in seconds that defines the baseline interval.
    data_span : tuple, list
        The range in seconds that defines the data (signal) interval.
    
    Return
    ------
    snr_range : list
        range of SNR values in your data.

    """

    if isinstance(mne_obj, EPOCH_INSTANCES):
        evoked = mne_obj.average()
    elif isinstance(mne_obj, EVOKED_INSTANCES):
        evoked = mne_obj
    else:
        msg = f'mne_obj is of type {type(mne_obj)} but should be mne.Evoked(Array) or mne.Epochs(Array).'
        raise ValueError(msg)
    
    
    data = np.squeeze(evoked.data)
    baseline_range = range(*[np.argmin(np.abs(evoked.times-base)) for base in baseline_span])
    data_range = range(*[np.argmin(np.abs(evoked.times-base)) for base in data_span])
    
    gfp = np.std(data, axis=0)
    snr_lo = gfp[data_range].min() / gfp[baseline_range].max() 
    snr_hi = gfp[data_range].max() / gfp[baseline_range].min()

    snr_range = [snr_lo, snr_hi]
    return snr_range

def repeat_newcol(x, n):
    ''' Repeat a list/numpy.ndarray x in n columns.'''
    out = np.zeros((len(x), n))
    for i in range(n):
        out[:,  i] = x
    return np.squeeze(out)


def get_n_order_indices(order, pick_idx, neighbors):
    ''' Iteratively performs region growing by selecting neighbors of 
    neighbors for <order> iterations.
    '''
    current_indices = np.array([pick_idx])

    if order == 0:
        return current_indices

    for _ in range(order):
        current_indices = np.append(current_indices, np.concatenate(neighbors[current_indices]))

    return np.unique(np.array(current_indices))

def gaussian(x, mu, sigma):
    ''' Gaussian distribution function.
    
    Parameters
    ----------
    x : numpy.ndarray, list
        The x-value.
    mu : float
        The mean of the gaussian kernel.
    sigma : float
        The standard deviation of the gaussian kernel.
    Return
    ------
    '''
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sigma, 2.)))

def get_triangle_neighbors(tris_lr):
    if not np.all(np.unique(tris_lr[0]) == np.arange(len(np.unique(tris_lr[0])))):
        for hem in range(2):
            old_indices = np.sort(np.unique(tris_lr[hem]))
            new_indices = np.arange(len(old_indices))
            for old_idx, new_idx in zip(old_indices, new_indices):
                tris_lr[hem][tris_lr[hem] == old_idx] = new_idx

        # print('indices were weird - fixed them.')
    numberOfDipoles = len(np.unique(tris_lr[0])) + len(np.unique(tris_lr[1]))
    neighbors = [list() for _ in range(numberOfDipoles)]
    # correct right-hemisphere triangles
    tris_lr_adjusted = deepcopy(tris_lr)
    # the right hemisphere indices start at zero, we need to offset them to start where left hemisphere indices end.
    tris_lr_adjusted[1] += int(numberOfDipoles/2)
    # left and right hemisphere
    for hem in range(2):
        for idx in range(numberOfDipoles):
            # Find the indices of the triangles where our current dipole idx is part of
            trianglesOfIndex = tris_lr_adjusted[hem][np.where(tris_lr_adjusted[hem] == idx)[0], :]
            for tri in trianglesOfIndex:
                neighbors[idx].extend(tri)
                # Remove self-index (otherwise neighbors[idx] is its own neighbor)
                neighbors[idx] = list(filter(lambda a: a != idx, neighbors[idx]))
            # Remove duplicates
            neighbors[idx] = list(np.unique(neighbors[idx]))
    return neighbors


def calculate_source(data_obj, fwd, baseline_span=(-0.2, 0.0), 
    data_span=(0, 0.5), n_samples=int(1e4), optimizer=None, learning_rate=0.001, 
    validation_split=0.1, n_epochs=100, metrics=None, device=None, delta=1, 
    batch_size=128, loss=None, false_positive_penalty=2, parallel=False, 
    verbose=False):
    ''' The all-in-one convenience function for esinet.
    
    Parameters
    ----------
    data_obj : mne.Epochs, mne.Evoked
        The mne object holding the M/EEG data. Can be Epochs or 
        averaged (Evoked) responses.
    fwd : mne.Forward
        The forward model
    baseline_span : tuple
        The time range of the baseline in seconds.
    data_span : tuple
        The time range of the data in seconds.
    n_samples : int
        The number of training samples to simulate. The 
        higher, the more accurate the inverse solution
    optimizer : tf.keras.optimizers
        The optimizer that for backpropagation.
    learning_rate : float
        The learning rate for training the neural network
    validation_split : float
        Proportion of data to keep as validation set.
    n_epochs : int
        Number of epochs to train. In one epoch all training samples 
        are used once for training.
    metrics : list/str
        The metrics to be used for performance monitoring during training.
    device : str
        The device to use, e.g. a graphics card.
    delta : float
        Controls the Huber loss.
    batch_size : int
        The number of samples to simultaneously calculate the error 
        during backpropagation.
    loss : tf.keras.losses
        The loss function.
    false_positive_penalty : float
        Defines weighting of false-positive predictions. Increase for conservative 
        inverse solutions, decrease for liberal prediction.
    verbose : int/bool
        Controls verbosity of the program.
    
    
    Return
    ------
    source_estimate : mne.SourceEstimate
        The source

    Example
    -------

    source_estimate = calculate_source(epochs, fwd)
    source_estimate.plot()

    '''
    # Calculate signal to noise ratio of the data
    target_snr = calc_snr_range(data_obj, baseline_span=baseline_span, data_span=data_span)
    # Simulate sample M/EEG data
    sim = simulation.Simulation(fwd, data_obj.info, settings=dict(duration_of_trial=0, target_snr=target_snr), parallel=parallel, verbose=True)
    sim.simulate(n_samples=n_samples)
    # Train neural network
    neural_net = net.Net(fwd, verbose=verbose)
    neural_net.fit(sim, optimizer=optimizer, learning_rate=learning_rate, 
        validation_split=validation_split, epochs=n_epochs, metrics=metrics,
        device=device, delta=delta, batch_size=batch_size, loss=loss,
        false_positive_penalty=false_positive_penalty)
    # Predict sources
    source_estimate = neural_net.predict(data_obj)

    return source_estimate

def get_eeg_from_source(stc, fwd, info, tmin=-0.2):
    ''' Get EEG from source by projecting source activity through the lead field.
    
    Parameters
    ----------
    stc : mne.SourceEstimate
        The source estimate object holding source data.
    fwd : mne.Forawrd
        The forward model.
    
    Return
    ------
    evoked : mne.EvokedArray
        The EEG data oject.
    '''
    fwd = deepcopy(fwd)
    fwd = fwd.copy().pick_channels(info['ch_names'])
    leadfield = fwd['sol']['data']
    eeg_hat = np.matmul(leadfield, stc.data)

    return mne.EvokedArray(eeg_hat, info, tmin=tmin)

def mne_inverse(fwd, epochs, method='eLORETA', snr=3.0, tmax=0, verbose=0):
    ''' Quickly compute inverse solution using MNE methods
    '''

    epochs.set_eeg_reference(projection=True, verbose=verbose).apply_baseline(baseline=(None, None), verbose=verbose)
    
    evoked = epochs.average()
    raw = mne.io.RawArray(evoked.data, evoked.info, verbose=verbose)
    
    
    if method.lower()=='beamformer' or method.lower()=='lcmv' or method.lower()=='beamforming':
        noise_cov = mne.make_ad_hoc_cov(evoked.info, std=dict(eeg=1), verbose=verbose)
        # noise_cov = mne.compute_covariance(epochs, method='empirical', verbose=verbose)
        # noise_cov = mne.compute_raw_covariance(raw, tmin=None, method='empirical', verbose=verbose)
        # noise_cov = mne.cov.regularize(noise_cov, raw.info, verbose=verbose)

        data_cov = mne.compute_covariance(epochs, method='empirical', verbose=verbose)
        # data_cov = mne.compute_raw_covariance(raw, method='empirical', verbose=verbose)
        # data_cov = mne.cov.regularize(data_cov, raw.info, verbose=verbose)
        
        lcmv_filter = mne.beamformer.make_lcmv(evoked.info, fwd, data_cov, reg=0.05, weight_norm='nai', noise_cov=noise_cov, verbose=verbose)
        # stc = mne.beamformer.apply_lcmv_raw(raw, lcmv_filter, verbose=verbose)
        stc = mne.beamformer.apply_lcmv(evoked, lcmv_filter, verbose=verbose)
    else:
        noise_cov = mne.make_ad_hoc_cov(evoked.info, std=dict(eeg=1), verbose=verbose)
        # noise_cov = mne.cov.regularize(noise_cov, raw.info, verbose=verbose)
        lambda2 = 1. / snr ** 2
        inverse_operator = mne.minimum_norm.make_inverse_operator(
            evoked.info, fwd, noise_cov, loose='auto', depth=None, fixed=True, 
            verbose=verbose)
            
        stc = mne.minimum_norm.apply_inverse(evoked, inverse_operator, lambda2,
                                    method=method, return_residual=False, verbose=verbose)
    return stc

def wrap_mne_inverse(fwd, sim, method='eLORETA', snr=3.0, tmax=0):
    ''' Wrapper that calculates inverse solutions to a bunch of simulated
        samples of a esinet.Simulation object
    '''
    eeg, sources = net.Net._handle_data_input((sim,))
    n_samples = eeg.get_data().shape[0]
    stcs = []

    for i in range(n_samples):
        stc = mne_inverse(fwd, eeg[i], method=method, snr=snr, tmax=tmax)
        stcs.append(stc)
    return stcs


def convert_simulation_temporal_to_single(sim):
    sim_single = deepcopy(sim)
    sim_single.temporal = False
    sim_single.settings['duration_of_trial'] = 0

    eeg_data_lstm = sim.eeg_data.get_data()
    # Reshape EEG data
    eeg_data_single = np.expand_dims(np.vstack(np.swapaxes(eeg_data_lstm, 1,2)), axis=-1)
    # Pack into mne.EpochsArray object
    epochs_single = mne.EpochsArray(eeg_data_single, sim.eeg_data.info, 
        tmin=sim.eeg_data.tmin, verbose=0)
    
    # Reshape Source data
    source_data = np.vstack(np.swapaxes(np.stack(
        [source.data for source in sim.source_data], axis=0), 1,2)).T
    # Pack into mne.SourceEstimate object
    source_single = deepcopy(sim.source_data[0])
    source_single.data = source_data
    
    # Copy new shaped data into the Simulation object:
    sim_single.eeg_data = epochs_single
    sim_single.source_data = source_single

    return sim_single

def collapse(x):
    ''' Collapse a  3D matrix (samples x dipoles x timepoints) into a 2D array 
    by combining the first and last dim.
    
    Parameters
    ----------
    x : numpy.ndarray
        Three-dimensional matrix, e.g. (samples x dipoles x timepoints)
    '''
    return np.swapaxes(x, 1,2).reshape(int(x.shape[0]*x.shape[2]), x.shape[1])

def custom_logger(logger_name, level=logging.DEBUG):
    """
    Creates a new log file to log to.
    
    Parameters
    ----------
    logger_name : str
        Name or path of the logger
    level : see logging module for further information
    
    Return
    ------
    logger : logging.getLogger 
        Logger handle

    Example
    -------
    logger1 = custom_logger('path_to_logger/mylogfile')
    logger1.info('Here is some info written to the file')

    logger2 = custom_logger('path_to_logger/mylogfile2')
    logger2.info('Here is some different info written to the file')

    
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    format_string = ("%(asctime)s — %(levelname)s: "
                    "%(message)s")
    log_format = logging.Formatter(format_string)

    file_handler = logging.FileHandler(logger_name, mode='a')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger

def load_net(path, name='instance'):
    import tensorflow as tf
    model = tf.keras.models.load_model(path)
    with open(path + f'\\{name}.pkl', 'rb') as f:
        net = pkl.load(f)
    net.model = model
    return net