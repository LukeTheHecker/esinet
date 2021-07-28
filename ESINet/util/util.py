from esinet.net.net import Net
import pickle as pkl
import mne
import numpy as np
import os
from ..simulation import Simulation
from ..net import Net

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
    number_of_dipoles = unpack_fwd(fwd)[1].shape[1]
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
    # If only one time point...
    # if data.shape[-1] == 1:
    #     # ...set sampling frequency to 1
    #     info['sfreq'] = 1
    # print(f'data.shape={data.shape}')
    epochs = mne.EpochsArray(data, info, verbose=0)
    try:
        epochs.set_eeg_reference('average', projection=True, verbose=0)
    except:
        pass

    return epochs

def rms(x):
    return np.sqrt(np.mean(np.square(x)))

def unpack_fwd(fwd):
    """Helper function that extract the most important data structures from the mne.Forward object
    Paramters:
    ----------
    fwd : mne.Forward, The mne Forward model

    Return:
    -------
    fwd_fixed : mne.Forward, Forward model for fixed dipole orientations
    leadfield : numpy.ndarray, the leadfield (gain matrix)
    pos : numpy.ndarray, the positions of dipoles in the source model
    tris : numpy.ndarray, the triangles that describe the source mmodel
    neighbors : numpy.ndarray, the neighbors of each dipole in the source model
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
    # neighbors = get_neighbors(fwd)

    return fwd_fixed, leadfield, pos, tris#, neighbors

def calc_snr_range(mne_obj, baseline_span=(-0.2, 0.0), data_span=(0.0, 0.5)):
    """ Calculate the signal to noise ratio (SNR) range of your mne object.
    
    Parameters:
    -----------
    mne_obj : mne.Epochs or mne.Evoked object. The mne object that contains your m/eeg data.
    baseline_span : tuple, list. The range in seconds that defines the baseline interval.
    data_span : tuple, list. The range in seconds that defines the data (signal) interval.
    
    Return:
    -------
    snr_range : list, range of SNR values in your data.

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
    # snr_mean = gfp[data_range].mean() / gfp[baseline_range].mean()
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

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def get_triangle_neighbors(tris_lr):
    if not np.all(np.unique(tris_lr[0]) == np.arange(len(np.unique(tris_lr[0])))):
        for hem in range(2):
            old_indices = np.sort(np.unique(tris_lr[hem]))
            new_indices = np.arange(len(old_indices))
            for old_idx, new_idx in zip(old_indices, new_indices):
                tris_lr[hem][tris_lr[hem] == old_idx] = new_idx

        print('indices were weird - fixed them.')
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
            # print(f'idx {idx} found in triangles: {neighbors[idx]}') 
    return neighbors


def calculate_source(data_obj, fwd, baseline_span=(-0.2, 0.0), data_span=(0, 0.5),
    n_samples=int(1e4), optimizer=None, learning_rate=0.001, 
        validation_split=0.1, n_epochs=100, metrics=None, device=None, delta=1, 
        batch_size=128, loss=None, parallel=False, verbose=1):
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
    loss : tf.kers.losses
        The loss function.
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
    simulation = Simulation(fwd, data_obj.info, settings=dict(duration_of_trial=0, target_snr=target_snr), parallel=parallel, verbose=True)
    simulation.simulate(n_samples=n_samples)
    # Train neural network
    net = Net(fwd, verbose=verbose)
    net.fit(simulation, optimizer=optimizer, learning_rate=learning_rate, 
        validation_split=validation_split, epochs=n_epochs, metrics=metrics,
        device=device, delta=delta, batch_size=batch_size, loss=loss)
    # Predict sources
    source_estimate = net.predict(data_obj)

    return source_estimate