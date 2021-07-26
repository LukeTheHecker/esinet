import pickle as pkl
import mne
import numpy as np
import os
from .. import simulations
# from ..simulations import get_triangle_neighbors

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
    if data.shape[-1] == 1:
        # ...set sampling frequency to 1
        info['sfreq'] = 1
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