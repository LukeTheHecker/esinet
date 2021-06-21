import pickle as pkl
import mne
import numpy as np
import os
from .. import simulations
# from ..simulations import get_triangle_neighbors

def load_info(pth_fwd):
    with open(pth_fwd + '/info.pkl', 'rb') as file:  
        info = pkl.load(file)
    return info

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