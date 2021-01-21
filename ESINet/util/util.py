import pickle as pkl
import mne
import numpy as np

def load_info(pth_fwd):
    with open(pth_fwd + '/info.pkl', 'rb') as file:  
        info = pkl.load(file)
    return info

def load_leadfield(pth_fwd):
    with open(pth_fwd + '/leadfield.pkl', 'rb') as file:  
        leadfield = pkl.load(file)
    return leadfield[0]

def load_fwd(pth_fwd):
    fwd = mne.read_forward_solution(pth_fwd + '/fsaverage-fwd.fif', verbose=0)
    return fwd

def source_to_sourceEstimate(data, pth_fwd, sfreq=1, subject='fsaverage', 
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

    src_template = mne.read_source_estimate(pth_fwd + "/ResSourceEstimate-lh.stc")
    number_of_dipoles = len(src_template.vertices[0]) + len(src_template.vertices[1])
    if data.shape[0] != number_of_dipoles:
        data = np.transpose(data)

    src = mne.SourceEstimate(data, src_template.vertices, tmin=tmin, tstep=1/sfreq, 
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

    epochs = mne.EpochsArray(data, info, verbose=0)
    epochs.set_eeg_reference('average', projection=True, verbose=0)

    return epochs

def rms(x):
    return np.sqrt(np.mean(np.square(x)))