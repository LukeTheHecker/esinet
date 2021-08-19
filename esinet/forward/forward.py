import os
import mne
import pickle as pkl
import numpy as np

def create_forward_model(sampling='ico3', info=None, verbose=0, fixed_ori=True):
    ''' Create a forward model using the fsaverage template form freesurfer.
    
    Parameters:
    ----------
    sampling : str
        the downsampling of the forward model. 
        Recommended are 'ico3' (small), 'ico4' (medium) or 
        'ico5' (large).
    info : mne.Info
        info instance which contains the desired 
        electrode names and positions. 
        This can be obtained e.g. from your processed mne.Raw.info, 
        mne.Epochs.info or mne.Evoked.info
        If info is None the Info instance is created where 
        electrodes are chosen automatically from the easycap-M10 
        layout.
    fixed_ori : bool
        Whether orientation of dipoles shall be fixed (set to True) 
        or free (set to False).

    Return
    ------
    fwd : mne.Forward
        The forward model object
    '''

    # Fetch the template files for our forward model
    fs_dir = mne.datasets.fetch_fsaverage(verbose=verbose)
    subjects_dir = os.path.dirname(fs_dir)

    # The files live in:
    subject = 'fsaverage'
    trans = os.path.join(fs_dir, 'bem', 'fsaverage-trans.fif')
    src = os.path.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif')
    bem = os.path.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')

 

    # Create our own info object, see e.g.:
    if info is None:
        info = get_info()
        
    # Create and save Source Model
    src = mne.setup_source_space(subject, spacing=sampling, surface='white',
                                        subjects_dir=subjects_dir, add_dist=False,
                                        n_jobs=-1, verbose=verbose)

    # Forward Model
    fwd = mne.make_forward_solution(info, trans=trans, src=src,
                                    bem=bem, eeg=True, mindist=5.0, n_jobs=-1,
                                    verbose=verbose)
    if fixed_ori:
        # Fixed Orientations
        fwd = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True,
                                            use_cps=True, verbose=verbose)

    return fwd


def get_info(kind='easycap-M10'):
    ''' Create some generic mne.Info object.
    
    Parameters
    ----------
    kind : str
        kind, for examples see:
            https://mne.tools/stable/generated/mne.channels.make_standard_montage.html#mne.channels.make_standard_montage

    Return
    ------
    info : mne.Info
        The mne.Info object
    '''
    # https://mne.tools/stable/generated/mne.create_info.html#mne.create_info
    # https://mne.tools/stable/auto_tutorials/simulation/plot_creating_data_structures.html

    montage = mne.channels.make_standard_montage(kind)
    sfreq = 1000 
    info = mne.create_info(montage.ch_names, sfreq, ch_types=['eeg']*len(montage.ch_names), verbose=0)
    info.set_montage(kind)
    return info