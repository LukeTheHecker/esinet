import os
import mne
import pickle as pkl
import numpy as np

def create_forward_model(savepath, sampling='ico3', info=None, verbose=0):
    ''' Create files of the forward model and source model. 
    Parameters:
    ----------
    savepath : str, path to store the forward model files 
    sampling : str, the downsampling of the forward model. 
        Recommended are 'ico3' (small), 'ico4' (medium) or 
        'ico5' (large).
    info : mne.Info, info instance which contains the desired 
        electrode names and positions. 
        This can be obtained e.g. from your processed mne.Raw.info, 
        mne.Epochs.info or mne.Evoked.info
        If info is None the Info instance is created where 
        electrodes are chosen automatically from the easycap-M10 
        layout.
    Return:
    -------
    <nothing is returned>
    '''
    
    if os.path.isdir(savepath) and len(os.listdir(savepath))>0:
        print(f'Model already exists at path {savepath}')
        return
    # Fetch the template files for our forward model
    fs_dir = mne.datasets.fetch_fsaverage(verbose=verbose)
    subjects_dir = os.path.dirname(fs_dir)

    # The files live in:
    subject = 'fsaverage'
    trans = os.path.join(fs_dir, 'bem', 'fsaverage-trans.fif')
    src = os.path.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif')
    bem = os.path.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')

    # Handling savepath
    if not savepath.endswith('/'):
        savepath += '/'

    if not os.path.isdir(savepath):
        os.makedirs(savepath)
    # Create our own info object, see e.g.:
    if info is None:
        info = get_info()
        

    # Create our own Epoch object
    epochs, evoked = create_fake_epochs(info)

    # Create and save Source Model
    src = mne.setup_source_space(subject, spacing=sampling, surface='white',
                                        subjects_dir=subjects_dir, add_dist=False,
                                        n_jobs=-1, verbose=verbose)

    src.save('{}\\{}-src.fif'.format(savepath, sampling), overwrite=True)

    # Forward Model
    fwd = mne.make_forward_solution(info, trans=trans, src=src,
                                    bem=bem, eeg=True, mindist=5.0, n_jobs=-1,
                                    verbose=verbose)
    mne.write_forward_solution(savepath+'\\{}-fwd.fif'.format(subject),fwd, 
                            overwrite=True, verbose=verbose)
    # Fixed Orientations
    fwd_fixed = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True,
                                            use_cps=True, verbose=verbose)

    ###############################################################################
    # Create Container for Source Estimate which is needed to plot data later on  #
    noise_cov = mne.compute_covariance(epochs, tmax=0., method=['empirical'], 
                                    rank=None, verbose=verbose)

    inv = mne.minimum_norm.make_inverse_operator(info, fwd, noise_cov,
                                        loose=0.2, depth=0.8, verbose=verbose)

    stc, residual = mne.minimum_norm.apply_inverse(evoked, inv, 0.05,
                                method="dSPM", pick_ori="normal",
                                return_residual=True, verbose=verbose)

    stc.save(savepath+"\\ResSourceEstimate".format(), ftype='stc', verbose=verbose)

    mne.minimum_norm.write_inverse_operator(savepath+"\\inverse-inv.fif",
                        inv, verbose=verbose)
    tris = inv['src'][0]['use_tris']
    # inv.save(, ftype='fif')
            


    leadfield = fwd_fixed['sol']['data']
    print(f'shape of leadfield: {leadfield.shape}')

    # Load source space file
    source = mne.read_source_spaces(savepath+"\"+sampling+"-src.fif", verbose=verbose) #changed / into \ . mm_2021-05-25
    pos_left = mne.vertex_to_mni(source[0]['vertno'], hemis=0, 
        subject='fsaverage', verbose=verbose)
    pos_right = mne.vertex_to_mni(source[0]['vertno'], hemis=1, 
        subject='fsaverage', verbose=verbose)
    pos = np.concatenate([pos_left, pos_right], axis=0)

    # save leadfield
    fn = "{}\\leadfield.pkl".format(savepath)
    with open(fn, 'wb') as f:
        pkl.dump([leadfield], f)

    # save pos
    fn = "{}\\pos.pkl".format(savepath)
    with open(fn, 'wb') as f:
        pkl.dump([pos], f)

    fn =f'{savepath}/info.pkl'
    with open(fn, 'wb') as f:
        pkl.dump(info, f)
    if verbose is not None and verbose!=0:
        print(f'All files for the forward model were saved to {savepath}')

def get_info():
    # https://mne.tools/stable/generated/mne.create_info.html#mne.create_info
    # https://mne.tools/stable/auto_tutorials/simulation/plot_creating_data_structures.html

    montage = mne.channels.make_standard_montage('easycap-M10')
    sfreq = 1000 
    info = mne.create_info(montage.ch_names, sfreq, ch_types=['eeg']*len(montage.ch_names), verbose=0)
    info.set_montage('easycap-M10')
    return info

def create_fake_epochs(info):
    n_time = 1000
    numberOfChannels = len(info.ch_names)
    fake_data = np.random.randn(5, numberOfChannels, n_time)
    epochs = mne.EpochsArray(fake_data, info, verbose=0)
    epochs.set_eeg_reference('average', projection=True, verbose=0)
    evoked = epochs.average()
    evoked.set_eeg_reference('average', projection=True, verbose=0)
    
    return epochs, evoked