import mne
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn.palettes import color_palette; sns.set(font_scale=1.25,style='ticks',context='notebook',font= 'georgia')
import mayavi

def splot(data, pth_fwd, del_below=0.1, backend='mayavi', title='title', \
        figure=None, alpha=1, transparent=True, smoothing_steps=10, cmap='inferno', \
        clim=None, sfreq=1000., correctionFactor=1e9):

    ''' quickly plot a source '''
    
    data = np.squeeze(np.array(data))
    if np.min(data) >= 0:
        data_range = 'pos'
    elif np.max(data) <=0:
        data_range = 'neg'
    else:
        data_range = 'mixed'
    # Convert from nAm to Am (Ampere-meter)
    data_corrected = data * correctionFactor
    maxabs = np.max(np.abs(data_corrected))
    
    if figure is None:
        if backend == 'matplotlib':
            figure = plt.figure()
        elif backend == 'mayavi:':
            figure = mayavi.mlab.figure()
 
    if clim is None:
        clim = {'kind': 'value',
                # 'lims': (-maxabs, 0, maxabs)
                'pos_lims': (0, maxabs/3, maxabs)
                }
    
    # Read some dummy object to assign the voxel values to
    a = mne.read_source_estimate(pth_fwd + "/ResSourceEstimate-lh.stc")
    # Get some info 
    n_voxels = a.data.shape[0]

    # Get it in shape
    if len(data_corrected.shape) == 1:
        data_corrected = np.expand_dims(data_corrected, axis=1)
    if data_corrected.shape[0] != n_voxels:
        data_corrected = np.transpose(data_corrected)
    n_timeframes = data_corrected.shape[1]

    # Delete values below threshold "del_below"
    if del_below > 0:
        if data_range == 'pos':
            for i in range(data_corrected.shape[1]):
                timeframe = data_corrected[:, i]
                mask_below_thr = timeframe < (np.max(timeframe) * del_below)
                data_corrected[mask_below_thr, i] = 0
        elif data_range == 'neg':
            for i in range(data_corrected.shape[1]):
                timeframe = data_corrected[:, i]
                mask_below_thr = timeframe > (np.min(timeframe) * del_below)
                data_corrected[mask_below_thr, i] = 0
        elif data_range == 'mixed':
            for i in range(data_corrected.shape[1]):
                timeframe = data_corrected[:, i]
                # negative values to be deleted
                negative_deletion_mask = np.logical_and((timeframe > np.min(timeframe) * del_below), (timeframe < 0))
                # positive values to be deleted
                positive_deletion_mask = np.logical_and((timeframe < np.max(timeframe) * del_below), (timeframe > 0))
                # delete them
                data_corrected[negative_deletion_mask, i] = 0
                data_corrected[positive_deletion_mask, i] = 0

            
    tmin, tmax = [0, (n_timeframes-1) / sfreq]

    stc = mne.SourceEstimate(data_corrected, a.vertices, tmin, tmax, subject='fsaverage', verbose=None)

    # Use stc-object plot function
    kwargs = dict(initial_time=tmin, surface="white", backend=backend, \
        title=title+'_lh' , clim=clim, transparent=transparent, figure=figure, \
        alpha=alpha, smoothing_steps=smoothing_steps, colormap=cmap, time_viewer=False)

    fleft = stc.plot(hemi='lh', **kwargs)
    fright = stc.plot(hemi='rh', **kwargs)
    return fleft, fright