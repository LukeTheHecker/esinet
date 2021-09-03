import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
from scipy.spatial.distance import cdist
from copy import deepcopy

def get_maxima_mask(y, pos, k_neighbors=5, threshold=0.1, min_dist=30,
    distance_matrix=None):
    ''' Returns the mask containing the source maxima (binary).
    
    Parameters
    ----------
    y : numpy.ndarray
        The source
    pos : numpy.ndarray
        The dipole position matrix
    k_neighbors : int
        The number of neighbors to incorporate for finding maximum
    threshold : float
        Proportion between 0 and 1. Defined the minimum value for a maximum to 
        be of significance. 0.1 -> 10% of the absolute maximum
    '''
    if distance_matrix is None:
        distance_matrix = cdist(pos, pos)

    mask = np.zeros((len(y)))
    y = np.abs(y)
    threshold = threshold*np.max(y)

    # find maxima that surpass the threshold:
    for i, _ in enumerate(y):
        distances = distance_matrix[i]
        close_idc = np.argsort(distances)[1:k_neighbors+1]
        if y[i] > np.max(y[close_idc]) and y[i] > threshold:
            mask[i] = 1

    # filter maxima
    maxima = np.where(mask==1)[0]
    distance_matrix_maxima = cdist(pos[maxima], pos[maxima])
    for i, _ in enumerate(maxima):
        distances_maxima = distance_matrix_maxima[i]
        close_maxima = maxima[np.where(distances_maxima < min_dist)[0]]
        # If there is a larger maximum in the close vicinity->delete maximum
        if np.max(y[close_maxima]) > y[maxima[i]]:
            mask[maxima[i]] = 0
    return mask
    
def get_maxima_pos(mask, pos):
    ''' Returns the positions of the maxima within mask.
    Parameters
    ----------
    mask : numpy.ndarray
        The source mask
    pos : numpy.ndarray
        The dipole position matrix
    '''
    return pos[np.where(mask==1)[0]]

def eval_mean_localization_error(y_true, y_est, pos, k_neighbors=5, 
    min_dist=30, threshold=0.1, ghost_thresh=40, distance_matrix=None):
    ''' Calculate the mean localization error for an arbitrary number of 
    sources.
    
    Parameters
    ----------
    y_true : numpy.ndarray
        The true source vector (1D)
    y_est : numpy.ndarray
        The estimated source vector (1D)
    pos : numpy.ndarray
        The dipole position matrix
    k_neighbors : int
        The number of neighbors to incorporate for finding maximum
    threshold : float
        Proportion between 0 and 1. Defined the minimum value for a maximum to 
        be of significance. 0.1 -> 10% of the absolute maximum
    min_dist : float/int
        The minimum viable distance in mm between maxima. The higher this 
        value, the more maxima will be filtered out.
    ghost_thresh : float/int
        The threshold distance between a true and a predicted source to not 
        belong together anymore. Predicted sources that have no true source 
        within the vicinity defined be ghost_thresh will be labeled 
        ghost_source.
    
    Return
    ------
    mean_localization_error : float
        The mean localization error between all sources in y_true and the 
        closest matches in y_est.
    '''
    y_true = deepcopy(y_true)
    y_est = deepcopy(y_est)
    maxima_true = get_maxima_pos(
        get_maxima_mask(y_true, pos, k_neighbors=k_neighbors, 
        threshold=threshold, min_dist=min_dist, 
        distance_matrix=distance_matrix), pos)
    maxima_est = get_maxima_pos(
        get_maxima_mask(y_est, pos, k_neighbors=k_neighbors,
        threshold=threshold, min_dist=min_dist, 
        distance_matrix=distance_matrix), pos)
    
    # Distance matrix between every true and estimated maximum
    distance_matrix = cdist(maxima_true, maxima_est)
    # For each true source find the closest predicted source:
    closest_matches = distance_matrix.min(axis=1)
    # Filter for ghost sources
    closest_matches = closest_matches[closest_matches<ghost_thresh]
    
    # No source left -> return nan
    if len(closest_matches) == 0:
        return np.nan
    mean_localization_error = np.mean(closest_matches)

    return mean_localization_error

def eval_ghost_sources(y_true, y_est, pos, k_neighbors=5, 
    min_dist=30, threshold=0.1, ghost_thresh=40):
    ''' Calculate the number of ghost sources in the estimated source.
    
    Parameters
    ----------
    y_true : numpy.ndarray
        The true source vector (1D)
    y_est : numpy.ndarray
        The estimated source vector (1D)
    pos : numpy.ndarray
        The dipole position matrix
    k_neighbors : int
        The number of neighbors to incorporate for finding maximum
    threshold : float
        Proportion between 0 and 1. Defined the minimum value for a maximum to 
        be of significance. 0.1 -> 10% of the absolute maximum
    min_dist : float/int
        The minimum viable distance in mm between maxima. The higher this 
        value, the more maxima will be filtered out.
    
    Return
    ------
    n_ghost_sources : int
        The number of ghost sources.
    '''
    y_true = deepcopy(y_true)
    y_est = deepcopy(y_est)
    maxima_true = get_maxima_pos(
        get_maxima_mask(y_true, pos, k_neighbors=k_neighbors, 
        threshold=threshold, min_dist=min_dist), pos)
    maxima_est = get_maxima_pos(
        get_maxima_mask(y_est, pos, k_neighbors=k_neighbors,
        threshold=threshold, min_dist=min_dist), pos)
    
    # Distance matrix between every true and estimated maximum
    distance_matrix = cdist(maxima_true, maxima_est)
    # For each true source find the closest predicted source:
    closest_matches = distance_matrix.min(axis=1)

    # Filter ghost sources
    ghost_sources = closest_matches[closest_matches>=ghost_thresh]
    n_ghost_sources = len(ghost_sources)

    return n_ghost_sources

def eval_found_sources(y_true, y_est, pos, k_neighbors=5, 
    min_dist=30, threshold=0.1, ghost_thresh=40):
    ''' Calculate the number of found sources in the estimated source.
    
    Parameters
    ----------
    y_true : numpy.ndarray
        The true source vector (1D)
    y_est : numpy.ndarray
        The estimated source vector (1D)
    pos : numpy.ndarray
        The dipole position matrix
    k_neighbors : int
        The number of neighbors to incorporate for finding maximum
    threshold : float
        Proportion between 0 and 1. Defined the minimum value for a maximum to 
        be of significance. 0.1 -> 10% of the absolute maximum
    min_dist : float/int
        The minimum viable distance in mm between maxima. The higher this 
        value, the more maxima will be filtered out.
    
    Return
    ------
    n_found_sources : int
        The Number of true found sources.
    '''
    y_true = deepcopy(y_true)
    y_est = deepcopy(y_est)
    maxima_true = get_maxima_pos(
        get_maxima_mask(y_true, pos, k_neighbors=k_neighbors, 
        threshold=threshold, min_dist=min_dist), pos)
    maxima_est = get_maxima_pos(
        get_maxima_mask(y_est, pos, k_neighbors=k_neighbors,
        threshold=threshold, min_dist=min_dist), pos)
    
    # Distance matrix between every true and estimated maximum
    distance_matrix = cdist(maxima_true, maxima_est)
    # For each true source find the closest predicted source:
    closest_matches = distance_matrix.min(axis=1)

    # Filter ghost sources
    found_sources = closest_matches[closest_matches<ghost_thresh]
    n_found_sources = len(found_sources)

    return n_found_sources


def eval_mse(y_true, y_est):
    '''Returns the mean squared error between predicted and true source. '''
    return np.mean((y_true-y_est)**2)

def eval_nmse(y_true, y_est):
    '''Returns the normalized mean squared error between predicted and true 
    source.'''
    
    y_true_normed = y_true / np.max(np.abs(y_true))
    y_est_normed = y_est / np.max(np.abs(y_est))
    return np.mean((y_true_normed-y_est_normed)**2)

def eval_auc(y_true, y_est, pos, n_redraw = 25, epsilon=0.05, plot_me=False):
    ''' Returns the area under the curve metric between true and predicted
    source. 
    Parameters
    ----------
    y_true : numpy.ndarray True source vector y_est : numpy.ndarray Estimated
        source vector pos : numpy.ndarray Dipole positions (points x dims)
        n_redraw : int Defines how often the negative samples are redrawn.
        epsilon : float Defines threshold on which sources are considered
        active.
    '''
    # Copy
    y_true = deepcopy(y_true)
    y_est = deepcopy(y_est)
    # Absolute values
    y_true = np.abs(y_true)
    y_est = np.abs(y_est)

    # Normalize values
    y_true /= np.max(y_true)
    y_est /= np.max(y_est)

    auc_close = np.zeros((n_redraw))
    auc_far = np.zeros((n_redraw))

    source_mask = (y_true>epsilon).astype(int)

    numberOfActiveSources = int(np.sum(source_mask))
    numberOfDipoles = pos.shape[0]
    # Draw from the 20% of closest dipoles to sources (~100)
    closeSplit = int(round(numberOfDipoles / 5))
    # Draw from the 50% of furthest dipoles to sources
    farSplit = int(round(numberOfDipoles / 2))
    distSortedIndices = find_indices_close_to_source(source_mask, pos)
    sourceIndices = np.where(source_mask==1)[0]
  
    
    for n in range(n_redraw):
        
        selectedIndicesClose = np.concatenate([sourceIndices, np.random.choice(distSortedIndices[:closeSplit], size=numberOfActiveSources) ])
        selectedIndicesFar = np.concatenate([sourceIndices, np.random.choice(distSortedIndices[-farSplit:], size=numberOfActiveSources) ])
        # print(f'redraw {n}:\ny_true={y_true[selectedIndicesClose]}\y_est={y_est[selectedIndicesClose]}')
        fpr_close, tpr_close, _ = roc_curve(source_mask[selectedIndicesClose], y_est[selectedIndicesClose])
   
        fpr_far, tpr_far, _  = roc_curve(source_mask[selectedIndicesFar], y_est[selectedIndicesFar])
        
        auc_close[n] = auc(fpr_close, tpr_close)
        auc_far[n] = auc(fpr_far, tpr_far)
    
    auc_far = np.mean(auc_far)
    auc_close = np.mean(auc_close)
    
    if plot_me:
        print("plotting")
        plt.figure()
        plt.plot(fpr_close, tpr_close, label='ROC_close')
        plt.plot(fpr_far, tpr_far, label='ROC_far')
        # plt.xlim(1, )
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'AUC_close={auc_close:.2f}, AUC_far={auc_far:.2f}')
        plt.legend()
        plt.show()
    

    return auc_close, auc_far

def find_indices_close_to_source(source_mask, pos):
    ''' Finds the dipole indices that are closest to the active sources. 
    Parameters:
    -----------
    simSettings : dict, retrieved from the simulate_source function
    pos : numpy.ndarray, list of all dipole positions in XYZ coordinates
    Return:
    -------
    ordered_indices : numpy.ndarray, ordered list of dipoles that are near active sources in ascending order with respect to their distance to         the next source.'''
    numberOfDipoles = pos.shape[0]

    sourceIndices = np.array([i[0] for i in np.argwhere(source_mask==1)])
    numberOfNans = 0
    min_distance_to_source = np.zeros((numberOfDipoles))
    for i in range(numberOfDipoles):
        if source_mask[i] == 1:
            min_distance_to_source[i] = np.nan
            numberOfNans +=1
        elif source_mask[i] == 0:
            distances = np.sqrt(np.sum((pos[sourceIndices, :] - pos[i, :])**2, axis=1))
            min_distance_to_source[i] = np.min(distances)
        else:
            print('source mask has invalid entries')
    
    # min_distance_to_source = min_distance_to_source[~np.isnan(min_distance_to_source)]
    ordered_indices = np.argsort(min_distance_to_source)
    # ordered_indices[np.where(~np.isnan(min_distance_to_source[ordered_indices]]
    return ordered_indices[:-numberOfNans]

# import numpy as np
# from esinet.evaluate import eval_auc
# y_true = np.random.randn(1284)
# y_est = y_true + np.random.randn(1284)*0.5
# pos = np.random.randn(1284,3)
# eval_auc(y_true, y_est, pos)