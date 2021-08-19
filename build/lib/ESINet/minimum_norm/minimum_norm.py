import numpy as np
from copy import deepcopy


def centeringMatrix(n):
    ''' Centering matrix, which when multiplied with a vector subtract the mean of the vector.'''
    C = np.identity(n) - (1/n) * np.ones((n, n))
    return C

def eloreta(x, leadfield, tikhonov=0.05, stopCrit=0.005, verbose=False):
    D, C = calc_eloreta_D(leadfield, tikhonov, stopCrit=stopCrit, verbose=verbose)
    
    K_elor = np.matmul( np.matmul(np.linalg.inv(D), leadfield.T), np.linalg.inv( np.matmul( np.matmul( leadfield, np.linalg.inv(D) ), leadfield.T) + (tikhonov**2 * C) ) )

    y_est = np.matmul(K_elor, x)
    return y_est

def calc_eloreta_D(leadfield, tikhonov, stopCrit=0.005, verbose=False):
    ''' Algorithm that optimizes weight matrix D as described in 
        Assessing interactions in the brain with exactlow-resolution electromagnetic tomography; Pascual-Marqui et al. 2011 and
        https://www.sciencedirect.com/science/article/pii/S1053811920309150
        '''
    numberOfElectrodes, numberOfVoxels = leadfield.shape
    # initialize weight matrix D with identity and some empirical shift (weights are usually quite smaller than 1)
    D = np.identity(numberOfVoxels)
    H = centeringMatrix(numberOfElectrodes)
    if verbose:
        print('Optimizing eLORETA weight matrix W...')
    cnt = 0
    while True:
        old_D = deepcopy(D)
        if verbose:
            print(f'\trep {cnt+1}')
        C = np.linalg.pinv( np.matmul( np.matmul(leadfield, np.linalg.inv(D)), leadfield.T ) + (tikhonov * H) )
        for v in range(numberOfVoxels):
            leadfield_v = np.expand_dims(leadfield[:, v], axis=1)
            D[v, v] = np.sqrt( np.matmul(np.matmul(leadfield_v.T, C), leadfield_v) )
        
        averagePercentChange = np.abs(1 - np.mean(np.divide(np.diagonal(D), np.diagonal(old_D))))
        if verbose:
            print(f'averagePercentChange={100*averagePercentChange:.2f} %')
        if averagePercentChange < stopCrit:
            if verbose:
                print('\t...converged...')
            break
        cnt += 1
    if verbose:
        print('\t...done!')
    return D, C