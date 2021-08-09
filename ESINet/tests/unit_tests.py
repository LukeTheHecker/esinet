from ..forward import create_forward_model
from ..simulations import run_simulations, create_eeg
from itertools import product

def create_forward_model_test(pth_fwd='temp/ico2/', sampling='ico2'):
    # Create a forward model
    try:
        create_forward_model(pth_fwd, sampling=sampling)
        
        return True
    except Exception as e:
        print(e)
        return False
    
def run_simulations_test(pth_fwd='temp/ico2/'):
    # Define parameters
    n_simulations = 2
    n_sources_list = [(1, 2), 3]
    extents_list = [(2, 3), 3] 
    amplitudes_list = [(5, 10), 3]
    shape_list = ['gaussian', 'flat', 'both']
    durOfTrial_list = [0, 1]
    
    sampleFreq_list = [1, 10]
    regionGrowing_list = [True, False]
    return_raw_data_list = [True, False]
    return_single_epoch_list = [True, False]
    
    snr_list = [1]
    n_trials_list = [1, 2]
    beta_list = [3]

    combinations_source = [n_sources_list, extents_list, amplitudes_list, 
                        shape_list, durOfTrial_list, sampleFreq_list, 
                        regionGrowing_list, return_raw_data_list, 
                        return_single_epoch_list]
    combinations_eeg = [snr_list, n_trials_list, beta_list]
    
    all_combinations = combinations_source
    all_combinations.extend(combinations_eeg)

    # try:
    cnt = 0
    for n_sources, extents, amplitudes, shape, durOfTrial, \
        sampleFreq, regionGrowing, return_raw_data, \
        return_single_epoch, snr, n_trials, beta \
        in product(*all_combinations):

        # print(f'durOfTrial={durOfTrial}, sampleFreq={sampleFreq}')
        cnt+= 1
        # print(cnt)
        # Simulate some source and EEG data
        sources_sim = run_simulations(pth_fwd, n_simulations=n_simulations, 
            n_sources=n_sources, extents=extents, amplitudes=amplitudes,
            shape=shape, durOfTrial=durOfTrial, sampleFreq=sampleFreq,
            regionGrowing=regionGrowing, return_raw_data=return_raw_data,
            return_single_epoch=return_single_epoch)
        eeg_sim = create_eeg(sources_sim, pth_fwd, snr=snr, n_trials=n_trials, beta=beta)
    return True
    # except Exception as e:
    #     print(e)
    #     return False