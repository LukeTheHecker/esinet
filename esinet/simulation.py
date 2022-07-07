from copy import deepcopy
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
# import pickle as pkl
import dill as pkl
import random
from joblib import Parallel, delayed
# from tqdm.notebook import tqdm
from tqdm import tqdm

import colorednoise as cn
import mne
from time import time
from . import util

DEFAULT_SETTINGS = {
    'method': 'standard',
    'number_of_sources': (1, 25),
    'extents':  (1, 50),  # in millimeters
    'amplitudes': (1e-3, 100),
    'shapes': 'mixed',
    'duration_of_trial': 1.0,
    'sample_frequency': 100,
    'target_snr': (1, 20),
    'beta': (0.5, 3),  # (0, 3),
    'exponent': 3,
    'source_spread': "mixed",
    'source_number_weighting': True,
    'source_time_course': "random",

}

class Simulation:
    ''' Simulate and hold source and M/EEG data.
    
    Attributes
    ----------
    settings : dict
        The Settings for the simulation. Keys:

        number_of_sources : int/tuple/list
            number of sources. Can be a single number or a list of two numbers 
            specifying a range.
        extents : int/float/tuple/list
            size of sources in mm. Can be a single number or a list of two 
            numbers specifying a range.
        amplitudes : int/float/tuple/list
            the current of the source in nAm
        shapes : str
            How the amplitudes evolve over space. Can be 'gaussian' or 'flat' 
            (i.e. uniform) or 'mixed'.
        duration_of_trial : int/float
            specifies the duration of a trial.
        sample_frequency : int
            specifies the sample frequency of the data.
        target_snr : float/tuple/list
            The desired average SNR of the simulation(s)
        beta : float/tuple/list
            The desired frequency spectrum slope (1/f**beta) of the noise. 
    fwd : mne.Forward
        The mne-python Forward object that contains the forward model
    source_data : mne.sourceEstimate
        A source estimate object from mne-python which contains the source 
        data.
    eeg_data : mne.Epochs
        A mne.Epochs object which contains the EEG data.
    n_jobs : int
        The number of jobs/cores to utilize.
    
    Methods
    -------
    simulate : Simulate source and EEG data
    plot : plot a random sample source and EEG

    '''
    def __init__(self, fwd, info, settings=DEFAULT_SETTINGS, n_jobs=-1, 
        parallel=False, verbose=False):
        self.settings = settings
        
        self.source_data = None
        self.eeg_data = None
        self.fwd = deepcopy(fwd)
        self.fwd.pick_channels(info['ch_names'])
        self.check_info(deepcopy(info))

        self.check_settings()
        self.settings['sample_frequency'] = info['sfreq']

        # self.info['sfreq'] = self.settings['sample_frequency']
        self.prepare_simulation_info()
        self.subject = self.fwd['src'][0]['subject_his_id']
        self.n_jobs = n_jobs
        self.parallel = parallel
        self.verbose = verbose
        self.diams = None
    
    def __add__(self, other):
        new_object = deepcopy(self)
        new_object.source_data.extend(other.source_data)
        new_object.eeg_data.extend(other.eeg_data)
        new_object.simulation_info.append(other.simulation_info)
        new_object.n_samples += other.n_samples
        if new_object.settings["method"] != other.settings["method"]:
            new_object.settings["method"] = "mixed"
        return new_object
        
    def check_info(self, info):
        self.info = info.pick_channels(self.fwd.ch_names, ordered=True)

    def prepare_simulation_info(self):
        self.simulation_info = pd.DataFrame(columns=['number_of_sources', 'positions', 'extents', 'amplitudes', 'shapes', 'target_snr', 'betas', 'duration_of_trials'])

    def simulate(self, n_samples=10000):
        ''' Simulate sources and EEG data'''
        self.n_samples = n_samples
        self.source_data = self.simulate_sources(n_samples)
        self.eeg_data = self.simulate_eeg()

        return self

    def plot(self):
        pass
    
    def simulate_sources(self, n_samples):
        
        n_dip = self.pos.shape[0]
        # source_data = np.zeros((n_samples, n_dip, n_time), dtype=np.float32)
        source_data = []
        if self.verbose:
                print(f'Simulate Source')


        if self.settings["method"] == "standard":
            print("Simulating data based on sparse patches.")
            if self.parallel:
                source_data = Parallel(n_jobs=self.n_jobs, backend='loky') \
                    (delayed(self.simulate_source)() 
                    for _ in tqdm(range(n_samples)))
            else:
                for i in tqdm(range(n_samples)):
                    source_data.append( self.simulate_source() )
            
        elif self.settings["method"] == "noise":
            print("Simulating data based on 1/f noise.")
            self.prepare_grid()
            if self.parallel:
                source_data = Parallel(n_jobs=self.n_jobs, backend='loky') \
                    (delayed(self.simulate_source_noise)() 
                    for _ in tqdm(range(n_samples)))
            else:
                for i in tqdm(range(n_samples)):
                    source_data.append( self.simulate_source_noise() )
        elif self.settings["method"] == "mixed":
            print("Simulating data based on 1/f noise and sparse patches.")
            self.prepare_grid()
            if self.parallel:
                source_data_tmp = Parallel(n_jobs=self.n_jobs, backend='loky') \
                    (delayed(self.simulate_source_noise)() 
                    for _ in tqdm(range(int(n_samples/2))))
                for single_source in source_data_tmp:
                    source_data.append( single_source )
                source_data_tmp = Parallel(n_jobs=self.n_jobs, backend='loky') \
                    (delayed(self.simulate_source)() 
                    for _ in tqdm(range(int(n_samples/2), n_samples)))

                for single_source in source_data_tmp:
                    source_data.append( single_source )
            else:
                for i in tqdm(range(int(n_samples/2))):
                    source_data.append( self.simulate_source_noise() )
                for i in tqdm(range(int(n_samples/2), n_samples)):
                    source_data.append( self.simulate_source() )
                    

        # Convert to mne.SourceEstimate
        if self.verbose:
            print(f'Converting Source Data to mne.SourceEstimate object')
        # if self.settings['duration_of_trial'] == 0:
        #     sources = util.source_to_sourceEstimate(source_data, self.fwd, 
        #         sfreq=self.settings['sample_frequency'], subject=self.subject) 
        # else:
        sources = self.sources_to_sourceEstimates(source_data)
        return sources

    def prepare_grid(self):
        n = 10
        n_time = np.clip(int(self.info['sfreq'] * np.max(self.settings['duration_of_trial'])), a_min=1, a_max=np.inf).astype(int)
        shape = (n,n,n,n_time)
        
        x = np.linspace(self.pos[:, 0].min(), self.pos[:, 0].max(), num=shape[0])
        y = np.linspace(self.pos[:, 1].min(), self.pos[:, 1].max(), num=shape[1])
        z = np.linspace(self.pos[:, 2].min(), self.pos[:, 2].max(), num=shape[2])
        k_neighbors = 5
        grid = np.stack(np.meshgrid(x,y,z, indexing='ij'), axis=0)
        grid_flat = grid.reshape(grid.shape[0], np.product(grid.shape[1:])).T
        neighbor_indices = np.stack([
            np.argsort(np.sqrt(np.sum((grid_flat - coords)**2, axis=1)))[:k_neighbors] for coords in self.pos
        ], axis=0)

        self.grid = {
            "shape": shape,
            "k_neighbors": k_neighbors,
            "exponent": self.settings["exponent"],
            "x": x,
            "y": y,
            "z": z,
            "grid": grid,
            "grid_flat": grid_flat,
            "neighbor_indices": neighbor_indices
        }
        

    def simulate_source_noise(self):
        exponent = self.get_from_range(self.grid["exponent"], dtype=float)
        src_3d = util.create_n_dim_noise(self.grid["shape"], exponent=exponent)

        duration_of_trial = self.get_from_range(
            self.settings['duration_of_trial'], dtype=float)
        n_time = np.clip(int(round(duration_of_trial * self.info['sfreq'])), 1, None)
        if len(src_3d.shape) == 3:
            src_3d = src_3d[:,:,:,np.newaxis]
        src = np.zeros((self.pos.shape[0], n_time))
        for i in range(n_time):
            src[:, i] = util.vol_to_src(self.grid["neighbor_indices"], src_3d[:, :, :, i], self.pos)
        
        d = dict(number_of_sources=np.nan, positions=[np.nan], extents=[np.nan], amplitudes=[np.nan], shapes=[np.nan], target_snr=0, duration_of_trials=duration_of_trial)
        self.simulation_info = self.simulation_info.append(d, ignore_index=True)
        return src
        
    def sources_to_sourceEstimates(self, source_data):
        template = util.source_to_sourceEstimate(source_data[0], 
                    self.fwd, sfreq=self.settings['sample_frequency'], 
                    subject=self.subject)
        sources = []
        for source in tqdm(source_data):
            tmp = deepcopy(template)
            tmp.data = source
            sources.append(tmp)
        return sources


    def simulate_source(self):
        ''' Returns a vector containing the dipole currents. Requires only a 
        dipole position list and the simulation settings.

        Parameters
        ----------
        pos : numpy.ndarray
            (n_dipoles x 3), list of dipole positions.
        number_of_sources : int/tuple/list
            number of sources. Can be a single number or a list of two 
            numbers specifying a range.
        extents : int/float/tuple/list
            diameter of sources (in mm). Can be a single number or a list of 
            two numbers specifying a range.
        amplitudes : int/float/tuple/list
            the current of the source in nAm
        shapes : str
            How the amplitudes evolve over space. Can be 'gaussian' or 'flat' 
            (i.e. uniform) or 'mixed'.
        duration_of_trial : int/float
            specifies the duration of a trial.
        sample_frequency : int
            specifies the sample frequency of the data.
        
        Return
        ------
        source : numpy.ndarray, (n_dipoles x n_timepoints), the simulated 
            source signal
        simSettings : dict, specifications about the source.

        Grova, C., Daunizeau, J., Lina, J. M., Bénar, C. G., Benali, H., & 
            Gotman, J. (2006). Evaluation of EEG localization methods using 
            realistic simulations of interictal spikes. Neuroimage, 29(3), 
            734-753.
        '''
        
        ###########################################
        # Select ranges and prepare some variables
        # Get number of sources:
        if not self.settings["source_number_weighting"] or isinstance(self.settings["number_of_sources"], (float, int)):
            number_of_sources = self.get_from_range(
                self.settings['number_of_sources'], dtype=int)
        else:
            population = np.arange(*self.settings["number_of_sources"])
            weights = 1 / population
            weights /= weights.sum()
            number_of_sources = random.choices(population=population,weights=weights,k=1)[0]

        
        if self.settings["source_spread"] == 'mixed':
            source_spreads = [np.random.choice(['region_growing', 'spherical']) for _ in range(number_of_sources)]
        else:
            source_spreads = [self.settings["source_spread"] for _ in range(number_of_sources)]

   
        extents = [self.get_from_range(self.settings['extents'], dtype=float) 
            for _ in range(number_of_sources)]
        
        # Decide shape of sources
        if self.settings['shapes'] == 'mixed':
            shapes = ['gaussian', 'flat']*number_of_sources
            np.random.shuffle(shapes)
            shapes = shapes[:number_of_sources]
            if type(shapes) == str:
                shapes = [shapes]

        elif self.settings['shapes'] == 'gaussian' or self.settings['shapes'] == 'flat':
            shapes = [self.settings['shapes']] * number_of_sources
        
        # Get amplitude gain for each source (amplitudes come in nAm)
        amplitudes = [self.get_from_range(self.settings['amplitudes'], dtype=float) * 1e-9 for _ in range(number_of_sources)]
        
        src_centers = np.random.choice(np.arange(self.pos.shape[0]), \
            number_of_sources, replace=False)
        duration_of_trial = self.get_from_range(
            self.settings['duration_of_trial'], dtype=float
        )
        signal_length = int(round(self.settings['sample_frequency']*duration_of_trial))

        if signal_length > 1:
            signals = []
            
            if self.settings["source_time_course"].lower() == "pulse":
                signals = [self.get_biphasic_pulse(signal_length) for _ in range(number_of_sources)]
            else:
                for _ in range(number_of_sources):
                    signal = cn.powerlaw_psd_gaussian(self.get_from_range(self.settings['beta'], dtype=float), signal_length) 
                    signal /= np.max(np.abs(signal))
                    signals.append(signal)
            
            sample_frequency = self.settings['sample_frequency']
        else:  # else its a single instance
            sample_frequency = 0
            signal_length = 1
            signals = list(np.random.choice([-1, 1], number_of_sources))
        
        
        # sourceMask = np.zeros((self.pos.shape[0]))
        source = np.zeros((self.pos.shape[0], signal_length))
        ##############################################
        # Loop through source centers (i.e. seeds of source positions)
        for i, (src_center, shape, amplitude, signal, source_spread) in enumerate(zip(src_centers, shapes, amplitudes, signals, source_spreads)):
            if source_spread == "region_growing":
                order = self.extents_to_orders(extents[i])
                d = np.array(get_n_order_indices(order, src_center, self.neighbors))
                # if isinstance(d, (int, float, np.int32)):
                #     d = [d,]
                dists = np.empty((self.pos.shape[0]))
                dists[:] = np.inf
                dists[d] = np.sqrt(np.sum((self.pos - self.pos[src_center, :])**2, axis=1))[d]
            else:
                # dists = np.sqrt(np.sum((self.pos - self.pos[src_center, :])**2, axis=1))
                dists = self.distance_matrix[src_center]
                d = np.where(dists<extents[i]/2)[0]
        
            if shape == 'gaussian':
                
                if len(d) < 2:                    
                    activity = np.zeros((len(dists), 1))
                    activity[d, 0] = amplitude
            
                    
                    activity = activity * signal
                else:
                    sd = np.clip(np.max(dists[d]) / 2, a_min=0.1, a_max=np.inf)  # <- works better
                    activity = np.expand_dims(util.gaussian(dists, 0, sd) * amplitude, axis=1) * signal
                source += activity
            elif shape == 'flat':
                if not isinstance(signal, (list, np.ndarray)):
                    signal = np.array([signal])
                    activity = util.repeat_newcol(amplitude * signal, len(d))
                    if len(activity.shape) == 0:
                        activity = np.array([activity]).T[:, np.newaxis]
                    else:
                        activity = activity.T[:, np.newaxis]
                    
                else:
                    activity = util.repeat_newcol(amplitude * signal, len(d)).T
                if len(activity.shape) == 1:
                    if len(d) == 1:
                        activity = np.expand_dims(activity, axis=0)    
                    else:
                        activity = np.expand_dims(activity, axis=1)
                source[d, :] += activity 
            else:
                msg = BaseException("shape must be of type >string< and be either >gaussian< or >flat<.")
                raise(msg)
        
        # Document the sample
        d = dict(number_of_sources=number_of_sources, positions=self.pos[src_centers], extents=extents, amplitudes=amplitudes, shapes=shapes, target_snr=0, duration_of_trials=duration_of_trial)
        self.simulation_info = self.simulation_info.append(d, ignore_index=True)
        # self.simulation_info = pd.concat([self.simulation_info, d])
        
        return source

    def simulate_eeg(self):
        ''' Create EEG of specified number of trials based on sources and some SNR.
        Parameters
        -----------
        sourceEstimates : list 
                        list containing mne.SourceEstimate objects
        fwd : mne.Forward
            the mne.Forward object
        target_snr : tuple/list/float, 
                    desired signal to noise ratio. Can be a list or tuple of two 
                    floats specifying a range.
        beta : float
            determines the frequency spectrum of the noise added to the signal: 
            power = 1/f^beta. 
            0 will yield white noise, 1 will yield pink noise (1/f spectrum)
        n_jobs : int
                Number of jobs to run in parallel. -1 will utilize all cores.
        return_raw_data : bool
                        if True the function returns a list of mne.SourceEstimate 
                        objects, otherwise it returns raw data

        Return
        -------
        epochs : list
                list of either mne.Epochs objects or list of raw EEG data 
                (see argument <return_raw_data> to change output)
        '''

        n_simulation_trials = 20
         
        # Desired Dim of sources: (samples x dipoles x time points)
        # unpack numpy array of source data
        # print(type(self.source_data))
        # if isinstance(self.source_data, (list, tuple)):
        #     sources = np.stack([source.data for source in self.source_data], axis=0)
        # else:
        #     sources = self.source_data.data.T

        # if there is no temporal dimension...
        for i, source in enumerate(self.source_data):
            if len(source.shape) == 1:
                self.source_data[i] = np.expand_dims(source, axis=-1)
        print('source data shape: ', self.source_data[0].shape, self.source_data[1].shape)
                

        # Load some forward model objects
        fwd_fixed, leadfield = util.unpack_fwd(self.fwd)[:2]
        n_elec = leadfield.shape[0]
        n_samples = np.clip(len(self.source_data), a_min=1, a_max=np.inf).astype(int)

        target_snrs = [self.get_from_range(self.settings['target_snr'], dtype=float) for _ in range(n_samples)]
        betas = [self.get_from_range(self.settings['beta'], dtype=float) for _ in range(n_samples)]

        # Document snr and beta into the simulation info
        
        self.simulation_info['betas'] = betas
        self.simulation_info['target_snr'] = target_snrs
    
        # Desired Dim for eeg_clean: (samples, electrodes, time points)
        if self.verbose:
            print(f'\nProject sources to EEG...')
        eeg_clean = self.project_sources(self.source_data)
        # print(type(eeg_clean), eeg_clean[0].shape)
        if self.verbose:
            print(f'\nCreate EEG trials with noise...')
        
        # Parallel processing was removed since it was extraordinarily slow:
        # if self.parallel:
        #     eeg_trials_noisy = Parallel(n_jobs=self.n_jobs, backend='loky') \
        #         (delayed(self.create_eeg_helper)(eeg_clean[sample], n_simulation_trials,
        #             target_snrs[sample], betas[sample]) 
        #         for sample in tqdm(range(n_samples)))
        # else:
        # eeg_trials_noisy = np.zeros((eeg_clean.shape[0], n_simulation_trials, *eeg_clean.shape[1:]))
        
        eeg_trials_noisy = []
        for sample in tqdm(range(n_samples)):
            eeg_trials_noisy.append( self.create_eeg_helper(eeg_clean[sample], 
                n_simulation_trials, target_snrs[sample], betas[sample]) 
            )
        for i, eeg_trial_noisy in enumerate(eeg_trials_noisy):
            if len(eeg_trial_noisy.shape) == 2:
                eeg_trials_noisy[i] = np.expand_dims(eeg_trial_noisy, axis=-1)
            if eeg_trial_noisy.shape[1] != n_elec:
                eeg_trials_noisy[i] = np.swapaxes(eeg_trial_noisy, 1, 2)
        
        if self.verbose:
            print(f'\nConvert EEG matrices to a single instance of mne.Epochs...')
        ERP_samples_noisy = [np.mean(eeg_trial_noisy, axis=0) for eeg_trial_noisy in eeg_trials_noisy]
        epochs = util.eeg_to_Epochs(ERP_samples_noisy, fwd_fixed, info=self.info)

        return epochs
    
    def create_eeg_helper(self, eeg_sample, n_simulation_trials, target_snr, beta):
        ''' Helper function for EEG simulation that transforms a clean 
            M/EEG signal to a bunch of noisy trials.

        Parameters
        ----------
        eeg_sample : numpy.ndarray
            data sample with dimension (time_points, electrodes)
        n_simulation_trials : int
            The number of trials desired
        target_snr : float
            The target signal-to-noise ratio
        beta : float
            The beta exponent of the 1/f**beta noise

        '''
        
        assert len(eeg_sample.shape) == 2, 'Length of eeg_sample must be 2 (time_points, electrodes)'
        
        eeg_sample = np.repeat(np.expand_dims(eeg_sample, 0), n_simulation_trials, axis=0)
        snr = target_snr / np.sqrt(n_simulation_trials)
        
        # Before: Add noise based on the GFP of all channels
        # noise_trial = self.add_noise(eeg_sample, snr, beta=beta)
        
        # NEW: ADD noise for different types of channels, separately
        # since they can have entirely different scales.
        coil_types = [ch['coil_type'] for ch in self.info['chs']]
        coil_types_set = list(set(coil_types))
        if len(coil_types_set)>1:
            msg = f'Simulations attempted with more than one channel type \
                ({coil_types_set}) may result in unexpected behavior. Please \
                select one channel type in your data only'
            raise ValueError(msg)
            
        coil_types_set = np.array([int(i) for i in coil_types_set])
        
        coil_type_assignments = np.array(
            [np.where(coil_types_set==coil_type)[0][0] 
                for coil_type in coil_types]
        )
        noise_trial = np.zeros(
            eeg_sample.shape
        )

        for i, coil_type in enumerate(coil_types_set):
            channel_indices = np.where(coil_type_assignments==i)[0]
            eeg_sample_temp = eeg_sample[:, channel_indices, :]
            noise_trial_subtype = self.add_noise(eeg_sample_temp, snr, beta=beta)
            noise_trial[:, channel_indices, :] = noise_trial_subtype


        

        return noise_trial
    
    def project_sources(self, sources):
        ''' Project sources through the leadfield to obtain the EEG data.
        Parameters
        ----------
        sources : numpy.ndarray
            3D array of shape (samples, dipoles, time points)
        
        Return
        ------

        '''
        fwd_fixed, leadfield = util.unpack_fwd(self.fwd)[:2]
        n_samples = len(sources)
        # n_elec, n_dipoles = leadfield.shape
        # eeg = np.zeros((n_samples, n_elec, n_timepoints))
        eeg = []
        # Swap axes to dipoles, samples, time_points
        # sources_tmp = np.swapaxes(sources, 0,1)
        # Collapse last two dims into one
        # short_shape = (sources_tmp.shape[0], 
            # sources_tmp.shape[1]*sources_tmp.shape[2])
        # sources_tmp = sources_tmp.reshape(short_shape)

        result = [np.matmul(leadfield, src.data) for src in sources]
        
        # Reshape result
        # result = result.reshape(result.shape[0], n_samples, n_timepoints)
        # swap axes to correct order
        # result = np.swapaxes(result,0,1)
        # Rescale
        # result /= scaler
        return result


    
    def add_noise(self, x, snr, beta=0):
        """ Add noise of given SNR to signal x.
        Parameters:
        -----------
        x : numpy.ndarray, 3-dimensional numpy array of dims (trials, channels, timepoints)
        Return:
        -------
        """
    
        # This looks inconvenient but we need to make sure that there is no empty dimension for the powerlaw noise function.
        x_shape = (x.shape[0], x.shape[1], np.clip(x.shape[2], a_min=2, a_max=np.inf).astype(int))
        noise = cn.powerlaw_psd_gaussian(beta, x_shape)
        
        # In case we added another entry in the 2nd dimension we have to remove it here again.
        if x_shape[2] != x.shape[2]:
            noise=noise[:, :, :1]
    
        noise_gfp = np.std(noise, axis=1)
        rms_noise = np.median(noise_gfp)  # rms(noise)
        
        x_gfp = np.std(x, axis=1)
        rms_x = np.median(x_gfp)  # np.mean(np.max(np.abs(x_gfp), axis=1))  # x.max()
        if rms_x == 0:  
            # in case most of the signal is zero, e.g. when using biphasic pulses
            rms_x = abs(x_gfp).max()
        # rms_noise = rms(noise-np.mean(noise))
        noise_scaler = rms_x / (rms_noise*snr)
        # print(f'rms_x = {rms_x}\nrms_noise = {rms_noise}\n\tScaling by {noise_scaler} to yield snr of {snr}')
        out = x + noise*noise_scaler  

        return out

    def check_settings(self):
        ''' Check if settings are complete and insert missing 
            entries if there are any.
        '''
        if self.settings is None:
            self.settings = DEFAULT_SETTINGS
        
        _, _, self.pos, _ = util.unpack_fwd(self.fwd)
        self.distance_matrix = cdist(self.pos, self.pos)

        # Check for wrong keys:
        for key in self.settings.keys():
            if not key in DEFAULT_SETTINGS.keys():
                msg = f'key {key} is not part of allowed settings. See DEFAULT_SETTINGS for reference: {DEFAULT_SETTINGS}'
                raise AttributeError(msg)
        
        # Check for missing keys and replace them from the DEFAULT_SETTINGS
        for key in DEFAULT_SETTINGS.keys():
            # Check if setting exists and is not None
            if not (key in self.settings.keys() and self.settings[key] is not None):
                self.settings[key] = DEFAULT_SETTINGS[key]
        
        if self.settings['duration_of_trial'] == 0:
            self.temporal = False
        else:
            self.temporal = True
        
        self.neighbors = self.calculate_neighbors()
        

            

    def calculate_neighbors(self):
        adj = mne.spatial_src_adjacency(self.fwd["src"]).toarray().astype(int)
        neighbors = np.array([np.where(a)[0] for a in adj], dtype=object)
        return neighbors

               
    @staticmethod
    def get_pulse(pulse_len):
        ''' Returns a pulse of given length. A pulse is defined as 
        half a revolution of a sine.
        
        Parameters
        ----------
        x : int
            the number of data points

        '''
        pulse_len = int(pulse_len)
        freq = (1/pulse_len) / 2
        time = np.arange(pulse_len)

        signal = np.sin(2*np.pi*freq*time)
        return signal
    
    @staticmethod
    def get_biphasic_pulse(pulse_len, center_fraction=1, temporal_jitter=0.):
        ''' Returns a biphasic pulse of given length.
        
        Parameters
        ----------
        x : int
            the number of data points

        '''
        pulse_len = int(pulse_len)
        freq = (1/pulse_len) *center_fraction#/ 2
        time = np.linspace(-pulse_len/2, pulse_len/2, pulse_len)
        
        jitter = np.random.randn()*temporal_jitter
        signal = np.sin(2*np.pi*freq*time + jitter)
        crop_start = int(pulse_len/2 - pulse_len/center_fraction/2)
        crop_stop = int(pulse_len/2 + pulse_len/center_fraction/2)
        
        # signal[(time<-1) | (time>1)] = 0
        signal[:crop_start] = 0
        signal[crop_stop:] = 0
        signal *= np.random.choice([-1,1])
        return signal

    @staticmethod
    def get_from_range(val, dtype=int):
        ''' If list of two integers/floats is given this method outputs a value in between the two values.
        Otherwise, it returns the value.
        
        Parameters
        ----------
        val : list/tuple/int/float

        Return
        ------
        out : int/float

        '''
        # If input is a function -> call it and return the result
        if callable(val):
            return val()
   

        if dtype==int:
            rng = random.randrange
        elif dtype==float:
            rng = random.uniform
        else:
            msg = f'dtype must be int or float, got {type(dtype)} instead'
            raise AttributeError(msg)

        if isinstance(val, (list, tuple, np.ndarray)):
            if val[0] == val[1]:
                out = dtype(val[0])
            else:
                out = rng(*val)
        else:
            # If input is only a single value anyway, there is no range and it can
            # be returned in the desired dtype.
            out = dtype(val)
        return out
    
    def save(self, file_name):
        ''' Store the simulation object.
        Parameters
        ----------
        file_name : str
            Filename or full path to store the object to.

        Example
        -------
        sim = Simulation().simulate()
        sim.save('C/Users/User/Desktop/simulation.pkl')
        '''

        with open(file_name, 'wb') as f:
            pkl.dump(self, f)

    def to_nontemporal(self):
        ''' Converts the internal data representation from temporal to 
        non-temporal. 
        
        Specifically, this changes the shape of sources from a
        list of mne.sourceEstimate to a single mne.sourceEstimate in which the 
        time dimension holds a concatenation of timepoints and samples.

        The eeg data is reshaped from (samples, channels, time points) to 
        (samples*time points, channels, 1).

        Parameters
        ----------
        

        Return
        ------
        self : esinet.Simulation
            Method returns itself for convenience

        '''
        if not self.temporal:
            print('This Simulation() instance is already non-temporal')
            return self

        self.temporal = False
        self.settings['duration_of_trial'] = 0

        eeg_data_lstm = self.eeg_data.get_data()
        # Reshape EEG data
        eeg_data_single = np.expand_dims(np.vstack(np.swapaxes(eeg_data_lstm, 1,2)), axis=-1)
        # Pack into mne.EpochsArray object
        epochs_single = mne.EpochsArray(eeg_data_single, self.eeg_data.info, 
            tmin=self.eeg_data.tmin, verbose=0)
        # Store the newly shaped data
        self.eeg_data = epochs_single
        
        # Reshape Source data
        source_data = np.vstack(np.swapaxes(np.stack(
            [source.data for source in self.source_data], axis=0), 1,2)).T
        # Pack into mne.SourceEstimate object
        source_single = deepcopy(self.source_data[0])
        source_single.data = source_data
        self.source_data = source_single
        
        return self
        
    def shuffle(self):
        ''' Shuffle the simulated samples.'''
        sources = self.source_data
        epochs = self.eeg_data
        df = self.simulation_info
        n_samples = len(epochs)

        # Shuffle everything
        new_order = np.arange(n_samples).astype(int)
        np.random.shuffle(new_order)
        
        epochs = [epochs[i] for i in new_order]
        sources = [sources[i] for i in new_order]
        
        df = df.reindex(new_order)

        # store back
        self.eeg_data = epochs
        self.source_data = sources
        self.simulation_info = df
        
    def crop(self, tmin=None, tmax=None, include_tmax=False, verbose=0):
        eeg_data = []
        source_data = []
        if tmax is None:
            tmax = self.eeg_data[0].tmax
        for i in range(self.n_samples):
            # print(self.eeg_data[i].tmax, tmax)
            cropped_source = self.source_data[i].crop(tmin=tmin, tmax=tmax, include_tmax=include_tmax)
            cropped_eeg = self.eeg_data[i].crop(tmin=tmin, tmax=tmax, include_tmax=include_tmax, verbose=verbose)
            # min_crop = (1/cropped_source.sfreq)
            # while len(cropped_source.times) > len(cropped_eeg.times):

            #     # print(f"cropping: {len(cropped_source.times)}")
            #     tmax -= min_crop
            #     cropped_source = cropped_source.crop(tmin=tmin, tmax=tmax-min_crop)
            #     # print(f"cropped: {len(cropped_source.times)}")


            source_data.append( cropped_source )
            eeg_data.append( cropped_eeg )

        
        self.source_data = source_data
        self.eeg_data = eeg_data

        return self


    def select(self, samples):
        ''' Select subset of samples.
        Parameters
        ----------
        samples : int/list/tuple
            If type int select the given number of samples, if type list select indices given by list
        Return
        ------

        '''
        print("not implemented yet")
        return self
    
    def extents_to_orders(self, extents):
        ''' Convert extents (source diameter in mm) to neighborhood orders.
        '''
        if self.diams is None:
            self.get_diams_per_order()
        if isinstance(extents, (int, float)):
            order = np.argmin(abs(self.diams-extents))
        else:
            order = (np.argmin(abs(self.diams-extents[0])), np.argmin(abs(self.diams-extents[1])))

        return order
    
    def get_diams_per_order(self):
        ''' Calculate the estimated source diameter per neighborhood order.
        '''
        diams = []
        diam = 0
        order = 0
        while diam<100:
            diam = util.get_source_diam_from_order(order, self.fwd, dists=deepcopy(self.distance_matrix))
            diams.append( diam )
            order += 1
        self.diams = np.array(diams)
    
    
def get_n_order_indices(order, pick_idx, neighbors):
    ''' Iteratively performs region growing by selecting neighbors of 
    neighbors for <order> iterations.
    '''
    assert order == round(order), "Neighborhood order must be a whole number"
    order = int(order)
    if order == 0:
        return [pick_idx,]
    flatten = lambda t: [item for sublist in t for item in sublist]
    # print("y")
    current_indices = [pick_idx,]
    for cnt in range(order):
        # current_indices = list(np.array( current_indices ).flatten())
        new_indices = [neighbors[i] for i in current_indices]
        new_indices = flatten( new_indices )
        current_indices.extend(new_indices)
        
        current_indices = list(set(current_indices))
    return current_indices
