from copy import deepcopy
import numpy as np
import pickle as pkl
import random
from joblib import Parallel, delayed
from tqdm.notebook import tqdm
import colorednoise as cn
import mne
from time import time
from . import util

DEFAULT_SETTINGS = {
            'number_of_sources': (1, 20),
            'extents': (1, 50),
            'amplitudes': (1, 10),
            'shapes': 'both',
            'duration_of_trial': 0,
            'sample_frequency': 100,
            'target_snr': 4,
            'beta': (0, 3),
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
            (i.e. uniform) or 'both'.
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
        settings['sample_frequency'] = info['sfreq']
        self.settings = settings
        self.check_settings()

        self.source_data = None
        self.eeg_data = None
        self.fwd = deepcopy(fwd)
        self.fwd.pick_channels(info['ch_names'])
        self.check_info(deepcopy(info))
        self.info['sfreq'] = self.settings['sample_frequency']
        self.subject = self.fwd['src'][0]['subject_his_id']
        self.n_jobs = n_jobs
        self.parallel = parallel
        self.verbose = verbose
        _, _, self.pos, _ = util.unpack_fwd(self.fwd)
    
    
    def check_info(self, info):
        self.info = info.pick_channels(self.fwd.ch_names, ordered=True)


    def simulate(self, n_samples=10000):
        ''' Simulate sources and EEG data'''
        self.source_data = self.simulate_sources(n_samples)
        self.eeg_data = self.simulate_eeg()

        return self

    def plot(self):
        pass
    
    def simulate_sources(self, n_samples):
        if self.verbose:
                print(f'Simulate Source')
        if self.parallel:
            source_data = np.stack(Parallel(n_jobs=self.n_jobs, backend='loky')
                (delayed(self.simulate_source)() 
                for _ in range(n_samples)))
        else:
            source_data = np.stack([self.simulate_source() 
                for _ in tqdm(range(n_samples))], axis=0)
        
        # Convert to mne.SourceEstimate
        if self.verbose:
            print(f'Converting Source Data to mne.SourceEstimate object')
        if self.settings['duration_of_trial'] == 0:
            sources = util.source_to_sourceEstimate(source_data, self.fwd, 
                sfreq=self.settings['sample_frequency'], subject=self.subject) 
        else:
            sources = self.sources_to_sourceEstimates(source_data)

        return sources

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
            (i.e. uniform) or 'both'.
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

        # Get number of sources is a range:
        number_of_sources = self.get_from_range(
            self.settings['number_of_sources'], dtype=int)

        # Get amplitudes for each source
        extents = [self.get_from_range(self.settings['extents'], dtype=float) 
            for _ in range(number_of_sources)]
        
        # Decide shape of sources
        if self.settings['shapes'] == 'both':
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

        if self.settings['duration_of_trial'] > 0:
            signal_length = int(self.settings['sample_frequency']*self.settings['duration_of_trial'])
            pulselen = self.settings['sample_frequency']/10
            # pulse = self.get_pulse(pulselen)
            
            signals = []
            for _ in range(number_of_sources):
                signal = cn.powerlaw_psd_gaussian(self.get_from_range(self.settings['beta'], dtype=float), signal_length) 
                # Old: have positive source values
                # signal += np.abs(np.min(signal))
                # signal /= np.max(signal)
                # New:
                signal /= np.max(np.abs(signal))

                signals.append(signal)
            
            sample_frequency = self.settings['sample_frequency']
        else:  # else its a single instance
            sample_frequency = 0
            signal_length = 1
            signals = [np.array([1])]*number_of_sources
        
        
        # sourceMask = np.zeros((self.pos.shape[0]))
        source = np.zeros((self.pos.shape[0], signal_length))
        
        ##############################################
        # Loop through source centers (i.e. seeds of source positions)
        for i, (src_center, shape, amplitude, signal) in enumerate(zip(src_centers, shapes, amplitudes, signals)):
            dists = np.sqrt(np.sum((self.pos - self.pos[src_center, :])**2, axis=1))
            d = np.where(dists<extents[i]/2)[0]

            if shape == 'gaussian':
                sd = np.clip(np.max(dists[d]) / 2, a_min=0.1, a_max=np.inf)  # <- works better
                activity = np.expand_dims(util.gaussian(dists, 0, sd) * amplitude, axis=1) * signal
                source += activity
            elif shape == 'flat':
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
            # sourceMask[d] = 1

        # if durOfTrial > 0:
        # n = np.clip(int(sample_frequency * self.settings['duration_of_trial']), a_min=1, a_max=None)
        # sourceOverTime = util.repeat_newcol(source, n)
        # source = np.squeeze(sourceOverTime * signal)
        # if len(source.shape) == 1:
        #     source = np.expand_dims(source, axis=1)
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
        if isinstance(self.source_data, (list, tuple)):
            sources = np.stack([source.data for source in self.source_data], axis=0)
        else:
            sources = self.source_data.data.T

        # if there is no temporal dimension...
        if len(sources.shape) < 3:
            # ...add empty temporal dimension
            sources = np.expand_dims(sources, axis=2)

        # Load some forward model objects
        fwd_fixed, leadfield = util.unpack_fwd(self.fwd)[:2]
        n_samples, _, _ = sources.shape
        n_elec = leadfield.shape[0]

        
        # Desired Dim for eeg_clean: (samples, electrodes, time points)
        if self.verbose:
            print(f'\nProject sources to EEG...')
        eeg_clean = self.project_sources(sources)

        if self.verbose:
            print(f'\nCreate EEG trials with noise...')
        if self.parallel:
            eeg_trials_noisy = np.stack(Parallel(n_jobs=self.n_jobs, backend='loky')
                (delayed(self.create_eeg_helper)(eeg_clean[sample], n_simulation_trials,
                self.settings['target_snr'], self.settings['beta']) 
                for sample in tqdm(range(n_samples))), axis=0)
        else:
            eeg_trials_noisy = np.stack(
                [self.create_eeg_helper(eeg_clean[sample], n_simulation_trials, 
                self.settings['target_snr'], self.settings['beta']) 
                for sample in tqdm(range(n_samples))], 
                axis=0)
            
        if n_simulation_trials == 1 and len(eeg_trials_noisy.shape) == 2:
            # Add empty dimension to contain the single trial
            eeg_trials_noisy = np.expand_dims(eeg_trials_noisy, axis=1)

        
        if len(eeg_trials_noisy.shape) == 3:
            eeg_trials_noisy = np.expand_dims(eeg_trials_noisy, axis=-1)
            
        if eeg_trials_noisy.shape[2] != n_elec:
            eeg_trials_noisy = np.swapaxes(eeg_trials_noisy, 1, 2)
        
        if self.verbose:
            print(f'\nConvert EEG matrices to a single instance of mne.Epochs...')
        ERP_samples_noisy = np.mean(eeg_trials_noisy, axis=1)
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
        target_snr : float/list/tuple
            The target signal-to-noise ratio, is converted to 
            single-trial SNR based on number of trials
        beta : float/list/tuple
            The beta exponent of the 1/f**beta noise

        '''
        target_snr = self.get_from_range(target_snr, dtype=float)
        beta = self.get_from_range(beta, dtype=float)
        
        assert len(eeg_sample.shape) == 2, 'Length of eeg_sample must be 2 (time_points, electrodes)'
        
        eeg_sample = np.repeat(np.expand_dims(eeg_sample, 0), n_simulation_trials, axis=0)
        snr = target_snr / np.sqrt(n_simulation_trials)
        
        # Before: Add noise based on the GFP of all channels
        # noise_trial = self.add_noise(eeg_sample, snr, beta=beta)
        
        # NEW: ADD noise for different types of channels, separately
        # since they can have entirely different scales.
        coil_types = [ch['coil_type'] for ch in self.info['chs']]
        coil_types_set = list(set(coil_types))
        coil_types_set = np.array([int(i) for i in coil_types_set])
        
        coil_type_assignments = np.array(
            [np.where(coil_types_set==coil_type)[0][0] 
                for coil_type in coil_types]
        )
        noise_trial = np.zeros(
            (eeg_sample.shape[0], eeg_sample.shape[1], eeg_sample.shape[2])
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
        n_samples, n_dipoles, n_timepoints = sources.shape
        n_elec = leadfield.shape[0]
        eeg = np.zeros((n_samples, n_elec, n_timepoints))

        # Swap axes to dipoles, samples, time_points
        sources_tmp = np.swapaxes(sources, 0,1)
        # Collapse last two dims into one
        short_shape = (sources_tmp.shape[0], 
            sources_tmp.shape[1]*sources_tmp.shape[2])
        sources_tmp = sources_tmp.reshape(short_shape)
        # Perform Matmul
        result = np.matmul(leadfield, sources_tmp)
        # Reshape result
        result = result.reshape(result.shape[0], n_samples, n_timepoints)
        # swap axes to correct order
        result = np.swapaxes(result,0,1)
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
        rms_noise = np.mean(noise_gfp)  # rms(noise)
        
        x_gfp = np.std(x, axis=1)
        rms_x = np.mean(np.max(np.abs(x_gfp), axis=1))  # x.max()
        
        # rms_noise = rms(noise-np.mean(noise))
        noise_scaler = rms_x / (rms_noise*snr)
        out = x + noise*noise_scaler  

        return out

    def check_settings(self):
        ''' Check if settings are complete and insert missing 
            entries if there are any.
        '''
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
        if dtype==int:
            rng = random.randrange
        elif dtype==float:
            rng = random.uniform
        else:
            msg = f'dtype must be int or float, got {type(dtype)} instead'
            raise AttributeError(msg)

        if isinstance(val, (list, tuple, np.ndarray)):
            out = rng(*val)
        elif isinstance(val, (int, float)):
            out = val
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
        
        