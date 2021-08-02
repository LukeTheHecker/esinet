from copy import deepcopy
import numpy as np
import mne
import random
from joblib import Parallel, delayed
from numpy.lib.utils import source
from scipy.sparse import data
from tqdm.notebook import tqdm
import colorednoise as cn

from .. import util


DEFAULT_SETTINGS = {
            'number_of_sources': (1, 5),
            'extents': (25, 35),
            'amplitudes': (1, 10),
            'shapes': 'both',
            'duration_of_trial': 0,
            'sample_frequency': 100,
            'target_snr': 4,
            'beta': 1.0,
        }

class Simulation:
    ''' Simulate and hold source and M/EEG data.
    
    Attributes
    ----------
    settings : dict
        The Settings for the simulation. Keys:

        n_sources : int/tuple/list
            number of sources. Can be a single number or a 
            list of two numbers specifying a range.
        extents : int/float/tuple/list
            size of sources in mm. 
            Can be a single number or a list of two numbers 
            specifying a range.
        amplitudes : int/float/tuple/list
            the current of the source in nAm
        shape : str
            How the amplitudes evolve over space. Can be 
            'gaussian' or 'flat' (i.e. uniform) or 'both'.
        durOfTrial : int/float
            specifies the duration of a trial.
        sample_frequency : int
            specifies the sample frequency of the data.
    fwd : mne.Forward
        The mne-python Forward object that contains the 
        forward model
    source_data : mne.sourceEstimate
        A source estimate object from mne-python which 
        contains the source data.
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
        if self.parallel:
            if self.verbose:
                print(f'Simulate Source')
            source_data = np.stack(Parallel(n_jobs=self.n_jobs, backend='loky')(
                delayed(self.simulate_source)() 
                for i in tqdm(range(n_samples))))
        else:
            source_data = np.stack([self.simulate_source() 
                for _ in tqdm(range(n_samples))], axis=0)
        
        # Convert to mne.SourceEstimate
        if self.settings['duration_of_trial'] == 0:
            if self.verbose:
                print(f'Converting Source Data to mne.SourceEstimate object')
            sources = util.source_to_sourceEstimate(source_data, self.fwd, 
                sfreq=self.settings['sample_frequency'], subject=self.subject) 
        else:
            if self.parallel:
                sources = Parallel(n_jobs=self.n_jobs, backend='loky')(
                    delayed(util.source_to_sourceEstimate)
                    (source, self.fwd, sfreq=self.settings['sample_frequency'], subject=self.subject) 
                    for source in tqdm(source_data))
            else:
                sources = [util.source_to_sourceEstimate(source, 
                    self.fwd, sfreq=self.settings['sample_frequency'], subject=self.subject) 
                    for source in tqdm(source_data)]

        return sources
    
    def simulate_source(self):
        ''' Returns a vector containing the dipole currents. Requires only a dipole 
        position list and the simulation settings.

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
        source : numpy.ndarray, (n_dipoles x n_timepoints), the simulated source signal
        simSettings : dict, specifications about the source.

        Grova, C., Daunizeau, J., Lina, J. M., Bénar, C. G., Benali, H., & Gotman, J. (2006). Evaluation of EEG localization methods using realistic simulations of interictal spikes. Neuroimage, 29(3), 734-753.
        '''
        
        # Handle input

        # Amplitudes come in nAm
        if isinstance(self.settings['amplitudes'], (list, tuple)):
            amplitudes = [amp* 1e-9  for amp in self.settings['amplitudes']] 
        else:
            amplitudes = self.settings['amplitudes'] * 1e-9

        if self.settings['duration_of_trial'] > 0:
            if self.settings['duration_of_trial'] < 0.5 :
                print(f'duration_of_trials should be either 0 or at least 0.5 seconds!')
                return
            
            signal_length = int(self.settings['sample_frequency']*self.settings['duration_of_trial'])
            pulselen = self.settings['sample_frequency']/10
            pulse = self.get_pulse(pulselen)
            signal = np.zeros((signal_length))
            start = int(np.floor((signal_length - pulselen) / 2))
            end = int(np.ceil((signal_length - pulselen) / 2))
            signal[start:-end] = pulse
            signal /= np.max(signal)
            sample_frequency = self.settings['sample_frequency']
        else:  # else its a single instance
            sample_frequency = 0
            signal = 1
        
        ###########################################
        # Select ranges and prepare some variables:
        sourceMask = np.zeros((self.pos.shape[0]))
        # If number_of_sources is a range:
        if isinstance(self.settings['number_of_sources'], (tuple, list)):
            number_of_sources = random.randrange(*self.settings['number_of_sources'])
        else:
            number_of_sources = self.settings['number_of_sources']

        if self.settings['shapes'] == 'both':
            shapes = ['gaussian', 'flat']*number_of_sources
            np.random.shuffle(shapes)
            shapes = shapes[:number_of_sources]
            if type(shapes) == str:
                shapes = [shapes]

        elif self.settings['shapes'] == 'gaussian' or self.settings['shapes'] == 'flat':
            shapes = [self.settings.shapes] * number_of_sources

        if isinstance(self.settings['extents'], (tuple, list)):
            extents = [random.randrange(*self.settings['extents']) for _ in range(number_of_sources)]
        else:
            extents = [self.settings.extents] * number_of_sources

        if isinstance(self.settings['amplitudes'], (tuple, list)):
            amplitudes = [random.uniform(*self.settings['amplitudes']) for _ in range(number_of_sources)]
        else:
            amplitudes = [self.settings['amplitudes']] * number_of_sources
        
        src_centers = np.random.choice(np.arange(self.pos.shape[0]), \
            number_of_sources, replace=False)

        
        source = np.zeros((self.pos.shape[0]))
        
        ##############################################
        # Loop through source centers (i.e. seeds of source positions)
        for i, (src_center, shape) in enumerate(zip(src_centers, shapes)):
            dists = np.sqrt(np.sum((self.pos - self.pos[src_center, :])**2, axis=1))
            d = np.where(dists<extents[i]/2)[0]

            if shape == 'gaussian':
                sd = np.clip(np.max(dists[d]) / 2, a_min=0.1, a_max=np.inf)  # <- works better
                source[:] += util.gaussian(dists, 0, sd) * amplitudes[i]
            elif shape == 'flat':
                source[d] += amplitudes[i]
            else:
                msg = BaseException("shape must be of type >string< and be either >gaussian< or >flat<.")
                raise(msg)
            sourceMask[d] = 1

        # if durOfTrial > 0:
        n = np.clip(int(sample_frequency * self.settings['duration_of_trial']), a_min=1, a_max=None)
        sourceOverTime = util.repeat_newcol(source, n)
        source = np.squeeze(sourceOverTime * signal)
        if len(source.shape) == 1:
            source = np.expand_dims(source, axis=1)
        
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
        n_samples, n_dipoles, n_timepoints = sources.shape
        n_elec = leadfield.shape[0]

        # Desired Dim for eeg_clean: (samples, electrodes, time points)
        eeg_clean = np.zeros((n_samples, n_elec, n_timepoints))
        for i, sample in enumerate(sources):
            for j, time_point in enumerate(sample.T):
                eeg_clean[i, :, j] = np.matmul(leadfield, time_point)

        # eeg_trials_noisy = np.zeros((n_samples, n_simulation_trials, n_elec, n_timepoints))

        print(f'\nCreate EEG trials with noise...')
        if self.parallel:
            eeg_trials_noisy = np.stack(Parallel(n_jobs=self.n_jobs, backend='loky')
                (delayed(self.create_eeg_helper)(eeg_clean[sample], n_simulation_trials,
                self.settings['target_snr'], self.settings['beta']) for sample in tqdm(range(n_samples))), axis=0)
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
        if isinstance(target_snr, (tuple, list)):
            target_snr = random.uniform(*target_snr)
        if isinstance(beta, (tuple, list)):
            beta = random.uniform(*beta)
        
        assert len(eeg_sample.shape) == 2, 'Length of eeg_sample must be 2 (time_points, electrodes)'
        
        eeg_sample = np.repeat(np.expand_dims(eeg_sample, 0), n_simulation_trials, axis=0)
        snr = target_snr / np.sqrt(n_simulation_trials)
        noise_trial = self.add_noise(eeg_sample, snr, beta=beta)

        return noise_trial
    
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

        for key in DEFAULT_SETTINGS.keys():
            # Check if setting exists and is not None
            if not (key in self.settings.keys() and self.settings[key] is not None):
                self.settings[key] = DEFAULT_SETTINGS[key]
        
               
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