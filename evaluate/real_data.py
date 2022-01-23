
import sys; sys.path.insert(0, '../')
import numpy as np
from copy import deepcopy

import mne
from mne.datasets.brainstorm import bst_raw
from mne.io import read_raw_ctf
import tensorflow as tf

from esinet import forward
from esinet import util
from esinet import net
from esinet import simulation

plot_params = dict(surface='white', hemi='both', verbose=0)


tmin, tmax, event_id = -0.1, 0.3, 2  # take right-hand somato
reject = dict(mag=4e-12, eog=250e-6)

data_path = bst_raw.data_path()

raw_path = (data_path + '/MEG/bst_raw/' +
            'subj001_somatosensory_20111109_01_AUX-f.ds')
# Here we crop to half the length to save memory
raw = read_raw_ctf(raw_path).crop(0, 180).load_data()

# set EOG channel
raw.set_channel_types({'EEG058': 'eog'})
raw.set_eeg_reference('average', projection=True)

# show power line interference and remove it
raw.notch_filter(np.arange(60, 181, 60), fir_design='firwin')

events = mne.find_events(raw, stim_channel='UPPT001')

# pick MEG channels
picks = mne.pick_types(raw.info, meg=True, eeg=False, stim=False, eog=True,
                       exclude='bads')

# Compute epochs
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), reject=reject, preload=False)

# compute evoked
evoked = epochs.average()

# remove physiological artifacts (eyeblinks, heartbeats) using SSP on baseline
evoked.add_proj(mne.compute_proj_evoked(evoked.copy().crop(tmax=0)))
evoked.apply_proj()

# fix stim artifact
mne.preprocessing.fix_stim_artifact(evoked)

# correct delays due to hardware (stim artifact is at 4 ms)
evoked.shift_time(-0.004)


info = deepcopy(evoked.info)
info["sfreq"] = 100  # change sfreq for lower computational expense
fwd = forward.create_forward_model(info=info)
fwd_free = forward.create_forward_model(info=info, fixed_ori=False)
pos = util.unpack_fwd(fwd)[2]

sim_short = simulation.Simulation(fwd, info, settings=dict(duration_of_trial=(0.01,2))).simulate(n_samples=10000)
sim_long = simulation.Simulation(fwd, info, settings=dict(duration_of_trial=(2,10))).simulate(n_samples=200)
sim = sim_short + sim_long
sim.shuffle()

train_params = dict(epochs=150, patience=2, loss=tf.keras.losses.CosineSimilarity(), 
    optimizer=tf.keras.optimizers.Adam() , return_history=True, 
    metrics=[tf.keras.losses.mean_squared_error], batch_size=8,
    validation_freq=2, validation_split=0.05,
    device='GPU:0')

model_params = {
    "LSTM Medium": dict(n_lstm_layers=2, n_lstm_units=85, n_dense_layers=0),
    "ConvDip Medium": dict(n_lstm_layers=2, n_dense_layers=3, n_dense_units=250, model_type='convdip')
}
for model_name, model_param in model_params.items():
    model = net.Net(fwd, **model_param).fit(sim, **train_params)[0]
    model.model.compile(optimizer='adam', loss='mean_squared_error')
    model.save(r'models', name=f'{model_name}_median_nerve_stimulation')
    # model.save(r'C:\Users\Lukas\Nextcloud\esinet_models', name=f'{model_name}_median_nerve_stimulation')
    del model