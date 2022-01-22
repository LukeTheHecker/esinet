

import sys; sys.path.insert(0, '../')
import pickle as pkl
import numpy as np
import pandas as pd
from copy import deepcopy
import mne
import seaborn as sns
import matplotlib.pyplot as plt
from esinet import util
from esinet import Simulation
from esinet import Net
from esinet import forward
from esinet import evaluate

info = forward.get_info()
info['sfreq'] = 100
fwd = forward.create_forward_model(info=info)
fwd_free = forward.create_forward_model(info=info, fixed_ori=False)

lstm_standard = util.load_net('models/LSTM Medium_1-1000points_standard-cosine_0')
dense_standard = util.load_net('models/Dense Medium_1-1000points_standard-cosine_0')
convdip_standard = util.load_net('models/ConvDip Medium_1-1000points_standard-cosine_0')

models = [lstm_standard, dense_standard, convdip_standard]
model_names = ['LSTM', 'Fully-Connected', 'ConvDip']

import seaborn as sns
sns.reset_orig()


model_names_tmp = deepcopy(model_names)
colormap = 'RdBu_r'

number_of_sources = 6
settings_eval = dict(
    method='standard', 
    number_of_sources=number_of_sources,
    duration_of_trial=2.0,
    extents=(25, 40),
    amplitudes=(7,10))
    


# Simulate new data
sim = Simulation(fwd, info, settings=settings_eval).simulate(2)
idx = 0

plot_params = dict(surface='white', hemi='split', size=(800*2,400*2), verbose=0, time_viewer=False, 
    background='w', colorbar=False, views=['lat', 'med'], 
    colormap=colormap, initial_time=0.0, transparent=True
)
fractions = [0., 0.05, 0.99]

snr = None
# print(sim.simulation_info)
# Predict sources using the esinet models
predictions = [model.predict(sim) for model in models]

# Predict sources with classical methods
# eLORETA
prediction_elor_data = util.wrap_mne_inverse(fwd, sim, method='eLORETA', 
    add_baseline=True, n_baseline=400)[idx].data.astype(np.float32)
prediction_elor = deepcopy(predictions[0][0])
prediction_elor.data = prediction_elor_data / np.abs(np.max(prediction_elor_data))
# MNE
prediction_mne_data = util.wrap_mne_inverse(fwd, sim, method='MNE', 
    add_baseline=True, n_baseline=400)[idx].data.astype(np.float32)
prediction_mne = deepcopy(predictions[0][0])
prediction_mne.data = prediction_mne_data / np.abs(np.max(prediction_mne_data))
# Beamformer
prediction_lcmv_data = util.wrap_mne_inverse(fwd, sim, method='lcmv', 
    add_baseline=True, n_baseline=400, parallel=False)[idx].data.astype(np.float32)
prediction_lcmv = deepcopy(predictions[0][0])
prediction_lcmv.data = prediction_lcmv_data / np.max(np.abs(prediction_lcmv_data))

# Get predictions and names in order
predictions.append([prediction_elor])
predictions.append([prediction_mne])
predictions.append([prediction_lcmv])

model_names_tmp.append('eLORETA')
model_names_tmp.append('MNE')
model_names_tmp.append('LCMV')

print(np.max(np.abs(sim.source_data[idx].data[:, 0])))
# Plot True Source
pos_lims = [np.max(np.abs(sim.source_data[idx].data[:, 0]))*frac for frac in fractions]
brain = sim.source_data[idx].plot(**plot_params, clim=dict(kind='value', pos_lims=pos_lims))
brain.add_text(0.1, 0.9, f'Ground Truth', 'title')
screenshot = brain.screenshot()
brain.close()
plt.figure()
plt.imshow(screenshot)
plt.axis('off')
util.multipage(f'C:/Users/lukas/Sync/lstm_inverse_problem/figures/results/brains/brain_ground_truth_{number_of_sources}_sources.pdf', png=True)
plt.close()

model_selection = model_names_tmp
# Plot predicted sources
pos = util.unpack_fwd(fwd)[2]
for model_name, prediction in zip(model_names_tmp, predictions):
    pos_lims = [np.max(np.abs(prediction[idx].data[:, 0]))*frac for frac in fractions]
    brain = prediction[idx].plot(**plot_params, clim=dict(kind='value', pos_lims=pos_lims))    
    title = f'{model_name}'
    print(title)
    auc = evaluate.eval_auc(sim.source_data[idx].data[:, 0], prediction[idx].data[:,0], pos, n_redraw=25, epsilon=0.25,)
    print(f'AUC: {auc}')
    brain.add_text(0.1, 0.9, title, 'title')
    
    screenshot = brain.screenshot()
    brain.close()
    plt.figure()
    plt.imshow(screenshot)
    plt.axis('off')
    util.multipage(f'C:/Users/lukas/Sync/lstm_inverse_problem/figures/results/brains/brain_{title}_{number_of_sources}_sources.pdf', png=True)
    plt.close()
# util.multipage(f'C:/Users/lukas/Sync/lstm_inverse_problem/figures/results/brain_rest.pdf', png=True)
    
    