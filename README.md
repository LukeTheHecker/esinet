# ESINet: Electric source imaging unsing artificial neural networks (ANNs)

ESINet let's you solve the EEG inverse problem using ANNs.

![Alt-Text](/assets/ESINet.png)

Installation from PyPi:

```
pip install ESINet
```
**Dependencies:**
* [mne 0.20.5]([mne.tools](https://mne.tools/stable/index.html))
* [Tensorflow 2.4.0](https://www.tensorflow.org/)

## Workflow
ESINet is a lightweight package that provides all building blocks to use an ANN
to solve the EEG inverse problem. It relies on
[mne]([mne.tools](https://mne.tools/stable/index.html)) to handle all tasks
related to EEG and [Tensorflow](https://www.tensorflow.org/) to create, train
and predict with the ANN.

## The forward model
Knowing how cerebral currents will project to the scalp electrodes requires solving the *forward problem*. Fortunately, this problem has a unique solution! We provide a function to quickly create a *forward model* which supplies all assets required for the following processing steps:

```
pth_fwd = 'forward_models/ico3/'
sampling = 'ico3'
create_forward_model(pth_fwd, sampling=sampling, info=epochs.info)
```
The sampling defines the number of dipoles in your source model and thereby the resolution of your inverse solution. If you don't have powerful hardware we encourage to leave the sampling at 'ico3'.

## Simulating EEG data
ANNs that find solutions to the inverse problem need to be trained to infer the dipole moments of your source model given the EEG.

You can create a set of sources using our high-level function:
```
sources_sim = run_simulations(pth_fwd, durOfTrial=0)
```

To simulate the corresponding EEG data with added noise you can use our function:
```
eeg_sim = create_eeg(sources_sim, pth_fwd)
```

You can visualize the simulated data with this code block:
```
%matplotlib qt
sample = 0  # index of the simulation
title = f'Simulation {sample}'
# ERP Plot
eeg_sim[sample].average().plot()
# Topographic plot
eeg_sim[sample].average().plot_topomap([0.5])
# Source plot
a = [sources_sim[sample].plot(hemi=hemi, initial_time=0.5, surface='white', colormap='inferno', figure=mlab.figure(title)) for hemi in ['lh', 'rh']]
```

## The EEG inverse problem
Now we want to train an ANN to infer sources given EEG data. Since we have simulated all the required data already we just have to load and train a ANN model.

To load our basic model:

```
# Find out input and output dimensions based on the shape of the leadfield 
input_dim, output_dim = load_leadfield(pth_fwd).shape
# Initialize the artificial neural network model
model = get_model(input_dim, output_dim)
```
Next, we train the model:
```
model, history = train_model(model, sources_sim, eeg_sim)
```

You have now trained your neural network - congratulations!

## Testing the ANN
Let's see how your ANN performs!
First, we have to simulate some data for evaluation:
```
# Simulate source
sources_eval = run_simulations(pth_fwd, 1, durOfTrial=0)
# Simulate corresponding EEG
eeg_eval = create_eeg(sources_eval, pth_fwd)
# Calculate average of the simulated trials (i.e. the event-related potential (ERP))
eeg_sample = np.squeeze( np.mean(eeg_eval, axis=1) )
```
Next, we use the trained ANN model to predict the source given the EEG.

```
# Predict
source_predicted = predict(model, eeg_sample, pth_fwd)
```

Now let's visualize the result:
```
# Plot ground truth source...
title = f'Ground Truth'
a = [sources_eval[0].plot(hemi=hemi, initial_time=0.5, surface='white', colormap='inferno', figure=mlab.figure(title)) for hemi in ['lh', 'rh']]

# Plot the simulated EEG
title = f'Simulated EEG'
eeg_eval[0].average().plot_topomap([0], title=title)

# Plot the predicted source
title = f'ConvDip Prediction'
b = [source_predicted.plot(hemi=hemi, initial_time=0.5, surface='white', colormap='inferno', figure=mlab.figure(title)) for hemi in ['lh', 'rh']]

```

## Feedback
Leave your feedback and bug reports at lukas_hecker@web.de.

## Literature
Cite us using our preprint (publication is in review):

Hecker, L., Rupprecht, R., van Elst, L. T., & Kornmeier, J. (2020). ConvDip: A convolutional neural network for better M/EEG Source Imaging. bioRxiv.