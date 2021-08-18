# esinet: Electric source imaging using artificial neural networks (ANNs)

**esinet** let's you solve the EEG inverse problem using ANNs. It currently supports two main architectures:
## Model 1

A fully connected neural network which is trained on single time instances of M/EEG data.

## Model 2

A temporal long-short-term memory (LSTM) model which is trained on sequences.

---

The package is based on our publication: [ConvDip: A Convolutional Neural Network for Better EEG Source Imaging](https://www.frontiersin.org/articles/10.3389/fnins.2021.569918/full)

![esinet](/assets/esinet.png)
Neural network design was created [here](http://alexlenail.me/NN-SVG/index.html)

<br/><br/>

## Dependencies:
* Python >= 3.8.3
* [mne 0.22.0](https://mne.tools/stable/index.html)
  * Follow the [installation guide](https://mne.tools/stable/install/mne_python.html#installing-mne-python-and-its-dependencies)
* [Tensorflow>=2.4.1](https://www.tensorflow.org/)
  * Follow the [installation guide](https://www.tensorflow.org/install)
* [Colorednoise](https://github.com/felixpatzelt/colorednoise)
* [joblib](https://joblib.readthedocs.io/en/latest/#)
* [pyvista>=0.24](https://docs.pyvista.org/)
* [pyvistaqt>=0.2.0](https://qtdocs.pyvista.org/)
* [tqdm](https://github.com/tqdm/tqdm)

<br/>

# Installation from PyPi
Use [pip](https://pip.pypa.io/en/stable/) to install esinet and all its
dependencies from [PyPi](https://pypi.org/):

```
pip install esinet
```

<br/>

# Quick start
The following code demonstrates how to use this package:

```
from esinet import Simulation, Net

# Simulate M/EEG data
settings = dict(duration_of_trial=0.2)
sim = Simulation(fwd, info, settings=settings)
sim.simulate(n_samples=10000)

# Train neural network on the simulated data
net = Net(fwd)
ann.fit(sim)

# Perform predictions on your data
stc = ann.predict(epochs)

```

# First steps

Check out one of the [tutorials](tutorials/) to learn how to use the package:

* [Tutorial 1](tutorials/tutorial_1.ipynb): The fastest way to get started with Model 1. This tutorial can be used as an entry point. If you want to dig deeper you should have a look at the next tutorials, too!
* [Tutorial 2](tutorials/tutorial_2.ipynb): Use esinet with low-level functions that allow for more control over your parameters with respect to simulations and training of the neural network.
* [Tutorial 3](tutorials/tutorial_3.ipynb): A demonstration of simulation parameters and how they affect the model performance.
* [Tutorial 4](tutorials/tutorial_4.ipynb): Learn to use the LSTM network (Model 2) to predict source time-series from EEG time-series.


# Feedback
Please leave your feedback and bug reports at lukas_hecker@web.de.

<br/>

# Literature
Please cite us with this publication:

@ARTICLE{10.3389/fnins.2021.569918,
AUTHOR={Hecker, Lukas and Rupprecht, Rebekka and Tebartz Van Elst, Ludger and Kornmeier, Jürgen},   
TITLE={ConvDip: A Convolutional Neural Network for Better EEG Source Imaging},      
JOURNAL={Frontiers in Neuroscience},      
VOLUME={15},      
PAGES={533},     
YEAR={2021},      
URL={https://www.frontiersin.org/article/10.3389/fnins.2021.569918},       
DOI={10.3389/fnins.2021.569918},      
ISSN={1662-453X}
}

# Troubleshooting
* Having problems with the installation? Check the [package requirements](requirements.txt)

