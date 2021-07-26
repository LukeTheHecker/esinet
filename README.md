# esinet: Electric source imaging using artificial neural networks (ANNs)

esinet let's you solve the EEG inverse problem using ANNs. 

It is based on our publication: [ConvDip: A Convolutional Neural Network for Better EEG Source Imaging](https://www.frontiersin.org/articles/10.3389/fnins.2021.569918/full)

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

# First steps

Check out one of the [tutorials](tutorials/) to learn how to use the package:

* [Base tutorial](tutorials/tutorial.ipynb): Simulate data and train a ANN to predict some sources. Shows the most important functions of this package and serves as the easiest entry point.
  
* [Brainstorm Auditory example](tutorials/brainstorm_auditory_example.ipynb): This tutorial shows you how to use esinet to predict the sources in word processing data. Code was partially used from the [MNE tutorials](https://mne.tools/stable/auto_tutorials/sample-datasets/plot_brainstorm_auditory.html?highlight=brainstorm)
 
<br/>

# Feedback
Leave your feedback and bug reports at lukas_hecker@web.de.

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

