# Changelog

## 0.2.2 02.05.2022
* Introducing region growing for source simulations. Region growing allows for
  more realistic source simulations that do not extend spherically but along a
  graph of connected surface dipoles. See esinet.Simulation.settings for
  details.

## 0.2.1 ??.04.2022
* Minor changes which were forgotten.


## 0.2.0 20.04.2022
* Support for LSTM, Fully-Connected and ConvDip models
* Most recent code basis which served for the LSTM bioRxiv preprint

## 0.1.0 19.08.2021
* Changed API from functional to object-oriented
* New Tutorials
* Models can now train on either single time instances of EEG data or on temporal data (LSTM Model)
* Optimized simulation through more intelligent matrix multiplications
* Im am sorry if this update broke your code. If you experience issues please either follow the tutorials on the new API or revert back to an older version.

## 0.0.12 15.06.2021
* Made tutorial notebook clearer and minor changes

## 0.0.11 08.06.2021
* Changed dependencies to be more rigid

## 0.0.10 08.06.2021
* Changed progress bar to juypter mode
* added option to have mixed gaussian and flat source shapes for higher simulation diversity

## 0.0.9 11.02.2021
* minor changes

## 0.0.8 - 10.02.2021
* Minor changes with indentation in forward.py
* Removed unused imports from tutorial notebooks
* removed a unused cell in tutorial.ipynb
* some minor changes

## 0.0.7 - 28.01.2021
* added required package: ['tensorflow>=2.4.1','mne>=0.22.0', 'scipy',
  'colorednoise', 'joblib', 'jupyter', 'ipykernel', 'matplotlib',
  'pyvista>=0.24', 'pyvistaqt>=0.2.0', 'vtk>=9.0.1', 'tqdm']
* Removed mayavi imports in tutorials

## 0.0.6 - 26.01.2021

* changed source plot functions in tutorial
* removed seaborn and mayavi as required package
* added tutorials to the repository
* changes to README
  

## 0.0.5 - 26.01.2021
### Installation fixes

* add packages to the install requirements: colorednoise, joblib, seaborn, mayavi
* changed required python version to >=3.8.3

