# Changelog

## 0.1.2 09.12.2021
* Fixed bug that occured for temporal simulations using the "duration_of_trial" key in the settings dictionary required for the simulation.Simulation object. Thanks to Winnie for the bug report! :-)

## 0.1.1 27.08.2021
* Reverted some changes that turned out not to be functional yet

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

