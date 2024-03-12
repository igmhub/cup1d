# cup1d

## Cosmology using P1D - small scale clustering of the Lyman alpha forest

This repository contains some tools to perform the last steps of a cosmological analysis of the 1D power spectrum (P1D) of the Lyman alpha forest. 

It uses the LaCE emulator (https://github.com/igmhub/LaCE), and some extra tools to run MCMC analyses on cosmological and IGM parameters for a mock P1D measurement.

If you would like to collaborate, please email Andreu Font-Ribera (afont@ifae.es) or Jonas Chaves-Montero (jchaves@ifae.es).
 

### Installation

- Create a new conda environment. It is usually better to follow python version one or two behind. In January 2024, the latest is 3.12, so we recommend 3.11.

```
conda create -n cup1d python=3.11 camb mpich mpi4py
conda activate cup1d
```
- Install cup1d:

```Follow the instructions from https://github.com/igmhub/cup1d```

- Clone the cup1d repo and perform an *editable* installation:

```
git clone https://github.com/igmhub/cup1d.git
cd cup1d
pip install -e .[jupyter]
``` 

### Notebooks / tutorials

You can start by plotting the many P1D measurements stored in the repo, by looking at `notebooks/p1d_measurements`

You can also redo old neutrino mass constraints by importance sampling WMAP and Planck chains, following `notebooks/planck`

You can also play with the LaCE emulator with the notebooks in `notebooks/emulator`

Finally, you can run your own analysis on mock data following the notebooks in `notebooks/likelihood`


### Forecasting script

You can run the script under scripts/forecast.py to forecast the constraints on linear power parameters for a given P1D covariance.

It marginalizes over 8 nuisance parameters, so it might take a while to run!
