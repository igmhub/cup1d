# cup1d

## Cosmology using P1D - small scale clustering of the Lyman alpha forest

This repository contains some tools to perform the last steps of a cosmological analysis of the 1D power spectrum (P1D) of the Lyman alpha forest. 

It uses the LaCE emulator (https://github.com/igmhub/LaCE), and some extra tools to run MCMC analyses on cosmological and IGM parameters for a mock P1D measurement.

If you would like to collaborate, please email Andreu Font-Ribera (afont@ifae.es)
 

### Installation

Start a fresh environment:

```
conda create -n cup1d python=3.8 pip matplotlib 
source activate cup1d
```

You will also need to install GPy and CAMB
```
pip install gpy camb
```

Clone the LaCE repo, set a LACE_REPO environment variable pointing to it, and install it:

```
cd $LACE_REPO
python setup.py install
```

Repeat the operation for the cup1d repo:

```
cd $CUP1D_PATH
python setup.py install
```

You will also need to install several other python packages:

```
pip install configobj configargparse emcee chainconsumer h5py getdist
```

### Notebooks / tutorials

You can start by plotting the many P1D measurements stored in the repo, by looking at `notebooks/p1d_measurements`.

You can also redo old neutrino mass constraints by importance sampling WMAP and Planck chains, following `notebooks/planck`.

You can also play with the LaCE emulator with the notebooks in `notebooks/emulator`

Finally, you can run your own analysis on mock data following the notebooks in `notebooks/likelihood`
