# cup1d

## Cosmology using P1D - small scale clustering of the Lyman alpha forest

This repository contains some tools to perform the last steps of a cosmological analysis of the 1D power spectrum (P1D) of the Lyman alpha forest. 

It uses the LaCE emulator (https://github.com/igmhub/LaCE), and some extra tools to run MCMC analyses on cosmological and IGM parameters for a mock P1D measurement.

If you would like to collaborate, please email Andreu Font-Ribera (afont@ifae.es) or Jonas Chaves-Montero (jchaves@ifae.es).
 

### Installation

- Create a new conda environment. It is usually better to follow python version one or two behind. In January 2024, the latest is 3.12, so we recommend 3.11.

```
conda create -n cup1d -c conda-forge python=3.11 camb mpich mpi4py fdasrsf pip=24.0
conda activate cup1d
```
- Install cup1d:

```Follow the instructions from https://github.com/igmhub/cup1d```

- Clone the cup1d repo and perform an *editable* installation:

```
git clone https://github.com/igmhub/cup1d.git
cd cup1d
pip install -e
``` 

- You also need to install the LaCE emulator. The instructions are in https://github.com/igmhub/LaCE, but repeat these below to avoid misunderstandings

```
git clone https://github.com/igmhub/LaCE.git
cd LacE
pip install -e .[explicit]
pip install -e .[gpy]
``` 

- If you want to use notebooks via JupyterHub, you'll also need to download `ipykernel`:

```
pip install ipykernel
python -m ipykernel install --user --name cup1d --display-name cup1d
```

#### NERSC users:

- You need to compile ``mpi4py`` package on NERSC (see [here](https://docs.nersc.gov/development/languages/python/parallel-python/#mpi4py-in-your-custom-conda-environment)).

```
MPICC="cc -shared" pip install --force-reinstall --no-cache-dir --no-binary=mpi4py mpi4py
```

#### Nyx users:

- You may need to add the Nyx path as an enviroment variable in your notebook kernel. The first is done writting in the kernel.json file:

```
 "env": {
  "NYX_PATH":"path_to_Nyx"
 }
```

For convenience, you could also add the Nyx path as an environment variable. The Nyx data is located at NERSC in 

```
NYX_PATH="/global/cfs/cdirs/desi/science/lya/y1-p1d/likelihood_files/nyx_files/"
```

### Notebooks / tutorials

You can run the notebooks in `notebooks`. You can find the main tutorial to run your own analysis in `notebooks/tutorials/sample_sim.ipynb`

You can also plot many P1D measurements stored in the repo, by looking at `notebooks/p1d_measurements`

You can also redo old neutrino mass constraints by importance sampling WMAP and Planck chains, following `notebooks/planck`

You can also play with the LaCE emulator with the notebooks in `notebooks/emulator`


### Forecasting script

You can use the script scripts/sam_sim.py to run you own analyses. It is fully parallelized using MPI.
