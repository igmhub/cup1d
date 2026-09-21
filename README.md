# cup1d

[![Documentation Status](https://readthedocs.org/projects/igmhubcup1d/badge/?version=latest)](https://igmhubcup1d.readthedocs.io/en/latest/)

## Cosmology using P1D - small-scale clustering of the Lyman alpha forest

This repository provides the data, theory, likelihood, inference, and
post-processing tools used for cosmological analyses of the one-dimensional
Lyman-alpha forest power spectrum (P1D). It supports observational data,
synthetic mocks, and forecasts through a common YAML-based interface.

It uses the [LaCE emulator](https://github.com/igmhub/LaCE) and supports
minimization and MCMC analyses of cosmological, IGM, contamination, and
instrumental-systematic parameters.

If you would like to collaborate, please email Andreu Font-Ribera (afont@ifae.es) or Jonas Chaves-Montero (jchaves@ifae.es).
 

### Documentation

The documentation covers installation, YAML configuration, the package layout,
tutorials, and the active API. Build it locally with:

- [Online documentation](https://igmhubcup1d.readthedocs.io/en/latest/)
- [Documentation source](https://github.com/igmhub/cup1d/tree/main/docs)

```
python -m pip install -e ".[docs]"
make docs
```

Open `docs/_build/html/index.html` after the build completes. The repository
also includes a Read the Docs configuration for publishing the same site.


### Installation

- Download and install Conda. You can find the instructions here https://docs.anaconda.com/miniconda/miniconda-install/

- Create a conda environment

```
conda create -n cup1d python=3.12
```
- Clone and install LaCE following the instructions [here](https://github.com/igmhub/LaCE) (do so within the environment created above):

```
git clone https://github.com/igmhub/LaCE.git
cd LaCE
make install
``` 

- Clone and install cup1d:

```
git clone https://github.com/igmhub/cup1d.git
cd cup1d
make install
``` 

#### NERSC users:

- You need to compile ``mpi4py`` package on NERSC (see [here](https://docs.nersc.gov/development/languages/python/parallel-python/#mpi4py-in-your-custom-conda-environment)).

```
MPICC="cc -shared" pip install --force-reinstall --no-cache-dir --no-binary=mpi4py mpi4py
```

#### Nyx users:

- You may need to add the Nyx path as an environment variable in your notebook kernel. The first is done by writing in the kernel.json file:

```
 "env": {
  "NYX_PATH":"path_to_Nyx"
 }
```

You also need to add the Nyx path as an environment variable. The Nyx data is located at NERSC in 

```
NYX_PATH="/global/cfs/cdirs/desi/science/lya/y1-p1d/likelihood_files/nyx_files/"
```

- Before running cup1d, please precompute all cosmological information needed using CAMB and save IGM histories. This is done by running the following scripts. *You do not need to do it* if you are in NERSC.

```
python LaCE/scripts/save_nyx_emu_cosmo.py
python LaCE/scripts/save_nyx_IGM.py
```

### Notebooks / tutorials


The main entry point is
[`notebooks/tutorials/dr1.py`](notebooks/tutorials/dr1.py), which runs the
baseline DESI DR1 analysis from its YAML configuration. The same directory
also contains compact tutorials for forecasts, mocks, and the Cobaya
likelihood interface.

The remaining notebooks are organized by purpose:

- `notebooks/data/observations` and `notebooks/data/mocks`: inspect input P1D
  measurements and synthetic data.
- `notebooks/likelihood`: likelihood diagnostics, initial-condition fits, and
  fit inspection.
- `notebooks/igm` and `notebooks/contaminants`: visualize physical and
  nuisance-model components.
- `notebooks/planck`: load Planck chains, perform importance sampling, and
  reproduce cosmological comparison figures.
- `notebooks/figures_CM26`: reproduce CM2026 figures and tables.
- `notebooks/wip` and `notebooks/old`: unfinished and preserved legacy work;
  these are not part of the supported tutorial path.

Notebook sources are paired with Jupyter files through Jupytext. To synchronize
a notebook after editing its Python source, run:

```
jupytext --sync notebooks/tutorials/dr1.py
```

- If you want to use notebooks via JupyterHub, you'll also need to download `ipykernel`:

```
pip install ipykernel
python -m ipykernel install --user --name cup1d --display-name cup1d
```

### YAML configurations

CM2026 defaults and sparse variations live under `configs/cm2026`. Dedicated
configurations for forecasts, mocks, high-resolution combinations, and other
papers live in their corresponding directories under `configs`.

```python
from cup1d import Analysis, Args

args = Args.from_yaml("configs/cm2026/cm2026_base.yaml", verbose=False)
analysis = Analysis(args)
```

See the [configuration documentation](https://igmhubcup1d.readthedocs.io/en/latest/configuration.html)
for layering rules and synthetic-data examples.


### Running tests

Install the test dependencies and run the complete suite without specifying
individual files:

```bash
python -m pip install -e ".[test]"
pytest -q
```
