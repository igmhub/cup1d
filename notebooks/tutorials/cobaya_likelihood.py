# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: lace
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Cobaya likelihood
#
# In order to use this likelihood:
#
# 1. git clone https://github.com/jchavesmontero/desi_dr1_p1d_cobaya_likelihood
# 2. cd desi_dr1_p1d_cobaya_likelihood
# 3. pip install -e .
# 4. Then run the notebook

# %%
# %load_ext autoreload
# %autoreload 2

import os
from cobaya.yaml import yaml_load_file
from cobaya.model import get_model
from cobaya_lya_p1d.cobaya_lya_p1d import Cobaya_lya_p1d

# %% [markdown]
# ### Load likelihood

# %%
import cobaya_lya_p1d
path_like = cobaya_lya_p1d.__path__

# %%
info_yaml = os.path.join(path_like[0], "desi_dr1.yaml")
packages_path = os.path.dirname(path_like[0])
info = {
    'likelihood': {'cobaya_lya_p1d.cobaya_lya_p1d.Cobaya_lya_p1d':yaml_load_file(info_yaml)},
    'theory': {'camb': None},
    'params': {
        "H0": 67.77,
        "mnu": 0,
        "omch2": 0.119,
        "ombh2": 0.0224,
        "As": 2.105e-09,
        "ns": 0.9665,
    },
    "sampler": {"mcmc": None},
}
info

# %% [markdown]
# ### Set model
#
# Evaluate the likelihood for a particular value of the H0 parameter, it can be done in the same for other cosmo params

# %%
model = get_model(info)

# %% [markdown]
# ### Evaluate model

# %%
model.logposterior({"H0":info["params"]["H0"]})

# %%

# %%
