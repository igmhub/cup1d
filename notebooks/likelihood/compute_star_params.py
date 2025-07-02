# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Compute star parameters

from cup1d.likelihood.pipeline import set_cosmo
from cup1d.likelihood import CAMB_model

# +

fid_cosmo_label="Planck18"
fid_cosmo = set_cosmo(cosmo_label=fid_cosmo_label)

blob = CAMB_model.CAMBModel(zs=[3], cosmo=fid_cosmo).get_linP_params()
blob
# -


