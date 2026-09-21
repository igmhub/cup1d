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
# # Inspect independent-redshift fits
#
# Initial conditions (ICs) for an analysis are generated with the YAML-driven
# script, rather than by this notebook:
#
# ```bash
# python scripts/create_at_a_time_initial_conditions.py \
#     configs/cm2026/variations/at_a_time_global_QMLE3.yaml
# ```
#
# The script fits every redshift bin, prints the goodness-of-fit summary, and
# saves the appropriate `mpg_ic_at_a_time.npy` or `nyx_ic_at_a_time.npy` file.
# This notebook is an interactive companion for inspecting one or a few local
# fits and their contamination contributions; it does not write IC files.

# %% [markdown]
# Load the analysis classes and the plotting tools used for the interactive
# diagnostics below.

# %%
# %load_ext autoreload
# %autoreload 2

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from cup1d import Analysis, Args
from cup1d.postprocessing.plotter import Plotter
from cup1d.utils.utils import get_path_repo


# %% [markdown]
# Select the P1D estimator and load its local-fit YAML configuration. The
# QMLE3 and FFT3_dir configurations both keep the background cosmology fixed,
# use one pivot parameter per fitted component, and do not inflate the
# statistical covariance errors.

# %%
# Choose either the QMLE3 or FFT3 direct-estimator local-fit configuration.
# Both use one pivot parameter per fitted IGM/contaminant component and do not
# inflate the statistical covariance.
data_label = "DESIY1_QMLE3"
# data_label = "DESIY1_FFT3_dir"

config_directory = Path(get_path_repo("cup1d")) / "configs" / "cm2026" / "variations"
config_files = {
    "DESIY1_QMLE3": "at_a_time_global_QMLE3.yaml",
    "DESIY1_FFT3_dir": "at_a_time_global_FFT3_dir.yaml",
}
config_path = config_directory / config_files[data_label]

args = Args.from_yaml(config_path, verbose=False)
analysis = Analysis(args)

# %% [markdown]
# Count the available P1D measurements in each redshift bin. These counts are
# useful for interpreting the local-fit goodness of fit.

# %%
key = args.data_label[0]

npoints = []
for ii in range(len(analysis.data[key].z)):
    npoints.append(len(analysis.data[key].k_kms[ii]))
npoints = np.array(npoints)
npoints

# %% [markdown]
# ## Fit selected redshift bins
#
# The production script fits every bin. Keep the short range below while
# interactively testing a single bin, or replace it with `range(len(...))` to
# inspect every local fit in this notebook.

# %%
out_mle = []
out_mle_cube = []
out_chi2 = []
out_pnames = []
# for ii in range(len(analysis.data[key].z)):
for ii in range(1):
    zmask = np.array([analysis.data[key].z[ii]])

    analysis = Analysis(args, out_folder=None)

    print()

    f_space_len = 14
    s_space_len = 5
    for p in analysis.like.free_params:
        print(
            p.name,
            (f_space_len - len(p.name)) * " ",
            "\t",
            np.round(p.value, 3),
            (s_space_len - len(str(np.round(p.value, 3)))) * " ",
            "\t",
            np.round(p.min_value, 3),
            (s_space_len - len(str(np.round(p.min_value, 3)))) * " ",
            "\t",
            np.round(p.max_value, 3),
            (s_space_len - len(str(np.round(p.max_value, 3)))) * " ",
            "\t",
            p.Gauss_priors_width,
        )

    print()

    print(ii, zmask)
    p0 = analysis.like.sampling_point_from_parameters().copy()
    analysis.run_minimizer(
        p0,
        zmask=zmask,
        restart=True,
    )
    out_pnames.append(analysis.like.free_param_names)
    out_mle.append(analysis.fitter.mle)
    out_mle_cube.append(analysis.fitter.mle_cube)
    out_chi2.append(analysis.fitter.mle_chi2)

# %% [markdown]
# ## Inspect contamination contributions
#
# This diagnostic figure compares the P1D residual after selectively removing
# contamination terms. It is not part of the IC-generation script. Set
# `zenodo_filename` only when deliberately exporting the plotted arrays.

# %%
# diru = "figs"
diru = None
plotter = Plotter(analysis.fitter, save_directory=diru, zmask=zmask)

store_data = plotter.plot_illustrate_contaminants_each(
    out_mle_cube[0].copy(),
    zmask,
    fontsize=22,
    store_data=True,
    zenodo_filename=None,
)
# %%
# store_data_ting = store_data
store_data_orig = store_data

# %% [markdown]
# ## Summarize the local fits
#
# This prints a compact LaTeX-ready table of the fitted redshift bins. The
# script prints the equivalent summary after it has processed all bins.

# %%
from cup1d.postprocessing.show_results import print_results
print_results(analysis.like, out_chi2, out_mle_cube)

# %%
