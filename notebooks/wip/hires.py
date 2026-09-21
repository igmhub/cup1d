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
# ### Trying to combine low- and high-res data
#
# Need to add possibility to control kmax_kms of hi-res data

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
import os, sys
import matplotlib.pyplot as plt
from cup1d import Analysis, Args
from cup1d.utils.utils import get_path_repo


# %% [markdown]
# ## Load P1D measurements and set likelihood

# %%
config_file = os.path.join(
    get_path_repo("cup1d"), "configs", "hires", "hires.yaml"
)
args = Args.from_yaml(config_file, verbose=False)
analysis = Analysis(args)

# %% [markdown]
# ## Plot P1D data
#
# Get parameters from a point of the parameter space close to the best fit

# %%
p0 = analysis.like.sampling_point_from_parameters().copy()
free_params = analysis.like.parameters_from_sampling_point(p0)
analysis.like.get_chi2(p0)

# %%
# kmax_kms = 0.05
pa = np.array(
    [
        2.75118584e-01,
        5.17774010e-01,
        5.30625889e-01,
        1.83371658e-01,
        3.14888265e-01,
        2.56233809e-04,
        4.49190221e-01,
        8.08802364e-01,
        2.75376246e-01,
        7.60464392e-01,
        7.80380692e-01,
        2.13933769e-01,
        9.76238948e-01,
        2.12625196e-01,
        8.36634280e-03,
        6.11667700e-04,
        4.83328967e-01,
        4.62507818e-01,
        4.42825501e-01,
        5.07518881e-01,
        5.77167635e-01,
        5.82047638e-01,
        5.73636776e-01,
        2.43226185e-01,
        7.31152406e-01,
        3.84519394e-01,
        5.72142838e-01,
        6.79952839e-01,
        5.84233785e-01,
        7.22705904e-01,
        4.60296343e-01,
        3.86725805e-01,
        2.67293134e-01,
        7.58941796e-01,
        9.45021016e-01,
        9.02367918e-01,
        2.21158462e-01,
        6.56859007e-01,
        8.39572131e-02,
        1.92424581e-01,
        6.38898130e-01,
        6.63939405e-01,
    ]
)

# kmax_kms = 0.1
pb = np.array(
    [
        7.33426056e-01,
        5.34725468e-01,
        3.21560156e-01,
        1.05118655e-01,
        9.22835705e-05,
        3.58845416e-06,
        6.17809858e-01,
        9.99986596e-01,
        9.99973173e-01,
        9.99879839e-01,
        8.49863134e-01,
        5.03862818e-01,
        5.42119000e-01,
        7.11304802e-01,
        5.86267647e-04,
        1.09726365e-05,
        2.83928640e-01,
        7.58647865e-06,
        4.26733863e-01,
        4.45587879e-01,
        5.78405300e-01,
        5.40943959e-01,
        5.38033093e-01,
        5.58162278e-01,
        7.12800869e-01,
        7.68411451e-01,
        5.68971975e-01,
        7.87179192e-01,
        5.83447873e-01,
        8.16674457e-01,
        4.46660740e-01,
        5.24128245e-01,
        2.86619375e-01,
        5.26975148e-01,
        9.15807378e-01,
        7.87923616e-01,
        7.19038867e-01,
        8.13452558e-01,
        7.00610726e-02,
        5.34434791e-02,
        6.25417538e-01,
        1.42502425e-02,
    ]
)

# forestflow DESI DR1
pc = np.array(
    [
        0.44370891,
        0.41896411,
        0.61988151,
        0.40441576,
        0.45867814,
        0.01096454,
        0.67756427,
        0.10833635,
        0.08650839,
        0.72379107,
        0.99996769,
        0.69281722,
        0.64864356,
        0.00276913,
        0.13844165,
        0.006074,
        0.30521981,
        0.85987199,
        0.45887747,
        0.52643935,
        0.58099397,
        0.57841591,
        0.58993933,
        0.31664493,
        0.7344514,
        0.42520576,
        0.56215998,
        0.66432898,
        0.59622537,
        0.6936141,
        0.46279051,
        0.39349512,
        0.26083371,
        0.73383231,
        0.84770478,
        0.80593018,
        0.17142966,
        0.76663517,
        0.54851977,
        0.32703703,
        0.57129636,
        0.62593543,
    ]
)


# forestflow DESI DR1 + high-res
pd = np.array(
    [
        3.61298262e-01,
        5.22894209e-01,
        4.94989807e-01,
        2.98809478e-01,
        3.04978420e-01,
        3.09442180e-04,
        9.87653865e-02,
        4.09004424e-01,
        4.59962792e-01,
        7.14441761e-01,
        9.99797440e-01,
        6.83919757e-01,
        7.91199875e-01,
        7.61970016e-04,
        2.88248284e-03,
        2.09967528e-03,
        4.97778270e-01,
        6.16933795e-02,
        4.41407264e-01,
        5.25453133e-01,
        5.74959555e-01,
        5.88260321e-01,
        5.77492490e-01,
        4.77830790e-01,
        7.36140573e-01,
        7.41602839e-01,
        5.68842364e-01,
        6.92868172e-01,
        5.78514223e-01,
        7.69904875e-01,
        4.58013030e-01,
        3.67787791e-01,
        2.63787399e-01,
        6.31085609e-01,
        8.89951239e-01,
        6.43353105e-01,
        7.22908393e-01,
        8.51732076e-01,
        1.17517034e-01,
        2.83270697e-01,
        6.10083973e-01,
        5.61151135e-01,
    ]
)

# %%
p0 = pa
# p0 = analysis.like.sampling_point_from_parameters().copy()
free_params = analysis.like.parameters_from_sampling_point(p0)
analysis.like.get_chi2(p0)

# %% [markdown]
# Plot model for these parameters

# %%
analysis.like.plot_p1d(p0)

# %% [markdown]
# ### Get predictions from the model

# %%
# list of model parameters

for par in analysis.like.free_params:
    print(par.name, par.value, par.min_value, par.max_value)

# %% [markdown]
# ## Run minimizer

# %% [markdown]
# Run minimizer starting from this point, it should stop the minimization soon

# %%
analysis.run_minimizer(p0)

# %%
p0 = analysis.like.sampling_point_from_parameters().copy()
free_params = analysis.like.parameters_from_sampling_point(p0)
analysis.like.get_chi2(p0)

# %% [markdown]
# Evaluate for the new best fit

# %%
p1 = analysis.fitter.mle_cube
analysis.like.plot_p1d(p1)

# %%
