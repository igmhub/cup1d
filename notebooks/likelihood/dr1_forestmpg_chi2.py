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
# # DESI DR1 Forest-MPG: χ² versus number of ForestFlow realizations
#
# This notebook builds only the `dr1_forestmpg` likelihood and evaluates its
# initial point.  ForestFlow averages latent cINN realizations before returning
# Arinyo coefficients.  Change `N_REALIZATIONS` below and rerun the evaluation
# cell to quantify the resulting change in χ².  The emulator uses a fixed seed,
# so a given realization count is reproducible.

# %%
from pathlib import Path

from cup1d import Analysis, Args
from cup1d.emulator.factory import set_emulator
from cup1d.utils.utils import get_path_repo


# %% [markdown]
# ## Configure the ForestFlow averaging

# %%
# Set this to the number of latent ForestFlow realizations to average for every
# emulator prediction.  Typical comparisons are 100, 500, 1000, and 3000.
N_REALIZATIONS = 1000


# %% [markdown]
# ## Build the DR1 Forest-MPG likelihood

# %%
config_file = Path(get_path_repo("cup1d")) / "configs" / "cm2026" / "cm2026_base_forestmpg.yaml"
args = Args.from_yaml(config_file, verbose=False)
# Keep this diagnostic out of the production DR1 output tree.
args.out_folder = "/tmp/cup1d_dr1_forestmpg_chi2"

# Build the same ForestFlow wrapper selected by the YAML, then set its P3D
# emulator's default number of latent realizations before cup1d evaluates it.
emulator = set_emulator(args.emulator_label)
emulator.emulator.Nrealizations = int(N_REALIZATIONS)
analysis = Analysis(args, emulator=emulator)
initial_point = analysis.fitter.initial_parameters()

print("Emulator:", analysis.emulator.emulator_label)
print("ForestFlow realizations:", analysis.emulator.emulator.Nrealizations)


# %% [markdown]
# ## Evaluate χ² at the standard initial point

# %%
def initial_chi2_for(n_realizations):
    """Evaluate the fixed initial point using a chosen ForestFlow average."""
    n_realizations = int(n_realizations)
    if n_realizations < 1:
        raise ValueError("n_realizations must be a positive integer")
    analysis.emulator.emulator.Nrealizations = n_realizations
    # Do not reuse any batched likelihood prediction after changing the
    # ForestFlow setting.
    analysis.emulator.clear_prediction_cache()
    return analysis.fitter.get_chi2(initial_point)


chi2 = initial_chi2_for(N_REALIZATIONS)
print(f"Nrealizations={N_REALIZATIONS}: chi2_initial={chi2:.6f}")


# %% [markdown]
# ## Optional realization-count scan
#
# Change this tuple (or set it to one value) and rerun the cell.  No parameters
# are fitted here: every entry evaluates exactly the same `initial_point`.

# %%
N_REALIZATIONS_SCAN = (100, 1000, 10000, 100000)
chi2_scan = {
    n_realizations: initial_chi2_for(n_realizations)
    for n_realizations in N_REALIZATIONS_SCAN
}
for n_realizations, value in chi2_scan.items():
    print(f"Nrealizations={n_realizations:5d}: chi2_initial={value:.6f}")

# %%
P1DEmulator
