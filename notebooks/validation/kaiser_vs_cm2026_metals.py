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
# # Can the Kaiser silicon model fit the CM2026 silicon model?
#
# Noiseless DESI DR1-like forecast with cup1d's analysis pipeline (`Analysis`):
#
# 1. **Mock**: the CM2026 analysis (`forest_mpg_gp`, `metal_model_type: McDonald`)
#    loads the `forest_mpg_gp` global initial conditions; its model at that point,
#    on the DR1 bins (same fine grid, `n_k_emu` interpolation and rebinning as any
#    fit), replaces the measured P1D. The DR1 covariance is kept.
# 2. **Fit**: the same analysis with the line-of-sight Kaiser silicon model
#    (`metal_model_type: Kaiser`), minimized with `Analysis.run_minimizer`.
#    Cosmology is fixed, and the IGM, HCD and resolution parameters are `fixed` at
#    the same initial conditions (`mask_pars=True`): only the Kaiser silicon
#    parameters vary, starting from the DR2 Vega biases, σ = 100 km/s, β = 0.5.
#
# The two analyses share everything but the silicon model, so the best-fit χ² is
# the bias of the Kaiser model with respect to the CM2026 one in DR1 units.

# %%
# %load_ext autoreload
# %autoreload 2

from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cup1d import Analysis, Args
from cup1d.utils.utils import get_path_repo, split_string

# %% [markdown]
# ## Settings
#
# `KAISER` sets the redshift histories of the Kaiser parameters; `BETA_FREE` also
# frees the silicon betas.

# %%
CONFIG = Path(get_path_repo("cup1d")) / "configs" / "desi_dr1_tests" / "cm2026_base_forestmpggp.yaml"
BETA_FREE = False  # False: betas fixed at 0.5 (the Kaiser default for a metal not varied)
N_NODES = 2  # redshift nodes of each Kaiser bias / sigma history
OLD_SILICON = ("f_Lya_SiIII", "s_Lya_SiIII", "f_Lya_SiII", "s_Lya_SiII",
               "f_SiIIa_SiIIb", "s_SiIIa_SiIIb", "f_SiIIa_SiIII", "f_SiIIb_SiIII")
KAISER = {"metal_model_type": "Kaiser"}
for family in OLD_SILICON:
    KAISER.update({f"n_{family}": 0, f"{family}_ztype": "pivot", f"{family}_fixed": True})
for family in ("b_SiIII", "b_SiIIa", "b_SiIIb", "b_SiIIc", "sig_SiIII", "sig_SiII"):
    KAISER.update({f"n_{family}": N_NODES, f"{family}_ztype": "interp_lin",
                   f"{family}_otype": "exp", f"{family}_fixed": False})
for family in ("beta_SiIII", "beta_SiII"):
    KAISER.update({f"n_{family}": int(BETA_FREE), f"{family}_ztype": "interp_lin" if BETA_FREE else "pivot",
                   f"{family}_otype": "const", f"{family}_fixed": not BETA_FREE})
# Everything but the Kaiser silicon parameters is fixed at the initial conditions.
HOLD = {
    "fix_cosmo": True,
    "fid_igm": {f"{family}_fixed": True for family in ("tau_eff", "sigT_kms", "gamma", "kF_kms")},
    "fid_syst": {"R_coeff_fixed": True},
}
HCD_FIXED = {f"HCD_damp{i}_fixed": True for i in range(1, 5)}
common = dict(verbose=False, fix_cosmo=True)
args_cm = Args.from_yaml(CONFIG, **common)
args_k = Args.from_yaml(CONFIG, fid_igm=HOLD["fid_igm"], fid_syst=HOLD["fid_syst"],
                        fid_cont={**KAISER, **HCD_FIXED}, **common)
print("emulator:", args_cm.emulator_label, "| k_rebin_factor:", args_cm.k_rebin_factor,
      "| n_k_emu:", args_cm.n_k_emu, "| fiducial cosmology:", args_cm.fid_cosmo_label,
      "| IC:", Path(args_cm.file_ic).name)

# %% [markdown]
# ## The CM2026 mock
#
# The CM2026 analysis loads the initial conditions; its model at that point
# replaces the data, and, as a check, has χ² = 0 on it.

# %%
analysis_cm = Analysis(args_cm, create_output=False)
truth = analysis_cm.fitter.initial_parameters()
model_cm = analysis_cm.like.get_p1d_kms(truth)[0]
data_mock = deepcopy(analysis_cm.data)
for key, dataset in data_mock.items():
    dataset.Pk_kms = [np.asarray(row, dtype=float) for row in model_cm[key]]
    if dataset.full_Pk_kms is not None:
        dataset.full_Pk_kms = np.concatenate(dataset.Pk_kms)
check = Analysis(args_cm, data=data_mock, emulator=analysis_cm.emulator, create_output=False)
print(f"CM2026 model on its own mock: chi2 = {check.like.get_chi2(truth):.2e}")

# %% [markdown]
# ## Fit the mock with the Kaiser model
#
# Nelder-Mead with restarts (`Fitter.run_minimizer`), only the Kaiser silicon
# parameters free (`mask_pars=True`).

# %%
analysis_k = Analysis(args_k, data=data_mock, emulator=analysis_cm.emulator, create_output=False)
free = [n for n, p in analysis_k.like.free_params.items() if not p["fixed"]]
print("free parameters:", free)
p0 = analysis_k.fitter.initial_parameters()
print(f"chi2 at the start: {analysis_k.like.get_chi2(p0):.2f}")
analysis_k.run_minimizer(p0, mask_pars=True, restart=True)
best = analysis_k.fitter.parameters_from_sampling_point(analysis_k.fitter.mle_cube)
chi2_best = analysis_k.like.get_chi2(best)
ndata = sum(len(row) for d in data_mock.values() for row in d.Pk_kms)
print(f"best-fit chi2 = {chi2_best:.4f} for {ndata} data points")

# %%
values = {name: (p["value"] if isinstance(p, dict) else p) for name, p in best.items()}
kaiser = analysis_k.like.theory.model_cont.metal_models["Si_kaiser"]
for z in (2.2, 2.33, 3.0, 4.2):
    coeff = kaiser.coefficients(z, values)
    print(f"z={z}:", {line: dict(bias=round(b, 5), beta=round(beta, 3), sigma_kms=round(s, 1))
                      for line, (b, beta, s) in coeff.items()})
print("DESI DR2 Vega baseline (z=2.33): b_SiIII=-0.0074, b_SiII(1190)=-0.0052, "
      "b_SiII(1193)=-0.0024, b_SiII(1260)=-0.0046")

# %% [markdown]
# ## Residuals and silicon factors
#
# Top: (Kaiser best fit − CM2026 mock)/σ per bin (σ from the diagonal of the DR1
# covariance). Bottom: the silicon factor of each model, the ratio of the full
# model to the model without silicon.

# %%
def without_silicon(point, analysis):
    """Copy of ``point`` with the silicon amplitudes switched off."""
    out = deepcopy(point)
    for name in out:
        if split_string(name)[0] in ("f_Lya_SiIII", "f_Lya_SiII", "f_SiIIa_SiIIb",
                                     "b_SiIII", "b_SiIIa", "b_SiIIb", "b_SiIIc"):
            if isinstance(out[name], dict):
                out[name]["value"] = -20.0  # amplitude / |b| -> 0
            else:
                out[name] = -20.0
    return out


fit_k = analysis_k.like.get_p1d_kms(best)[0]
no_si_k = analysis_k.like.get_p1d_kms(without_silicon(best, analysis_k))[0]
no_si_cm = analysis_cm.like.get_p1d_kms(without_silicon(truth, analysis_cm))[0]
key = next(iter(data_mock))
dset = data_mock[key]
nz, ncol = len(dset.z), 4
nrow = int(np.ceil(nz / ncol))
fig, axes = plt.subplots(2 * nrow, ncol, figsize=(4.2 * ncol, 3.0 * 2 * nrow), sharex=True)
for iz in range(nz):
    r, c = divmod(iz, ncol)
    k = dset.k_kms[iz]
    sigma = np.sqrt(np.diag(dset.cov_Pk_kms[iz]))
    axes[2 * r, c].plot(k, (fit_k[key][iz] - model_cm[key][iz]) / sigma, "o", ms=3)
    axes[2 * r, c].axhline(0, color="k", lw=0.5)
    axes[2 * r, c].set_title(f"z={dset.z[iz]:.1f}", fontsize=9)
    axes[2 * r + 1, c].plot(k, model_cm[key][iz] / no_si_cm[key][iz], label="CM2026 (mock)")
    axes[2 * r + 1, c].plot(k, fit_k[key][iz] / no_si_k[key][iz], "--", label="Kaiser best fit")
    if c == 0:
        axes[2 * r, c].set_ylabel("(Kaiser - mock) / σ")
        axes[2 * r + 1, c].set_ylabel("with / without Si")
for ax in axes[-1]:
    ax.set_xlabel(r"$k_\parallel$ [s/km]")
for extra in range(nz, nrow * ncol):
    r, c = divmod(extra, ncol)
    axes[2 * r, c].set_visible(False)
    axes[2 * r + 1, c].set_visible(False)
axes[1, 0].legend(fontsize=8)
fig.suptitle(f"Kaiser fit to the CM2026 silicon mock: chi2 = {chi2_best:.2f} / {ndata}")
plt.tight_layout()
plt.show()

