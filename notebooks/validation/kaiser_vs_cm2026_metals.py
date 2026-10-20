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
# Noiseless DESI DR1-like forecast:
#
# 1. **Mock**: P1D from the `forest_mpg_gp` emulator with the CM2026 silicon model
#    (`metal_model_type: McDonald`, `si_mult` + `si_add`) on the DR1 k bins.
# 2. **Fit**: the same P1D model with the line-of-sight Kaiser silicon model
#    (`metal_model_type: Kaiser`), varying **only** the Kaiser metal parameters.
#
# Cosmology, IGM, HCD and resolution parameters are the same in both and fixed to
# the `forest_mpg_gp` global initial conditions (the CM2026 metal parameters of the
# mock come from them too). The Lyα P1D and the Kaiser moments are projected once,
# with the same quadrature, and shared, so the only difference between mock and
# fit is the metal model. The DR1 covariance (data, systematics, emulator) is used
# for the χ²; there is no noise, so the best-fit χ² is the bias of the Kaiser
# model with respect to the CM2026 one in DR1 units.

# %%
# %load_ext autoreload
# %autoreload 2

import os
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

import forestflow
from forestflow.emulator import load_emulator
from cup1d import Args
from cup1d.emulator.factory import set_emulator
from cup1d.likelihood.external import (
    ExternalP1DLikelihood, PredictionContext, make_integrator, project_arinyo_request,
    project_metal_moments)
from cup1d.likelihood.parameters import set_free_likelihood_parameters
from cup1d.models.contaminants.metals.si_kaiser import FAMILIES as KAISER_FAMILIES
from cup1d.models.contaminants.model_contaminants import Contaminants
from cup1d.models.contaminants.model_systematics import Systematics
from cup1d.p1ds.factory import set_p1d
from cup1d.theory.factory import set_theory
from cup1d.utils.ic_files import get_ic_path
from cup1d.utils.utils import get_path_repo, split_string

# %% [markdown]
# ## Settings
#
# `KAISER` sets the redshift histories of the Kaiser parameters (two nodes, as the
# DR1 metal parameters). `BETA_FREE` also frees the silicon betas (one node each).

# %%
CONFIG = Path(get_path_repo("cup1d")) / "configs" / "desi_dr1_tests" / "cm2026_base_forestmpggp.yaml"
BETA_FREE = True
OLD_SILICON = ("f_Lya_SiIII", "s_Lya_SiIII", "f_Lya_SiII", "s_Lya_SiII",
               "f_SiIIa_SiIIb", "s_SiIIa_SiIIb", "f_SiIIa_SiIII", "f_SiIIb_SiIII")
KAISER = {"metal_model_type": "Kaiser"}
for family in OLD_SILICON:
    KAISER.update({f"n_{family}": 0, f"{family}_ztype": "pivot", f"{family}_fixed": True})
for family in ("b_SiIII", "b_SiIIa", "b_SiIIb", "b_SiIIc", "sig_SiIII", "sig_SiII"):
    KAISER.update({f"n_{family}": 2, f"{family}_ztype": "interp_lin",
                   f"{family}_otype": "exp", f"{family}_fixed": False})
for family in ("beta_SiIII", "beta_SiII"):
    KAISER.update({f"n_{family}": int(BETA_FREE), f"{family}_ztype": "interp_lin" if BETA_FREE else "pivot",
                   f"{family}_otype": "const", f"{family}_fixed": not BETA_FREE})

args_cm = Args.from_yaml(CONFIG, verbose=False)
args_k = Args.from_yaml(CONFIG, verbose=False, fid_cont=KAISER)
names_cm = set_free_likelihood_parameters(args_cm, emulator_label=args_cm.emulator_label)
names_k = set_free_likelihood_parameters(args_k, emulator_label=args_k.emulator_label)
print("emulator:", args_cm.emulator_label, "| k_rebin_factor:", args_cm.k_rebin_factor,
      "| fiducial cosmology:", args_cm.fid_cosmo_label)

# %% [markdown]
# ## Parameters from the `forest_mpg_gp` initial conditions

# %%
ic = np.load(get_ic_path(args_cm.data_label, args_cm.emulator_label, "global"), allow_pickle=True).item()


def ic_values(args, names):
    """Values of the IC histories at the redshift nodes of ``names``."""
    out, missing = {}, []
    for name in names:
        if name in ("As", "ns"):
            continue  # not stored in the IC file: fiducial cosmology
        family, index = split_string(name)
        for section in (args.fid_igm, args.fid_cont, args.fid_syst):
            if family + "_znodes" in section:
                znode = section[family + "_znodes"][int(index)]
                break
        if family in ic:
            out[name] = float(np.interp(znode, ic[family]["z"], ic[family]["val"]))
        elif family == "HCD_const":
            out[name] = 0.0
        else:
            missing.append(name)
    return out, missing


point_cm, missing_cm = ic_values(args_cm, names_cm)
point_k, missing_k = ic_values(args_k, names_k)
assert not missing_cm
assert set(missing_k) == {n for n in names_k if split_string(n)[0] in KAISER_FAMILIES}
shared = {n: point_cm[n] for n in point_k}  # IGM, HCD, resolution: identical in both
assert shared == point_k
print("free Kaiser parameters:", missing_k)

# %% [markdown]
# ## Lyα P1D and Kaiser moments, shared by mock and fit
#
# The emulator inputs come from cup1d's own theory (IGM model and fiducial
# cosmology), the Arinyo coefficients from the `forest_mpg_gp` GP, and the
# projection from cup1d's external route with ForestFlow's default (converged)
# quadrature.

# %%
data = {label: set_p1d(args_cm, label) for label in args_cm.data_label}
emulator = set_emulator(emulator_label=args_cm.emulator_label)
gp = load_emulator(args_cm.emulator_label)
error = np.load(Path(forestflow.__file__).resolve().parents[1] / "data" / "covariance"
                / f"l1O_cov_{args_cm.emulator_label}.npy", allow_pickle=True).item()
zs = np.array(sorted({float(z) for d in data.values() for z in d.z}))
theory = set_theory(args_cm, emulator, names_cm, use_hull=False, zs=zs)
emu_call, M_of_z = theory.get_emulator_calls(zs, like_params=point_cm)
cosmology = theory.get_cosmology(point_cm)


class Snapshot:
    """LaCE cosmology with the k-range check of cup1d's external projection."""

    def __init__(self, cosmo):
        self._cosmo = cosmo

    def __getattr__(self, name):
        return getattr(self._cosmo, name)

    def validate_k(self, k_iMpc):
        if np.any(np.asarray(k_iMpc) > self._cosmo.get_kmax_linP_Mpc()):
            raise ValueError("k outside the linear-power coverage")


snapshot = Snapshot(cosmology)
inputs = [{k: float(emu_call[k][i]) for k in gp.input_labels} for i in range(len(zs))]
ari = gp.evaluate(inputs)
arinyo = {float(z): {k: float(np.atleast_1d(ari[k])[i]) for k in gp.output_labels}
          for i, z in enumerate(zs)}
M = dict(zip(zs, np.asarray(M_of_z, dtype=float)))


def backend(args, data_used, name):
    """cup1d external P1D likelihood with the contaminants of ``args``."""
    names = set_free_likelihood_parameters(args, emulator_label=args.emulator_label)
    return ExternalP1DLikelihood(
        data_used,
        Contaminants(free_param_names=names, pars_cont=args.fid_cont, ic_correction=args.ic_correction),
        Systematics(free_param_names=names, pars_syst=args.fid_syst),
        cov_factor=args.cov_factor, emulator_covariance=error,
        fiducial_conversion=lambda z: M[float(z)], emu_cov_type=args.emu_cov_type,
        k_rebin_factor=args.k_rebin_factor, configuration_id=name)


backend_cm = backend(args_cm, data, "cm2026")
request = backend_cm.get_prediction_request()
integrator = make_integrator({"rule": "k_mu"})
p1d_lya = project_arinyo_request(request, snapshot, arinyo, kmax_iMpc=gp.kmax_1d_iMpc, integrator=integrator)
moments = project_metal_moments(request, snapshot, arinyo, kmax_iMpc=gp.kmax_1d_iMpc, integrator=integrator)


def context(be):
    return PredictionContext(be.request.identity, tuple(zs), tuple(emu_call["mF"]), tuple(M[z] for z in zs))


nuisance_cm = {n: v for n, v in point_cm.items() if split_string(n)[0] not in args_cm.igm_params}

# %% [markdown]
# ## The CM2026 mock
#
# The mock replaces the measured P1D of every data set (the covariance is kept).
# As a check, the CM2026 model fits its own mock with χ² = 0.

# %%
mock = backend_cm.apply_observation_model(p1d_lya, context(backend_cm), nuisance_cm)
data_mock = deepcopy(data)
for key, dataset in data_mock.items():
    dataset.Pk_kms = [np.asarray(row, dtype=float) for row in mock[key]]
    if dataset.full_Pk_kms is not None:
        dataset.full_Pk_kms = np.concatenate(dataset.Pk_kms)
check = backend(args_cm, data_mock, "cm2026-mock").evaluate_from_p1d(
    p1d_lya, context(backend(args_cm, data_mock, "cm2026-mock")), nuisance_cm)
print(f"CM2026 model on its own mock: chi2 = {check.chi2_data:.2e} ({check.ndata} points)")

# %% [markdown]
# ## Fit the mock with the Kaiser model
#
# Only the Kaiser parameters vary, within their cup1d priors (ln|b|, ln σ/(km/s)
# and β). They start from the DESI DR2 Vega biases, σ = 100 km/s and β = 0.5.

# %%
backend_k = backend(args_k, data_mock, "kaiser-mock")
ctx_k = context(backend_k)
fixed_k = {n: v for n, v in point_k.items() if split_string(n)[0] not in args_k.igm_params}
fit_names = list(missing_k)
priors = args_k.fid_cont["flat_priors"]
bounds = [tuple(priors[split_string(n)[0]][-1]) for n in fit_names]
start = np.array([args_k.fid_cont[split_string(n)[0]][int(split_string(n)[1])] for n in fit_names])


def kaiser_chi2(theta):
    values = dict(fixed_k, **dict(zip(fit_names, theta)))
    return backend_k.evaluate_from_p1d(p1d_lya, ctx_k, values, moments).chi2_data


print(f"chi2 at the start: {kaiser_chi2(start):.2f}")
best = None
for method in ("L-BFGS-B", "Nelder-Mead", "L-BFGS-B"):
    x0 = start if best is None else best.x
    result = minimize(kaiser_chi2, x0, method=method, bounds=bounds,
                      options={"maxiter": 20000, "maxfev": 20000} if method == "Nelder-Mead" else {"maxiter": 5000})
    if best is None or result.fun < best.fun:
        best = result
    print(f"{method:12s} chi2 = {result.fun:.4f}  ({result.nfev} evaluations)")
best_values = dict(fixed_k, **dict(zip(fit_names, best.x)))
print(f"best-fit chi2 = {best.fun:.4f} for {check.ndata} data points")

# %%
kaiser = backend_k.model_cont.metal_models["Si_kaiser"]
display_table = {}
for z in (2.2, 2.33, 3.0, 4.2):
    coeff = kaiser.coefficients(z, best_values)
    display_table[z] = {line: dict(bias=round(b, 5), beta=round(beta, 3), sigma_kms=round(s, 1))
                        for line, (b, beta, s) in coeff.items()}
for z, row in display_table.items():
    print(f"z={z}:", row)
print("DESI DR2 Vega baseline (z=2.33): b_SiIII=-0.0074, b_SiII(1190)=-0.0052, "
      "b_SiII(1193)=-0.0024, b_SiII(1260)=-0.0046")

# %% [markdown]
# ## Residuals and metal contributions
#
# Top: (Kaiser best fit − CM2026 mock)/σ per bin (σ from the diagonal of the DR1
# covariance). Bottom: the metal contamination of each model, as the ratio of the
# full model to the model without silicon, on the data bins.

# %%
fit_obs = backend_k.apply_observation_model(p1d_lya, ctx_k, best_values, moments)
remove_k = {n: v for n, v in best_values.items()}
for name in fit_names:
    if name.startswith("b_"):
        remove_k[name] = -20.0  # |b| -> 0
no_metal_k = backend_k.apply_observation_model(p1d_lya, ctx_k, remove_k, moments)
remove_cm = dict(nuisance_cm)
for name in remove_cm:
    if split_string(name)[0] in ("f_Lya_SiIII", "f_Lya_SiII", "f_SiIIa_SiIIb"):
        remove_cm[name] = -20.0  # amplitudes -> 0
no_metal_cm = backend_cm.apply_observation_model(p1d_lya, context(backend_cm), remove_cm)

key = next(iter(data_mock))
dset = data_mock[key]
nz = len(dset.z)
ncol = 4
nrow = int(np.ceil(nz / ncol))
fig, axes = plt.subplots(2 * nrow, ncol, figsize=(4.2 * ncol, 3.0 * 2 * nrow), sharex=True)
for iz in range(nz):
    r, c = divmod(iz, ncol)
    k = dset.k_kms[iz]
    sigma = np.sqrt(np.diag(dset.cov_Pk_kms[iz]))
    ax = axes[2 * r, c]
    ax.plot(k, (fit_obs[key][iz] - mock[key][iz]) / sigma, "o", ms=3)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title(f"z={dset.z[iz]:.1f}", fontsize=9)
    if c == 0:
        ax.set_ylabel("(Kaiser - mock) / σ")
    ax = axes[2 * r + 1, c]
    ax.plot(k, mock[key][iz] / no_metal_cm[key][iz], label="CM2026 (mock)")
    ax.plot(k, fit_obs[key][iz] / no_metal_k[key][iz], "--", label="Kaiser best fit")
    if c == 0:
        ax.set_ylabel("with / without Si")
for ax in axes[-1]:
    ax.set_xlabel(r"$k_\parallel$ [s/km]")
for extra in range(nz, nrow * ncol):
    r, c = divmod(extra, ncol)
    axes[2 * r, c].set_visible(False)
    axes[2 * r + 1, c].set_visible(False)
axes[1, 0].legend(fontsize=8)
fig.suptitle(f"Kaiser fit to the CM2026 silicon mock: chi2 = {best.fun:.2f} / {check.ndata}")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Interpretation
#
# The best-fit χ² is the χ² a perfect, noiseless DR1 measurement generated by the
# CM2026 silicon model would have under the Kaiser model with the other
# parameters at their true values. Values well below 1 mean the Kaiser model
# reproduces the CM2026 silicon signal within the DR1 precision; larger values
# point to the scales and redshifts (residual panels) where the two differ.
# Letting the IGM, cosmology and other nuisances vary as well would absorb part
# of any difference.
