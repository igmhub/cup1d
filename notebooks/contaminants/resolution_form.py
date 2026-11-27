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
# # Resolution correction: first-order expansion vs Gaussian kernel
#
# The spectrograph-resolution correction multiplies the model P1D by
#
# $$C_R(k, z) = \exp(x) \quad\text{(exponential, now the default)}
# \qquad\text{or}\qquad 1 + x \quad\text{(linear, the former model)},$$
#
# with $x = 2\,R_\mathrm{coeff}(z)\,R(z)^2 k^2$ and $R(z)$ the velocity width of
# a 0.8 Å pixel. The exponential is the exact correction for a Gaussian kernel
# whose variance is off by $-2 R_\mathrm{coeff} R^2$; both agree to first order
# in $x$. At $k = 0.5\,k_\mathrm{Nyq}$, $k^2R^2 \approx 2.4$ at every redshift,
# so the prior edges $R_\mathrm{coeff} = \pm 0.1$ give $x \approx \pm 0.49$,
# outside the validity of the expansion.
#
# 1. the two corrections over the DESI DR1 $k$ range, at the prior edges and at
#    the best fit;
# 2. $\chi^2$ of the baseline best fit with each form;
# 3. optionally, a minimization with each form (`RUN_MINIMIZATION`).
#
# Set `RESULT` to the baseline minimizer result to use.

# %%
# %load_ext autoreload
# %autoreload 2

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cup1d.inference.analysis import Analysis
from cup1d.models.contaminants.resolution import (
    get_Rz_Naim,
    resolution_correction,
)

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix"})

RESULT = Path(
    "~/Proyectos/projects/lya/data/out_DESI_DR1/DESIY1_QMLE3/global_opt/lace_mpg/chain_1/minimizer_results.npy"
).expanduser()
RUN_MINIMIZATION = True

# %% [markdown]
# ## Best fit with the default (exponential) form

# %%
analysis = Analysis.from_results(RESULT)
fitter, like = analysis.fitter, analysis.fitter.like
model = like.theory.model_syst.resolution_model
print("resolution form:", model.form)
point = fitter.parameters_from_sampling_point(fitter.mle_cube)
data = next(iter(like.data.values()))
zs = np.asarray(data.z)
r_best = np.atleast_1d(model.get_value("R_coeff", zs, like_params=point))

# %% [markdown]
# ## The two corrections

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
colors = plt.cm.viridis(np.linspace(0, 1, len(zs)))
for ax, r_label in zip(axes, ("prior edges", "best fit")):
    for iz, z in enumerate(zs):
        k = np.asarray(data.k_kms[iz])
        values = (-0.1, 0.1) if r_label == "prior edges" else (r_best[iz],)
        for r in values:
            ax.plot(k, resolution_correction(r, z, k, "exponential"), color=colors[iz])
            ax.plot(k, resolution_correction(r, z, k, "linear"), color=colors[iz], ls="--")
    ax.set_xlabel(r"$k\,[\mathrm{s\,km^{-1}}]$", fontsize=14)
    ax.set_title(f"$R_\\mathrm{{coeff}}$ at the {r_label}", fontsize=14)
axes[0].set_ylabel(r"$C_R(k, z)$ (solid: exponential, dashed: linear)", fontsize=13)
fig.tight_layout()
plt.show()

print("x = 2 R_coeff R^2 k^2 at the largest k of each redshift, for R_coeff = 0.1:")
for iz, z in enumerate(zs):
    print(f"  z = {z:.1f}: x = {2 * 0.1 * get_Rz_Naim(z) ** 2 * np.max(data.k_kms[iz]) ** 2:.3f}")

# %% [markdown]
# ## $\chi^2$ of the best fit with each form

# %%
chi2 = {}
for form in ("exponential", "linear"):
    model.form = form
    chi2[form] = like.get_chi2(point)
model.form = "exponential"
print(f"chi2 at the best fit: exponential {chi2['exponential']:.3f}, linear {chi2['linear']:.3f}, "
      f"difference {chi2['linear'] - chi2['exponential']:+.3f}")

# %% [markdown]
# ## Minimization with each form
#
# A few minutes each. The starting point is the best fit above.

# %%
if RUN_MINIMIZATION:
    fits = {}
    start = np.array(fitter.mle_cube, copy=True)  # both forms start from the baseline best fit
    for form in ("exponential", "linear"):
        model.form = form
        fitter.run_minimizer(
            log_func_minimize=fitter.minus_log_prob, p0=start, restart=True, estimate_errors=True
        )
        star = fitter.like.blind.unblind(dict(fitter.mle_cosmo))
        errors = np.sqrt(np.diag(fitter.mle_cosmo_covariance))
        fits[form] = (fitter.mle_chi2, star, errors)
    model.form = "exponential"
    for form, (c2, star, errors) in fits.items():
        print(f"{form:12s} chi2 = {c2:.3f}, Delta2_star = {star['Delta2_star']:.4f} +- {errors[0]:.4f}, "
              f"n_star = {star['n_star']:.4f} +- {errors[1]:.4f}")

# %%
