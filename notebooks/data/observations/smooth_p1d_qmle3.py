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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Smooth P1D of the DESI DR1 QMLE3 measurement
#
# The emulator error enters the likelihood as a relative covariance times
# $P_\mathrm{smooth}(k_i)\,P_\mathrm{smooth}(k_j)$. So far $P_\mathrm{smooth}$ is
# the `PSMOOTH` column of the measurement file, which lies 3-5% above the
# measured `PLYA` (it seems to precede the continuum correction). Scaling by the
# noisy measurement instead (as for the DR1 paper) makes the covariance depend on
# the data noise and lowers the $\chi^2$.
#
# Here we compute our own smooth P1D from `PLYA` and its covariance, with three
# methods, one redshift at a time:
#
# 1. **Smoothing spline** in $(\ln k, \ln P)$, diagonal weights
#    (`scipy.interpolate.make_smoothing_spline`), smoothing $\lambda$ = 0.3 times
#    the generalized cross-validation (GCV) optimum, fitted to the **whole
#    measurement in the file, before any scale cut** (no border effects at the
#    cuts). **Baseline**: computed by `cup1d.p1ds.smooth_p1d` every time a
#    measurement is read and cached next to the measurement file
#    (`<file>.smooth_p1d.npz`, with its inputs); `data.Pksmooth_kms` is this
#    curve. Other smoothing factors (`variants`, 1 = plain GCV) are computed
#    here on the same full measurement.
# 2. **B-spline, generalized least squares**: cubic B-spline in $\ln k$ fitted to
#    $\ln P$ with the full covariance of the redshift bin; number of knots chosen
#    by BIC.
# 3. **Polynomial, generalized least squares**: polynomial in $\ln k$ fitted the
#    same way; degree chosen by BIC.
#
# Methods 2 and 3 are fitted to the data after the scale cuts.
#
# A good smooth P1D follows the data within the errors ($\chi^2/\mathrm{dof}
# \approx 1$, no trend in the residuals) without following the noise. The fit is
# redone with the baseline.

# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.interpolate import BSpline

from cup1d.configuration.args import Args
from cup1d.p1ds.factory import set_p1d
from astropy.io import fits

from cup1d.p1ds.observations.data_DESIY1 import set_p1d_filename
from cup1d.p1ds.smooth_p1d import read_cache, smooth_rows

rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "STIXGeneral"

# %% [markdown]
# ## Measurement
#
# Read with cup1d's reader, so the $k$ range and the covariance (statistical plus
# systematic, per redshift) are those of the likelihood. Choose the covariance
# used by the smoothing: `"total"` (as in the likelihood) or `"stat"` (systematic
# modes, e.g. continuum errors, are smooth in $k$ and should be followed by the
# smooth curve rather than penalized).

# %%
args = Args(pre_defined="CM2026", system="local")
data = set_p1d(args, "DESIY1_QMLE3")
cov_kind = "total"  # or "stat" (the cup1d baseline always uses the total)

zs = np.asarray(data.z)
ks = [np.asarray(k) for k in data.k_kms]
pks = [np.asarray(p) for p in data.Pk_kms]
covs = [np.asarray(c) for c in (data.cov_Pk_kms if cov_kind == "total" else data.covstat_Pk_kms)]
psmooth_cup1d = [np.asarray(p) for p in data.Pksmooth_kms]  # baseline, from the cache
print("redshifts:", zs)
print("points per redshift:", [len(k) for k in ks])

# full measurement (before scale cuts) used by the baseline, from its cache entry
fname = set_p1d_filename("QMLE3")
full = None
for entry in read_cache(fname).values():
    rows = [np.flatnonzero((entry["z"] == z) & np.isin(entry["k"], k)) for z, k in zip(zs, ks)]
    if all(len(r) == len(k) for r, k in zip(rows, ks)) and all(
            np.allclose(entry["smooth"][r], p) for r, p in zip(rows, psmooth_cup1d)):
        full, cut_rows = entry, rows
assert full is not None, "no cache entry matches the data: read the data again"
print(f"full measurement: {len(full['k'])} rows, {np.isin(np.arange(len(full['k'])), np.concatenate(cut_rows)).sum()} kept by the cuts")

# PSMOOTH column of the file, at the same rows
with fits.open(fname) as hdu:
    table = hdu["P1D_BLIND"].data
    psmooth_file = [np.asarray(table["PSMOOTH"])[r] for r in cut_rows]

# %% [markdown]
# ## Smoothing methods
#
# All fit $y = \ln P$ against $x = \ln k$; the covariance of $y$ is
# $C_{ij} / (P_i P_j)$.

# %%
def log_covariance(pk, cov):
    """Covariance of ln P from that of P (first order)."""
    return cov / np.outer(pk, pk)


def gls(design, y, cov_y):
    """Generalized least squares: coefficients and chi2 with the full covariance."""
    icov = np.linalg.inv(cov_y)
    coeffs = np.linalg.solve(design.T @ icov @ design, design.T @ icov @ y)
    r = y - design @ coeffs
    return coeffs, float(r @ icov @ r)


def bspline_basis(x, n_interior, x_range):
    """Cubic B-spline basis with n_interior knots equally spaced in x."""
    inner = np.linspace(*x_range, n_interior + 2)
    knots = np.r_[[x_range[0]] * 3, inner, [x_range[1]] * 3]
    return knots, BSpline.design_matrix(np.clip(x, *x_range), knots, 3).toarray()


def bspline_gls(k, pk, cov, n_interior):
    """Cubic B-spline in ln k fitted to ln P with the full covariance."""
    x, y = np.log(k), np.log(pk)
    x_range = (x.min(), x.max())
    knots, design = bspline_basis(x, n_interior, x_range)
    coeffs, chi2 = gls(design, y, log_covariance(pk, cov))
    spline = BSpline(knots, coeffs, 3, extrapolate=False)
    return (lambda kk: np.exp(spline(np.clip(np.log(kk), *x_range)))), chi2, len(coeffs)


def polynomial_gls(k, pk, cov, degree):
    """Polynomial in ln k (centred) fitted to ln P with the full covariance."""
    x, y = np.log(k), np.log(pk)
    x0, dx = x.mean(), np.ptp(x) / 2
    design = np.vander((x - x0) / dx, degree + 1)
    coeffs, chi2 = gls(design, y, log_covariance(pk, cov))
    return (lambda kk: np.exp(np.vander((np.log(kk) - x0) / dx, degree + 1) @ coeffs)), chi2, degree + 1


def select_by_bic(fit, k, pk, cov, options, penalty="bic"):
    """Fit with every option and keep the lowest chi2 + n_par x penalty.

    penalty: "bic" (ln n_data), "aic" (2) or a number; smaller penalties allow
    more parameters, i.e. curves closer to the data.
    """
    per_par = {"bic": np.log(len(k)), "aic": 2.0}.get(penalty, penalty)
    best = None
    for option in options:
        model, chi2, npar = fit(k, pk, cov, option)
        bic = chi2 + npar * per_par
        if best is None or bic < best[-1]:
            best = (model, chi2, npar, option, bic)
    return best


# %% [markdown]
# ## Fit every redshift

# %%
variants = {"smoothing spline (GCV)": 1.0}  # name: lam_factor, on the full measurement
full_variants = {name: smooth_rows(full["z"], full["k"], full["pk"], full["var"], lam_factor=f)
                 for name, f in variants.items()}

methods = ["smoothing spline", *variants, "B-spline GLS", "polynomial GLS", "PSMOOTH (file)"]
baseline = "smoothing spline"  # used by cup1d; marked "(baseline)" in the legends
labels = {m: f"{m} (baseline)" if m == baseline else m for m in methods}
smooth = {m: [] for m in methods}
choice = {m: [] for m in methods}

for i, (k, pk, cov, pfile, pcup1d) in enumerate(zip(ks, pks, covs, psmooth_file, psmooth_cup1d)):
    smooth["smoothing spline"].append(pcup1d)
    choice["smoothing spline"].append("0.3 GCV")
    for name, factor in variants.items():
        smooth[name].append(full_variants[name][cut_rows[i]])
        choice[name].append(f"{factor:g} GCV")
    model, _, npar, n_knots, _ = select_by_bic(bspline_gls, k, pk, cov, range(1, 9))
    smooth["B-spline GLS"].append(model(k))
    choice["B-spline GLS"].append(f"{n_knots} knots")
    model, _, npar, degree, _ = select_by_bic(polynomial_gls, k, pk, cov, range(2, 9))
    smooth["polynomial GLS"].append(model(k))
    choice["polynomial GLS"].append(f"deg {degree}")
    smooth["PSMOOTH (file)"].append(pfile)
    choice["PSMOOTH (file)"].append("-")


def block_chi2(pk, model, cov):
    r = pk - model
    return float(r @ np.linalg.solve(cov, r))


print(f"{'z':>4} " + " ".join(f"{m:>26}" for m in methods))
for i, z in enumerate(zs):
    cells = []
    for m in methods:
        chi2 = block_chi2(pks[i], smooth[m][i], covs[i])
        cells.append(f"{chi2:7.1f}/{len(ks[i]):3d} {choice[m][i]:>10}")
    print(f"{z:4.1f} " + " ".join(f"{c:>26}" for c in cells))
print("\nchi2 / n_data of data vs each smooth curve (covariance: " + cov_kind + ")")

# %% [markdown]
# ## Residuals
#
# Data / smooth - 1 with the diagonal errors. A good smooth curve leaves
# residuals scattered around zero without trends in $k$.

# %%
colors = {"smoothing spline": "C0", "B-spline GLS": "C1", "polynomial GLS": "C2", "PSMOOTH (file)": "k"}
colors.update({name: f"C{4 + j}" for j, name in enumerate(variants)})
fig, axes = plt.subplots(4, 3, figsize=(15, 13), sharex=True)
for i, (ax, z) in enumerate(zip(axes.ravel(), zs)):
    err = np.sqrt(np.diag(covs[i]))
    for m in methods:
        ratio = pks[i] / smooth[m][i] - 1
        ax.errorbar(ks[i], ratio, err / smooth[m][i], fmt="o", ms=2.5, lw=0.8, color=colors[m],
                    alpha=0.7, label=labels[m])
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_title(f"z = {z:.1f}", fontsize=13)
    ax.set_ylim(-0.25, 0.15)
axes.ravel()[-1].set_visible(False)
axes.ravel()[0].legend(fontsize=9)
fig.supxlabel(r"$k_\parallel\,[\mathrm{s\,km^{-1}}]$", fontsize=15)
fig.supylabel(r"$P_\mathrm{1D}/P_\mathrm{smooth} - 1$", fontsize=15)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Whole measurement and scale cuts
#
# The splines are fitted to every row of the file; the shaded band is the $k$
# range kept by the likelihood. The smooth curve should be regular across the
# cuts (no border effects).

# %%
fig, axes = plt.subplots(4, 3, figsize=(15, 13))
for i, (ax, z) in enumerate(zip(axes.ravel(), zs)):
    rows = np.flatnonzero(full["z"] == z)
    rows = rows[np.argsort(full["k"][rows])]
    k, pk, var = full["k"][rows], full["pk"][rows], full["var"][rows]
    ok = np.isfinite(pk) & (var > 0) & (pk > 0)
    base = full["smooth"][rows]
    ax.errorbar(k[ok], pk[ok] / base[ok] - 1, np.sqrt(var[ok]) / base[ok], fmt="o", ms=2, lw=0.6,
                color="0.5", label="data / baseline - 1")
    for name in variants:
        ax.plot(k, full_variants[name][rows] / base - 1, color=colors[name], lw=1.5, label=name)
    ax.axvspan(ks[i].min(), ks[i].max(), color="C0", alpha=0.08)
    ax.axhline(0, color="C0", lw=1)
    ax.set_xscale("log")
    ax.set_ylim(-0.3, 0.3)
    ax.set_title(f"z = {z:.1f}", fontsize=13)
axes.ravel()[-1].set_visible(False)
axes.ravel()[0].legend(fontsize=9)
fig.supxlabel(r"$k_\parallel\,[\mathrm{s\,km^{-1}}]$", fontsize=15)
fig.supylabel(r"$P/P_\mathrm{baseline} - 1$", fontsize=15)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Smooth curves against the measurement

# %%
fig, axes = plt.subplots(4, 3, figsize=(15, 13), sharex=True)
for i, (ax, z) in enumerate(zip(axes.ravel(), zs)):
    err = np.sqrt(np.diag(covs[i]))
    ax.errorbar(ks[i], ks[i] * pks[i] / np.pi, ks[i] * err / np.pi, fmt="o", ms=2.5, color="0.5",
                label="PLYA")
    for m in methods:
        ax.plot(ks[i], ks[i] * smooth[m][i] / np.pi, color=colors[m], lw=1.5, label=labels[m])
    ax.set_title(f"z = {z:.1f}", fontsize=13)
    ax.set_yscale("log")
axes.ravel()[-1].set_visible(False)
axes.ravel()[0].legend(fontsize=9)
fig.supxlabel(r"$k_\parallel\,[\mathrm{s\,km^{-1}}]$", fontsize=15)
fig.supylabel(r"$k_\parallel P_\mathrm{1D}/\pi$", fontsize=15)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Pulls and offsets
#
# Pulls $(P - P_\mathrm{smooth}) / \sigma$ of all redshifts (should be close to a
# unit Gaussian with some correlation between neighbours), and the mean ratio
# of every smooth curve to `PSMOOTH` (the change of the emulator-error
# amplitude is twice it).

# %%
fig, ax = plt.subplots(figsize=(7, 4.5))
bins = np.linspace(-5, 5, 41)
for m in methods:
    pulls = np.concatenate([(pks[i] - smooth[m][i]) / np.sqrt(np.diag(covs[i])) for i in range(len(zs))])
    ax.hist(pulls, bins=bins, histtype="step", lw=2, color=colors[m], density=True,
            label=f"{labels[m]}: mean {pulls.mean():+.2f}, rms {pulls.std():.2f}")
x = np.linspace(-5, 5, 200)
ax.plot(x, np.exp(-x**2 / 2) / np.sqrt(2 * np.pi), "k:", label="unit Gaussian")
ax.set_xlabel(r"$(P - P_\mathrm{smooth})/\sigma$", fontsize=14)
ax.legend(fontsize=8)
fig.tight_layout()
plt.show()

print(f"{'z':>4}  mean(smooth / PSMOOTH) per method")
for i, z in enumerate(zs):
    print(f"{z:4.1f}  " + "  ".join(f"{m}: {np.mean(smooth[m][i] / psmooth_file[i]):.4f}"
                                     for m in methods[:-1]))

# %% [markdown]
# ## Smoothness
#
# Ratio of every new smooth curve to the polynomial one: wiggles here show a
# method following the noise.

# %%
fig, axes = plt.subplots(4, 3, figsize=(15, 13), sharex=True)
for i, (ax, z) in enumerate(zip(axes.ravel(), zs)):
    for m in methods:
        ax.plot(ks[i], smooth[m][i] / smooth["polynomial GLS"][i] - 1, color=colors[m], lw=1.5, label=labels[m])
    ax.set_title(f"z = {z:.1f}", fontsize=13)
    ax.set_ylim(-0.1, 0.1)
axes.ravel()[-1].set_visible(False)
axes.ravel()[0].legend(fontsize=9)
fig.supxlabel(r"$k_\parallel\,[\mathrm{s\,km^{-1}}]$", fontsize=15)
fig.supylabel(r"$P_\mathrm{smooth}/P_\mathrm{poly} - 1$", fontsize=15)
fig.tight_layout()
plt.show()

# %%

# %%

# %%
