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
# # Kaiser vs CM2026 silicon at z = 2.2: one ingredient at a time
#
# Single-redshift (z = 2.2) noiseless DR1-like tests with cup1d's analysis
# pipeline, to find which ingredient of the CM2026 silicon model the Kaiser model
# cannot reproduce. For each test:
#
# 1. **Mock**: the CM2026 analysis (`forest_mpg_gp`, `metal_model_type: McDonald`)
#    at the `forest_mpg_gp` global initial conditions, with only some silicon
#    ingredients switched on (the others "zero"); its model on the DR1 bin (same
#    fine grid, `n_k_emu` interpolation and rebinning as any fit) replaces the
#    measured P1D, keeping the DR1 covariance.
# 2. **Fit**: the Kaiser analysis (`metal_model_type: Kaiser`) with only the
#    matching silicon species, minimized with `Analysis.run_minimizer`.
#    Cosmology is fixed and the IGM, HCD and resolution parameters are `fixed` at
#    the same initial conditions (`mask_pars=True`). The Kaiser Lya x metal
#    damping has the functional form `KAISER_DAMPING` (default: the Gaussian
#    `exp(-(k sigma)^2 / 2)`, as in Vega; "sigmoid" gives the CM2026
#    `2 - 2 / (1 + exp(-k sigma))`). What the mock does not have,
#    the fit does not have either: SiIIc is off (CM2026 has no SiIIc), and the
#    SiII x SiII crosses are excluded unless the mock has `f_SiIIa_SiIIb` (the
#    only SiII x SiII cross of CM2026, in `Si_add`).
#
# Ingredients of CM2026 (coefficients of `otype: exp` families are logarithms):
# amplitudes `f_Lya_SiIII`, `f_Lya_SiII`, `f_SiIIa_SiIIb` and their sigmoid
# dampings `s_*`; "zero" is a coefficient of -20, which `si_mult` sets to exactly
# 0 (for a damping scale, 0 means no damping). The ratio factors `f_SiIIa_SiIII`
# and `f_SiIIb_SiIII` scale the optically-thin line ratios; 0 (factor 1) is the
# optically-thin limit.

# %%
# %load_ext autoreload
# %autoreload 2

from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cup1d import Analysis, Args
from cup1d.utils.utils import get_path_repo, split_string

plt.rcParams.update({"mathtext.fontset": "stix", "font.family": "STIXGeneral", "font.size": 15})

# %% [markdown]
# ## Settings and the CM2026 analysis

# %%
CONFIG = Path(get_path_repo("cup1d")) / "configs" / "desi_dr1_tests" / "cm2026_base_forestmpggp.yaml"
Z = 2.2  # the only redshift (z_max = Z selects the bin and its covariance block)
BETA_FREE = False  # False: betas fixed at 0.5 (the Kaiser default for a metal not varied)
KAISER_DAMPING = "gaussian"  # Kaiser Lya x metal damping: "gaussian" (as Vega) or "sigmoid" (as CM2026)
KAISER_DAMPING = "gaussian"
COMMON = dict(verbose=False, fix_cosmo=True, z_max=Z)
DATA_COSMO = "Planck18_noBAO"  # fixed fiducial cosmology of the fits to the measured P1D (section 7)
IGM = ("tau_eff", "sigT_kms", "gamma", "kF_kms")
HCD = tuple(f"HCD_damp{i}" for i in range(1, 5))
OFF = -20.0  # "zero" for otype-exp coefficients
AMPLITUDES = ("f_Lya_SiIII", "f_Lya_SiII", "f_SiIIa_SiIIb")
DAMPINGS = ("s_Lya_SiIII", "s_Lya_SiII", "s_SiIIa_SiIIb")
RATIOS = ("f_SiIIa_SiIII", "f_SiIIb_SiIII")
OLD_SILICON = AMPLITUDES + DAMPINGS + RATIOS
SIIxSII_CROSSES = {"SiIIa_SiIIb", "SiIIa_SiIIc", "SiIIb_SiIIc"}  # Kaiser SiII x SiII cross terms

args_cm = Args.from_yaml(CONFIG, **COMMON)
analysis_cm = Analysis(args_cm, create_output=False)
truth = analysis_cm.fitter.initial_parameters()  # the forest_mpg_gp initial conditions
print("emulator:", args_cm.emulator_label, "| k_rebin_factor:", args_cm.k_rebin_factor,
      "| n_k_emu:", args_cm.n_k_emu, "| IC:", Path(args_cm.file_ic).name)
print("CM2026 silicon at the initial conditions:",
      {n: round(p["value"], 3) for n, p in truth.items() if split_string(n)[0] in OLD_SILICON})

# %% [markdown]
# ## Helpers

# %%
def cm2026_point(keep, optically_thin):
    """Initial conditions with only the silicon families in ``keep`` switched on."""
    point = deepcopy(truth)
    for name, parameter in point.items():
        family = split_string(name)[0]
        if family in AMPLITUDES + DAMPINGS and family not in keep:
            parameter["value"] = OFF
        elif family in RATIOS and optically_thin:
            parameter["value"] = 0.0  # factor 1: optically-thin line ratio
    return point


def make_mock(point):
    """CM2026 model at ``point`` replacing the measured P1D (covariance kept)."""
    model = analysis_cm.like.get_p1d_kms(point)[0]
    data = deepcopy(analysis_cm.data)
    for key, dataset in data.items():
        dataset.Pk_kms = [np.asarray(row, dtype=float) for row in model[key]]
        if dataset.full_Pk_kms is not None:
            dataset.full_Pk_kms = np.concatenate(dataset.Pk_kms)
    return model, data


def fixed_nuisance_args(cont):
    """Configuration with the contaminant overrides ``cont``; IGM, HCD and resolution fixed."""
    cont = {**cont, **{f"{f}_fixed": True for f in HCD}}
    return Args.from_yaml(
        CONFIG, fid_cont=cont, fid_syst={"R_coeff_fixed": True},
        fid_igm={f"{f}_fixed": True for f in IGM}, **COMMON)


def free_nuisance_args(cont, z=Z):
    """Configuration at redshift ``z`` with the contaminant overrides ``cont`` and every
    non-cosmology parameter free.

    IGM, HCD, resolution and the free families of ``cont`` get one coefficient each, at a node
    at ``z`` (so the global initial conditions are interpolated at ``z``); cosmology is fixed to
    DATA_COSMO.
    """
    def one_node(families):
        return {k: v for f in families for k, v in
                {f"n_{f}": 1, f"{f}_ztype": "interp_lin", f"{f}_znodes": [z], f"{f}_fixed": False}.items()}
    free_cont = [k[2:] for k, v in cont.items() if k.startswith("n_") and v > 0]
    cont = {**cont, **one_node(HCD + tuple(free_cont))}
    return Args.from_yaml(
        CONFIG, fid_cont=cont, fid_syst=one_node(["R_coeff"]), fid_igm=one_node(IGM),
        **{**COMMON, "fid_cosmo_label": DATA_COSMO, "z_min": z, "z_max": z})


def kaiser_cont(species, beta_free=BETA_FREE, off_lines=(), sii_free=True):
    """Kaiser contaminant overrides with only ``species`` ('SiIII', 'SiII') switched on.

    The transitions in ``off_lines`` (e.g. 'SiIIc') are switched off: bias fixed at zero.
    ``sii_free`` frees the SiII x SiII amplitude factor and decorrelation
    (``f_SiIIxSiII``, ``sig_SiIIxSiII``); otherwise these terms follow the biases. It only
    applies when SiII is on. The fiducial Kaiser model is ``kaiser_cont({"SiIII", "SiII"})``
    (Gaussian damping, ``KAISER_DAMPING``; free SiII x SiII factor and decorrelation; betas
    fixed at 0.5 unless ``beta_free``).
    """
    lines = {"SiIII": ("SiIII",), "SiII": ("SiIIa", "SiIIb", "SiIIc")}
    cont = {"metal_model_type": "Kaiser"}
    for family in OLD_SILICON:
        cont.update({f"n_{family}": 0, f"{family}_ztype": "pivot", f"{family}_fixed": True})
    for sp in ("SiIII", "SiII"):
        on = sp in species
        families = {f"b_{line}": on and line not in off_lines for line in lines[sp]}
        families[f"sig_{sp}"] = on
        for family, free in families.items():
            cont.update({f"n_{family}": int(free), f"{family}_ztype": "interp_lin" if free else "pivot",
                         f"{family}_otype": "exp", f"{family}_fixed": not free})
        free_beta = on and beta_free
        cont.update({f"n_beta_{sp}": int(free_beta), f"beta_{sp}_ztype": "interp_lin" if free_beta else "pivot",
                     f"beta_{sp}_otype": "const", f"beta_{sp}_fixed": not free_beta})
    sii_free = sii_free and "SiII" in species
    for family in ("f_SiIIxSiII", "sig_SiIIxSiII"):
        cont.update({f"n_{family}": int(sii_free), f"{family}_ztype": "interp_lin" if sii_free else "pivot",
                     f"{family}_otype": "exp", f"{family}_fixed": not sii_free})
    return cont


def kaiser_args(species, beta_free=BETA_FREE, off_lines=(), sii_free=True):
    """Kaiser configuration (see ``kaiser_cont``) with IGM, HCD and resolution fixed."""
    return fixed_nuisance_args(kaiser_cont(species, beta_free, off_lines, sii_free))


CM2026_CONT = {f"n_{family}": 1 for family in OLD_SILICON}  # one coefficient per silicon family


def without_silicon(point):
    """Copy of ``point`` with every silicon amplitude switched off."""
    out = deepcopy(point)
    for name, parameter in out.items():
        if split_string(name)[0] in AMPLITUDES + ("b_SiIII", "b_SiIIa", "b_SiIIb", "b_SiIIc"):
            if isinstance(parameter, dict):
                parameter["value"] = OFF
            else:
                out[name] = OFF
    return out


def silicon_values(analysis, point, families, z=Z):
    """``{family: (coefficients, value at z)}`` of the metal ``families`` in ``analysis``.

    ``value`` is after the ``otype`` transform (exp for amplitudes, dampings and ratios);
    a family switched off (all coefficients ``OFF``) has value 0, as ``si_mult`` uses it.
    """
    values = {n: (p["value"] if isinstance(p, dict) else p) for n, p in point.items()}
    out = {}
    for model in analysis.like.theory.model_cont.metal_models.values():
        for family in model.list_coeffs:
            if family in families and family not in out:
                coeff = [values[n] for n in values if split_string(n)[0] == family]
                off = len(coeff) > 0 and all(c <= OFF for c in coeff)
                out[family] = (coeff, 0.0 if off else float(model.get_value(family, z, values)))
    return {family: out[family] for family in families if family in out}


def print_cm2026(point, header="CM2026 input", analysis=analysis_cm, z=Z):
    """Silicon parameters of a CM2026 ``point``: coefficients and value at ``z``."""
    print(f"  {header} (z = {z}):")
    print(f"    {'parameter':16s} {'coefficients':>18s} {'value at z':>12s}")
    for family, (coeff, value) in silicon_values(analysis, point, OLD_SILICON, z).items():
        coeff = ", ".join(f"{c:.3f}" for c in coeff)
        print(f"    {family:16s} {coeff:>18s} {value:12.4e}")


def kaiser_values(model, best, z=Z):
    """Per-line bias, beta and sigma of the Kaiser ``model`` at ``z`` (lines switched on only),
    and the SiII x SiII amplitude factor and decorrelation under the key ``SiIIxSiII``."""
    out = {line: dict(bias=b, beta=beta, sigma_kms=s)
           for line, (b, beta, s) in model.coefficients(z, best).items() if abs(b) > 1e-8}
    out["SiIIxSiII"] = dict(factor=float(model.get_value("f_SiIIxSiII", z, best)),
                            sigma_kms=float(model.get_value("sig_SiIIxSiII", z, best)))
    return out


def print_free(best, free, exclude=()):
    """Best-fit values of the ``free`` parameters whose family is not in ``exclude``."""
    names = [n for n in free if split_string(n)[0] not in exclude]
    for i in range(0, len(names), 4):
        print("    " + ", ".join(f"{n} = {best[n]:.4f}" for n in names[i:i + 4]))


def print_kaiser_best(best, free, coeff, z=Z):
    """Free Kaiser parameters at the best fit and the per-line bias, beta and sigma at ``z``."""
    print(f"  Kaiser best fit (z = {z}, {KAISER_DAMPING} Lya x metal damping), free parameters:")
    print_free(best, free)
    print(f"    {'line':8s} {'bias':>12s} {'beta':>8s} {'sigma [km/s]':>14s}")
    for line, c in coeff.items():
        if line != "SiIIxSiII":
            print(f"    {line:8s} {c['bias']:12.4e} {c['beta']:8.3f} {c['sigma_kms']:14.1f}")
    sii = coeff["SiIIxSiII"]
    if "f_SiIIxSiII_0" in free or "sig_SiIIxSiII_0" in free:
        print(f"    SiII x SiII terms: factor = {sii['factor']:.4f}, sigma = {sii['sigma_kms']:.1f} km/s")
    else:
        print("    SiII x SiII terms: set by the biases (no extra factor or decorrelation)")


def plot_fit(title, k, sigma, ref, models):
    """P1D, residuals and with/without-Si ratio, stacked vertically.

    ``ref`` (the data or the mock, plotted with errors ``sigma``) and each of ``models``
    are dicts with ``label``, ``p1d`` and ``no_si`` (P1D without silicon; None to skip).
    """
    fig, axes = plt.subplots(3, 1, figsize=(9, 13), sharex=True)
    axes[0].errorbar(k, k * ref["p1d"] / np.pi, k * sigma / np.pi, fmt="o", ms=3, color="k",
                     label=ref["label"])
    if ref["no_si"] is not None:
        axes[2].plot(k, ref["p1d"] / ref["no_si"], "k-", label=ref["label"])
    for i, model in enumerate(models):
        color = f"C{i}"
        axes[0].plot(k, k * model["p1d"] / np.pi, color=color, label=model["label"])
        axes[1].plot(k, (model["p1d"] - ref["p1d"]) / sigma, "o-", ms=3, lw=0.8, color=color)
        axes[2].plot(k, model["p1d"] / model["no_si"], "--", color=color, label=model["label"])
    axes[0].set(ylabel=r"$k P_{1D}/\pi$")
    axes[0].legend()
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].set(ylabel=rf"(model $-$ {ref['label']}) / $\sigma$")
    axes[2].set(ylabel="with / without Si")
    axes[2].legend()
    axes[2].set_xlabel(r"$k_\parallel$ [s/km]")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


RESULTS = {}


def run_test(label, keep, species, optically_thin=True, beta_free=BETA_FREE, exclude=(), sii_free=True):
    """Generate the CM2026 mock, fit it with the Kaiser model and plot the result.

    ``exclude`` lists Kaiser terms (``SiKaiser.p1d_terms`` names) omitted from the fit model;
    ``sii_free`` frees the Kaiser SiII x SiII amplitude factor and decorrelation (not when the
    mock has no SiII x SiII term, where they would be degenerate). Terms absent from the mock
    are absent from the fit: SiIIc is always off, and without ``f_SiIIa_SiIIb`` in ``keep``
    the SiII x SiII crosses are excluded.
    """
    point_cm = cm2026_point(keep, optically_thin)
    model_cm, data = make_mock(point_cm)
    if "f_SiIIa_SiIIb" not in keep:
        exclude = set(exclude) | SIIxSII_CROSSES
        sii_free = False
    analysis_k = Analysis(kaiser_args(species, beta_free, off_lines={"SiIIc"}, sii_free=sii_free), data=data,
                          emulator=analysis_cm.emulator, create_output=False)
    analysis_k.like.theory.model_cont.metal_models["Si_kaiser"].exclude = set(exclude)
    analysis_k.like.theory.model_cont.metal_models["Si_kaiser"].damping = KAISER_DAMPING
    free = [n for n, p in analysis_k.like.free_params.items() if not p["fixed"]]
    p0 = analysis_k.fitter.initial_parameters()
    analysis_k.run_minimizer(p0, mask_pars=True, restart=True)
    best = analysis_k.fitter.parameters_from_sampling_point(analysis_k.fitter.mle_cube)  # name -> value
    chi2 = analysis_k.like.get_chi2(best)
    coeff = kaiser_values(analysis_k.like.theory.model_cont.metal_models["Si_kaiser"], best)
    key = next(iter(data))
    ndata = len(data[key].Pk_kms[0])
    RESULTS[label] = dict(chi2=chi2, ndata=ndata, kaiser=coeff,
                          cm2026={f: v for f, (_, v) in silicon_values(analysis_cm, point_cm, OLD_SILICON).items()},
                          analysis=analysis_k, best=best, point_cm=point_cm)  # for the SiII x SiII plot
    print(f"{label}: free {free}\n  chi2 = {chi2:.3f} / {ndata}")
    print_cm2026(point_cm)
    print_kaiser_best(best, free, coeff)

    fit = analysis_k.like.get_p1d_kms(best)[0]
    no_si_k = analysis_k.like.get_p1d_kms(without_silicon(best))[0]
    no_si_cm = analysis_cm.like.get_p1d_kms(without_silicon(point_cm))[0]
    k = data[key].k_kms[0]
    sigma = np.sqrt(np.diag(data[key].cov_Pk_kms[0]))
    plot_fit(f"z = {Z}, {label}: $\\chi^2$ = {chi2:.2f} / {ndata}", k, sigma,
             dict(label="mock", p1d=model_cm[key][0], no_si=no_si_cm[key][0]),
             [dict(label="Kaiser best fit", p1d=fit[key][0], no_si=no_si_k[key][0])])


def fit_data(label, args, exclude=(), z=Z):
    """Fit the measured QMLE3 P1D at redshift ``z`` (that of ``args``); print the best fit and
    return its curves and the data.

    Starts from the global initial conditions (interpolated at ``z``) for every family they hold.

    ``exclude`` lists Kaiser terms omitted from the model (Kaiser configurations only).
    """
    analysis = Analysis(args, emulator=analysis_cm.emulator, create_output=False)  # loads the data at z
    metal_models = analysis.like.theory.model_cont.metal_models
    kaiser = "Si_kaiser" in metal_models
    if kaiser:
        metal_models["Si_kaiser"].exclude = set(exclude)
        metal_models["Si_kaiser"].damping = KAISER_DAMPING
    free = [n for n, p in analysis.like.free_params.items() if not p["fixed"]]
    p0 = analysis.fitter.initial_parameters()
    analysis.run_minimizer(p0, mask_pars=True, restart=True)
    best = analysis.fitter.parameters_from_sampling_point(analysis.fitter.mle_cube)  # name -> value
    chi2 = analysis.like.get_chi2(best)
    key = next(iter(analysis.data))
    ndata = len(analysis.data[key].Pk_kms[0])
    print(f"{label}: free {free}\n  chi2 = {chi2:.3f} / {ndata}")
    RESULTS[label] = dict(chi2=chi2, ndata=ndata, z=z, best=best)
    if kaiser:
        coeff = kaiser_values(metal_models["Si_kaiser"], best, z)
        RESULTS[label]["kaiser"] = coeff
        print_kaiser_best(best, free, coeff, z)
    else:
        RESULTS[label]["cm2026"] = {f: v for f, (_, v) in silicon_values(analysis, best, OLD_SILICON, z).items()}
        print("  free parameters other than silicon:")
        print_free(best, free, exclude=OLD_SILICON)
        print_cm2026(best, "CM2026 best fit", analysis, z)
    dataset = analysis.data[key]
    return dict(label=f"{label}: $\\chi^2$ = {chi2:.1f}", z=z,
                p1d=analysis.like.get_p1d_kms(best)[0][key][0],
                no_si=analysis.like.get_p1d_kms(without_silicon(best))[0][key][0],
                k=dataset.k_kms[0], data=np.asarray(dataset.Pk_kms[0]),
                sigma=np.sqrt(np.diag(dataset.cov_Pk_kms[0])))


def plot_data(fits, title):
    """Measured P1D against the best fits ``fits`` (``fit_data`` outputs at one redshift)."""
    ref = fits[0]
    plot_fit(f"z = {ref['z']}, {title}", ref["k"], ref["sigma"],
             dict(label="data", p1d=ref["data"], no_si=None), fits)


# %% [markdown]
# ## 1. SiIII amplitude only (no damping)

# %%
run_test("1. SiIII amplitude", keep={"f_Lya_SiIII"}, species={"SiIII"})

# %% [markdown]
# ## 2. SiIII amplitude and damping

# %%
run_test("2. SiIII amplitude + damping", keep={"f_Lya_SiIII", "s_Lya_SiIII"}, species={"SiIII"})

# %% [markdown]
# ## 3. SiII amplitude only (no damping)

# %%
run_test("3. SiII amplitude", keep={"f_Lya_SiII"}, species={"SiII"})

# %% [markdown]
# ## 4. SiII amplitude and damping

# %%
run_test("4. SiII amplitude + damping", keep={"f_Lya_SiII", "s_Lya_SiII"}, species={"SiII"})

# %% [markdown]
# ## 5. All amplitudes and dampings, optically-thin line ratios
#
# Includes the SiII doublet (`f_SiIIa_SiIIb`, `s_SiIIa_SiIIb`); the ratio factors
# are 1. Kaiser here has its SiII x SiII terms tied to the biases
# (`sii_free=False`); the fiducial Kaiser model, with free SiII x SiII factor and
# decorrelation, is 5e onwards.

# %%
run_test("5. amplitudes + dampings, optically thin", keep=set(AMPLITUDES + DAMPINGS),
         species={"SiIII", "SiII"}, optically_thin=True, sii_free=False)

# %% [markdown]
# ### 5b. The same with the silicon betas free
#
# Is the redshift-space distortion parameter (`beta_SiIII`, `beta_SiII`, prior
# [0, 3]) relevant? Compare with the fixed beta = 0.5 fit above.

# %%
run_test("5b. as 5, betas free", keep=set(AMPLITUDES + DAMPINGS),
         species={"SiIII", "SiII"}, optically_thin=True, beta_free=True, sii_free=False)

# %% [markdown]
# ### 5c. SiIII and SiII amplitudes + dampings, no SiII x SiII, beta = 0.5
#
# Mock: as test 5 without the SiII doublet term (`f_SiIIa_SiIIb`, `s_SiIIa_SiIIb`
# off), which is the only SiII x SiII cross of CM2026. Kaiser: the SiII x SiII
# crosses are therefore excluded too (as in tests 3 and 4). Both keep the metal
# autos and the SiIII x SiII crosses.

# %%
NO_SIIxSII = {"f_Lya_SiIII", "s_Lya_SiIII", "f_Lya_SiII", "s_Lya_SiII"}
run_test("5c. no SiIIxSiII, beta = 0.5", keep=NO_SIIxSII, species={"SiIII", "SiII"},
         optically_thin=True, beta_free=False)

# %% [markdown]
# ### 5d. The same with the silicon betas free

# %%
run_test("5d. no SiIIxSiII, betas free", keep=NO_SIIxSII, species={"SiIII", "SiII"},
         optically_thin=True, beta_free=True)

# %% [markdown]
# ### 5e. As 5, with a free SiII x SiII amplitude and decorrelation
#
# Mock: as test 5. Kaiser: the SiII x SiII terms (SiII autos and crosses
# between SiII lines) are multiplied by a free factor `f_SiIIxSiII` and a
# Gaussian decorrelation `exp(-(k sigma)^2 / 2)` with free `sig_SiIIxSiII`,
# decoupling them from the Lya x SiII biases as in CM2026 (`f_SiIIa_SiIIb`,
# `s_SiIIa_SiIIb`). If the SiII x SiII coupling is what the Kaiser model cannot
# reproduce, the chi2 should drop from that of test 5 to about that of 5c.

# %%
run_test("5e. as 5, free SiIIxSiII amplitude + decorrelation", keep=set(AMPLITUDES + DAMPINGS),
         species={"SiIII", "SiII"}, optically_thin=True, sii_free=True)

# %% [markdown]
# ## 6. Full CM2026 model
#
# Mock: all CM2026 ingredients, including the fitted line ratios. Kaiser: the
# full model of 5e (SiIII and SiII biases and decorrelations, plus the free
# SiII x SiII amplitude and decorrelation).

# %%
run_test("6. full CM2026", keep=set(AMPLITUDES + DAMPINGS), species={"SiIII", "SiII"},
         optically_thin=False, sii_free=True)

# %% [markdown]
# ## 7. The measured QMLE3 P1D at z = 2.2
#
# Fits to the DR1 QMLE3 measurement instead of a mock, with every parameter of
# the model free except cosmology, which is fixed to `Planck18_noBAO`
# (`DATA_COSMO`): IGM (`tau_eff`, `sigT_kms`, `gamma`, `kF_kms`), HCD
# (`HCD_damp1-4`), resolution (`R_coeff`) and silicon, one coefficient each at a
# node at z = 2.2. The fits start from the `forest_mpg_gp` global initial
# conditions interpolated at z = 2.2 (the Kaiser silicon families, not in that
# file, from their fiducial values).
#
# ### 7a. CM2026
#
# All eight silicon families free.

# %%
FIT_7A = fit_data("7a. QMLE3, CM2026", free_nuisance_args(CM2026_CONT))
plot_data([FIT_7A], "QMLE3: CM2026")

# %% [markdown]
# ### 7b. Full Kaiser model
#
# The fiducial Kaiser model, `kaiser_cont({"SiIII", "SiII"})`: SiIII and SiII
# (SiIIa, SiIIb, SiIIc) biases, the Gaussian Lya x metal dampings (`KAISER_DAMPING`)
# and the SiII x SiII amplitude factor and decorrelation free; betas as `BETA_FREE`.

# %%
FIT_7B = fit_data("7b. QMLE3, Kaiser", free_nuisance_args(kaiser_cont({"SiIII", "SiII"})))
plot_data([FIT_7A, FIT_7B], "QMLE3: CM2026 and Kaiser")

# %% [markdown]
# ### 7c. Full Kaiser model, betas free
#
# As 7b with the metal betas (`beta_SiIII`, `beta_SiII`, prior [0, 3]) free too.

# %%
FIT_7C = fit_data("7c. QMLE3, Kaiser, betas free",
                  free_nuisance_args(kaiser_cont({"SiIII", "SiII"}, beta_free=True)))
plot_data([FIT_7A, FIT_7B, FIT_7C], "QMLE3: CM2026 and Kaiser")

# %% [markdown]
# ## 8. The measured QMLE3 P1D at the other redshifts
#
# The fits of 7a (CM2026) and 7b (full Kaiser) at each of the other DR1
# redshifts, one at a time: every parameter but cosmology free, one coefficient
# each at a node at that redshift, starting from the global initial conditions
# there. Each redshift is plotted as in 7b; the last figure compares the two
# models across redshift.

# %%
# Z_OTHER = [round(z, 1) for z in np.arange(Z + 0.2, 4.2 + 1e-6, 0.2)]
Z_OTHER = [2.6]
LABELS_8 = {Z: ("7a. QMLE3, CM2026", "7b. QMLE3, Kaiser")}
for z in Z_OTHER:
    LABELS_8[z] = (f"8. z = {z}, QMLE3, CM2026", f"8. z = {z}, QMLE3, Kaiser")
    fit_cm = fit_data(LABELS_8[z][0], free_nuisance_args(CM2026_CONT, z), z=z)
    fit_k = fit_data(LABELS_8[z][1], free_nuisance_args(kaiser_cont({"SiIII", "SiII"}, sii_free=True), z), z=z)
    plot_data([fit_cm, fit_k], "QMLE3: CM2026 and Kaiser")

# %%
# Z_OTHER = [round(z, 1) for z in np.arange(Z, 4.2 + 1e-6, 0.2)]
Z_OTHER = [round(z, 1) for z in np.arange(Z, 3.0 + 1e-6, 0.2)]
LABELS_8 = {Z: ("7a. QMLE3, CM2026", "7b. QMLE3, Kaiser")}
for z in Z_OTHER:
    LABELS_8[z] = (f"8. z = {z}, QMLE3, CM2026", f"8. z = {z}, QMLE3, Kaiser")
    fit_cm = fit_data(LABELS_8[z][0], free_nuisance_args(CM2026_CONT, z), z=z)
    fit_k = fit_data(LABELS_8[z][1], free_nuisance_args(kaiser_cont({"SiIII", "SiII"}, sii_free=True), z), z=z)
    plot_data([fit_cm, fit_k], "QMLE3: CM2026 and Kaiser")

# %%
zs = np.array(list(LABELS_8))
chi2_cm = np.array([RESULTS[cm]["chi2"] for cm, _ in LABELS_8.values()])
chi2_k = np.array([RESULTS[k]["chi2"] for _, k in LABELS_8.values()])
ndata = np.array([RESULTS[cm]["ndata"] for cm, _ in LABELS_8.values()])
print(f"{'z':>4s} {'ndata':>6s} {'chi2 CM2026':>12s} {'chi2 Kaiser':>12s} {'Kaiser - CM2026':>16s}")
for z, n, c, k in zip(zs, ndata, chi2_cm, chi2_k):
    print(f"{z:4.1f} {n:6d} {c:12.2f} {k:12.2f} {k - c:16.2f}")
print(f"total: CM2026 {chi2_cm.sum():.1f}, Kaiser {chi2_k.sum():.1f} for {ndata.sum()} data points")

fig, axes = plt.subplots(2, 1, figsize=(9, 9), sharex=True)
axes[0].plot(zs, chi2_cm / ndata, "o-", label="CM2026")
axes[0].plot(zs, chi2_k / ndata, "s--", label="Kaiser")
axes[0].set(ylabel=r"$\chi^2 / N_{\rm data}$")
axes[0].legend()
axes[1].plot(zs, chi2_k - chi2_cm, "o-", color="k")
axes[1].axhline(0, color="k", lw=0.5)
axes[1].set(xlabel="$z$", ylabel=r"$\chi^2_{\rm Kaiser} - \chi^2_{\rm CM2026}$")
fig.suptitle("QMLE3, one redshift at a time, cosmology fixed to " + DATA_COSMO)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summary

# %%
for label, result in RESULTS.items():
    print(f"{label:45s} chi2 = {result['chi2']:8.3f} / {result['ndata']}")


# %% [markdown]
# ## SiII x SiII term: CM2026 vs Kaiser
#
# The SiII x SiII contribution to P1D in the test 5 mock and in the Kaiser best
# fits of tests 5 (terms set by the biases) and 5e (free amplitude and
# decorrelation). Each term is the model P1D minus the same model without its
# SiII x SiII terms: `f_SiIIa_SiIIb` off in CM2026 (its `Si_add` term, the SiIIa
# and SiIIb autos and their cross with optically-thin ratio), and the SiII autos
# and SiII x SiII crosses excluded in Kaiser. In CM2026 it is a constant times
# `1 + r^2 + 2 r cos(k dv)` times `exp(-s^2 k^2)`; in Kaiser it follows the
# projected 3D power (the moments `I_j`) times the decorrelation. The ratio panel
# shows the shape difference, the last one its weight in the chi2.

# %%
SIIxSII_KAISER = {"SiIIa_SiIIa", "SiIIb_SiIIb", "SiIIc_SiIIc", "SiIIa_SiIIb", "SiIIa_SiIIc", "SiIIb_SiIIc"}


def sii_term_cm2026(point):
    """CM2026 SiII x SiII contribution to P1D at ``point`` (Si_add, from f_SiIIa_SiIIb)."""
    off = deepcopy(point)
    for name, parameter in off.items():
        if split_string(name)[0] == "f_SiIIa_SiIIb":
            parameter["value"] = OFF
    key = next(iter(analysis_cm.data))
    return analysis_cm.like.get_p1d_kms(point)[0][key][0] - analysis_cm.like.get_p1d_kms(off)[0][key][0]


def sii_term_kaiser(analysis, best):
    """Kaiser SiII x SiII contribution to P1D at ``best`` (SiII autos and SiII x SiII crosses)."""
    model = analysis.like.theory.model_cont.metal_models["Si_kaiser"]
    key = next(iter(analysis.data))
    full = analysis.like.get_p1d_kms(best)[0][key][0]
    exclude = model.exclude
    model.exclude = set(exclude) | SIIxSII_KAISER
    try:
        without = analysis.like.get_p1d_kms(best)[0][key][0]
    finally:
        model.exclude = exclude
    return full - without


LABEL_5 = "5. amplitudes + dampings, optically thin"
LABEL_5E = "5e. as 5, free SiIIxSiII amplitude + decorrelation"
key = next(iter(analysis_cm.data))
k = analysis_cm.data[key].k_kms[0]
sigma = np.sqrt(np.diag(analysis_cm.data[key].cov_Pk_kms[0]))
term_cm = sii_term_cm2026(RESULTS[LABEL_5]["point_cm"])
terms_k = {f"Kaiser {label.split('.')[0]}: " + ("free factor + decorrelation" if label == LABEL_5E else "set by the biases"):
           sii_term_kaiser(RESULTS[label]["analysis"], RESULTS[label]["best"]) for label in (LABEL_5, LABEL_5E)}

fig, axes = plt.subplots(3, 1, figsize=(9, 13), sharex=True)
axes[0].plot(k, k * term_cm / np.pi, "k-", label="CM2026 (mock 5)")
for i, (label, term) in enumerate(terms_k.items()):
    axes[0].plot(k, k * term / np.pi, "--", color=f"C{i}", label=label)
    axes[1].plot(k, term / term_cm, "--", color=f"C{i}")
    axes[2].plot(k, (term - term_cm) / sigma, "o-", ms=3, lw=0.8, color=f"C{i}")
axes[0].set(ylabel=r"$k\,\Delta P_{1D}^{\rm SiII\times SiII}/\pi$")
axes[0].legend()
axes[1].axhline(1, color="k", lw=0.5)
axes[1].set(ylabel="Kaiser / CM2026")
axes[2].axhline(0, color="k", lw=0.5)
axes[2].set(ylabel=r"(Kaiser $-$ CM2026) / $\sigma$")
axes[2].set_xlabel(r"$k_\parallel$ [s/km]")
fig.suptitle(f"z = {Z}, SiII x SiII contribution to P1D")
plt.tight_layout()
plt.show()
