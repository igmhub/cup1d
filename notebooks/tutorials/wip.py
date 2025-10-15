# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Tutorial: sample data
#
# This notebook shows an illustrative example of how to run cup1d for extracting cosmological constraints from P1D data:
#
# - Set mock data
# - Set emulator
# - Set likelihood
# - Set sampler
# - Run sample for a small number of steps
#
# All these steps are implemented in cup1d/cup1d/likelihood/samplerpipeline.py. If you are interested in running cup1d, please take a look at cup1d/scripts/sam_sim.py. That script is parallelized using MPI and includes a bookkeeper taking care of all relevant options.

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
import time, os, sys
import matplotlib.pyplot as plt

# our own modules
from lace.cosmo import camb_cosmo
from lace.emulator.emulator_manager import set_emulator
from cup1d.likelihood import lya_theory, likelihood
from cup1d.likelihood.fitter import Fitter
from cup1d.likelihood.plotter import Plotter

from cup1d.likelihood.input_pipeline import Args
from cup1d.likelihood.pipeline import Pipeline

from cup1d.p1ds.data_DESIY1 import P1D_DESIY1
from astropy.io import fits


from corner import corner

from cup1d.utils.utils import get_path_repo

from scipy.stats import chi2 as chi2_scipy
from cup1d.pipeline.set_archive import set_archive


# %%

# %%
from cup1d.plots.plots_corner import plots_chain

# %%
base = "/home/jchaves/Proyectos/projects/lya/data/out_DESI_DR1/"

variations = {
    "DESIY1_QMLE3_mpg": ["Fiducial", "DESIY1_QMLE3/global_opt/CH24_mpgcen_gpr/chain_1/"],
    "zmax": ["Data: $z \\leq 3.4$", "DESIY1_QMLE3/zmax/CH24_mpgcen_gpr/chain_1/"],
    "zmin": ["Data: $z \\geq 2.6$", "DESIY1_QMLE3/zmin/CH24_mpgcen_gpr/chain_1/"],
    
    "DESIY1_QMLE_mpg": ["Data: w/ low SNR", "DESIY1_QMLE/global_opt/CH24_mpgcen_gpr/chain_1/"],
    "DESIY1_FFT3_dir_mpg": ["Data: FFT", "DESIY1_FFT3_dir/global_opt/CH24_mpgcen_gpr/chain_1/"],
    # "DESIY1_FFT_dir_mpg": ["Data: FFT w/ low SNR", "DESIY1_FFT_dir/global_opt/CH24_mpgcen_gpr/chain_1/"],
    
    "no_emu_cov": ["Cov: w/o emu err", "DESIY1_QMLE3/no_emu_cov/CH24_mpgcen_gpr/chain_1/"],
    "no_inflate": ["Cov: w/o 5\% err", "DESIY1_QMLE3/no_inflate/CH24_mpgcen_gpr/chain_1/"],
    "no_inflate_no_emu_cov": ["Cov: w/o emu, 5\% err", "DESIY1_QMLE3/no_inflate_no_emu_cov/CH24_mpgcen_gpr/chain_1/"],
    
    "DESIY1_QMLE3_nyx": ["Model: lace-lyssa", "DESIY1_QMLE3/global_opt/CH24_nyxcen_gpr/chain_1/"],
    
    "cosmo": ["Model: $\omega_0\omega_a$CDM", "DESIY1_QMLE3/cosmo/CH24_mpgcen_gpr/chain_1/"],
    "cosmo_high": ["Model: high $\Omega_\mathrm{M}h^2$", "DESIY1_QMLE3/cosmo_high/CH24_mpgcen_gpr/chain_1/"],
    "cosmo_low": ["Model: low $\Omega_\mathrm{M}h^2$", "DESIY1_QMLE3/cosmo_low/CH24_mpgcen_gpr/chain_1/"],
    
    "more_igm": ["Model: IGM $n_z=8$", "DESIY1_QMLE3/more_igm/CH24_mpgcen_gpr/chain_1/"],
    "less_igm": ["Model: IGM $n_z=4$", "DESIY1_QMLE3/less_igm/CH24_mpgcen_gpr/chain_1/"],
    "Turner24": ["Model: $\\bar{F}\\, n_z=1$", "DESIY1_QMLE3/Turner24/CH24_mpgcen_gpr/chain_1/"],
    
    "hcd_z": ["Model: HCD $n_z=2$", "DESIY1_QMLE3/hcd_z/CH24_mpgcen_gpr/chain_1/"],
    "dlas": ["Model: only DLAs", "DESIY1_QMLE3/DLAs/CH24_mpgcen_gpr/chain_1/"],
    
    "metals_z": ["Model: metals $n_z=2$", "DESIY1_QMLE3/metals_z/CH24_mpgcen_gpr/chain_1/"],
    "metal_trad": ["Model: trad metal", "DESIY1_QMLE3/metal_trad/CH24_mpgcen_gpr/chain_1/"],
    "metal_thin": ["Model: metal thin", "DESIY1_QMLE3/metal_thin/CH24_mpgcen_gpr/chain_1/"],
    "metal_deco": ["Model: no metal decorr", "DESIY1_QMLE3/metal_deco/CH24_mpgcen_gpr/chain_1/"],
    "metal_si2": ["Model: no SiII-SiII", "DESIY1_QMLE3/metal_si2/CH24_mpgcen_gpr/chain_1/"],
    
    "no_res": ["Model: no resolution", "DESIY1_QMLE3/no_res/CH24_mpgcen_gpr/chain_1/"],
    
    # "sim_mpg_central": ["Val: mpg-central simulation", "DESIY1_QMLE3/sim_mpg_central/CH24_mpgcen_gpr/chain_1/"],
    # "sim_mpg_central_igm": ["Val: mpg-central simulation only IGM", "DESIY1_QMLE3/sim_mpg_central_igm/CH24_mpgcen_gpr/chain_1/"],
    # "sim_mpg_central_igm0": ["Val: mpg-central simulation only cosmo", "DESIY1_QMLE3/sim_mpg_central_igm0/CH24_mpgcen_gpr/chain_1/"],
    # "sim_nyx_central": ["Val: nyx-central simulation", "DESIY1_QMLE3/sim_nyx_central/CH24_mpgcen_gpr/chain_1/"],
    # "sim_sherwood": ["Val: sherwood simulation", "DESIY1_QMLE3/sim_sherwood/CH24_mpgcen_gpr/chain_1/"],
}
variations = {
    "DESIY1_QMLE3_mpg": ["Fiducial", "DESIY1_QMLE3/global_opt/CH24_mpgcen_gpr/chain_1/"],
}

for ii, var in enumerate(variations):
    folder = os.path.join(base, variations[var][1])
    plots_chain(folder)

# %%
from cup1d.utils.utils import purge_chains

# %%
help(purge_chains)

# %%
base = "/home/jchaves/Proyectos/projects/lya/data/out_DESI_DR1/"
folder = "DESIY1_QMLE3/global_opt/CH24_mpgcen_gpr/chain_1/"
# chain = np.load(base + folder + "chain.npy")
chain = 0
# blobs = np.load(base + folder + "blobs.npy")
# lnprob = np.load(base + folder + "lnprob.npy")

# mask, other = purge_chains(lnprob)

# for ii in range(other.shape[0]):
# # for ii in range(5):
#     # print(np.median(lnprob[:, other[ii]]))
#     plt.plot(lnprob[:, other[ii]])

# %%
from cup1d.plots.plots_corner import prepare_data
from cup1d.utils.various_dicts import param_dict

# %%
labels, lnprob, dat, priors, dat_Asns = prepare_data(base + folder)

# %%
dat = dat.reshape(-1, dat.shape[-1])

for ii in range(2, dat.shape[-1]):
    if ("sigT_kms" in labels[ii]) | ("gamma" in labels[ii]) | ("R_coeff_" in labels[ii]):
        pass
    else:
        dat[:, ii] = np.exp(dat[:, ii])
        if labels[ii].startswith("s_"):
            dat[:, ii] = 1 / dat[:, ii]

# %%
# Compute percentiles
p16, p50, p84 = np.percentile(dat, [16, 50, 84], axis=0)

# Asymmetric errors
err_minus = p50 - p16
err_plus  = p84 - p50


# %%
def format_value_with_error(m, ep, em):
    if ep == 0 or em == 0 or np.isnan(ep) or np.isnan(em):
        return f"${m:.2f}^{{+{ep:.2f}}}_{{-{em:.2f}}}$"
    
    # Use scientific notation if either error is < 1e-3
    if min(ep, em) < 1e-3:
        exp = int(np.floor(np.log10(abs(m)))) if m != 0 else 0
        scale = 10**exp
        m_s, ep_s, em_s = m/scale, ep/scale, em/scale
        return f"${m_s:.2f}^{{+{ep_s:.2f}}}_{{-{em_s:.2f}}}\\times10^{{{exp}}}$"
    
    # Otherwise, use two significant digits for the errors
    exp_err = int(np.floor(np.log10(min(ep, em))))
    ndec = max(-exp_err + 1, 0)
    m_str  = f"{m:.{ndec}f}"
    ep_str = f"{ep:.{ndec}f}"
    em_str = f"{em:.{ndec}f}"
    return f"${m_str}^{{+{ep_str}}}_{{-{em_str}}}$"



# %%

# # Print each row
# ii = 0
# for label, m, ep, em in zip(labels, p50, err_plus, err_minus):
#     formatted = format_value_with_error(m, ep, em)
#     if ii > 10:
#         print(f"{param_dict[label]:<45} & {formatted}\\\\")
#     ii += 1

# %%

imin = 20
n = len(labels) - imin
half = (n + 1) // 2  # split index
for ii in range(half):
    i = ii + imin
    left_label = param_dict[labels[i]]
    left_value = format_value_with_error(p50[i], err_plus[i], err_minus[i])
    if i > 1:
        if i + half < n + imin:
            right_label = param_dict[labels[i + half]]
            right_value = format_value_with_error(p50[i + half], err_plus[i + half], err_minus[i + half])
            print(f"{left_label:<10} & {left_value:<20} & {right_label:<10} & {right_value} \\\\")
        else:
            # Odd number of parameters
            print(f"{left_label:<10} & {left_value:<20} & & \\\\")

# %%

# %%
base = "/home/jchaves/Proyectos/projects/lya/data/out_DESI_DR1/"
# folder = base + "sim_mpg_central/CH24_mpgcen_gpr/chain_3/"
# folder = base + "sim_nyx_central/CH24_mpgcen_gpr/chain_1/"
# folder = base + "sim_nyx_central/CH24_mpgcen_gpr/chain_2/"
# folder = base + "sim_sherwood/CH24_mpgcen_gpr/chain_2/"

# folders = [
#     "DESIY1_QMLE3/global_opt/CH24_mpgcen_gpr/chain_1/",
#     "DESIY1_QMLE3/global_opt/CH24_nyxcen_gpr/chain_2/",
#     "DESIY1_FFT3_dir/global_opt/CH24_mpgcen_gpr/chain_3/",
#     "DESIY1_FFT_dir/global_opt/CH24_mpgcen_gpr/chain_5/",
#     "DESIY1_QMLE/global_opt/CH24_mpgcen_gpr/chain_6/",
#     "DESIY1_QMLE3/Turner24/CH24_mpgcen_gpr/chain_3/",
#     "DESIY1_QMLE3/cosmo/CH24_mpgcen_gpr/chain_4/",
#     "DESIY1_QMLE3/cosmo_high/CH24_mpgcen_gpr/chain_2/",
#     "DESIY1_QMLE3/cosmo_low/CH24_mpgcen_gpr/chain_2/",
#     "DESIY1_QMLE3/hcd_z/CH24_mpgcen_gpr/chain_4/",
#     "DESIY1_QMLE3/less_igm/CH24_mpgcen_gpr/chain_4/",
#     "DESIY1_QMLE3/more_igm/CH24_mpgcen_gpr/chain_4/",
#     "DESIY1_QMLE3/metal_deco/CH24_mpgcen_gpr/chain_4/",
#     "DESIY1_QMLE3/metal_si2/CH24_mpgcen_gpr/chain_4/",
#     "DESIY1_QMLE3/metal_thin/CH24_mpgcen_gpr/chain_3/",
#     "DESIY1_QMLE3/metal_trad/CH24_mpgcen_gpr/chain_4/",
#     "DESIY1_QMLE3/metals_z/CH24_mpgcen_gpr/chain_4/",
#     "DESIY1_QMLE3/no_emu_cov/CH24_mpgcen_gpr/chain_4/",
#     "DESIY1_QMLE3/no_inflate/CH24_mpgcen_gpr/chain_4/",
#     "DESIY1_QMLE3/no_inflate_no_emu_cov/CH24_mpgcen_gpr/chain_4/",
#     "DESIY1_QMLE3/zmax/CH24_mpgcen_gpr/chain_3/",
#     "DESIY1_QMLE3/zmin/CH24_mpgcen_gpr/chain_3/",
#     "DESIY1_QMLE3/no_res/CH24_mpgcen_gpr/chain_4/",
# ]

fid_vals = {}

table = []
corr_fid = 0.186

for ii, var in enumerate(variations):
    folder = os.path.join(base, variations[var][1])
    # print(var, np.load(folder + "lnprob.npy").shape)
    # print(var,variations[var][1])
    # plots_chain(folder)
    data = np.load(
        os.path.join(folder, "summary.npy"), allow_pickle=True
    ).item()
    delta2_star = data["delta2_star_16_50_84"][1]
    n_star = data["n_star_16_50_84"][1]
    if ii == 0:
        fid_vals["delta2_star"] = delta2_star
        fid_vals["delta2_star_2dcen"] = data["delta2_star_2dcen"]
        fid_vals["delta2_star_err"] = data["delta2_star_err"]
        fid_vals["n_star"] = n_star
        fid_vals["nstar_2dcen"] = data["nstar_2dcen"]
        fid_vals["n_star_err"] = data["n_star_err"]

    row = []
    row.append(variations[var][0])
    row.append(delta2_star - fid_vals["delta2_star"])
    top = data["delta2_star_16_50_84"][2] - data["delta2_star_16_50_84"][1]
    bot = data["delta2_star_16_50_84"][1] - data["delta2_star_16_50_84"][0]
    row.append(top)
    row.append(bot)
    row.append(n_star - fid_vals["n_star"])
    top = data["n_star_16_50_84"][2] - data["n_star_16_50_84"][1]
    bot = data["n_star_16_50_84"][1] - data["n_star_16_50_84"][0]
    row.append(top)
    row.append(bot)

    diffx = data['delta2_star_2dcen'] - fid_vals["delta2_star_2dcen"]
    errx = np.max([data["delta2_star_err"], fid_vals["delta2_star_err"]])
    diffy = data['nstar_2dcen'] - fid_vals["nstar_2dcen"]
    erry = np.max([data["n_star_err"], fid_vals["n_star_err"]])
    consist = 1/(1-corr_fid**2)*(diffx**2/errx**2 - 2*corr_fid*diffx*diffy/errx/erry + diffy**2/erry**2)
    print(var, consist)
    row.append(chi2_scipy.sf(consist, 2))

    row.append(data["chi2"])
    row.append(data["prob_chi2"])

    table.append(row)

# %%
# chi2_scipy.sf(1, 1)

# %%
# blobs = np.load(base + "DESIY1_QMLE3/global_opt/CH24_mpgcen_gpr/chain_1/blobs.npy")
# np.corrcoef(blobs["Delta2_star"].reshape(-1), blobs["n_star"].reshape(-1))[1,0]
blobs = 0
chain = 0


# %%

# %%

# %%
def format_last_column(values):
    """Format last column with trailing zeros or LaTeX scientific notation."""
    formatted = []
    for val in values:
        if abs(val) >= 1e-3:
            s = f"{val:.4f}"   # fixed 4 decimals
        else:
            coeff, exp = f"{val:.1e}".split("e")
            exp = int(exp)
            s = f"${coeff}\\times10^{{{exp}}}$"
        formatted.append(s)
    width = max(len(s) for s in formatted)
    return [f"{s:>{width}}" for s in formatted]


def format_column(values, sigfigs=2, force_decimals=True, one_decimal=False, two_decimals=False):
    formatted = []
    for val in values:
        if one_decimal:
            s = f"{val:.1f}"
        elif two_decimals:
            s = f"{val:.2f}"
        elif force_decimals:
            s = f"{val:.3f}"
        else:
            s = f"{val:.{sigfigs}g}"
        formatted.append(s)
    width = max(len(s) for s in formatted)
    return [f"{s:>{width}}" for s in formatted]




# %%
import math

def match_precision(x, xp, xm, sig=2):
    """
    Return LaTeX string "$x^{+xp}_{-xm}$" with x and errors rounded so that
    errors have `sig` significant figures.
    If x is positive, add LaTeX thin space prefix for alignment: '\;\;\,'.
    """
    err = max(abs(xp), abs(xm))
    if err == 0:
        s = f"${x:.3f}$"
    else:
        exp_err = int(math.floor(math.log10(err)))
        digits = sig - exp_err - 1
        digits = max(digits, 0)
        fmt = f"{{:.{digits}f}}"
        x_s  = fmt.format(x)
        xp_s = fmt.format(xp)
        xm_s = fmt.format(xm)
        s = f"${x_s}^{{+{xp_s}}}_{{-{xm_s}}}$"

    if x >= 0:
        s = r"$\;\;\," + s[1:]  # prepend LaTeX thin space, keep $...$
    return s


def format_last(val):
    """Scientific notation if |val| < 1e-3 (and val != 0), else 4 decimals."""
    if val == 0:
        return "0.00"
    if abs(val) < 1e-2:
        coeff, exp = f"{val:.1e}".split("e")
        exp = int(exp)
        return f"${coeff}\\times10^{{{exp}}}$"
    return f"{val:.2f}"


def make_latex_table(table, color_threshold=0.25, color="red"):
    """
    Print aligned LaTeX rows from `table`.
    Each row: [name, x1, x1p, x1m, x2, x2p, x2m, val3, val4, val5]
    - columns 2 & 3: $value^{+err}_{-err}$; positive values get '\;\;\,' padding
    - column 4 (val3) -> formatted as .2f, triggers coloring if < color_threshold
    - column 5 (val4) -> formatted as .1f
    - column 6 (val5) -> .4f or scientific if <1e-3
    """
    rows_plain = []
    for row in table:
        name = str(row[0])
        x1, x1p, x1m = row[1:4]
        x2, x2p, x2m = row[4:7]
        val3, val4, val5  = row[7:]

        p1 = match_precision(x1, x1p, x1m, sig=2)
        p2 = match_precision(x2, x2p, x2m, sig=2)
        s3 = f"{val3:.2f}"
        s4 = f"{val4:.1f}"
        s5 = format_last(val5)

        # store val3 for threshold checking
        rows_plain.append([name, p1, p2, s3, s4, s5, val3])

    # compute column widths
    ncols = 6
    col_widths = [0]*ncols
    for r in rows_plain:
        for i in range(ncols):
            col_widths[i] = max(col_widths[i], len(str(r[i])))


    for r in rows_plain:
        name, p1, p2, s3, s4, s5, val3num = r
    
        # decide if coloring is needed
        colorize = float(val3num) < color_threshold
    
        if colorize:
            # no padding
            cells = [
                f"\\textcolor{{{color}}}{{{name}}}",
                f"\\textcolor{{{color}}}{{{p1}}}",
                f"\\textcolor{{{color}}}{{{p2}}}",
                f"\\textcolor{{{color}}}{{{s3}}}",
                f"\\textcolor{{{color}}}{{{s4}}}",
                f"\\textcolor{{{color}}}{{{s5}}}",
            ]
        else:
            # pad for neat alignment
            cells = [
                str(name).ljust(col_widths[0]),
                p1.ljust(col_widths[1]),
                p2.ljust(col_widths[2]),
                s3.ljust(col_widths[3]),
                s4.ljust(col_widths[4]),
                s5.ljust(col_widths[5]),
            ]
    
        line = " & ".join(cells) + " \\\\"
        print(line)


# %%

# %%
make_latex_table(table)

# %%

# %%
emulator_label = "CH24_mpgcen_gpr"
# emulator_label = "CH24_nyxcen_gpr"


data_label = "mpg_central"
# data_label = "nyx_central"
# data_label = "nyx_seed"
# data_label = "nyx_cgan_base"
# data_label = "accel2"
# data_label = "sherwood"

if data_label == "mpg_central":
    zmin=2.25
    zmax=4.25
elif data_label == "nyx_central":
    zmin=2.2
    zmax=4.2
else:
    zmin=2.2
    zmax=4.2

true_cosmo_label = data_label
fid_cosmo_label = data_label
name_variation= "sim_" + data_label
name_variation= "sim_" + data_label + "_igm"
name_variation= "sim_" + data_label + "_igm0"
fit_type = "global_opt"

args = Args(
    data_label=data_label, 
    cov_label="DESIY1_QMLE3", 
    emulator_label=emulator_label,
    true_cosmo_label=true_cosmo_label,
    apply_smoothing=True,
    # add_noise=True,
    # seed_noise=0,
)
args.set_baseline(
    fit_type=fit_type, 
    fix_cosmo=False, 
    P1D_type="DESIY1_QMLE3",
    fid_cosmo_label=fid_cosmo_label,
    name_variation=name_variation,
    z_min=zmin,
    z_max=zmax,
    mcmc_conf="test"
)

# %%
args.nyx_training_set = "models_Nyx_Sept2025_include_Nyx_fid_rseed"

# %%
archive_mock = set_archive(training_set=args.nyx_training_set)

# %%
archive_mock = set_archive(training_set="Cabayol23")

# %%
central = archive_mock.get_testing_data("nyx_central")

# %%
seed = archive_mock.get_testing_data("nyx_seed")

# %%
mpg_central = archive_mock.get_testing_data("mpg_central")
mpg_seed = archive_mock.get_testing_data("mpg_seed")

# %%
# mpg_central[-2]

# %%
_ = (central[0]["k_Mpc"] < 2) & (central[0]["k_Mpc"] > 0.1)
mean1 = np.median(central[0]["p1d_Mpc"][_]/seed[1]["p1d_Mpc"][_])
plt.plot(central[0]["k_Mpc"][_], (central[0]["p1d_Mpc"]/seed[1]["p1d_Mpc"])[_] - mean1, label="nyx-central/nyx-seed-1")
std1 = np.std((central[0]["p1d_Mpc"]/seed[1]["p1d_Mpc"])[_] - mean1)

_ = (mpg_central[-2]["k_Mpc"] < 2) & (mpg_central[-2]["k_Mpc"] > 0.1)
mean2 = np.median(mpg_central[-2]["p1d_Mpc"][_]/mpg_seed[-2]["p1d_Mpc"][_])
plt.plot(mpg_central[-2]["k_Mpc"][_], (mpg_central[-2]["p1d_Mpc"]/mpg_seed[-2]["p1d_Mpc"])[_]-mean2, label="mpg-central/mpg-seed-1")
std2 = np.std((mpg_central[-2]["p1d_Mpc"]/mpg_seed[-2]["p1d_Mpc"])[_]-mean2)

plt.legend()
plt.axhline(0, color="k")
plt.xlim(0.1,4)
plt.ylim(-0.04,0.04)
plt.xscale("log")
plt.xlabel("k[1/Mpc]")
plt.ylabel("Relative difference")
plt.savefig("cosmic_variance_nyx_mpg.png")

# %%
87.7/29.3

# %%
std2

# %%
std1/std2

# %%
for ii, sim in archive_mock.data:
    if sim["sim_label"] == "nyx_seed":
        print(sim)
        break

# %%
# pip = Pipeline(args, archive=archive_mock)
pip = Pipeline(args)

# %%
# np.load("/home/jchaves/Proyectos/projects/lya/data/nyx/nyx_emu_cosmo_models_Nyx_Mar2025_with_CGAN_val_3axes.npy", allow_pickle=True).item()

# %%
# pip.fitter.like.data.plot_p1d()

# %%
p0 = pip.fitter.like.sampling_point_from_parameters()
pip.fitter.like.get_chi2(p0)

# %%
# args.fid_cont

# %%
for par in pip.fitter.like.free_params:
    print(par.name, par.value, par.min_value, par.max_value)

# %%
name_variation

# %%
pip.fitter.like.plot_p1d()

# %%
65.9

# %%
pip.run_minimizer(p0, restart=True)

# %%
pip.fitter.like.plot_p1d(pip.fitter.mle_cube)

# %%
2.5 vs 0.8

# %%
pip.fitter.mle_cosmo

# %%
pip.fitter.mle_cosmo

# %%
# pip.plotter.plot_igm()

# %%
# pip.run_sampler(pini=p0)

# %%
from cup1d.likelihood.cosmologies import set_cosmo

from cup1d.likelihood import CAMB_model

# %%
pip.fitter.like.data

# %%
cosmo = set_cosmo(cosmo_label="mpg_central")
# cosmo = set_cosmo(cosmo_label="nyx_central")
# cosmo = set_cosmo(cosmo_label="sherwood")
# cosmo = set_cosmo(cosmo_label="sherwood")
like_cosmo = CAMB_model.CAMBModel(np.array([3]), cosmo=cosmo)
true_cosmo = like_cosmo.get_linP_params()
true_cosmo

# %%

# %%
# file = "/home/jchaves/Proyectos/projects/lya/data/out_DESI_DR1/sim_mpg_central/chain_1/fitter_results.npy" # pedersen
# file = "/home/jchaves/Proyectos/projects/lya/data/out_DESI_DR1/sim_mpg_central/chain_2/fitter_results.npy" # cabayol
file = "/home/jchaves/Proyectos/projects/lya/data/out_DESI_DR1/sim_mpg_central/chain_3/fitter_results.npy" # cabayol sm
# file = "/home/jchaves/Proyectos/projects/lya/data/out_DESI_DR1/sim_nyx_central/chain_1/fitter_results.npy" # nyx
# file = "/home/jchaves/Proyectos/projects/lya/data/out_DESI_DR1/sim_nyx_central/chain_2/fitter_results.npy" # nyx sm
# file = "/home/jchaves/Proyectos/projects/lya/data/out_DESI_DR1/sim_sherwood/chain_1/fitter_results.npy" # sherwood sm
dat = np.load(file, allow_pickle=True).item()
dat["fitter"]

# %%

# %%
for par in dat["fitter"]["mle_cosmo"]:
    p1 = dat["fitter"]["mle_cosmo"][par]
    # p1 = pip.fitter.mle_cosmo[par]
    p2 = true_cosmo[par]
    print(par, np.round(p1, 3), np.round(p2, 3), np.round(p1-p2, 3))

# %%
p0 = dat["fitter"]["mle_cube"]

# %%
# pip.fitter.like.plot_igm()

# %%
p0 = pip.fitter.like.sampling_point_from_parameters()
pip.fitter.like.get_chi2(p0)

# %%
pip.run_minimizer(p0, restart=True)

# %%
p0 = pip.fitter.mle_cube.copy()

# %%
pip.fitter.like.plot_p1d(p0, residuals=True, plot_panels=True)

# %%
# Turner24

# %%

# args = Args(data_label="DESIY1_QMLE3", emulator_label="CH24_nyxcen_gpr")
# args = Args(data_label="DESIY1_FFT_dir", emulator_label="CH24_nyxcen_gpr")
# args.set_baseline(fit_type="andreu2", fix_cosmo=True)

# args.set_baseline(fit_type="global_all", fix_cosmo=True)

# fid 0.18697989940945323
# HCD depend z, 0.13580926057347983
# same num IGM, new fid, 0.1755366851767883

# name_variation = None
# name_variation = "no_res"
# name_variation = "Turner24"
variations = [
    "fid",
    "no_inflate",  # no increase errors for 3, 3.6, and 4
    "all_inflate",
    "cosmo",  # different fiducial cosmo
    "metal_trad",  # 2 params for metals like eBOSS
    "metal_si2",  # no SiII-SiII cont
    "metal_deco",  # no decorrelation metals
    "metal_thin",  # no desviation from optically-thin limit
    "no_res",  # no resolution correction
    "Turner24",  # mF from Turner24 with 1 free param to scale
    "more_igm",  # 8 params for IGM evolution
    "less_igm",  # 4 params for IGM evolution
    "metals_z",  # 2 params for z ev metals
    "hcd_z",  # 2 params for z ev hcd
]

# metals_z 3.89548175069871

# name_variation = variations[12]
# name_variation = "metals_z"
# name_variation = "all_inflate"
# name_variation = "Turner24"

data_label = "DESIY1_QMLE3"
# data_label = "DESIY1_FFT3_dir"
name_variation = None
# name_variation = "no_inflate"
# name_variation = "no_emu_cov"
# name_variation = "no_inflate_no_emu_cov"

# name_variation = "metal_deco"
# name_variation = "metal_si2"
# name_variation = "no_res"

args = Args(data_label=data_label, emulator_label="CH24_mpgcen_gpr")
args.set_baseline(
    fit_type="global_opt", 
    fix_cosmo=False, 
    P1D_type=data_label, 
    name_variation=name_variation, 
)

pip = Pipeline(args, out_folder=args.out_folder)


# %%
data_lab = "DESIY1_QMLE3"
fit_type = "global_opt"
emu = "mpg"
folder = "/home/jchaves/Proyectos/projects/lya/data/out_DESI_DR1/"+data_lab+"/"+fit_type+"/CH24_"+emu+"cen_gpr/chain_1/"
data = np.load(folder + "fitter_results.npy", allow_pickle=True).item()
p0 = data["fitter"]["mle_cube"]
free_params = pip.fitter.like.parameters_from_sampling_point(p0)

# %%
base = "/home/jchaves/Proyectos/projects/lya/data/out_DESI_DR1/"
folder = "DESIY1_QMLE3/global_opt/CH24_mpgcen_gpr/chain_1/"
chain = np.load(base + folder + "chain.npy")

# %%
0.0075/0.00345

# %%

# %%
0.0061/(0.5*(0.0045 + 0.0036))

# %%
zstar = 3
pip.fitter.like.plot_metal_cont_mult(chain=chain, zstar=zstar, save_directory="figs")

# %%
pip.fitter.like.plot_metal_cont_add(free_params=free_params, chain=chain, save_directory="figs")

# %%
zstar = 3
pip.fitter.like.plot_hcd_cont(p0=p0, chain=chain, zstar=zstar, save_directory="figs")

# %%

# %%

# %%

# %%
chain = chain.reshape(-1, chain.shape[-1])
p0 = np.median(chain, axis=0);

# %%
ind = np.random.permutation(np.arange(0, chain.shape[0]))[:20000]

# %%
# 
free_params = pip.fitter.like.parameters_from_sampling_point(p0)
tab_out = pip.fitter.like.plot_igm(
    free_params=free_params,
    plot_fid=True,
    plot_type="tau_sigT",
    cloud=False,
    ftsize=20,
    chain=chain[ind, :],
    save_directory="figs",
)


# %% editable=true slideshow={"slide_type": ""}
def format_asym_error(arr):
    """Format median + errors from percentiles (arr[0]=16th, arr[1]=50th, arr[2]=84th)."""
    lower = arr[1] - arr[0]
    upper = arr[2] - arr[1]
    
    # Determine number of decimal places based on the smaller error (2 significant figures)
    err = min(lower, upper)
    if err == 0:
        ndec = 2
    else:
        exp = int(np.floor(np.log10(abs(err))))
        ndec = max(-exp + 1, 0)
    
    m_str = f"{arr[1]:.{ndec}f}"
    upper_str = f"{upper:.{ndec}f}"
    lower_str = f"{lower:.{ndec}f}"
    
    return f"${m_str}^{{+{upper_str}}}_{{-{lower_str}}}$"


# %%
z, mF, T0, gamma = tab_out

# %%
print(r"\begin{tabular}{cccc}")
print(r"$z$ & $\bar{F}$ & $T_0[K]/10^4$ & $\gamma$ \\")
print(r"\hline")

for i in range(z.shape[0]):
    mF_str = format_asym_error(mF[:, i])
    T0_str = format_asym_error(T0[:, i])
    gamma_str = format_asym_error(gamma[:, i])
    print(f"{z[i]:.2f} & {mF_str} & {T0_str} & {gamma_str} \\\\")
    # print(r"\hline")

print(r"\end{tabular}")

# %%
for ii in range(len(z)):
    print(

# %%
# std_mpg = np.sqrt(np.diag(pip.fitter.like.emu_full_cov_Pk_kms)).copy()
# std_nyx = np.sqrt(np.diag(pip.fitter.like.emu_full_cov_Pk_kms)).copy()
# np.mean(std_nyx/std_mpg)

# %%
pip.fitter.like.free_param_names[26]

# %%

# %%

# %%

p0 = pip.fitter.like.sampling_point_from_parameters().copy()
free_params = pip.fitter.like.parameters_from_sampling_point(p0)
pip.fitter.like.get_chi2(p0)

# %%

# %%

# %%
# pip.fitter.like.theory.model_cont.metal_models["Si_add"].coeffs

# %%
pip.fitter.like.plot_p1d(p0)
# pip.fitter.like.plot_cov_to_pk(fname="figs/nyx_err2p1d_qmle3")

# %%
# pip.fitter.like.plot_p1d(p0, residuals=True, plot_panels=True, print_chi2=False)
pip.fitter.like.plot_p1d(p0, residuals=True, plot_panels=True, print_chi2=False, fix_cosmo=False, plot_fname="figs/residual_fid_opt_global")

# pip.fitter.like.plot_p1d(p0, residuals=True, plot_panels=True, print_chi2=False)

# %%
npoints = 0
for ii in range(len(pip.fitter.like.data.z)):
    npoints += len(pip.fitter.like.data.k_kms[ii])
npoints - len(pip.fitter.like.free_param_names)

# %%
# data_lab = "DESIY1_QMLE3"
# fit_type = "global_opt"
# emu = "mpg"
# folder = "/home/jchaves/Proyectos/projects/lya/data/out_DESI_DR1/"+data_lab+"/"+fit_type+"/CH24_"+emu+"cen_gpr/"
# data_cen = np.load(folder + "best_dircosmo.npy", allow_pickle=True).item()


# ii = 556
# type_prof = "prof_2d_deep"
# data_best = np.load(folder + type_prof + "/profile_"+str(ii)+ ".npy", allow_pickle=True).item()

# data_best.keys()

# %%
data_best["chi2"]

# %%

tar_cosmo = pip.fitter.like.apply_unblinding(data_best["blind_cosmo"])
pip.fitter.like.theory.rescale_fid_cosmo(tar_cosmo)

# %%
p0 = data_cen["mle_cube"].copy()
# p0 = data_best["mle_cube"].copy()
pip.fitter.like.get_chi2(p0)

# %%

# %%
# from cup1d.likelihood.cosmologies import set_cosmo

# cosmos = set_cosmo("mpg_central", return_all=True)

# p1 = np.zeros(len(cosmos))
# p2 = np.zeros(len(cosmos))
# for ii, key in enumerate(cosmos):
#     p1[ii] = cosmos[key]["star_params"]['Delta2_star']
#     p2[ii] = cosmos[key]["star_params"]['n_star']

# plt.scatter(p1, p2)

# print(p1.min(), p1.max())
# print(p2.min(), p2.max())

# %%

# %%

# %%

# %%

# %%

# %%

# %%
# from cup1d.nuisance.resolution_class import get_Rz_Naim

# # k_kms = np.geomspace(0.003, 0.04, 100) 
# nz = len(pip.fitter.like.data.z)
# fig, ax = plt.subplots(4, 3, figsize=(8, 6), sharex=True, sharey=True)
# ax = ax.reshape(-1)
# for jj in range(nz):
#     bias = 0.015
    
#     k_kms = pip.fitter.like.data.k_kms[jj]
#     z = pip.fitter.like.data.z[jj]
#     cont = bias * 2 * get_Rz_Naim(z)**2 * k_kms**2
#     pk = pip.fitter.like.data.Pk_kms[jj]
#     err = np.sqrt(np.diag(pip.fitter.like.cov_Pk_kms[jj]))
#     ax[jj].set_title("z=" + str(z))
#     # ax[jj].plot(k_kms, pk * cont)
#     # ax[jj].plot(k_kms, err)
#     ax[jj].plot(k_kms, pk * cont/err)
#     ax[jj].set_xscale("log")
#     ax[jj].axhline(0.5)
# plt.tight_layout()

# %%

pip.fitter.like.get_chi2(p0)

# %%
# pip.fitter.like.data.plot_p1d(fname="figs/p1d_qmle3")
# pip.fitter.like.data.plot_p1d()
# pip.fitter.like.plot_cov_to_pk(fname="figs/err2p1d_qmle3")

# %%
# cov_stat = pip.fitter.like.data.covstat_Pk_kms.copy()
# covfull_stat = pip.fitter.like.data.cov_Pk_kms.copy()

# %%
# cov_stat3 = pip.fitter.like.data.covstat_Pk_kms.copy()
# covfull_stat3 = pip.fitter.like.data.cov_Pk_kms.copy()

# %%
# res = 0
# for ii in range(len(cov_stat)):
#     y1 = covfull_stat[ii] - cov_stat[ii]
#     y2 = covfull_stat3[ii] - cov_stat3[ii]
#     res += np.mean(np.sqrt(np.diag(y1))/np.sqrt(np.diag(y2)))
# res/=len(cov_stat)
# res

# %%

# %%
677.23 - 593.67

# %%
# p0 = pip.fitter.like.sampling_point_from_parameters()
# p0[2] = 0.5
# p0[14] = 0.5
# p0 = p0min
pip.run_minimizer(p0, restart=True)

# %%
Delta2_star 0.46458
n_star -2.26928

# %%
p0 = pip.fitter.mle_cube

# %%
# p0 = pip.fitter.like.sampling_point_from_parameters()
# free_params = pip.fitter.like.parameters_from_sampling_point(p0min)
free_params = pip.fitter.like.parameters_from_sampling_point(p0)

# %%
# with resolution
Delta2_star 0.48258
n_star -2.26846
alpha_star -0.21803
prob 3.6517204717834764
chi2 703.49

# with no resolution
chi2
Delta2_star 0.50047
n_star -2.2727
alpha_star -0.21803

# %%
# pip.fitter.like.args

# %%

# pip.fitter.like.plot_p1d(residuals=True, plot_panels=True, glob_full=True, fontsize=18, chi2_nozcov=True)

# %%



plt.errorbar(z, Fmean, dFmean)
plt.errorbar(z_tu24, mF_tu24, emF_tu24)
# plt.errorbar(z, T0, dT0)
# plt.errorbar(z, gamma, dgamma)

# %%
# 
pip.fitter.like.plot_igm(free_params=free_params, plot_fid = False, plot_type="tau_sigT", cloud=False, ftsize=20, save_directory="figs")

# %%

# %%

# %%
# pip.fitter.like.plot_igm(free_params=free_params, plot_fid = False, plot_type="tau_sigT", cloud=True, ftsize=20)
# pip.fitter.like.plot_igm(cloud=True, ftsize=5)
pip.fitter.like.plot_igm(free_params=free_params, cloud=True, ftsize=5)

# %%
p0min = pip.fitter.mle_cube.copy()
pip.fitter.like.plot_p1d(p0min, residuals=True, plot_panels=True, print_chi2=False)

# %%
p0min = pip.fitter.mle_cube.copy()
pip.fitter.like.plot_p1d(p0min, residuals=True, plot_panels=True, print_chi2=False, plot_fname="figs/residual_opt_global")
# pip.fitter.like.plot_p1d(p0min, residuals=True, plot_panels=True, print_chi2=False, plot_fname=None)

# %%
np.exp(-11)

# %%
pip.fitter.mle

# %%
pip.fitter.mle

# %%
pip.fitter.mle_cosmo

# %%

# %%
fit_type = "global_opt"
data_lab = "DESIY1_QMLE3"
emu = "mpg"

folder = "/home/jchaves/Proyectos/projects/lya/data/out_DESI_DR1/"+data_lab+"/"+fit_type+"/CH24_"+emu+"cen_gpr/"
data_cen = np.load(folder + "best_dircosmo.npy", allow_pickle=True).item()
# tar_cosmo = pip.fitter.like.apply_unblinding(data_cen['mle_cosmo_cen'])
# pip.fitter.like.theory.rescale_fid_cosmo(tar_cosmo)

# %%
data_cen["mle_cosmo_cen"]

# %%
p0 = pip.fitter.like.sampling_point_from_parameters()
pip.run_minimizer(p0, restart=True)
p0min = pip.fitter.mle_cube.copy()

# %% [markdown]
# Turner24 807

# %%
772

# %%
p0min = pip.fitter.mle_cube.copy()

# %%
pip.fitter.mle

# %%
pip.fitter.mle

# %%
# p0 = data_cen["mle_cube"].copy()
# print(data_cen['best_chi2'], pip.fitter.like.get_chi2(p0min))

# %%
pip.fitter.like.plot_p1d(p0min, residuals=True, plot_panels=True, plot_fname="figs/glob_opt_qmle3", print_chi2=False)

# %%
# pip.fitter.like.data.plot_p1d(fname="figs/p1d_qmle3")
pip.fitter.like.plot_cov_to_pk(fname="figs/err2p1d_qmle3")


# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Get some data for Sec results

# %%
def mle_to_table(mle):
    rbc = (1193.28 * 0.575)/(1260.42 * 1.22)
    rb3 = (1193.28 * 0.575)/(1206.51 * 1.67)
    
    parslatex = {
        '$\\mathrm{ln}\\,f($\\mathrm{Ly}\x07lpha-\\mathrm{SiIII}$_0)$': "$\log\,f_{\lyasiii}=", 
        '$\\mathrm{ln}\\,s($\\mathrm{Ly}\x07lpha-\\mathrm{SiIII}$_0)$': "$k_{\lyasiii}=", 
        '$\\mathrm{ln}\\,f($\\mathrm{Ly}\x07lpha-\\mathrm{SiII}$_0)$': "$\log\,f_{\lyasii}=", 
        '$\\mathrm{ln}\\,s($\\mathrm{Ly}\x07lpha-\\mathrm{SiII}$_0)$': "$k_{\lyasii}=", 
        '$\\mathrm{ln}\\,f($\\mathrm{SiIIa}_\\mathrm{SiIII}$_0)$': "$r_{\siia}=", 
        '$\\mathrm{ln}\\,f($\\mathrm{SiIIb}_\\mathrm{SiIII}$_0)$': "$r_{\siisiii}=", 
        '$\\mathrm{ln}\\,f($\\mathrm{SiIIa}_\\mathrm{SiIIb}$_0)$': "$\log\,f_{\siisii}=", 
        '$\\mathrm{ln}\\,s($\\mathrm{SiIIa}_\\mathrm{SiIIb}$_0)$': "$k_{\siisii}=", 
        '$f_{\rm HCD1}_0$': "$\log\,f_\mathrm{LLS}^\mathrm{HCD}=", 
        '$f_{\rm HCD4}_0$': "$\log\,f_\mathrm{large DLA}^\mathrm{HCD}=",
    }

    res = []

    for par in mle:
        if par in parslatex:
            if parslatex[par] == "$\log\,f_{\siisii}=":
                val = np.log(np.exp(mle[par])**2 * rbc**2)
            elif parslatex[par] == "$r_{\siia}=":
                val = np.sqrt(np.exp(mle[par]))
            elif parslatex[par] == "$r_{\siisiii}=":
                val = np.exp(mle[par]) * rb3
            elif parslatex[par] == "$\log\,f_{\lyasii}=":
                val = np.log(np.exp(mle[par]) * rb3)
            elif "k_" in parslatex[par]:
                val = 1/np.exp(mle[par])
            else:
                val = mle[par]
            print(parslatex[par], np.round(val, 3))

            if "k_" in parslatex[par]:
                spar = str(np.round(val, 3)) + r"\,\mathrm{km}^{-1}\mathrm{s}$"
            elif "r_" in parslatex[par]:
                spar = str(np.round(val, 2)) + r"$"
            else:
                spar = str(np.round(val, 2)) + r"$"
            res.append(parslatex[par]+spar)
    return res


# %%
res = mle_to_table(data_cen["mle"])
print()
for ss in res:
    print(ss + ",")

# %%
print(np.exp(-1.25)*100, np.exp(-4.1)*100, (1-np.exp(-1.25)-np.exp(-4.1))*100)

# %%
np.exp(-4.13)/np.exp(-4.94)

# %%
k = np.geomspace(0.005, 0.04, 100)
k1 = 0.007
k2 = 0.004
k3 = 0.01
y1 = 2-2/(1+np.exp(-k/k1))
ind = np.argwhere(y1<0.1)[0,0]
print(np.round(k[ind], 3))
y2 = 2-2/(1+np.exp(-k/k2))
ind = np.argwhere(y2<0.1)[0,0]
print(np.round(k[ind], 3))
y3 = np.exp(-k**2/k3**2)
ind = np.argwhere(y3<0.1)[0,0]
print(np.round(k[ind], 3))
plt.plot(k, y1)
plt.plot(k, y2)
plt.plot(k, y3)
plt.yscale("log")
plt.ylim(0.1)

# %% [markdown]
# ### Run variations

# %%
var_deg = 657

results_var = {
    "fiducial":   np.array([ 785.4, 0.435, -2.283]),
    "HCD":        np.array([ 844.907, 0.54528, -2.33871]),
    "metal_trad": np.array([1985.443, 0.47164, -2.29145]),
    "metal_si2":  np.array([ 884.533, 0.53581, -2.28978]),
    "metal_deco": np.array([1805.704, 0.36753, -2.28083]),
    "metal_thin": np.array([ 863.954, 0.41014, -2.28247]),
    "cosmo":      np.array([ 785.870, 0.42266, -2.28021]),
    "cov":        np.array([ 649.170, 0.42287, -2.28212]),
}
ndeg = {
    "fiducial": var_deg,
    "HCD": var_deg+2,
    "metal_trad": var_deg+6,
    "metal_si2": var_deg+2,
    "metal_deco": var_deg+2,
    "metal_thin": var_deg+2,
    "cosmo": var_deg,
    "cov": var_deg,
}

err = np.array([0.036, 0.014])


for key in results_var:
    print()
    diffp = results_var[key][1:] - results_var["fiducial"][1:]
    print(key, np.round(diffp, 2))
    chi2 = results_var[key][0]
    prob = chi2_scipy.sf(chi2, ndeg[key]) * 100
    consist = np.sum(diffp**2/err**2)
    prob_var = chi2_scipy.sf(consist, 2) * 100
    print(np.round(prob_var, 1), np.round(chi2, 0), f'{prob:.1e}')


# %%
fit_type = "global_opt"
data_lab = "DESIY1_QMLE3"
emu = "mpg"
name_variation = None         # chi2  785.471, Delta2_star 0.42385, n_star -2.28224
# name_variation = "HCD"        # chi2  844.907, Delta2_star 0.54528, n_star -2.33871
# name_variation = "metal_trad" # chi2 1905.443, Delta2_star 0.47164, n_star -2.29145
# name_variation = "metal_si2"  # chi2  884.533, Delta2_star 0.53581, n_star -2.28978
# name_variation = "metal_deco" # chi2 1805.704, Delta2_star 0.36753, n_star -2.28083
# name_variation = "metal_thin" # chi2  863.954, Delta2_star 0.41014, n_star -2.28247
# name_variation = "cosmo"      # chi2  649.170, Delta2_star 0.42287, n_star -2.28212

args = Args(data_label=data_lab, emulator_label="CH24_"+emu+"cen_gpr")
args.set_baseline(fit_type=fit_type, fix_cosmo=False, name_variation=name_variation)
pip = Pipeline(args)

# name_variation = "cov"        # chi2  785.471, Delta2_star 0.42385, n_star -2.28224
if name_variation == "cov":
    pip.fitter.like.full_icov_Pk_kms /= 1.1**2

folder = "/home/jchaves/Proyectos/projects/lya/data/out_DESI_DR1/"+data_lab+"/"+fit_type+"/CH24_"+emu+"cen_gpr/"
data_cen = np.load(folder + "best_dircosmo.npy", allow_pickle=True).item()
p0 = pip.fitter.like.sampling_point_from_parameters()
print(data_cen['best_chi2'], pip.fitter.like.get_chi2(p0))
print(len(p0))

# %%
# pip.fitter.like.icov_Pk_kms

# %%
pip.fitter.like.get_chi2(p0)

# %%
pip.run_minimizer(p0, restart=True)

# %%

# %%

# %%

# %%

# %%

# %% [markdown]
# # OLD

# %% [markdown]
# ### Set emulator

# %% [markdown]
# #### Set either mock data or real data

# %%
# for forecast, just start label of observational data with mock
# choose_forecast = True 
choose_forecast = False
# to analyze data from simulations
choose_mock = False
# choose_mock = True
# to analyze data from observations
choose_data = False
# to analyze data from mock challenge
choose_challenge = False
# to analyze data from desiy1
choose_desiy1 = True

if choose_forecast:
    args.data_label_hires = None
    # for forecast, just start label of observational data with mock
    # args.data_label = "mock_Chabanier2019"
    # args.data_label="mock_Karacayli2024"
    # args.data_label_hires = "mock_Karacayli2022"
    args.data_label="mock_challenge_DESIY1"
    # args.p1d_fname="/home/jchaves/Proyectos/projects/lya/data/cup1d/obs/desi_y1_baseline_p1d_sb1subt_qmle_power_estimate.fits"
    # args.p1d_fname="/home/jchaves/Proyectos/projects/lya/data/cup1d/obs/p1d_fft_y1_measurement_kms_v3.fits"
    version = "9fx"
    folder = "/home/jchaves/Proyectos/projects/lya/data/mock_challenge/MockChallengeSnapshot/mockchallenge-0."+version+"/"
    fname = "mockchallenge-0."+version+"_nonoise_fiducial.fits.gz"
    args.p1d_fname=folder + fname

    # # you need to provide true cosmology, IGM history, and contaminants
    # true_cosmo = set_cosmo(cosmo_label="nyx_central")
    # args.true_igm_label="nyx_central"
    # true_sim = "nyx_seed"
    true_sim = "nyx_seed_val"
    # true_sim = "sherwood"
    # true_sim = "accel2"
    # true_sim = "nyx_central"
    # true_sim = "mpg_central"
    # true_sim_cosmo = "mpg_central"
    args.mF_model_type="chunks"
    true_cosmo = set_cosmo(cosmo_label=true_sim)
    args.true_label_mF=true_sim
    args.true_label_T=true_sim
    
    # true_sim = "nyx_central"
    args.true_label_kF=true_sim
    
    # true_sim_cosmo = "Planck18_low"
    # args.true_label_kF="kF_both"
    
    # true_cosmo = set_cosmo(cosmo_label="mpg_22")
    # true_cosmo = set_cosmo(cosmo_label="Planck18")
    # args.true_igm_label="nyx_central"
    # args.true_igm_label="mpg_22"
    # from -11 to -4
    # args.true_SiIII=[[0, 0], [-10, -10]]
    # args.true_SiII=[[0, 0], [-10, -10]]
    # # from -7 to 0
    # args.true_HCD=[0, -6]
    # # from -5 to 2
    # args.true_SN=[0, -4]
    # # from -5 to 1.5
    # args.true_AGN=[0, -5]
    args.z_min = 2.1
    args.z_max = 4.3
elif choose_mock:    
    # true_cosmo=None
    # to analyze data from simulations
    true_sim = "nyx_central"
    # true_sim = "nyx_seed"
    # true_sim = "nyx_central"
    # args.data_label = "mock_DESIY1"
    args.true_cosmo_label=true_sim
    args.true_label_mF=true_sim
    # args.mF_model_type="pivot"
    args.mF_model_type="chunks"
    args.true_label_T=true_sim
    args.true_label_kF=true_sim
    # args.data_label="nyx_seed"
    # args.data_label_hires="mpg_central"
    args.data_label_hires = None
    # args.apply_smoothing=True
    args.apply_smoothing=False

    # args.cov_label = "DESIY1"
    version = "9fx"
    folder = "/home/jchaves/Proyectos/projects/lya/data/mock_challenge/MockChallengeSnapshot/mockchallenge-0."+version+"/"
    fname = "mockchallenge-0."+version+"_nonoise_fiducial.fits.gz"
    # fname = "mockchallenge-0."+version+"_nonoise_ACCEL2_6144_160.fits.gz"
    # fname = "mockchallenge-0."+version+"_nonoise_CGAN_4096_base.fits.gz"
    # fname = "mockchallenge-0."+version+"_nonoise_Sherwood_2048_40.fits.gz"
    args.cov_fname = folder + fname
    # args.p1d_fname = folder + fname
    # args.p1d_fname = "/home/jchaves/Proyectos/projects/lya/data/cup1d/obs/p1d_fft_y1_measurement_kms_v3.fits"

    # provide cosmology only to cull the data
    # args.true_cosmo_label="mpg_central"
    # args.true_cosmo_label="nyx_central"
    true_cosmo = set_cosmo(cosmo_label=true_sim)
    # args.true_cosmo_label="nyx_seed"

    # you may provide contaminants
    # from 1 to 6, -11 to -4
    # args.true_SiIII=[[0, 0], [2, -10]]
    # args.true_SiII=[[0, 0], [2, -10]]
    # # from -5 to 0
    # args.true_HCD=[0, -4]
    # # from -5 to 2
    # args.true_SN=[0, -4]
    # args.true_AGN=[0, -5]
    args.z_max=4.3
elif choose_data:    
    true_cosmo=None
    args.data_label = "Chabanier2019"
    # args.data_label="Karacayli2024"
    args.data_label_hires = "Karacayli2022"
    args.z_max = 3.9
elif choose_challenge:
    args.data_label = "challenge_DESIY1"
    # version = "1.1qh"
    # version = "1.9fsh"
    version = "1.10qsh"
    folder = "/home/jchaves/Proyectos/projects/lya/data/mock_challenge/MockChallengeSnapshot/mockchallenge-"+version+"/"
    fname = "mockchallenge-"+version+"_nonoise_fiducial.fits.gz"
    # fname = "mockchallenge-"+version+"_fsiiii1.3e-03_fsiii1.1e-03_nonoise_fiducial.fits.gz"
    # fname = "mockchallenge-"+version+"_noise-42-0_fiducial.fits.gz"
    # fname = "mockchallenge-0."+version+"_nonoise_bar_ic_grid_3.fits.gz"
    # fname = "mockchallenge-0."+version+"_noise-42-0_fiducial.fits.gz"
    # fname = "mockchallenge-0."+version+"_nonoise_ACCEL2_6144_160.fits.gz"
    # fname = "mockchallenge-0."+version+"_nonoise_CGAN_4096_base.fits.gz"
    # fname = "mockchallenge-0."+version+"_nonoise_Sherwood_2048_40.fits.gz"
    args.p1d_fname = folder + fname
    if "fiducial" in args.p1d_fname:
        true_sim_label = "nyx_central"
        args.true_label_mF=true_sim_label
        args.true_label_T=true_sim_label
        args.true_label_kF=true_sim_label
    elif "CGAN" in args.p1d_fname:
        true_sim_label = "nyx_seed"        
        args.true_label_mF=true_sim_label
        args.true_label_T=true_sim_label
        args.true_label_kF=true_sim_label
    elif "grid_3" in args.p1d_fname:
        true_sim_label = "nyx_3"
        args.true_label_mF=true_sim_label
        args.true_label_T=true_sim_label
        args.true_label_kF=true_sim_label
    elif "Sherwood_2048_40" in args.p1d_fname:
        true_sim_label = "Sherwood_2048_40"
        args.true_label_mF=true_sim_label
        args.true_label_T="nyx_central"
        args.true_label_kF="nyx_central"
    elif "ACCEL2_6144_160" in args.p1d_fname:
        true_sim_label = "ACCEL2_6144_160"
        args.true_label_mF=true_sim_label
        args.true_label_T=true_sim_label
        args.true_label_kF="nyx_central"
    else:
        true_sim_label = None
    # true_sim_label = "nyx_central"

    true_cosmo = set_cosmo(cosmo_label=true_sim_label)
    
    args.z_min = 2.1
    # args.z_max = 4.3
    args.z_max = 2.7
    # args.z_min = 2.8
    # args.z_max = 3.2
elif choose_desiy1:
    true_cosmo = None
    args.true_igm_label= None
    args.data_label = "DESIY1"
    # args.cov_syst_type = "xred"
    # args.cov_syst_type = "fid"
    args.cov_syst_type = "red"
    folder = "/home/jchaves/Proyectos/projects/lya/data/DESI-DR1/"
    # in NERSC
    # /global/cfs/cdirs/desicollab/science/lya/y1-p1d/iron-baseline/qmle_measurement/DataProducts/
    # QMLE /global/cfs/cdirs/desicollab/users/naimgk/my-reductions/data/iron-v3/DataProducts/desi_y1_baseline_p1d_sb1subt_qmle_power_estimate.fits
    # FFT /global/cfs/cdirs/desi/science/lya/y1-p1d/fft_measurement/v0/plots/baseline/notebook/measurement/p1d_fft_y1_measurement_kms.fits
    
    # args.p1d_fname=folder + "/qmle_measurement/DataProducts/v3/desi_y1_baseline_p1d_sb1subt_qmle_power_estimate_contcorr_v3.fits"
    args.p1d_fname= folder + "/qmle_measurement/DataProducts/v3/desi_y1_snr3_p1d_sb1subt_qmle_power_estimate_contcorr_v3.fits"
    
    # args.p1d_fname = folder + "/fft_measurement/p1d_fft_y1_measurement_kms_v7_direct_metal_subtraction.fits"
    
    args.z_min = 2.1
    # args.z_min = 3.5
    args.z_max = 4.3
    # args.z_max = 2.7
    # args.z_max = 3.3
    # args.z_max = 2.9

# you do not need to provide the archive for obs data 
data = {"P1Ds": None, "extra_P1Ds": None}

data["P1Ds"] = set_P1D(
    args,
    archive=archive,
    true_cosmo=true_cosmo,
    emulator=emulator,
    cull_data=False
)
if args.data_label_hires is not None:
    data["extra_P1Ds"] = set_P1D(
        args,
        archive=archive,
        true_cosmo=true_cosmo,
        emulator=emulator,
        cull_data=False
    )

# %%

# %% [markdown]
# ## Optimize parameters of pipeline with z at a time fits 

# %%
from cup1d.optimize.baseline_ztime import run_grow_model_atz

# %%
# always run with zs starting with 2.2, need to change
zs = np.arange(2.2, 4.4, 0.2)
# zs = np.arange(2.2, 2.4, 0.2)
folder = "/home/jchaves/Proyectos/projects/lya/cup1d/notebooks/tutorials/qmle3_lpo/"
select_props = run_grow_model_atz(folder, zs, verbose=False)

for key in select_props:
    print(np.round(select_props[key]["z"], 2), np.sort(select_props[key]["name"]))

# %% [markdown]
# ## Run a z at a time fit

# %% [markdown]
# #### Same baseline for all z

# %%
# args.set_baseline(fit_type="all", fix_cosmo=False)
# args.set_baseline(fit_type="full")

args = Args(emulator_label="CH24_mpgcen_gpr", training_set="Cabayol23")
# args.set_baseline(fit_type="at_a_time_igm")
args.set_baseline(fit_type="at_a_time_orig")

like = set_like(
    data["P1Ds"],
    emulator,
    args,
    data_hires=data["extra_P1Ds"],
)

f_space_len = 14
s_space_len = 5
for p in like.free_params:
    print(
        p.name, (f_space_len-len(p.name)) * " ", "\t", 
        np.round(p.value, 3), (s_space_len-len(str(np.round(p.value, 3)))) * " ", '\t', 
        np.round(p.min_value, 3), (s_space_len-len(str(np.round(p.min_value, 3)))) * " ", '\t', 
        np.round(p.max_value, 3), (s_space_len-len(str(np.round(p.max_value, 3)))) * " ", '\t', 
        p.Gauss_priors_width
    )

print()

fitter = Fitter(
    like=like,
    rootdir=output_dir,
    nburnin=args.n_burn_in,
    nsteps=args.n_steps,
    parallel=args.parallel,
    explore=args.explore,
    fix_cosmology=args.fix_cosmo,
)

# %%

out_mle = []
out_mle_cube = []
out_chi2 = []
# for ii in range(len(like.data.z)): 
# for ii in range(1): 
for ii in range(7, 8): 
# for ii in range(2,3): 
# for ii in range(9, 10): 
# for ii in range(2, 3): 
    zmask = np.array([like.data.z[ii]])
    
    print()
    print(ii, like.data.z[ii])
    p0 = np.array(list(like.fid["fit_cube"].values()))
    # p0 = np.array([0.5, 0.8])
    # fitter.run_minimizer(log_func_minimize=fitter.like.get_chi2, p0=p0_arr[ii], zmask=zmask, restart=True)
    fitter.run_minimizer(log_func_minimize=fitter.like.minus_log_prob, p0=p0, zmask=zmask, restart=True, nsamples=4)
    # fitter.run_minimizer(log_func_minimize=fitter.like.get_chi2, zmask=zmask, restart=True)
    out_mle.append(fitter.mle)
    out_mle_cube.append(fitter.mle_cube)
    out_chi2.append(fitter.mle_chi2)
    # plotter = Plotter(fitter, zmask=zmask, save_directory="mpg_baseline/"+str(ii))
    # plotter.plots_minimizer()

# %%

ndeg_all = 0
props = []
chi2_all = 0
for ii in range(len(out_chi2)):
    ndeg = len(like.data.k_kms[ii]) - len(out_mle_cube[ii])
    prob = chi2_scipy.sf(out_chi2[ii], ndeg)
    print(like.data.z[ii], '&', np.round(out_chi2[ii], 2), '&', ndeg, '&', np.round(prob*100, 2), '\\\\')
    ndeg_all += ndeg
    chi2_all += out_chi2[ii]
    props.append(prob)
prob = chi2_scipy.sf(chi2_all, ndeg_all)
print()
print("All", '&', np.round(chi2_all, 2), '&', ndeg_all, '&', np.round(prob*100, 2), '\\\\')
prob

# %%

# diru = "igm_1z_snr3_nocosmo"
# diru = "orig_1z_snr3_nocosmo"
diru = None
plotter = Plotter(fitter, save_directory=diru, zmask=zmask)
plotter.plot_p1d(values=out_mle_cube, plot_panels=True, residuals=True, z_at_time=True)

# %%

plotter.plot_illustrate_contaminants_cum(out_mle_cube[0].copy(), np.array([3.6]))

# %%

# plotter.plot_p1d(residuals=True, zmask=zmask)
# plotter.plot_illustrate_contaminants(out_mle_cube[0].copy(), [2.2], lines_use=lines_use)
# plotter.plot_illustrate_contaminants(out_mle_cube[0].copy(), [2.4], lines_use=lines_use)
# plotter.plot_illustrate_contaminants_each(out_mle_cube[0].copy(), np.array([2.2]))
# plotter.plot_illustrate_contaminants(test, [2.4], lines_use=lines_use)

# %%
ii = 0

zmask = np.array([data["P1Ds"].z[ii]])

args = Args(emulator_label="CH24_mpgcen_gpr", training_set="Cabayol23")
args.set_baseline(ztar=data["P1Ds"].z[ii], fit_type="at_a_time")

like = set_like(
    data["P1Ds"],
    emulator,
    args,
    data_hires=data["extra_P1Ds"],
)

print()

f_space_len = 14
s_space_len = 5
for p in like.free_params:
    
    # if p.name[:-2] == "HCD_damp4":
    #     if p.name[:-2] in vals_modify:
    #         p.value = vals_modify[p.name[:-2]][ii]
    #         p.min_value = p.value - 1e-3
    #         p.max_value = p.value + 1e-3
        
    print(
        p.name, (f_space_len-len(p.name)) * " ", "\t", 
        np.round(p.value, 3), (s_space_len-len(str(np.round(p.value, 3)))) * " ", '\t', 
        np.round(p.min_value, 3), (s_space_len-len(str(np.round(p.min_value, 3)))) * " ", '\t', 
        np.round(p.max_value, 3), (s_space_len-len(str(np.round(p.max_value, 3)))) * " ", '\t', 
        p.Gauss_priors_width
    )

# %% [markdown]
# #### Different baseline as a function of z

# %%

out_mle = []
out_mle_cube = []
out_chi2 = []
for ii in range(len(data["P1Ds"].z)): 
# for ii in range(1): 
# for ii in range(10, 11): 
# for ii in range(1): 
# for ii in range(2,3): 
# for ii in range(7, 8): 
# for ii in range(2, 3): 
    zmask = np.array([data["P1Ds"].z[ii]])

    args = Args(emulator_label="CH24_mpgcen_gpr", training_set="Cabayol23")
    args.set_baseline(ztar=data["P1Ds"].z[ii], fit_type="at_a_time")

    like = set_like(
        data["P1Ds"],
        emulator,
        args,
        data_hires=data["extra_P1Ds"],
    )
    
    print()
    
    f_space_len = 14
    s_space_len = 5
    for p in like.free_params:
        
        # if p.name[:-2] == "HCD_damp4":
        #     if p.name[:-2] in vals_modify:
        #         p.value = vals_modify[p.name[:-2]][ii]
        #         p.min_value = p.value - 1e-3
        #         p.max_value = p.value + 1e-3
            
        print(
            p.name, (f_space_len-len(p.name)) * " ", "\t", 
            np.round(p.value, 3), (s_space_len-len(str(np.round(p.value, 3)))) * " ", '\t', 
            np.round(p.min_value, 3), (s_space_len-len(str(np.round(p.min_value, 3)))) * " ", '\t', 
            np.round(p.max_value, 3), (s_space_len-len(str(np.round(p.max_value, 3)))) * " ", '\t', 
            p.Gauss_priors_width
        )

    
    print()
    
    fitter = Fitter(
        like=like,
        rootdir=output_dir,
        nburnin=args.n_burn_in,
        nsteps=args.n_steps,
        parallel=args.parallel,
        explore=args.explore,
        fix_cosmology=args.fix_cosmo,
    )
    
    print(ii, like.data.z[ii])
    p0 = np.array(list(like.fid["fit_cube"].values()))
    # fitter.run_minimizer(log_func_minimize=fitter.like.get_chi2, p0=p0_arr[ii], zmask=zmask, restart=True)
    fitter.run_minimizer(log_func_minimize=fitter.like.minus_log_prob, p0=p0, zmask=zmask, restart=True, nsamples=6)
    # fitter.run_minimizer(log_func_minimize=fitter.like.get_chi2, zmask=zmask, restart=True)
    out_mle.append(fitter.mle)
    out_mle_cube.append(fitter.mle_cube)
    out_chi2.append(fitter.mle_chi2)
    # plotter = Plotter(fitter, zmask=zmask, save_directory="mpg_baseline/"+str(ii))
    # plotter.plots_minimizer()

# %%
print_results(like, out_chi2, out_mle_cube)

# %%
print_results(like, out_chi2, out_mle_cube)

# %%
# xjulia
args = Args(data_label="DESIY1_QMLE3", emulator_label="CH24_mpgcen_gpr")
args.set_baseline(fit_type="at_a_time_global", fix_cosmo=True)
pip = Pipeline(args, out_folder=None)

# %%

out_mle = []
out_mle_cube = []
out_chi2 = []
out_pnames = []
for ii in range(len(pip.fitter.like.data.z)): 
# for ii in range(1): 
# for ii in range(10, 11): 
# for ii in range(1): 
# for ii in range(3, 4): 
# for ii in range(7, 8): 
# for ii in range(2, 3): 
    zmask = np.array([pip.fitter.like.data.z[ii]])

    pip = Pipeline(args, out_folder=None)
    
    print()
    
    f_space_len = 14
    s_space_len = 5
    for p in pip.fitter.like.free_params:
        
        # if p.name[:-2] == "HCD_damp4":
        #     if p.name[:-2] in vals_modify:
        #         p.value = vals_modify[p.name[:-2]][ii]
        #         p.min_value = p.value - 1e-3
        #         p.max_value = p.value + 1e-3
            
        print(
            p.name, (f_space_len-len(p.name)) * " ", "\t", 
            np.round(p.value, 3), (s_space_len-len(str(np.round(p.value, 3)))) * " ", '\t', 
            np.round(p.min_value, 3), (s_space_len-len(str(np.round(p.min_value, 3)))) * " ", '\t', 
            np.round(p.max_value, 3), (s_space_len-len(str(np.round(p.max_value, 3)))) * " ", '\t', 
            p.Gauss_priors_width
        )

    
    print()
    
    print(ii, zmask)
    p0 = np.array(list(pip.fitter.like.fid["fit_cube"].values()))
    # fitter.run_minimizer(log_func_minimize=fitter.like.get_chi2, p0=p0_arr[ii], zmask=zmask, restart=True)
    pip.fitter.run_minimizer(log_func_minimize=pip.fitter.like.minus_log_prob, p0=p0, zmask=zmask, restart=True)
    # fitter.run_minimizer(log_func_minimize=fitter.like.get_chi2, zmask=zmask, restart=True)
    out_pnames.append(pip.fitter.like.free_param_names)
    out_mle.append(pip.fitter.mle)
    out_mle_cube.append(pip.fitter.mle_cube)
    out_chi2.append(pip.fitter.mle_chi2)
    # plotter = Plotter(fitter, zmask=zmask, save_directory="mpg_baseline/"+str(ii))
    # plotter.plots_minimizer()

# %%

# %%
dir_out = {
    "z":pip.fitter.like.data.z,
    "pnames":out_pnames,
    "mle_cube":out_mle_cube,
    "mle":out_mle,
    "chi2":out_chi2,
}
np.save("ics/at_a_time_andreu2.npy", dir_out)

# np.array(list(pip.fitter.mle.values()))[:-3]

# %%

# %%
mF= pip.fitter.like.theory.model_igm.models["F_model"].get_mean_flux(2.8)
kms = pip.fitter.like.data.k_kms[-1]

# pip.fitter.mle_cube[-4] = 2.16368962e-01 
# pip.fitter.mle_cube[-3] = 2.93090546e-01
# params = pip.fitter.like.parameters_from_sampling_point(pip.fitter.mle_cube)
# res = pip.fitter.like.theory.model_cont.metal_models["Si_mult"].get_contamination(np.array([2.8]), [kms], [mF], like_params=params)
# plt.plot(kms, res[0])

# pip.fitter.mle_cube[-4] = 0.8
# pip.fitter.mle_cube[-3] = 0.8
# params = pip.fitter.like.parameters_from_sampling_point(pip.fitter.mle_cube)
# res = pip.fitter.like.theory.model_cont.metal_models["Si_mult"].get_contamination(np.array([2.8]), [kms], [mF], like_params=params)
# plt.plot(kms, res[0])


pip.fitter.mle_cube[-2] = 9.31404223e-01
pip.fitter.mle_cube[-1] = 7.50958585e-01
# pip.fitter.mle_cube[-1] = 0.5
params = pip.fitter.like.parameters_from_sampling_point(pip.fitter.mle_cube)
res1 = pip.fitter.like.theory.model_cont.hcd_model.get_contamination(np.array([2.8]), [kms], like_params=params)
plt.plot(kms, res1)

pip.fitter.mle_cube[-2] = 1
# pip.fitter.mle_cube[-1] = 1
params = pip.fitter.like.parameters_from_sampling_point(pip.fitter.mle_cube)
res2 = pip.fitter.like.theory.model_cont.hcd_model.get_contamination(np.array([2.8]), [kms], like_params=params)
plt.plot(kms, res2)


# %%
res2 = res.copy()

# %%
np.log(0.00051312)
# np.exp(2.5)

# %%
from cup1d.optimize.show_results import print_results
print_results(pip.fitter.like, out_chi2, out_mle_cube)

# %%
from cup1d.optimize.show_results import print_results
print_results(pip.fitter.like, out_chi2, out_mle_cube)

# %%
from cup1d.optimize.show_results import reformat_cube

# %%
out_mle_cube_reformat = reformat_cube(args, data, emulator, out_mle_cube)

# %%
ii = 0
args.set_baseline(ztar=data["P1Ds"].z[ii], fit_type="at_a_time")
like1 = set_like(
    data["P1Ds"],
    emulator,
    args,
    data_hires=data["extra_P1Ds"],
)

out_mle_cube_reformat = []
for ii in range(len(data["P1Ds"].z)):
    args.set_baseline(ztar=data["P1Ds"].z[ii], fit_type="at_a_time")
    like2 = set_like(
        data["P1Ds"],
        emulator,
        args,
        data_hires=data["extra_P1Ds"],
    )
    _cube = np.zeros(len(all_props))
    for jj, prop in enumerate(like1.free_param_names):
        if prop in like2.free_param_names:
            ind = np.argwhere(prop == np.array(like2.free_param_names))[0,0]
            _cube[jj] = out_mle_cube[ii][ind]
    out_mle_cube_reformat.append(np.array(_cube))

# %%
diru = "first_1z_snr3_nocosmo"
# diru = None

args = Args(emulator_label="CH24_mpgcen_gpr", training_set="Cabayol23")
ii = 0
args.set_baseline(ztar=data["P1Ds"].z[ii], fit_type="at_a_time")

like = set_like(
    data["P1Ds"],
    emulator,
    args,
    data_hires=data["extra_P1Ds"],
)

fitter = Fitter(
    like=like,
    rootdir=output_dir,
    nburnin=args.n_burn_in,
    nsteps=args.n_steps,
    parallel=args.parallel,
    explore=args.explore,
    fix_cosmology=args.fix_cosmo,
)
fitter.mle_cube = out_mle_cube[ii]

plotter = Plotter(fitter, save_directory=diru, zmask=zmask)
plotter.plot_p1d(values=out_mle_cube_reformat, plot_panels=True, residuals=True, z_at_time=True)

# %%
for iz in range(len(like.data.z)):
    args.set_baseline(ztar=data["P1Ds"].z[iz], fit_type="at_a_time")
    like = set_like(
        data["P1Ds"],
        emulator,
        args,
        data_hires=data["extra_P1Ds"],
    )
    fitter = Fitter(
        like=like,
        rootdir=output_dir,
        nburnin=args.n_burn_in,
        nsteps=args.n_steps,
        parallel=args.parallel,
        explore=args.explore,
        fix_cosmology=args.fix_cosmo,
    )
    fitter.mle_cube = out_mle_cube[iz]
    
    plotter = Plotter(fitter, save_directory=diru, zmask=[like.data.z[iz]])
    plotter.plot_illustrate_contaminants_cum(out_mle_cube[iz].copy(), np.array([like.data.z[iz]]))

# %%
dir_out = {
    "mle_cube":out_mle_cube,
    "mle":out_mle,
    "chi2":out_chi2,
    "mle_cube_reformat":out_mle_cube_reformat,
}
np.save("first_1z_snr3_nocosmo/res.npy", dir_out)

# %%
dir_out = np.load("first_1z_snr3_nocosmo/res.npy", allow_pickle=True).item()
out_mle_cube = dir_out["mle_cube"]
out_mle_cube_reformat = dir_out["mle_cube_reformat"]

# %%

# %%

# %%

# %%
args = Args(emulator_label="CH24_mpgcen_gpr", training_set="Cabayol23")
ii = 0
args.set_baseline(ztar=data["P1Ds"].z[ii], fit_type="at_a_time")

like = set_like(
    data["P1Ds"],
    emulator,
    args,
    data_hires=data["extra_P1Ds"],
)

fitter = Fitter(
    like=like,
    rootdir=output_dir,
    nburnin=args.n_burn_in,
    nsteps=args.n_steps,
    parallel=args.parallel,
    explore=args.explore,
    fix_cosmology=args.fix_cosmo,
)

mask = np.arange(11)

like_params = []
for mle_cube in out_mle_cube_reformat:
    like_params.append(fitter.like.parameters_from_sampling_point(mle_cube))

fold0 = "/home/jchaves/Proyectos/projects/lya/cup1d/notebooks/tutorials/first_1z_snr3_nocosmo/"
folder = fold0 + "taueff"
oFmodel, ocFmodel = fitter.like.theory.model_igm.models["F_model"].plot_parameters(data["P1Ds"].z[mask], like_params, folder=folder)
folder = fold0 + "sigT"
oTmodel, ocTmodel = fitter.like.theory.model_igm.models["T_model"].plot_parameters(data["P1Ds"].z[mask], like_params, folder=folder)
folder = fold0 + "Simult"
oSimult, ocSimult = fitter.like.theory.model_cont.metal_models["Si_mult"].plot_parameters(data["P1Ds"].z[mask], like_params, folder=folder)
folder = fold0 + "Siadd"
oSiadd, ocSiadd = fitter.like.theory.model_cont.metal_models["Si_add"].plot_parameters(data["P1Ds"].z[mask], like_params, folder=folder)
folder = fold0 + "HCD"
oHCD, ocHCD = fitter.like.theory.model_cont.hcd_model.plot_parameters(data["P1Ds"].z[mask], like_params, folder=folder)

models = [ocFmodel, ocTmodel, ocSimult, ocSiadd, ocHCD]
param_attime_all = {}
for mod in models:
    for key in mod:
        param_attime_all[key] = mod[key]

# %%
fitter.like.theory.model_igm.models["F_model"].list_coeffs

# %%
np.save("first_1z_snr3_nocosmo/fit_baseline_param_attime.npy", param_attime_all)

# %%

# %%

# %%
# plotter.plot_p1d(zmask=zmask)
# plotter.plot_metal_cont(smooth_k=False, plot_data=True, zrange=[2.3, 2.5], plot_panels=False)
# plotter.plot_metal_cont(smooth_k=False, plot_data=True, zrange=[2.9, 3.1], plot_panels=False)

# %% [markdown]
# ### Plot evolution of parameters with z

# %%
from cup1d.optimize.plot_params_ztime import plot_z_at_time_params

weak1_priors = plot_z_at_time_params(fitter, out_mle)

# weak2_priors = plot_z_at_time_params(fitter, out_mle)

# %%
np.log10(10**-1.5 / 5)

# %%
np.save("first_1z_snr3_nocosmo/weak_priors.npy", weak1_priors)

# %% [markdown]
# #### Redo 1 z at time fits using weak_priors

# %%
weak_priors = weak1_priors.copy()

p0 = np.array(list(like.fid["fit_cube"].values()))
out_mle = []
out_mle_cube = []
out_chi2 = []
list_fix = ["tau_eff_0", "sigT_kms_0", "gamma_0", "kF_kms_0"]

for ii in range(len(like.data.z)): 
# for ii in range(10, 11):
    print(ii)

    zmask = np.array([data["P1Ds"].z[ii]])

    args.set_baseline(ztar=data["P1Ds"].z[ii], fit_type="at_a_time")

    like = set_like(
        data["P1Ds"],
        emulator,
        args,
        data_hires=data["extra_P1Ds"],
    )    
    
    for par in like.free_params:
        if par.name not in list_fix:
            par.value = weak_priors[par.name + "_cen"][ii]
            par.min_value = weak_priors[par.name + "_cen"][ii] - 2 * weak_priors[par.name + "_std"]
            par.max_value = weak_priors[par.name + "_cen"][ii] + 2 * weak_priors[par.name + "_std"]
        else:
            if (par.value < par.max_value) & (par.value > par.min_value):
                par.value = weak_priors[par.name + "_cen"][ii]
        print(par.name, par.value, par.min_value, par.max_value)
    
    fitter = Fitter(
        like=like,
        rootdir=output_dir,
        nburnin=args.n_burn_in,
        nsteps=args.n_steps,
        parallel=args.parallel,
        explore=args.explore,
        fix_cosmology=args.fix_cosmo,
    )

    p0 = like.sampling_point_from_parameters().copy()
            
    # fitter.run_minimizer(log_func_minimize=fitter.like.get_chi2, p0=p0_arr[ii], zmask=zmask, restart=True)
    fitter.run_minimizer(log_func_minimize=fitter.like.minus_log_prob, p0=p0, zmask=zmask, restart=True, nsamples=5)
    # fitter.run_minimizer(log_func_minimize=fitter.like.get_chi2, zmask=zmask, restart=True)
    out_mle.append(fitter.mle)
    out_mle_cube.append(fitter.mle_cube)
    out_chi2.append(fitter.mle_chi2)
    # plotter = Plotter(fitter, zmask=zmask, save_directory="mpg_baseline/"+str(ii))
    # plotter.plots_minimizer()

# %%

ndeg_all = 0
props = []
chi2_all = 0
for ii in range(len(out_chi2)):
    ndeg = len(like.data.k_kms[ii]) - len(out_mle_cube[ii])
    prob = chi2_scipy.sf(out_chi2[ii], ndeg)
    print(like.data.z[ii], '&', np.round(out_chi2[ii], 2), '&', ndeg, '&', np.round(prob*100, 2), '\\\\')
    ndeg_all += ndeg
    chi2_all += out_chi2[ii]
    props.append(prob)
prob = chi2_scipy.sf(chi2_all, ndeg_all)
print()
print("All", '&', np.round(chi2_all, 2), '&', ndeg_all, '&', np.round(prob*100, 2), '\\\\')
prob

# %%

args = Args(emulator_label="CH24_mpgcen_gpr", training_set="Cabayol23")
ii = 0
args.set_baseline(ztar=data["P1Ds"].z[ii], fit_type="at_a_time")
like1 = set_like(
    data["P1Ds"],
    emulator,
    args,
    data_hires=data["extra_P1Ds"],
)

out_mle_cube_reformat = []
for ii in range(len(data["P1Ds"].z)):
# for ii in range(10,11):
    args = Args(emulator_label="CH24_mpgcen_gpr", training_set="Cabayol23")
    args.set_baseline(ztar=data["P1Ds"].z[ii], fit_type="at_a_time")
    like2 = set_like(
        data["P1Ds"],
        emulator,
        args,
        data_hires=data["extra_P1Ds"],
    )
    
    for par in like2.free_params:
        if par.name not in list_fix:
            par.value = weak_priors[par.name + "_cen"][ii]
            par.min_value = weak_priors[par.name + "_cen"][ii] - 2 * weak_priors[par.name + "_std"]
            par.max_value = weak_priors[par.name + "_cen"][ii] + 2 * weak_priors[par.name + "_std"]
        else:
            if (par.value < par.max_value) & (par.value > par.min_value):
                par.value = weak_priors[par.name + "_cen"][ii]
    _cube = np.zeros(len(all_props))
    for jj, prop in enumerate(like1.free_param_names):
        if prop in like2.free_param_names:
            ind = np.argwhere(prop == np.array(like2.free_param_names))[0,0]
            value = like2.free_params[ind].value_from_cube(out_mle_cube[ii][ind])
            in_cube = like1.free_params[jj].get_value_in_cube(value)
            print(prop, like1.free_params[jj].name)
            if in_cube < 0:
                in_cube = 0
            _cube[jj] = in_cube
    out_mle_cube_reformat.append(np.array(_cube))

# %%
diru = "priors_1z_snr3_nocosmo"

args = Args(emulator_label="CH24_mpgcen_gpr", training_set="Cabayol23")
ii = 0
args.set_baseline(ztar=data["P1Ds"].z[ii], fit_type="at_a_time")

like = set_like(
    data["P1Ds"],
    emulator,
    args,
    data_hires=data["extra_P1Ds"],
)

fitter = Fitter(
    like=like,
    rootdir=output_dir,
    nburnin=args.n_burn_in,
    nsteps=args.n_steps,
    parallel=args.parallel,
    explore=args.explore,
    fix_cosmology=args.fix_cosmo,
)
fitter.mle_cube = out_mle_cube[ii]

plotter = Plotter(fitter, save_directory=diru, zmask=zmask)
plotter.plot_p1d(values=out_mle_cube_reformat, plot_panels=True, residuals=True, z_at_time=True)

# %%

# %%
dir_out = {
    "mle_cube":out_mle_cube,
    "mle":out_mle,
    "chi2":out_chi2,
    "mle_cube_reformat":out_mle_cube_reformat,
}
np.save("priors_1z_snr3_nocosmo/res.npy", dir_out)

# %%
args = Args(emulator_label="CH24_mpgcen_gpr", training_set="Cabayol23")
ii = 0
args.set_baseline(ztar=data["P1Ds"].z[ii], fit_type="at_a_time")

like = set_like(
    data["P1Ds"],
    emulator,
    args,
    data_hires=data["extra_P1Ds"],
)

for par in like.free_params:
    if par.name not in list_fix:
        par.value = weak_priors[par.name + "_cen"][ii]
        par.min_value = weak_priors[par.name + "_cen"][ii] - 2 * weak_priors[par.name + "_std"]
        par.max_value = weak_priors[par.name + "_cen"][ii] + 2 * weak_priors[par.name + "_std"]
    else:
        if (par.value < par.max_value) & (par.value > par.min_value):
            par.value = weak_priors[par.name + "_cen"][ii]

fitter = Fitter(
    like=like,
    rootdir=output_dir,
    nburnin=args.n_burn_in,
    nsteps=args.n_steps,
    parallel=args.parallel,
    explore=args.explore,
    fix_cosmology=args.fix_cosmo,
)

mask = np.arange(11)

like_params = []
for mle_cube in out_mle_cube_reformat:
    like_params.append(fitter.like.parameters_from_sampling_point(mle_cube))

fold0 = "/home/jchaves/Proyectos/projects/lya/cup1d/notebooks/tutorials/priors_1z_snr3_nocosmo/"
folder = fold0 + "taueff"
oFmodel, ocFmodel = fitter.like.theory.model_igm.models["F_model"].plot_parameters(data["P1Ds"].z[mask], like_params, folder=folder)
folder = fold0 + "sigT"
oTmodel, ocTmodel = fitter.like.theory.model_igm.models["T_model"].plot_parameters(data["P1Ds"].z[mask], like_params, folder=folder)
folder = fold0 + "Simult"
oSimult, ocSimult = fitter.like.theory.model_cont.metal_models["Si_mult"].plot_parameters(data["P1Ds"].z[mask], like_params, folder=folder)
folder = fold0 + "Siadd"
oSiadd, ocSiadd = fitter.like.theory.model_cont.metal_models["Si_add"].plot_parameters(data["P1Ds"].z[mask], like_params, folder=folder)
folder = fold0 + "HCD"
oHCD, ocHCD = fitter.like.theory.model_cont.hcd_model.plot_parameters(data["P1Ds"].z[mask], like_params, folder=folder)

models = [ocFmodel, ocTmodel, ocSimult, ocSiadd, ocHCD]
param_attime_all = {}
for mod in models:
    for key in mod:
        param_attime_all[key] = mod[key]

# %%
np.save("priors_1z_snr3_nocosmo/fit_baseline_param_attime.npy", param_attime_all)

# %%
weak2_priors = plot_z_at_time_params(fitter, out_mle)

# %%
weak_priors = weak2_priors.copy()

p0 = np.array(list(like.fid["fit_cube"].values()))
out_mle = []
out_mle_cube = []
out_chi2 = []
list_fix = ["tau_eff_0", "sigT_kms_0", "gamma_0", "kF_kms_0"]

for ii in range(len(like.data.z)): 
# for ii in range(10, 11):
    print(ii)

    zmask = np.array([data["P1Ds"].z[ii]])

    args.set_baseline(ztar=data["P1Ds"].z[ii], fit_type="at_a_time")

    like = set_like(
        data["P1Ds"],
        emulator,
        args,
        data_hires=data["extra_P1Ds"],
    )    
    
    for par in like.free_params:
        if par.name not in list_fix:
            par.value = weak_priors[par.name + "_cen"][ii]
            par.min_value = weak_priors[par.name + "_cen"][ii] - 2 * weak_priors[par.name + "_std"]
            par.max_value = weak_priors[par.name + "_cen"][ii] + 2 * weak_priors[par.name + "_std"]
        else:
            if (par.value < par.max_value) & (par.value > par.min_value):
                par.value = weak_priors[par.name + "_cen"][ii]
        print(par.name, par.value, par.min_value, par.max_value)
    
    fitter = Fitter(
        like=like,
        rootdir=output_dir,
        nburnin=args.n_burn_in,
        nsteps=args.n_steps,
        parallel=args.parallel,
        explore=args.explore,
        fix_cosmology=args.fix_cosmo,
    )

    p0 = like.sampling_point_from_parameters().copy()
            
    # fitter.run_minimizer(log_func_minimize=fitter.like.get_chi2, p0=p0_arr[ii], zmask=zmask, restart=True)
    fitter.run_minimizer(log_func_minimize=fitter.like.minus_log_prob, p0=p0, zmask=zmask, restart=True, nsamples=5)
    # fitter.run_minimizer(log_func_minimize=fitter.like.get_chi2, zmask=zmask, restart=True)
    out_mle.append(fitter.mle)
    out_mle_cube.append(fitter.mle_cube)
    out_chi2.append(fitter.mle_chi2)
    # plotter = Plotter(fitter, zmask=zmask, save_directory="mpg_baseline/"+str(ii))
    # plotter.plots_minimizer()

# %%
from cup1d.optimize.show_results import print_results

# %%
from cup1d.optimize.show_results import print_results
print_results(like, out_chi2, out_mle_cube)

# %%

# %%

# %%
weak3_priors = plot_z_at_time_params(fitter, out_mle)

# %% [markdown]
# ## Read results from chains

# %%
folder_data = "/home/jchaves/Proyectos/projects/lya/data/obs/QMLE3/CH24_mpgcen_gpr/fid/chain_5/2.2/"
folder_priors = "/home/jchaves/Proyectos/projects/lya/data/obs/QMLE3/CH24_mpgcen_gpr/priors/chain_3/2.2/"
file = "fitter_results.npy"
# sampler_data = np.load(folder + file, allow_pickle=True).item()

# %%
plotter = Plotter(fname_chain=folder_data + file, fname_priors=folder_priors + file)
# plotter.fitter.like.plot_p1d(residuals=True)

# %% [markdown]
# ### Normal corner

# %%
# plotter.fitter.like.plot_p1d()
# plotter.save_directory = folder_data
# plotter.save_directory = None
# plotter.plot_corner()

# %% [markdown]
# ### Corner natural units

# %%
plotter.save_directory = folder_data

only_plot = [
    '$\\mathrm{ln}\\,\\tau_0$',
    '$\\mathrm{ln}\\,\\sigma^T_0$',
    '$\\mathrm{ln}\\,\\gamma_0$',
    '$\\mathrm{ln}\\,k^F_0$',
    '$\\mathrm{R}_0$'
]
plotter.plot_corner_1z_natural(2.2, only_plot=only_plot)

# %%
