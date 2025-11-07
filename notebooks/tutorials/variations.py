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

# # Variations

# +
# %load_ext autoreload
# %autoreload 2

import numpy as np
import time, os, sys
import matplotlib.pyplot as plt
from cup1d.utils.fit_ellipse import fit_ellipse, plot_ellipse
from scipy.interpolate import griddata
import matplotlib.patches as mpatches
from scipy.stats import chi2 as chi2_scipy


from matplotlib import rcParams

rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "STIXGeneral"

# +
cont = np.array([0, 1, 2, 3, 4, 5])
prob_levels = np.zeros(len(cont))
chi2_levels = np.zeros(len(cont))

for ii in range(len(cont)):
    prob = chi2_scipy.cdf(cont[ii]**2, 1)
    chi2 = chi2_scipy.ppf(prob, 2)
    print(cont[ii], cont[ii]**2, chi2, prob)
    prob_levels[ii] = prob
    chi2_levels[ii] = chi2

print(prob_levels)
print(chi2_levels)
# -

# ### Set priors

from lace.cosmo import camb_cosmo
from cup1d.likelihood import CAMB_model
def rescale_star(fid_cosmo, new_cosmo, kp_Mpc, ks_Mpc=0.05):
    """Fast computation of blob when running with fixed background"""

    # differences in primordial power (at CMB pivot point)
    ratio_As = new_cosmo["As"] / fid_cosmo["As"]
    delta_ns = new_cosmo["ns"] - fid_cosmo["ns"]
    delta_nrun = new_cosmo["nrun"] - fid_cosmo["nrun"]

    # logarithm of ratio of pivot points
    ln_kp_ks = np.log(kp_Mpc / ks_Mpc)

    # rescale blobs
    delta_alpha_star = delta_nrun
    delta_n_star = delta_ns + delta_nrun * ln_kp_ks
    ln_ratio_A_star = (
        np.log(ratio_As)
        + (delta_ns + 0.5 * delta_nrun * ln_kp_ks) * ln_kp_ks
    )

    out_star={
        "alpha_star": fid_cosmo["alpha_star"] + delta_alpha_star,
        "n_star": fid_cosmo["n_star"] + delta_n_star,
        "Delta2_star": fid_cosmo["Delta2_star"] * np.exp(ln_ratio_A_star)
    }

    return out_star


# +
cosmo_camb = camb_cosmo.get_cosmology(
    H0=67.66,
    mnu=0.0,
    omch2=0.119,
    ombh2=0.0224,
    omk=0.0,
    As=2.105e-09,
    ns=0.9665,
    nrun=0.0,
    pivot_scalar=0.05,
    w=-1,
)

fun_cosmo = CAMB_model.CAMBModel(
    zs=[3.],
    cosmo=cosmo_camb,
    z_star=3,
    kp_kms=0.009,
    fast_camb=False
)
results_camb = fun_cosmo.get_linP_params()
results_camb

kp_kms = 0.009
dkms_dMpc = fun_cosmo.dkms_dMpc(3.)
kp_Mpc = kp_kms * dkms_dMpc

fid_cosmo = {
    "As":2.105e-09,
    "ns":0.9665,
    "nrun":0,
    "Delta2_star":results_camb["Delta2_star"],
    "n_star":results_camb["n_star"],
    "alpha_star":results_camb["alpha_star"],
}

# +
nn = 50
As = np.linspace(1.026970414602751e-09, 3.4382747625135757e-09, nn)
ns = np.linspace(0.6922930599534987, 1.2746555165892985, nn)
As_grid, ns_grid = np.meshgrid(As, ns)
res_fit = np.zeros((nn, nn, 2))

As2 = np.linspace(8.062826641805509e-10, 4.658124996088607e-09, nn)
ns2 = np.linspace(0.6465761801374897, 1.3448138385530823, nn)
As_grid2, ns_grid2 = np.meshgrid(As2, ns2)
res_fit2 = np.zeros((nn, nn, 2))

for ii in range(nn):
    for jj in range(nn):
        new_cosmo = {
            "As":As_grid[ii, jj],
            "ns":ns_grid[ii, jj],
            "nrun":0
        }
        res = rescale_star(fid_cosmo, new_cosmo, kp_Mpc, ks_Mpc=0.05)
        res_fit[ii, jj, 0] = res["Delta2_star"]
        res_fit[ii, jj, 1] = res["n_star"]
        
        new_cosmo = {
            "As":As_grid2[ii, jj],
            "ns":ns_grid2[ii, jj],
            "nrun":0
        }
        res = rescale_star(fid_cosmo, new_cosmo, kp_Mpc, ks_Mpc=0.05)
        res_fit2[ii, jj, 0] = res["Delta2_star"]
        res_fit2[ii, jj, 1] = res["n_star"]
        
res_fit = res_fit.reshape(-1, 2)
res_fit2 = res_fit2.reshape(-1, 2)

# +
import alphashape
from shapely.geometry import Polygon, MultiPolygon
alpha = 1.0

# Compute alpha shape (concave hull)
alpha_shape = alphashape.alphashape(res_fit, alpha)
alpha_shape2 = alphashape.alphashape(res_fit2, alpha)

# Extract boundary coordinates
if isinstance(alpha_shape, Polygon):
    boundary = np.array(alpha_shape.exterior.coords)
elif isinstance(alpha_shape, MultiPolygon):
    # Take largest polygon if multiple disconnected regions
    largest = max(alpha_shape.geoms, key=lambda p: p.area)
    boundary = np.array(largest.exterior.coords)

    
if isinstance(alpha_shape2, Polygon):
    boundary2 = np.array(alpha_shape2.exterior.coords)
elif isinstance(alpha_shape2, MultiPolygon):
    # Take largest polygon if multiple disconnected regions
    largest = max(alpha_shape2.geoms, key=lambda p: p.area)
    boundary2 = np.array(largest.exterior.coords)


# from scipy.spatial import ConvexHull
# hull = ConvexHull(res_fit)
# hull_points = res_fit[hull.vertices]
# hull_points = np.append(hull_points, hull_points[:1, :])
# hull_points = hull_points.reshape(-1, 2)
# -

from cup1d.likelihood.cosmologies import set_cosmo

mpg_all = set_cosmo("mpg_0", return_all=True)
nyx_all = set_cosmo("nyx_0", return_all=True)

# +
# plt.scatter(res_fit[:,0], res_fit[:,1], alpha=0.1)
sim_dat = np.zeros((31, 2))
ii = 0
for lab in mpg_all:
    if lab[-1].isdigit() | (lab == "mpg_central"):
        _cosmo = mpg_all[lab]
        plt.scatter(_cosmo["star_params"]['Delta2_star'], _cosmo["star_params"]['n_star'], color="C0")
        sim_dat[ii, 0] = _cosmo["star_params"]['Delta2_star']
        sim_dat[ii, 1] = _cosmo["star_params"]['n_star']
        ii += 1
       
sim_dat2 = np.zeros((18, 2))
ii = 0 
for lab in nyx_all:
    if (lab[-1].isdigit() and (lab != "accel2")) | (lab == "nyx_central"):
        if lab[-2:] == "14":
            continue
        _cosmo = nyx_all[lab]
        plt.scatter(_cosmo["star_params"]['Delta2_star'], _cosmo["star_params"]['n_star'], color="C1")
        sim_dat2[ii, 0] = _cosmo["star_params"]['Delta2_star']
        sim_dat2[ii, 1] = _cosmo["star_params"]['n_star']
        ii += 1
plt.plot(boundary[:,0], boundary[:,1], "C0")
# plt.scatter(res_fit2[:,0], res_fit2[:,1], alpha=0.1)
plt.plot(boundary2[:,0], boundary2[:,1], "C1")
# -

# #### Contours from chains

from cup1d.likelihood.cosmologies import set_cosmo
from cup1d.likelihood import CAMB_model
import matplotlib.cm as cm



# +
base = "/home/jchaves/Proyectos/projects/lya/data/out_DESI_DR1/DESIY1_QMLE3/"
folder = base + "sim_mpg_central/CH24_mpgcen_gpr/chain_2/"
dat_mpg_sim = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()
sum_mpg_sim = np.load(folder + "summary.npy", allow_pickle=True).item()

folder = base + "sim_mpg_central_igm/CH24_mpgcen_gpr/chain_1/"
dat_mpg_igm = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()
sum_mpg_igm = np.load(folder + "summary.npy", allow_pickle=True).item()

folder = base + "sim_mpg_central_igm0/CH24_mpgcen_gpr/chain_1/"
dat_mpg_igm0 = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()
sum_mpg_igm0 = np.load(folder + "summary.npy", allow_pickle=True).item()

# folder = base + "sim_nyx_central/CH24_mpgcen_gpr/chain_2/"
# dat_nyx_sim = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()
folder = base + "sim_nyx_central/CH24_mpgcen_gpr/chain_2/"
dat_nyx_sim = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()

folder = base + "sim_sherwood/CH24_mpgcen_gpr/chain_1/"
dat_sherwood = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()

# +
print(sum_mpg_sim["delta2_star_err"]/sum_mpg_igm["delta2_star_err"])
print(sum_mpg_sim["n_star_err"]/sum_mpg_igm["n_star_err"])

print(sum_mpg_sim["delta2_star_err"]/sum_mpg_igm0["delta2_star_err"])
print(sum_mpg_sim["n_star_err"]/sum_mpg_igm0["n_star_err"])

# +

print(1-sum_qmle["delta2_star_err"]/sum_mpg["delta2_star_err"])
print(1-sum_qmle["n_star_err"]/sum_mpg["n_star_err"])

# +

print(sum_nyx["delta2_star_err"]/sum_mpg["delta2_star_err"])
print(sum_nyx["n_star_err"]/sum_mpg["n_star_err"])
# +

# folder = base + "DESIY1_QMLE3/global_opt/CH24_mpgcen_gpr/chain_5/"
# dat_mpg2 = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()
# sum_mpg2 = np.load(folder + "summary.npy", allow_pickle=True).item()
# # log2 = np.load(folder + "lnprob.npy")

# +
base = "/home/jchaves/Proyectos/projects/lya/data/out_DESI_DR1/"

folder = base + "DESIY1_QMLE3/global_opt/CH24_mpgcen_gpr/chain_1/"
dat_mpg = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()
sum_mpg = np.load(folder + "summary.npy", allow_pickle=True).item()
dat_mpg_Asns = np.load(folder + "line_sigmas_Asns.npy", allow_pickle=True).item()

## emu
folder = base + "DESIY1_QMLE3/global_opt/CH24_nyxcen_gpr/chain_3/"
dat_nyx = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()
sum_nyx = np.load(folder + "summary.npy", allow_pickle=True).item()

## data

folder = base + "DESIY1_FFT3_dir/global_opt/CH24_mpgcen_gpr/chain_1/"
dat_fft3 = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()

folder = base + "DESIY1_QMLE/global_opt/CH24_mpgcen_gpr/chain_1/"
dat_qmle = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()
sum_qmle = np.load(folder + "summary.npy", allow_pickle=True).item()

# folder = base + "DESIY1_FFT_dir/global_opt/CH24_mpgcen_gpr/chain_1/"
# dat_fft = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()

folder = base + "DESIY1_QMLE3/zmax/CH24_mpgcen_gpr/chain_1/"
dat_zmax = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()

folder = base + "DESIY1_QMLE3/zmin/CH24_mpgcen_gpr/chain_1/"
dat_zmin = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()

## cov

folder = base + "DESIY1_QMLE3/no_inflate/CH24_mpgcen_gpr/chain_1/"
dat_no_inflate = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()

folder = base + "DESIY1_QMLE3/no_emu_cov/CH24_mpgcen_gpr/chain_1/"
dat_no_emu_cov = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()

folder = base + "DESIY1_QMLE3/emu_diag/CH24_mpgcen_gpr/chain_2/"
dat_emu_diag = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()

folder = base + "DESIY1_QMLE3/emu_block/CH24_mpgcen_gpr/chain_2/"
dat_emu_block = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()

folder = base + "DESIY1_QMLE3/data_syst_diag/CH24_mpgcen_gpr/chain_1/"
dat_syst_diag = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()

## IGM

folder = base + "DESIY1_QMLE3/more_igm/CH24_mpgcen_gpr/chain_1/"
dat_more_igm = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()

## cosmo

folder = base + "DESIY1_QMLE3/cosmo/CH24_mpgcen_gpr/chain_1/"
dat_cosmo = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()
folder = base + "DESIY1_QMLE3/cosmo/CH24_mpgcen_gpr/chain_1/"
dat_cosmo_Asns = np.load(folder + "line_sigmas_Asns.npy", allow_pickle=True).item()

folder = base + "DESIY1_QMLE3/cosmo_high/CH24_mpgcen_gpr/chain_1/"
dat_cosmo_high = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()
folder = base + "DESIY1_QMLE3/cosmo_high/CH24_mpgcen_gpr/chain_1/"
dat_cosmo_high_Asns = np.load(folder + "line_sigmas_Asns.npy", allow_pickle=True).item()

folder = base + "DESIY1_QMLE3/cosmo_low/CH24_mpgcen_gpr/chain_1/"
dat_cosmo_low = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()
folder = base + "DESIY1_QMLE3/cosmo_low/CH24_mpgcen_gpr/chain_1/"
dat_cosmo_low_Asns = np.load(folder + "line_sigmas_Asns.npy", allow_pickle=True).item()

folder = base + "DESIY1_QMLE3/cosmo_h74/CH24_mpgcen_gpr/chain_1/"
dat_cosmo_h74 = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()
folder = base + "DESIY1_QMLE3/cosmo_h74/CH24_mpgcen_gpr/chain_1/"
dat_cosmo_h74_Asns = np.load(folder + "line_sigmas_Asns.npy", allow_pickle=True).item()

folder = base + "DESIY1_QMLE3/cosmo_mnu/CH24_mpgcen_gpr/chain_2/"
dat_cosmo_mnu = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()
folder = base + "DESIY1_QMLE3/cosmo_mnu/CH24_mpgcen_gpr/chain_2/"
dat_cosmo_mnu_Asns = np.load(folder + "line_sigmas_Asns.npy", allow_pickle=True).item()

## ingredients

folder = base + "DESIY1_QMLE3/DLAs/CH24_mpgcen_gpr/chain_1/"
dat_dlas = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()

folder = base + "DESIY1_QMLE3/HCD0/CH24_mpgcen_gpr/chain_1/"
dat_HCD0 = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()

folder = base + "DESIY1_QMLE3/HCD_BOSS/CH24_mpgcen_gpr/chain_1/"
dat_HCD_BOSS = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()

folder = base + "DESIY1_QMLE3/metal_deco/CH24_mpgcen_gpr/chain_1/"
dat_metal_deco = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()

folder = base + "DESIY1_QMLE3/metal_si2/CH24_mpgcen_gpr/chain_1/"
dat_metal_si2 = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()

folder = base + "DESIY1_QMLE3/metal_trad/CH24_mpgcen_gpr/chain_1/"
dat_metal_trad = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()

folder = base + "DESIY1_QMLE3/metal_thin/CH24_mpgcen_gpr/chain_1/"
dat_metal_thin = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()

folder = base + "DESIY1_QMLE3/Metals_Ma2025/CH24_mpgcen_gpr/chain_1/"
dat_Metals_Ma2025 = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()

## no more

# folder = base + "DESIY1_QMLE3/Turner24/CH24_mpgcen_gpr/chain_1/"
# dat_turner = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()


# folder = base + "DESIY1_QMLE3/hcd_z/CH24_mpgcen_gpr/chain_1/"
# dat_hcd_z = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()

# folder = base + "DESIY1_QMLE3/less_igm/CH24_mpgcen_gpr/chain_1/"
# dat_less_igm = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()



# folder = base + "DESIY1_QMLE3/metals_z/CH24_mpgcen_gpr/chain_1/"
# dat_metals_z = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()



# folder = base + "DESIY1_QMLE3/no_inflate_no_emu_cov/CH24_mpgcen_gpr/chain_1/"
# dat_no_inflate_no_emu_cov = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()


# folder = base + "DESIY1_QMLE3/no_res/CH24_mpgcen_gpr/chain_1/"
# dat_no_res = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()



# folder = base + "DESIY1_QMLE3/kF_kms/CH24_mpgcen_gpr/chain_2/"
# dat_kF = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()

# +
from cup1d.likelihood.cosmologies import set_cosmo
from cup1d.likelihood import CAMB_model
import matplotlib.cm as cm

from matplotlib.path import Path
from matplotlib.patches import PathPatch

# Suppose you already have:
# boundary: (N,2) array of alpha shape boundary points (closed polygon)
# res_fit: (M,2) array of data points

def return_patch_priors(boundary, col="0.5"):
    # Create a large rectangle that covers the whole plot area
    xmin, xmax = boundary[:,0].min(), boundary[:,0].max()
    ymin, ymax = boundary[:,1].min(), boundary[:,1].max()
    outer = np.array([
        [xmin, ymin], [xmin, ymax],
        [xmax, ymax], [xmax, ymin],
        [xmin, ymin]
    ])
    
    # Build compound path: outer rectangle + inner polygon
    verts = np.concatenate([outer, boundary])
    codes = (
        [Path.MOVETO] +
        [Path.LINETO] * (len(outer) - 2) + [Path.CLOSEPOLY] +
        [Path.MOVETO] +
        [Path.LINETO] * (len(boundary) - 2) + [Path.CLOSEPOLY]
    )
    path = Path(verts, codes)
    
    # Add shaded patch (the "outside" area)
    patch = PathPatch(path, facecolor=col, edgecolor='none', alpha=0.5)

    return patch


# +

ls = ["-", "--"]

lw = [3, 2]
col = [0.7, 0.3]
ftsize = 28
cmaps = ["Blues", "Oranges", "Greens", "Reds", "Purples"]

dict_trans = {
    "DESIY1_QMLE3_mpg":"Fiducial", 
    
    "DESIY1_QMLE_mpg":"Data: w/ low SNR", 
    "DESIY1_FFT3_dir_mpg": "Data: FFT",
    # "DESIY1_FFT_dir_mpg":"Data: FFT w/ low SNR", 
    "zmin": "Data: $z \geq 2.6$",  # restricted zrange
    "zmax": "Data: $z \leq 3.4$",  # restricted zrange
    
    "no_emu_cov":"Cov: w/o emu", # no emu error
    "emu_block":"Cov: emu block diag", # no emu error
    "emu_diag":"Cov: emu diag", # no emu error
    "no_inflate":"Cov: w/o 5% err",
    "dat_syst_diag": "Cov: uncorr syst", # systematics data uncorrelated
    
    "DESIY1_QMLE3_nyx":"Emulator: lace-lyssa",
    
    "cosmo": "Cosmo: $\omega_0\omega_a$CDM",  # different fiducial cosmo
    "cosmo_low": "Cosmo: low $\Omega_\mathrm{cdm}h^2$",  # different fiducial cosmo
    "cosmo_high": "Cosmo: high $\Omega_\mathrm{cdm}h^2$",  # different fiducial cosmo
    "cosmo_h74": "Cosmo: $h=0.74$",  # different fiducial cosmo
    "cosmo_mnu": r"Cosmo: $\sum m_\nu=0.3$ eV", # different fiducial cosmo
    
    "more_igm": "IGM: $n_z=8$",  # 6 params for IGM evolution
    
    "DLAs": "HCD: only DLAs",  # no LLS, sub-DLA
    "HCD0": "HCD: w/ $f_\\mathrm{const}^\\mathrm{HCD}$", # w/ constant term
    "HCD_BOSS": "HCD: BOSS",
    
    "metal_si2": "Metals: no SiII-SiII",  # no SiII-SiII cont
    "metal_deco": "Metals: no H-Si decorr",  # no decorrelation metals
    "metal_thin": "Metals: opt thin",  # no desviation from optically-thin limit ERROR
    
    "metal_trad": "Metals: BOSS",  # 2 params for metals like eBOSS
    "Metals_Ma2025": "Metals: Ma+2025",

    "sim_mpg_central": "mpg-central", 
    "sim_mpg_central_all": "Model: cosmo, IGM, cont, syst", 
    "sim_mpg_central_igm": "Model: cosmo, IGM",
    "sim_mpg_central_igm0": "Model: cosmo", 
    "sim_nyx_central": "lyssa-central", 
    "sim_sherwood": "sherwood", 
}


fname = [
    "data_type",
    "zsplit",
    "cov_data",
    "cov_emu",
    "cosmo",
    "cosmo_Asns",
    "metals_ing",
    "metals_models",
    "DLAs",
    "igm",
    "emu",
    "val_sims",
    "val_sims_model",
    "cosmo2",
    "cosmo_Asns2",
    # "test",
]

for image in range(15):

    # if image in [3, 4, 5]:
    #     ftsize = 26
    # else:
    #     ftsize = 22
    factx = 1

    if image == 0:
        variations = ["DESIY1_QMLE3_mpg", "DESIY1_QMLE_mpg", "DESIY1_FFT3_dir_mpg"]
        dats = [dat_mpg, dat_qmle, dat_fft3]
    elif image == 1:
        variations = ["DESIY1_QMLE3_mpg", "zmin", "zmax"]
        dats = [dat_mpg, dat_zmin, dat_zmax]
    elif image == 2:
        variations = ["DESIY1_QMLE3_mpg", "no_inflate", "dat_syst_diag"]
        dats = [dat_mpg, dat_no_inflate, dat_syst_diag]
    elif image == 3:
        variations = ["DESIY1_QMLE3_mpg", "emu_block", "emu_diag", "no_emu_cov"]
        dats = [dat_mpg, dat_emu_block, dat_emu_diag, dat_no_emu_cov]
    elif image == 4:
        variations = ["DESIY1_QMLE3_mpg", "cosmo", "cosmo_h74", "cosmo_mnu"]
        dats = [dat_mpg, dat_cosmo, dat_cosmo_h74, dat_cosmo_mnu]
    elif image == 5:
        variations = ["DESIY1_QMLE3_mpg", "cosmo", "cosmo_h74", "cosmo_mnu"]
        dats = [dat_mpg_Asns, dat_cosmo_Asns, dat_cosmo_h74_Asns, dat_cosmo_mnu_Asns]
        factx = 1e9
    elif image == 6:
        variations = ["DESIY1_QMLE3_mpg", "metal_deco", "metal_thin", "metal_si2"]
        dats = [dat_mpg, dat_metal_deco, dat_metal_thin, dat_metal_si2]
    elif image == 7:
        variations = ["DESIY1_QMLE3_mpg", "metal_trad", "Metals_Ma2025"]
        dats = [dat_mpg, dat_metal_trad, dat_Metals_Ma2025]
    elif image == 8:
        variations = ["DESIY1_QMLE3_mpg", "HCD0", "DLAs", "HCD_BOSS"]
        dats = [dat_mpg, dat_HCD0, dat_dlas, dat_HCD_BOSS]
    elif image == 9:
        variations = ["DESIY1_QMLE3_mpg", "more_igm"]
        dats = [dat_mpg, dat_more_igm]
    elif image == 10:
        variations = ["DESIY1_QMLE3_mpg", "DESIY1_QMLE3_nyx"]
        dats = [dat_mpg, dat_nyx]
    elif image == 11:
        variations = ["sim_mpg_central", "sim_nyx_central", "sim_sherwood"]
        dats = [dat_mpg_sim, dat_nyx_sim, dat_sherwood]
    elif image == 12:
        variations = ["sim_mpg_central_all", "sim_mpg_central_igm", "sim_mpg_central_igm0"]
        dats = [dat_mpg_sim, dat_mpg_igm, dat_mpg_igm0]
    elif image == 13:
        variations = ["DESIY1_QMLE3_mpg", "cosmo_low", "cosmo_high"]
        dats = [dat_mpg, dat_cosmo_low, dat_cosmo_high]
    elif image == 14:
        variations = ["DESIY1_QMLE3_mpg", "cosmo_low", "cosmo_high"]
        dats = [dat_mpg_Asns, dat_cosmo_low_Asns, dat_cosmo_high_Asns]
        factx = 1e9
    else:
        continue


    dict_diff = {
        "xcen": np.median(dats[0][0.68][0][0]),
        "ycen": np.median(dats[0][0.68][0][1]),
    }

    fig, ax = plt.subplots(figsize=(8, 6))
    
    fit_type = "global_opt"
    xlim = [1, -1]
    ylim =  [1, -1]
    for ii, var in enumerate(variations):
        print()
        dat = dats[ii].copy()
        cmap = plt.colormaps[cmaps[ii]]

        if var.startswith("sim_"):
            if "mpg_central" in var:
                clabel = "mpg_central"
            else:
                clabel = var[4:]
            cosmo = set_cosmo(cosmo_label=clabel)
            like_cosmo = CAMB_model.CAMBModel(np.array([3]), cosmo=cosmo)
            true_cosmo = like_cosmo.get_linP_params()
            ds_diff = true_cosmo["Delta2_star"]
            ns_diff = true_cosmo["n_star"]
            print(var, ds_diff, ns_diff)
        else:
            ds_diff = dict_diff["xcen"]
            ns_diff = dict_diff["ycen"]

        # if (ii == 0):
        #     if variations[1] == "DESIY1_QMLE3_nyx":
        #         new_boundary = boundary2.copy()
        #         new_boundary[:,0] -= ds_diff
        #         new_boundary[:,1] -= ns_diff
        #         patch = return_patch_priors(new_boundary, col="0.3")
        #         ax.add_patch(patch)
        #         ax.fill(new_boundary[:, 0], new_boundary[:, 1], "white")
        #         ax.scatter(sim_dat2[:,0] - ds_diff, sim_dat2[:,1] - ns_diff, color="C1")
        #     new_boundary = boundary.copy()
        #     new_boundary[:,0] -= ds_diff
        #     new_boundary[:,1] -= ns_diff
        #     patch = return_patch_priors(new_boundary)
        #     ax.add_patch(patch)
        #     ax.fill(new_boundary[:, 0], new_boundary[:, 1], "white")

        for inum, num in enumerate([0.68, 0.95]):
            if inum == 0:
                label=dict_trans[var]
            else:
                label=None
            for jj in range(len(dat[num])):
                x = (dat[num][jj][0] - ds_diff) * factx
                y = dat[num][jj][1] - ns_diff
                ax.plot(x, y, color=cmap(col[inum]), label=label, lw=lw[inum], alpha=0.75)
                ax.fill(x, y, color=cmap(col[inum]), alpha=0.5)

                if x.min() < xlim[0]:
                    xlim[0] = x.min()
                if x.max() > xlim[1]:
                    xlim[1] = x.max()
                if y.min() < ylim[0]:
                    ylim[0] = y.min()
                if y.max() > ylim[1]:
                    ylim[1] = y.max()

    dx = xlim[1] - xlim[0]
    ax.set_xlim(xlim[0] - dx*0.05, xlim[1] + dx*0.05)
    dx = ylim[1] - ylim[0]
    ax.set_ylim(ylim[0] - dx*0.05, ylim[1] + dx*0.05)
    
    if fname[image].startswith("cosmo_Asns"):
        ax.set_xlabel(r"$\Delta A_s[\times 10^{-9}]$", fontsize=ftsize+2)
        ax.set_ylabel(r"$\Delta n_s$", fontsize=ftsize+2)
    else:
        ax.set_xlabel(r"$\Delta(\Delta^2_\star)$", fontsize=ftsize+2)
        ax.set_ylabel(r"$\Delta n_\star$", fontsize=ftsize+2)
    ax.tick_params(
        axis="both", which="major", labelsize=ftsize - 2
    )
    ax.axhline(color="k", ls=":")
    ax.axvline(color="k", ls=":")

     
    if fname[image] in ["cosmo", "cosmo_Asns","cosmo2", "cosmo_Asns2","metals_ing", "model_ing_no",  "DLAs"]:
        ymin, ymax = plt.ylim()
        yrange = ymax - ymin
        ax.set_ylim(ymin, ymax + 0.2 * yrange)

        
    if fname[image] in ["data"]:
        loc = "lower right"
    elif fname[image] in ["val_sims"]:
        loc = "upper left"
    else:
        loc = "upper right"

    # if variations[1] == "DESIY1_QMLE3_nyx":
    #     ax.scatter(sim_dat2[:,0] - ds_diff, sim_dat2[:,1] - ns_diff, color="C1")
    # ax.scatter(sim_dat[:,0] - ds_diff, sim_dat[:,1] - ns_diff, color="C0")
    
    plt.legend(fontsize=ftsize-6, loc=loc, ncol=1)
    plt.tight_layout()
    plt.savefig("figs/vars/variations_"+fname[image]+".pdf")
    plt.savefig("figs/vars/variations_"+fname[image]+".png")
# -


# from matplotlib.patches import Ellipse


# +
fig, ax = plt.subplots(figsize=(8, 6))
ftsize = 20
ls = ["-", "--"]

variations = ["sim_nyx_central"]
# variations = ["sim_mpg_central", "sim_nyx_central"]
dict_trans = {
    "sim_mpg_central":"mpg-central", 
    "sim_nyx_central":"nyx-central", 
    "sim_sherwood":"sherwood", 
}
var_deg = [550-26, 681-26, 670-26]



fit_type = "global_opt"
x0 = 0
y0 = 0
for ii, var in enumerate(variations):
    print()
    file = "out_pl/"+ var + ".npy"
    out_dict = np.load(file, allow_pickle=True).item()
    
    prob = chi2_scipy.sf(out_dict['chi2'], var_deg[ii]) * 100
    print(var, np.round(out_dict['chi2'], 1), f'{prob:.1e}')

    cosmo = set_cosmo(cosmo_label=var[4:])
    like_cosmo = CAMB_model.CAMBModel(np.array([3]), cosmo=cosmo)
    true_cosmo = like_cosmo.get_linP_params()

    consist = 0
    for key in ["Delta2_star", "n_star"]:
        print(np.round(out_dict[key], 3), np.round(out_dict["err_" + key], 3))
        print("diff", np.round(out_dict[key] - true_cosmo[key], 3), np.round(out_dict["err_" + key], 3))
        consist += (out_dict[key] - true_cosmo[key])**2/out_dict["err_" + key]**2

    prob_var = chi2_scipy.sf(consist, 2) * 100
    print(np.round(prob_var, 1))

    col = "C"+str(ii)
    ax.scatter(
        out_dict["Delta2_star"] - true_cosmo["Delta2_star"], 
        out_dict["n_star"] - true_cosmo["n_star"], 
        color=col, marker="x")

    for jj in range(1, 2):
        if jj == 1:
            lab = dict_trans[var]
        else:
            lab= None
        ax.plot(
            out_dict["xell"+str(jj)]- true_cosmo["Delta2_star"], 
            out_dict["yell"+str(jj)]- true_cosmo["n_star"], 
            col+ls[jj-1], lw=3, label=lab+" w/o errors")

    nseed = 400
    xy_all = np.zeros((nseed, 2))
    for jj in range(nseed):
        folder = "/home/jchaves/Proyectos/projects/lya/data/out_DESI_DR1/DESIY1_QMLE3/"+var+"/CH24_mpgcen_gpr/"
        data_cen = np.load(folder + "seed_" + str(jj) + "/best_dircosmo.npy", allow_pickle=True).item()
        x = data_cen["mle_cosmo_cen"]["Delta2_star"] - true_cosmo["Delta2_star"]
        y = data_cen["mle_cosmo_cen"]["n_star"] - true_cosmo["n_star"]
        # print(data_cen["mle"]['$f_{\rm HCD1}_0$'])
        xy_all[jj, 0] = x
        xy_all[jj, 1] = y
        plt.scatter(x, y, marker=".", color="C1")


    # plot ellipse containing 68%
    mean = xy_all.mean(axis=0)
    cov = np.cov(xy_all, rowvar=False)
    rho = cov[0,1]/np.sqrt(cov[0,0] * cov[1,1])
    
    # Eigen-decomposition of covariance
    # vals, vecs = np.linalg.eigh(cov)
    # order = vals.argsort()[::-1]
    # vals, vecs = vals[order], vecs[:, order]
    
    # # Scale to the 68% quantile of chi-square with 2 dof
    # chi2_val = chi2_scipy.ppf(0.68, df=2)
    # _ = vals < 0
    # width, height = 2 * np.sqrt(vals * chi2_val)
    # angle = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    
    # ellipse = Ellipse(mean, width, height, angle, edgecolor='red', facecolor='none', lw=2)
    # ax.add_patch(ellipse)
    
    plot_ellipse(
        np.sqrt(cov[0, 0]),
        np.sqrt(cov[1, 1]),
        rho,
        [mean[0], mean[1]],
        ax=ax,
        label=lab + " noisy realizations"
    )


ax.axhline(0, color="k", linestyle="--")
ax.axvline(0, color="k", linestyle="--")

ax.set_xlim(-0.1, 0.15)
ax.set_ylim(-0.1, 0.15)


ax.set_ylabel(r"$\Delta(n_\star)$", fontsize=ftsize)
ax.set_xlabel(r"$\Delta(\Delta^2_\star)$", fontsize=ftsize)
ax.tick_params(
    axis="both", which="major", labelsize=ftsize - 2
)

plt.legend(fontsize=ftsize-2, loc="upper right")
plt.tight_layout()
# plt.savefig("figs/validation_2d.pdf")
# plt.savefig("figs/validation_2d.png")
# -


