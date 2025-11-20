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
# # Importance sampling of Planck chains with mock DESI Lya likelihoods
#
# Here we read one of the extended Planck chains with linear power parameters, and add an extra likelihood coming from a fake measurement of DESI Lya P1D.

# %%
# %load_ext autoreload
# %autoreload 2
import numpy as np
import os
from getdist import plots, loadMCSamples
import matplotlib.pyplot as plt
from cup1d.planck import planck_chains
from cup1d.planck import add_linP_params
from cup1d.likelihood import marg_lya_like
from cup1d.utils.utils import get_path_repo

from matplotlib import rcParams

rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "STIXGeneral"

# %% [markdown]
# ### Read an extended Planck chains and plot linear power parameters

# %%

root_dir=os.path.join(get_path_repo("cup1d"), "data", "planck_linP_chains")

folder = root_dir + "/crisjagq/base_mnu/desi-bao-all_planck2018-lowl-TT-clik_planck2018-lowl-EE-clik_planck-NPIPE-highl-CamSpec-TTTEEE_planck-act-dr6-lensing_linP/base_mnu_desi-bao-all_planck2018-lowl-TT-clik_planck2018-lowl-EE-clik_planck-NPIPE-highl-CamSpec-TTTEEE_planck-act-dr6-lensing_linP"
cmb_mnu2 = {} 
cmb_mnu2["samples"] = loadMCSamples(folder)

cmb = planck_chains.get_planck_2018(
    model='base',
    data='plikHM_TTTEEE_lowl_lowE_linP',
    root_dir=root_dir,
    linP_tag=None
)

cmb_nrun = planck_chains.get_planck_2018(
    model='base_nrun',
    data='plikHM_TTTEEE_lowl_lowE_linP',
    root_dir=root_dir,
    linP_tag=None
)

cmb_nrun_nrunrun = planck_chains.get_planck_2018(
    model='base_nrun_nrunrun',
    data='plikHM_TTTEEE_lowl_lowE_linP',
    root_dir=root_dir,
    linP_tag=None
)

cmb_mnu = planck_chains.get_planck_2018(
    model='base_mnu',
    data='plikHM_TTTEEE_lowl_lowE_linP',
    root_dir=root_dir,
    linP_tag=None
)

cmb_nnu = planck_chains.get_planck_2018(
    model='base_nnu',
    data='plikHM_TTTEEE_lowl_lowE_linP',
    root_dir=root_dir,
    linP_tag=None
)


# cmb_tau = planck_chains.get_planck_2018(
#     model='base',
#     data='plikHM_TTTEEE_lowl_linP',
#     root_dir=root_dir,
#     linP_tag=None
# )


# cmb_omega_k = planck_chains.get_planck_2018(
#     model='base_omegak',
#     data='plikHM_TTTEEE_lowl_lowE_linP',
#     root_dir=root_dir,
#     linP_tag=None
# )

# cmb_w_wa = planck_chains.get_planck_2018(
#     model='base_w_wa',
#     data='plikHM_TTTEEE_lowl_lowE_BAO_linP',
#     root_dir=root_dir,
#     linP_tag=None
# )


# cmb_r = planck_chains.get_planck_2018(
#     model='base_r',
#     data='plikHM_TTTEEE_lowl_lowE_linP',
#     root_dir=root_dir,
#     linP_tag=None
# )


# cmb_nrun_nnu_w_mnu = planck_chains.get_planck_2018(
#     model='base_nrun_nnu_w_mnu',
#     data='plikHM_TTTEEE_lowl_lowE_BAO_Riess18_Pantheon18_lensing_linP',
#     root_dir=root_dir,
#     linP_tag=None
# )

# %%
base_notebook = "/home/jchaves/Proyectos/projects/lya/cup1d/notebooks/tutorials/"

base = "/home/jchaves/Proyectos/projects/lya/data/out_DESI_DR1/"
folder = base + "DESIY1_QMLE3/global_opt/CH24_mpgcen_gpr/chain_7/"

dat_mpg = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()
sum_mpg = np.load(folder + "summary.npy", allow_pickle=True).item()

blobs = np.load(folder + "blobs.npy")
ds = blobs["Delta2_star"].reshape(-1)
ns = blobs["n_star"].reshape(-1)

# per_ds = np.percentile(ds, [5, 16, 50, 84, 95])
# per_ns = np.percentile(ns, [5, 16, 50, 84, 95])
# _ = (ds > per_ds[0]) & (ds < per_ds[-1]) & (ns > per_ns[0]) & (ns < per_ns[-1])
# corr = np.corrcoef(ds[_], ns[_])
corr = np.corrcoef(ds, ns)
r = corr[0, 1]
r

# %%
# def subsample_correlations(ds, ns, A=2, nrepeat=2000, seed=0):
#     """
#     subsample without replacement: select n//A indices per repeat by permutation.
#     If the chain has weights, warn and use weighted bootstrap instead.
#     method: 'pearson' or 'spearman'
#     winsorize_frac: fraction to winsorize for Pearson (e.g. 0.05), or None.
#     Returns: array of correlation estimates (length nrepeat)
#     """
#     rng = np.random.default_rng(seed)
#     n = len(ds)
#     subsz = max(2, n // A)   # ensure at least 2
#     corr_vals = np.empty(nrepeat)
#     # Unweighted: do permutation + take first subsz (no repeats inside each subsample)
#     for i in range(nrepeat):
#         idx = rng.permutation(n)[:subsz]
#         xs = ds[idx]
#         ys = ns[idx]
#         corr_vals[i] = np.corrcoef(xs, ys)[0, 1]
#     return corr_vals

# corr_samps = subsample_correlations(ds, ns, nrepeat=100, A=2)
# percen = np.percentile(corr_samps, [16, 50, 84])
# print(np.round(percen[1], 4), np.round(percen[1]-percen[0], 4), np.round(percen[2]-percen[1], 4))

# %%
# from corner import corner
# corner(np.array([ds[_], ns[_]]).T);

# %%

# %% [markdown]
# ### Prepare unblinding

# %%
fake_blinding = {
    'Delta2_star': sum_mpg["delta2_star_16_50_84"][1]-np.median(cmb["samples"]["linP_DL2_star"]),
     'n_star': sum_mpg["n_star_16_50_84"][1]-np.median(cmb["samples"]["linP_n_star"]),
}
real_blinding = np.load(base_notebook + "blinding.npy", allow_pickle=True).item()

blinding = real_blinding
# blinding = fake_blinding

# %%
desi_dr1 = {
    "Delta2_star":sum_mpg["delta2_star_16_50_84"][1] - blinding["Delta2_star"],
    "n_star":sum_mpg["n_star_16_50_84"][1] - blinding["n_star"],
    "r":r,
    "Delta2_star_err":sum_mpg['delta2_star_err'],
    "n_star_err":sum_mpg["n_star_err"],
}

# %%
desi_dr1

# %%
ds -= blinding["Delta2_star"]
ns -= blinding["n_star"]

# %% [markdown]
# #### Get pk from cmb chains

# %%
from lace.cosmo import camb_cosmo

def get_pk(icmb, nn=100):
    
    ind = np.random.permutation(np.arange(len(icmb['samples'][0])))[:nn]
    for ii in range(nn):
        if ii % 10 == 0:
            print(ii)
        pars_median = icmb['samples'].getParamSampleDict(ii)
        _cosmo = camb_cosmo.get_cosmology_from_dictionary(pars_median)
        k_kms, zs_out, _P_kms = camb_cosmo.get_linP_kms(_cosmo)
        if ii == 0:
            P_kms = np.zeros((nn, k_kms[0].shape[0]))
        P_kms[ii] = _P_kms[0]
    return k_kms[0], P_kms


run_code = False

if run_code:
    for ii in range(len(cmb_all)):
        k_kms, P_kms = get_pk(cmb_all[ii], nn=500)

# %%

# %%
ftsize = 22
kp_kms = 0.009
fact = kp_kms**3 / (2 * np.pi**2)
hatch =  ["", "", "/", "", "/", ""]


_ds = cmb['samples'][cmb['samples'].index['linP_DL2_star']]
_ns = cmb['samples'][cmb['samples'].index['linP_n_star']]
# p16, p50, p84 = np.percentile(vals, [16, 50, 84])
A_fid, sigma_A = np.median(np.log(_ds/fact)), np.std(np.log(_ds/fact))
# B_fid, sigma_B = np.median(_ns), np.std(_ns)
# rho = np.corrcoef(_ds, _ns)[0][1]

x = np.geomspace(0.5 * kp_kms, 2 * kp_kms, 200)
y_samp_fid = np.median(np.log(_ds/fact)[:,None] + _ns[:,None] * np.log(x[None,:]/kp_kms), axis=0)

cmb_all = [
    cmb, 
    # cmb_tau, 
    cmb_mnu, 
    cmb_nnu, 
    cmb_nrun, 
    cmb_nrun_nrunrun,
]
cmb_labs = [
    r"$\mathit{Planck}$ T&E: $\Lambda$CDM", 
    # r"Planck+18 $\Lambda$CDM (no lowE)",
    r"$\mathit{Planck}$ T&E: $\sum m_\nu$",
    r"$\mathit{Planck}$ T&E: $N_\mathrm{eff}$",
    r"$\mathit{Planck}$ T&E: $\alpha_\mathrm{s}$",
    r"$\mathit{Planck}$ T&E: $\alpha_\mathrm{s}, \,\beta_\mathrm{s}$",
]

# icmb = cmb
# pars_median = {}
# for par in icmb['samples'].index:
#     pars_median[par] = np.median(icmb['samples'][icmb['samples'].index[par]])
# i_best = cmb['samples'].loglikes.argmax()
# pars_median = cmb['samples'].getParamSampleDict(i_best)
# _cosmo = camb_cosmo.get_cosmology_from_dictionary(pars_median)
# k_kms, zs_out, P_kms = camb_cosmo.get_linP_kms(_cosmo)

fig, ax = plt.subplots(3, 1, figsize=(10, 12), sharex=True, sharey=True)

for ii in range(len(cmb_all)):
# for ii in range(5):

    if ii == 0:
        nn = 20000
        ind = np.random.permutation(np.arange(len(ds)))[:nn]
        A_samp = np.log(ds[ind]/fact)
        B_samp = ns[ind]
        
        # x-grid and sampled lines
        # x = np.geomspace(0.001, 0.04, 200)
        x = np.geomspace(0.5 * kp_kms, 2 * kp_kms, 200)
        y_samp = np.exp(A_samp[:,None] + B_samp[:,None] * np.log(x[None,:]/kp_kms))/np.exp(y_samp_fid)
        
        # mean and percentiles
        y_med = np.median(y_samp, axis=0)
        y_p16 = np.percentile(y_samp, 16, axis=0)
        y_p84 = np.percentile(y_samp, 84, axis=0)
        # y_err = np.std(y_samp)
        
        # plot

        for kk in range(3):
            if kk == 0:
                lab = r"DESI-Ly$\alpha$"
            else:
                lab = None
            ax[kk].scatter(
                kp_kms, 
                np.median(np.exp(A_samp)/np.exp(A_fid)), 
                # np.std(np.exp(A_samp)/np.exp(A_fid)),
                # marker="o", 
                color="C0"
            )
            # ax[kk].plot(x, y_med, color="C0")
            ax[kk].fill_between(x, y_p16, y_p84, alpha=0.3, label=lab, color="C0")
    
        P_kms = np.load("P_kms_" + str(ii) + ".npy")
        y16, ynorm, y84 = np.percentile(P_kms, [16, 50, 84], axis=0)
        fid_k_kms = 10**np.arange(-5.888706504390846, -0.41158524967118454, 0.0054826)
        ax[0].fill_between(fid_k_kms, y16/ynorm, y84/ynorm, alpha=0.2, color="C1", label=cmb_labs[ii])
        ax[0].plot(fid_k_kms, ynorm/ynorm, lw=2, color="C1")
    else:
        if ii < 3:
            kk = 1
        else:
            kk = 2
        # icmb = cmb_all[ii]
        # pars_median = {}
        # for par in icmb['samples'].index:
        #     pars_median[par] = np.median(icmb['samples'][icmb['samples'].index[par]])
        # i_best = icmb['samples'].loglikes.argmax()
        # pars_median = icmb['samples'].getParamSampleDict(i_best)
        # _cosmo = camb_cosmo.get_cosmology_from_dictionary(pars_median)
        # k_kms, zs_out, P_kms = camb_cosmo.get_linP_kms(_cosmo)
        P_kms = np.load("P_kms_" + str(ii) + ".npy")
        y16, y50, y84 = np.percentile(P_kms, [16, 50, 84], axis=0)
        ax[kk].fill_between(fid_k_kms, y16/ynorm, y84/ynorm, alpha=0.2, color="C"+str(ii+1), label=cmb_labs[ii], hatch=hatch[ii+1])
        ax[kk].plot(fid_k_kms, y50/ynorm, lw=2, color="C"+str(ii+1))

for kk in range(3):
    ax[kk].axhline(1, ls=":", color="k", alpha=0.5)
    ax[kk].set_xlim(1.5e-5, 0.06)
    ax[kk].set_ylim(0.85, 1.2)
    ax[kk].set_xscale("log")
    # plt.yscale("log")
    ax[kk].legend(fontsize=ftsize, loc="upper left")
    ax[kk].tick_params(axis="both", which="major", labelsize=ftsize)
    
    x1, x2 = 0.00125, 0.04
    ax[kk].plot([x1, x2], [0.92, 0.92], lw=2, color="k")
    ax[kk].text(5e-3, 0.87, r"Ly$\alpha$ $P_\mathrm{1D}$", fontsize=ftsize)
    x1, x2 = 2.85e-5, 0.0025
    ax[kk].plot([x1, x2], [0.93, 0.93], lw=2, color="k")
    ax[kk].text(1.2e-4, 0.87, r"CMB T&E", fontsize=ftsize)
    

    
fig.supylabel("$P_\mathrm{lin}(k, z=3)/P^\mathrm{\Lambda CDM}_\mathrm{lin}(k, z=3)$", fontsize=ftsize)
ax[2].set_xlabel(r"$k\,[\mathrm{km}^{-1}\mathrm{s}]$", fontsize=ftsize)
plt.tick_params()
plt.savefig("figs/Plin_extra.pdf", bbox_inches="tight")
plt.savefig("figs/Plin_extra.png", bbox_inches="tight")

# %%
# desi_dr12 = {
#     "Delta2_star":0.664,
#     "n_star":-2.474,
#     "r":-0.002,
#     "Delta2_star_err":0.056,
#     "n_star_err":0.019,
# }

# %%

# %%

# %%

# %%
ftsize = 26
cmap = plt.colormaps["Blues"]
col = [0.7, 0.3]
lw = [3, 2]
cmb_all = [
    cmb, 
    # cmb_tau, 
    cmb_mnu, 
    cmb_nnu, 
    cmb_nrun, 
    cmb_nrun_nrunrun,
]
cmb_labs = [
    r"$\mathit{Planck}$ T&E: $\Lambda$CDM", 
    # r"Planck+18 $\Lambda$CDM (no lowE)",
    r"$\mathit{Planck}$ T&E: $\sum m_\nu$",
    r"$\mathit{Planck}$ T&E: $N_\mathrm{eff}$",
    r"$\mathit{Planck}$ T&E: $\alpha_\mathrm{s}$",
    r"$\mathit{Planck}$ T&E: $\alpha_\mathrm{s}, \,\beta_\mathrm{s}$",
]

g = plots.getSinglePlotter(width_inch=10)
g.settings.num_plot_contours = 2

for ii, icmb in enumerate(cmb_all):

    new_samples=icmb['samples'].copy()
    
    if ii == 0:
        filled = False
        lwu = 3
    else:
        filled = False
        lwu = 2
    g.plot_2d(
        new_samples, 
        ['linP_DL2_star', 'linP_n_star'], 
        colors=["C"+str(ii+1)], 
        lws=2, 
        alphas=[0.8, 0.5],
        filled=filled,
    )

ax = g.subplots[0,0]


for inum, num in enumerate([0.68, 0.95]):
    if inum == 0:
        label=r"DESI-Ly$\alpha$"
    else:
        label=None
    for jj in range(len(dat_mpg[num])):
        x = dat_mpg[num][jj][0] - blinding["Delta2_star"]
        y = dat_mpg[num][jj][1] - blinding["n_star"]
        ax.plot(x, y, color=cmap(col[inum]), label=label, lw=lw[inum], alpha=0.75)
        ax.fill(x, y, color=cmap(col[inum]), alpha=0.5)

# ax.axhline(y=1,color='k',lw=lw,label=r"DESI-DR1 Ly$\alpha$ (this work)")
for ii, icmb in enumerate(cmb_all):
    ax.axhline(y=1,color="C"+str(ii+1),lw=lw[0],label=cmb_labs[ii])    


ax.set_xlim(0.25, 0.45)
ax.set_ylim(-2.36, -2.24)
ax.set_ylabel(r"$n_\star$", fontsize=ftsize)
ax.set_xlabel(r"$\Delta^2_\star$", fontsize=ftsize)
ax.tick_params(axis="both", which="major", labelsize=ftsize)

plt.legend(fontsize=ftsize-1, loc="upper left", ncol=2)
plt.tight_layout()

plt.savefig("figs/star_planck_mine.png", bbox_inches='tight')
plt.savefig("figs/star_planck_mine.pdf", bbox_inches='tight')


# %%

# %% [markdown]
# ### Add mock DESI Lya likelihood
#
# We will generate three fake DESI likelihoods, for different fiducial values, and see how they affect the cosmo params.

# %%
def gaussian_chi2_mock_DESI(neff, DL2, true_DL2=0.35, true_neff=-2.3, DL2_err=0.003, neff_err=0.002, r=0.55):
    """Compute Gaussian Delta chi^2 for a particular point(s) (neff,DL2),
    using a mock measurement from DESI (complete made up at this point).
    """
    return marg_lya_like.gaussian_chi2(neff, DL2, true_neff, true_DL2, neff_err, DL2_err, r)


# %%
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FormatStrFormatter
from getdist.mcsamples import MCSamples

def plot_combine(chain_type, desi_dr1, param_name, fontsize=24):
    latex_param = {
        "nnu": r"$N_\mathrm{eff}$",
        "tau": r"$\tau$",
        "mnu": r"$\sum m_\nu$",
        "nrun": r"$\alpha_\mathrm{s}$",
        "nrunrun": r"$\beta_\mathrm{s}$"
    }
    all_chains = []
    all_labels = []
    # samples_DESI=[]
    labels_DESI=[]
    colors = []
    for ii in range(len(chain_type)):
        # overwrite, but no idea how to make a deep copy
        new_samples = chain_type[ii]['samples'].copy()

        # print before weighting
        # if param_name == "mnu":
        #     print("2 sigma", param_name, new_samples.getInlineLatex(param_name, limit=2))
        # else:
        #     percen = np.percentile(new_samples[param_name], [16, 50, 84])
        #     print(
        #         np.round(percen[1], 4), 
        #         np.round(percen[1]-percen[0], 4),
        #         np.round(percen[2]-percen[1], 4)
        #     )
        #     if (param_name == "nrunrun"): 
        #         percen = np.percentile(new_samples["nrun"], [16, 50, 84])
        #         print(
        #             np.round(percen[1], 4), 
        #             np.round(percen[1]-percen[0], 4),
        #             np.round(percen[2]-percen[1], 4)
        #         )
        
        p = new_samples.getParams()

        # log unnormalised weights
        logw = 0.5 * gaussian_chi2_mock_DESI(
            p.linP_n_star, 
            p.linP_DL2_star, 
            true_neff=desi_dr1["n_star"], 
            true_DL2=desi_dr1["Delta2_star"], 
            neff_err=desi_dr1["n_star_err"],
            DL2_err=desi_dr1["Delta2_star_err"],
            r=desi_dr1["r"]
        )
        
        # # numerical stabilisation: subtract max
        # logw -= logw.max()
        # w_unnorm = np.exp(logw)      # safe: values won't overflow
        # w = w_unnorm / w_unnorm.sum()  # normalised weights
        
        # # effective sample size (ESS)
        # ess = 1.0 / np.sum(w**2)
        # print("ESS", ess)
        
        new_samples.reweightAddingLogLikes(logw) #re-weight cut_samples to account for the new likelihood
        
        # samples_DESI.append(new_samples)
        if ii == 0:
            all_chains.append(chain_type[ii]['samples'])
            all_labels.append(r"$\mathit{Planck}$ T&E")
            colors.append("C0")
            labels_DESI.append(r"$\mathit{Planck}$ T&E + DESI-Ly$\alpha$")
        elif ii == 1:
            labels_DESI.append(
                r"CMB-SPA + DESI-BAO"
                "\n"
                r"+ DESI-Ly$\alpha$"
            )
        colors.append("C"+str(ii + 1))
        all_chains.append(new_samples)
        all_labels.append(labels_DESI[ii])

    for ii in range(len(all_chains)):
        # print after weighting
        if param_name == "mnu":
            print("2 sigma", param_name, all_chains[ii].getInlineLatex(param_name, limit=2))
        else:
            print("1 sigma", param_name, all_chains[ii].getInlineLatex(param_name, limit=1))
            if (param_name == "nrunrun"): 
                print("1 sigma", "nrun", all_chains[ii].getInlineLatex("nrun", limit=1))
    
    g = plots.getSubplotPlotter(width_inch=8)
    g.settings.axes_fontsize = fontsize
    g.settings.legend_fontsize = fontsize

    if param_name == "nrunrun":
        arr_plot = ['linP_DL2_star','linP_n_star', "nrun", param_name]
    else:
        arr_plot = ['linP_DL2_star','linP_n_star', param_name]

    
    mm = len(all_labels)
    
    g.triangle_plot(
        all_chains,
        arr_plot,
        legend_labels=all_labels,
        legend_loc='upper right',
        colors=colors,
        filled=True,
        lws=[3] * mm,
        alphas=[0.8] * mm,
        line_args=[{"color": f"C{i}", "lw": 3, "alpha": 0.8} for i in range(mm)],
    )
            

    if (param_name != "nrunrun"): 
        ndim = 3
    else:
        ndim = 4
        
    for jj in range(ndim):
        g.subplots[ndim-1, jj].tick_params(
            axis="both", which="major", labelsize=fontsize
        )
        g.subplots[-1, jj].xaxis.set_major_locator(MaxNLocator(nbins=3))
        g.subplots[-1, jj].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    for jj in range(1, ndim):
        g.subplots[jj, 0].tick_params(
            axis="both", which="major", labelsize=fontsize
        )            
        g.subplots[jj, 0].yaxis.set_major_locator(MaxNLocator(nbins=3))
        g.subplots[jj, 0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        
    g.subplots[-1, 0].set_xlabel(r"$\Delta^2_\star$", fontsize=fontsize)
    g.subplots[-1, 1].set_xlabel(r"$n_\star$", fontsize=fontsize)
    g.subplots[1, 0].set_ylabel(r"$n_\star$", fontsize=fontsize)

    if (param_name != "nrunrun"): 
        g.subplots[-1, 2].set_xlabel(latex_param[param_name], fontsize=fontsize)
        g.subplots[2, 0].set_ylabel(latex_param[param_name], fontsize=fontsize)
    else:
        g.subplots[-1, 2].set_xlabel(latex_param["nrun"], fontsize=fontsize)
        g.subplots[-1, 3].set_xlabel(latex_param[param_name], fontsize=fontsize)
        g.subplots[2, 0].set_ylabel(latex_param["nrun"], fontsize=fontsize)
        g.subplots[3, 0].set_ylabel(latex_param[param_name], fontsize=fontsize)
    

    for ax in g.subplots[-1]:  # last row of panels
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha('right')

    if param_name == "nnu":
        g.subplots[-1, -1].axvline(3.046, ls="--", color = "black")
        g.subplots[-1, 0].axhline(3.046, ls="--", color = "black")
        g.subplots[-1, 1].axhline(3.046, ls="--", color = "black")
    elif param_name == "mnu":
        pass
        g.subplots[-1, -1].axvline(0.06, ls="--", color = "black")
        g.subplots[-1, 0].axhline(0.06, ls="--", color = "black")
        g.subplots[-1, 1].axhline(0.06, ls="--", color = "black")
        g.subplots[-1, -1].axvline(0.1, ls="--", color = "black")
        g.subplots[-1, 0].axhline(0.1, ls="--", color = "black")
        g.subplots[-1, 1].axhline(0.1, ls="--", color = "black")
        # g.subplots[-1, -1].set_xlim(0.01, 0.4)
        g.subplots[-1, -1].set_xlim(0., 0.4)
    elif param_name == "nrun":
        g.subplots[-1, -1].axvline(0., ls="--", color = "black")
        g.subplots[-1, 0].axhline(0., ls="--", color = "black")
        g.subplots[-1, 1].axhline(0., ls="--", color = "black")
    elif param_name == "nrunrun":
        g.subplots[-1, -1].axvline(0., ls="--", color = "black")
        g.subplots[-1, 0].axhline(0., ls="--", color = "black")
        g.subplots[-1, 1].axhline(0., ls="--", color = "black")
        g.subplots[-1, 2].axhline(0., ls="--", color = "black")
        g.subplots[-1, 2].axvline(0., ls="--", color = "black")
        g.subplots[-2, 0].axhline(0., ls="--", color = "black")
        g.subplots[-2, 1].axhline(0., ls="--", color = "black")
        g.subplots[-2, 2].axvline(0., ls="--", color = "black")

        
    plt.tight_layout()
    plt.savefig("figs/import_"+param_name+".png", bbox_inches="tight")
    plt.savefig("figs/import_"+param_name+".pdf", bbox_inches="tight")

# %%
cmb_mnu = planck_chains.get_planck_2018(
    model='base_mnu',
    data='plikHM_TTTEEE_lowl_lowE_linP',
    root_dir=root_dir,
    linP_tag=None
)

root_dir=os.path.join(get_path_repo("cup1d"), "data", "planck_linP_chains")
folder = root_dir + "/crisjagq/base_mnu/desi-bao-all_planck2018-lowl-TT-clik_planck2018-lowl-EE-clik_planck-NPIPE-highl-CamSpec-TTTEEE_planck-act-dr6-lensing_linP/base_mnu_desi-bao-all_planck2018-lowl-TT-clik_planck2018-lowl-EE-clik_planck-NPIPE-highl-CamSpec-TTTEEE_planck-act-dr6-lensing_linP"
cmb_mnu2 = {} 
cmb_mnu2["samples"] = loadMCSamples(folder)

chains = [cmb_mnu, cmb_mnu2]
plot_combine(chains, desi_dr1, "mnu")

# %%
np.round(100*(1-0.0046/0.0067), 2)

# %%
cmb_nrun = planck_chains.get_planck_2018(
    model='base_nrun',
    data='plikHM_TTTEEE_lowl_lowE_linP',
    root_dir=root_dir,
    linP_tag=None
)

chains = [cmb_nrun]
plot_combine(chains, desi_dr1, "nrun")

# %%
print(np.round(100*(1-0.0049/0.010), 2))
np.round(100*(1-0.0050/0.013), 2)

# %%

cmb_nrun_nrunrun = planck_chains.get_planck_2018(
    model='base_nrun_nrunrun',
    data='plikHM_TTTEEE_lowl_lowE_linP',
    root_dir=root_dir,
    linP_tag=None
)


plot_combine([cmb_nrun_nrunrun], desi_dr1, "nrunrun")

# %%
np.round(100*(1-0.16/0.19), 2)

# %%

cmb_nnu = planck_chains.get_planck_2018(
    model='base_nnu',
    data='plikHM_TTTEEE_lowl_lowE_linP',
    root_dir=root_dir,
    linP_tag=None
)

plot_combine([cmb_nnu], desi_dr1, "nnu")

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %% [markdown]
# #### Almost no constraints on As, ns

# %%
samples_DESI=[]
true_DL2=[0.33,0.35,0.37]
true_neff=[-2.295,-2.30,-2.305]
# DL2_err = out_dict["err_Delta2_star"]
# neff_err = out_dict["err_n_star"]
# r=out_dict["rho"]
DL2_err = sum_mpg['delta2_star_err']
neff_err = sum_mpg['n_star_err']
# r=0.1
coeff_mult = 0.5

chain_type = cmb


labels_DESI=[]
for ii in range(len(true_DL2)):
    new_samples=chain_type['samples'].copy()
    p=new_samples.getParams()
    new_loglike = coeff_mult * gaussian_chi2_mock_DESI(
        p.linP_n_star, 
        p.linP_DL2_star, 
        true_DL2=true_DL2[ii],
        true_neff=true_neff[ii],
        neff_err=neff_err,
        DL2_err=DL2_err,
        r=r
    )
    new_samples.reweightAddingLogLikes(new_loglike) #re-weight cut_samples to account for the new likelihood
    samples_DESI.append(new_samples)
    labels_DESI.append(r'+ DESI Ly$\alpha \,(\Delta^2_\ast = ${} $n_\ast = ${})'.format(true_DL2[ii], true_neff[ii]))


g = plots.getSubplotPlotter(width_inch=16)
g.settings.axes_fontsize = 10
g.settings.legend_fontsize = 14
g.triangle_plot([chain_type['samples'],samples_DESI[0],samples_DESI[1],samples_DESI[2]],
                ['linP_DL2_star','linP_n_star'],
                legend_labels=['Planck 2018',labels_DESI[0],labels_DESI[1],labels_DESI[2]])
# g.triangle_plot([chain_type['samples'],samples_DESI[0],samples_DESI[1],samples_DESI[2]],
#                 ['linP_DL2_star','linP_n_star','logA','ns'],
#                 legend_labels=['Planck 2018 + SDSS',labels_DESI[0],labels_DESI[1],labels_DESI[2]])
# g.triangle_plot([chain_type['samples'],samples_DESI[0],samples_DESI[1],samples_DESI[2]],
#                 ['linP_DL2_star','linP_n_star','logA','ns', "tau"],
#                 legend_labels=['Planck 2018 + SDSS',labels_DESI[0],labels_DESI[1],labels_DESI[2]])

# %% [markdown]
# #### In tau

# %%

# %%
samples_DESI=[]
true_DL2=[0.33,0.35,0.37]
true_neff=[-2.295,-2.30,-2.305]
# DL2_err = out_dict["err_Delta2_star"]
# neff_err = out_dict["err_n_star"]
# r=out_dict["rho"]
DL2_err = sum_mpg['delta2_star_err']
neff_err = sum_mpg['n_star_err']
coeff_mult = 0.5
# coeff_mult = 1

chain_type = cmb_tau


labels_DESI=[]
for ii in range(len(true_DL2)):
    new_samples=chain_type['samples'].copy()
    p=new_samples.getParams()
    new_loglike = coeff_mult * gaussian_chi2_mock_DESI(
        p.linP_n_star, 
        p.linP_DL2_star, 
        true_DL2=true_DL2[ii],
        true_neff=true_neff[ii],
        neff_err=neff_err,
        DL2_err=DL2_err,
        r=r
    )
    new_samples.reweightAddingLogLikes(new_loglike) #re-weight cut_samples to account for the new likelihood
    samples_DESI.append(new_samples)
    labels_DESI.append(r'+ DESI Ly$\alpha \,(\Delta^2_\ast = ${} $n_\ast = ${})'.format(true_DL2[ii], true_neff[ii]))


g = plots.getSubplotPlotter(width_inch=16)
g.settings.axes_fontsize = 10
g.settings.legend_fontsize = 14
g.triangle_plot([chain_type['samples'],samples_DESI[0],samples_DESI[1],samples_DESI[2]],
                ['linP_DL2_star','linP_n_star', "tau"],
                legend_labels=['Planck 2018 (no lowE)',labels_DESI[0],labels_DESI[1],labels_DESI[2]])
# g.triangle_plot([chain_type['samples'],samples_DESI[0],samples_DESI[1],samples_DESI[2]],
#                 ['linP_DL2_star','linP_n_star','logA','ns'],
#                 legend_labels=['Planck 2018 + SDSS',labels_DESI[0],labels_DESI[1],labels_DESI[2]])
# g.triangle_plot([chain_type['samples'],samples_DESI[0],samples_DESI[1],samples_DESI[2]],
#                 ['linP_DL2_star','linP_n_star','logA','ns', "tau"],
#                 legend_labels=['Planck 2018 + SDSS',labels_DESI[0],labels_DESI[1],labels_DESI[2]])
plt.savefig("figs/import_tau.png")
plt.savefig("figs/import_tau.pdf")

# %% [markdown]
# #### No constraints on w0, wa

# %%
samples_DESI=[]
true_DL2=[0.33,0.35,0.37]
true_neff=[-2.295,-2.30,-2.305]
# DL2_err = 0.056
# neff_err = 0.022
# r=-0.134
# coeff_mult = 2.3 # should it be 0.5?
# DL2_err = 0.043
# neff_err = 0.025
# r=0.1
DL2_err = sum_mpg['delta2_star_err']
neff_err = sum_mpg['n_star_err']
coeff_mult = 0.5

chain_type = cmb_w_wa


labels_DESI=[]
for ii in range(len(true_DL2)):
    new_samples=chain_type['samples'].copy()
    p=new_samples.getParams()
    new_loglike = coeff_mult * gaussian_chi2_mock_DESI(
        p.linP_n_star, 
        p.linP_DL2_star, 
        true_DL2=true_DL2[ii],
        true_neff=true_neff[ii],
        neff_err=neff_err,
        DL2_err=DL2_err,
        r=r
    )
    new_samples.reweightAddingLogLikes(new_loglike) #re-weight cut_samples to account for the new likelihood
    samples_DESI.append(new_samples)
    labels_DESI.append(r'+ DESI Ly$\alpha \,(\Delta^2_\ast = ${} $n_\ast = ${})'.format(true_DL2[ii], true_neff[ii]))


g = plots.getSubplotPlotter(width_inch=16)
g.settings.axes_fontsize = 10
g.settings.legend_fontsize = 14
g.triangle_plot([chain_type['samples'],samples_DESI[0],samples_DESI[1],samples_DESI[2]],
                ['linP_DL2_star','linP_n_star','w','wa'],
                legend_labels=['Planck 2018 + SDSS',labels_DESI[0],labels_DESI[1],labels_DESI[2]])

# %% [markdown]
# ### Almost no constraints on omega_k

# %%
samples_DESI=[]
true_DL2=[0.33,0.35,0.37]
true_neff=[-2.295,-2.30,-2.305]
# DL2_err = 0.056
# neff_err = 0.022
# r=-0.134
# coeff_mult = 2.3 # should it be 0.5?
DL2_err = sum_mpg['delta2_star_err']
neff_err = sum_mpg['n_star_err']
coeff_mult = 0.5

chain_type = cmb_omega_k

labels_DESI=[]
for ii in range(len(true_DL2)):
    new_samples=chain_type['samples'].copy()
    p=new_samples.getParams()
    new_loglike = coeff_mult * gaussian_chi2_mock_DESI(
        p.linP_n_star, 
        p.linP_DL2_star, 
        true_DL2=true_DL2[ii],
        true_neff=true_neff[ii],
        neff_err=neff_err,
        DL2_err=DL2_err,
        r=r
    )
    new_samples.reweightAddingLogLikes(new_loglike) #re-weight cut_samples to account for the new likelihood
    samples_DESI.append(new_samples)
    labels_DESI.append(r'+ DESI Ly$\alpha \,(\Delta^2_\ast = ${} $n_\ast = ${})'.format(true_DL2[ii], true_neff[ii]))


g = plots.getSubplotPlotter(width_inch=16)
g.settings.axes_fontsize = 10
g.settings.legend_fontsize = 14
g.triangle_plot([chain_type['samples'],samples_DESI[0],samples_DESI[1],samples_DESI[2]],
                ['linP_DL2_star','linP_n_star','omegak'],
                legend_labels=['Planck 2018',labels_DESI[0],labels_DESI[1],labels_DESI[2]])

# %% [markdown]
# ### Constraints on mnu

# %%

# %%
samples_DESI=[]
# true_DL2=[0.33,0.35,0.37]
# true_neff=[-2.295,-2.30,-2.305]
# DL2_err = 0.056
# neff_err = 0.022
# r=-0.134
# coeff_mult = 2.3 # should it be 0.5?
# DL2_err = 0.043
# neff_err = 0.025
# r=0.1
# DL2_err = sum_mpg['delta2_star_err']
# neff_err = sum_mpg['n_star_err']
# coeff_mult = 0.5

chain_type = cmb_mnu


# labels_DESI=[]
# for ii in range(len(true_DL2)):
#     new_samples=chain_type['samples'].copy()
#     p=new_samples.getParams()
#     new_loglike = coeff_mult * gaussian_chi2_mock_DESI(
#         p.linP_n_star, 
#         p.linP_DL2_star, 
#         true_DL2=true_DL2[ii],
#         true_neff=true_neff[ii],
#         neff_err=neff_err,
#         DL2_err=DL2_err,
#         r=r
#     )
#     new_samples.reweightAddingLogLikes(new_loglike) #re-weight cut_samples to account for the new likelihood
#     samples_DESI.append(new_samples)
#     labels_DESI.append(r'+ DESI Ly$\alpha \,(\Delta^2_\ast = ${} $n_\ast = ${})'.format(true_DL2[ii], true_neff[ii]))

labels_DESI=[]
for ii in range(1):
    new_samples=chain_type['samples'].copy()
    p=new_samples.getParams()
    new_loglike = 0.5 * gaussian_chi2_mock_DESI(
        p.linP_n_star, 
        p.linP_DL2_star, 
        true_neff=desi_dr1["n_star"], 
        true_DL2=desi_dr1["Delta2_star"], 
        neff_err=desi_dr1["n_star_err"],
        DL2_err=desi_dr1["Delta2_star_err"],
        r=desi_dr1["r"]
    )
    new_samples.reweightAddingLogLikes(new_loglike) #re-weight cut_samples to account for the new likelihood
    samples_DESI.append(new_samples)
    labels_DESI.append(r"+ DESI DR1")


# g = plots.getSubplotPlotter(width_inch=8)
# g.settings.axes_fontsize = 12
# g.settings.legend_fontsize = 14
# g.triangle_plot([chain_type['samples'],samples_DESI[0],samples_DESI[1],samples_DESI[2]],
#                 ['linP_DL2_star','linP_n_star', 'sigma8', 'mnu'],
#                 legend_labels=["Planck 2018",labels_DESI[0],labels_DESI[1],labels_DESI[2]])


# g = plots.getSubplotPlotter(width_inch=8)
# g.settings.axes_fontsize = 12
# g.settings.legend_fontsize = 14
# g.triangle_plot([chain_type['samples'],samples_DESI[0],samples_DESI[1],samples_DESI[2]],
#                 ['linP_DL2_star','linP_n_star','mnu'],
#                 legend_labels=["Planck 2018",labels_DESI[0],labels_DESI[1],labels_DESI[2]])
g = plots.getSubplotPlotter(width_inch=8)
g.settings.axes_fontsize = 12
g.settings.legend_fontsize = 14
g.triangle_plot([chain_type['samples'],samples_DESI[0]],
                ['linP_DL2_star','linP_n_star','mnu'],
                legend_labels=["Planck 2018",labels_DESI[0]])


print(chain_type['samples'].getInlineLatex('mnu',limit=1))
print(samples_DESI[0].getInlineLatex('mnu',limit=1))
# print(samples_DESI[1].getInlineLatex('mnu',limit=1))
# print(samples_DESI[2].getInlineLatex('mnu',limit=1))
# print("")
print(chain_type['samples'].getInlineLatex('mnu',limit=2))
print(samples_DESI[0].getInlineLatex('mnu',limit=2))
# print(samples_DESI[1].getInlineLatex('mnu',limit=2))
# print(samples_DESI[2].getInlineLatex('mnu',limit=2))

plt.tight_layout()
plt.savefig("figs/import_neutrinos.png")
plt.savefig("figs/import_neutrinos.pdf")

# %% [markdown]
# #### Constraints on nrun

# %%
samples_DESI=[]
true_DL2=[0.33,0.35,0.37]
true_neff=[-2.295,-2.30,-2.305]
# DL2_err = out_dict["err_Delta2_star"]
# neff_err = out_dict["err_n_star"]
# r=out_dict["rho"]
# coeff_mult = 1 # should it be 0.5?
# DL2_err = 0.043
# neff_err = 0.025
# r=0.1
DL2_err = sum_mpg['delta2_star_err']
neff_err = sum_mpg['n_star_err']
coeff_mult = 0.5

chain_type = cmb_nrun


labels_DESI=[]
for ii in range(len(true_DL2)):
    new_samples=chain_type['samples'].copy()
    p=new_samples.getParams()
    new_loglike = coeff_mult * gaussian_chi2_mock_DESI(
        p.linP_n_star, 
        p.linP_DL2_star, 
        true_DL2=true_DL2[ii],
        true_neff=true_neff[ii],
        neff_err=neff_err,
        DL2_err=DL2_err,
        r=r
    )
    new_samples.reweightAddingLogLikes(new_loglike) #re-weight cut_samples to account for the new likelihood
    samples_DESI.append(new_samples)
    labels_DESI.append(r'+ DESI Ly$\alpha \,(\Delta^2_\ast = ${} $n_\ast = ${})'.format(true_DL2[ii], true_neff[ii]))


# g = plots.getSubplotPlotter(width_inch=12)
# g.settings.axes_fontsize = 12
# g.settings.legend_fontsize = 14
# g.triangle_plot([chain_type['samples'],samples_DESI[0],samples_DESI[1],samples_DESI[2]],
#                 ['linP_DL2_star','linP_n_star', 'omegach2', 'theta', 'tau', 'logA', 'ns', 'nrun'],
#                 legend_labels=["Planck 2018",labels_DESI[0],labels_DESI[1],labels_DESI[2]])

g = plots.getSubplotPlotter(width_inch=8)
g.settings.axes_fontsize = 12
g.settings.legend_fontsize = 14
g.triangle_plot([chain_type['samples'],samples_DESI[0],samples_DESI[1],samples_DESI[2]],
                ['linP_DL2_star','linP_n_star','nrun'],
                legend_labels=["Planck 2018",labels_DESI[0],labels_DESI[1],labels_DESI[2]])

fontsize = 20
ax_D2S_NS = g.subplots[1, 0]
ax_D2S_new = g.subplots[2, 0]
ax_NS_new = g.subplots[2, 1]
ax_new = g.subplots[2, 2]
axs = [ax_D2S_NS, ax_D2S_new, ax_NS_new, ax_new]
for ax in axs:
    ax.tick_params(
        axis="both", which="major", labelsize=fontsize
    )
    
ax_D2S_NS.set_ylabel(r"$n_\star$", fontsize=fontsize)
ax_D2S_new.set_xlabel(r"$\Delta^2_\star$", fontsize=fontsize)
ax_NS_new.set_xlabel(r"$n_\star$", fontsize=fontsize)

ax_NS_new.set_ylabel(r"$n_\star$", fontsize=fontsize)
ax_new.set_xlabel(r"$n_\star$", fontsize=fontsize)


# print(chain_type['samples'].getInlineLatex('nrun',limit=1))
# print(samples_DESI[1].getInlineLatex('nrun',limit=1))

plt.tight_layout()
plt.savefig("figs/import_nrun.png")
plt.savefig("figs/import_nrun.pdf")

# %%

# %%

# %% [markdown]
# #### Constraints on nnu

# %%
samples_DESI=[]
true_DL2=[0.33,0.35,0.37]
true_neff=[-2.295,-2.30,-2.305]
# DL2_err = out_dict["err_Delta2_star"]
# neff_err = out_dict["err_n_star"]
# r=out_dict["rho"]
DL2_err = sum_mpg['delta2_star_err']
neff_err = sum_mpg['n_star_err']
coeff_mult = 0.5 # should it be 0.5?

chain_type = cmb_nnu

labels_DESI=[]
for ii in range(len(true_DL2)):
    new_samples=chain_type['samples'].copy()
    p=new_samples.getParams()
    new_loglike = coeff_mult * gaussian_chi2_mock_DESI(
        p.linP_n_star, 
        p.linP_DL2_star, 
        true_DL2=true_DL2[ii],
        true_neff=true_neff[ii],
        neff_err=neff_err,
        DL2_err=DL2_err,
        r=r
    )
    new_samples.reweightAddingLogLikes(new_loglike) #re-weight cut_samples to account for the new likelihood
    samples_DESI.append(new_samples)
    labels_DESI.append(r'+ DESI Ly$\alpha \,(\Delta^2_\ast = ${} $n_\ast = ${})'.format(true_DL2[ii], true_neff[ii]))


g = plots.getSubplotPlotter(width_inch=8)
g.settings.axes_fontsize = 12
g.settings.legend_fontsize = 14
g.triangle_plot([chain_type['samples'],samples_DESI[0],samples_DESI[1],samples_DESI[2]],
                ['linP_DL2_star','linP_n_star','nnu'],
                legend_labels=["Planck 2018",labels_DESI[0],labels_DESI[1],labels_DESI[2]])


# g.triangle_plot([chain_type['samples'],samples_DESI[0],samples_DESI[1],samples_DESI[2]],
#                 ['linP_DL2_star','linP_n_star', 'logA', 'ns', 'nnu'],
#                 legend_labels=["Planck 2018",labels_DESI[0],labels_DESI[1],labels_DESI[2]])


print(chain_type['samples'].getInlineLatex('nnu',limit=1))
print(samples_DESI[1].getInlineLatex('nnu',limit=1))

plt.tight_layout()
plt.savefig("figs/import_nnu.png")
plt.savefig("figs/import_nnu.pdf")

# %% [markdown]
# #### Constraints on nrun and nrunrun

# %%

# %%

# %%
samples_DESI=[]
true_DL2=[0.33,0.35,0.37]
true_neff=[-2.295,-2.30,-2.305]
# DL2_err = out_dict["err_Delta2_star"]
# neff_err = out_dict["err_n_star"]
# r=out_dict["rho"]
# coeff_mult = 1 # should it be 0.5?
# DL2_err = 0.043
# neff_err = 0.025
# r=0.1
DL2_err = sum_mpg['delta2_star_err']
neff_err = sum_mpg['n_star_err']
coeff_mult = 0.5

chain_type = cmb_nrun_nrunrun


labels_DESI=[]
for ii in range(len(true_DL2)):
    new_samples=chain_type['samples'].copy()
    p=new_samples.getParams()
    new_loglike = coeff_mult * gaussian_chi2_mock_DESI(
        p.linP_n_star, 
        p.linP_DL2_star, 
        true_DL2=true_DL2[ii],
        true_neff=true_neff[ii],
        neff_err=neff_err,
        DL2_err=DL2_err,
        r=r
    )
    new_samples.reweightAddingLogLikes(new_loglike) #re-weight cut_samples to account for the new likelihood
    samples_DESI.append(new_samples)
    labels_DESI.append(r'+ DESI Ly$\alpha \,(\Delta^2_\ast = ${} $n_\ast = ${})'.format(true_DL2[ii], true_neff[ii]))


g = plots.getSubplotPlotter(width_inch=8)
g.settings.axes_fontsize = 12
g.settings.legend_fontsize = 14
g.triangle_plot([chain_type['samples'],samples_DESI[0],samples_DESI[1],samples_DESI[2]],
                ['linP_DL2_star','linP_n_star','nrun','nrunrun'],
                legend_labels=["Planck 2018",labels_DESI[0],labels_DESI[1],labels_DESI[2]])
#plt.savefig(plot_label+'.pdf')
#plt.savefig(plot_label+'.png')

print(chain_type['samples'].getInlineLatex('nrun',limit=1))
print(samples_DESI[1].getInlineLatex('nrun',limit=1))
print(chain_type['samples'].getInlineLatex('nrunrun',limit=1))
print(samples_DESI[1].getInlineLatex('nrunrun',limit=1))

plt.tight_layout()
plt.savefig("figs/import_nrun_nrunrun.png")
plt.savefig("figs/import_nrun_nrunrun.pdf")

# %% [markdown]
# ### Constraints on nrun_nnu_w_mnu

# %%
samples_DESI=[]
true_DL2=[0.33,0.35,0.37]
true_neff=[-2.295,-2.30,-2.305]
# DL2_err = 0.056
# neff_err = 0.022
# r=-0.134
# coeff_mult = 2.3 # should it be 0.5?
DL2_err = 0.043
neff_err = 0.025
r=0.1
coeff_mult = 0.5

chain_type = cmb_nrun_nnu_w_mnu


labels_DESI=[]
for ii in range(len(true_DL2)):
    new_samples=chain_type['samples'].copy()
    p=new_samples.getParams()
    new_loglike = coeff_mult * gaussian_chi2_mock_DESI(
        p.linP_n_star, 
        p.linP_DL2_star, 
        true_DL2=true_DL2[ii],
        true_neff=true_neff[ii],
        neff_err=neff_err,
        DL2_err=DL2_err,
        r=r
    )
    new_samples.reweightAddingLogLikes(new_loglike) #re-weight cut_samples to account for the new likelihood
    samples_DESI.append(new_samples)
    labels_DESI.append(r'+ DESI Ly$\alpha \,(\Delta^2_\ast = ${} $n_\ast = ${})'.format(true_DL2[ii], true_neff[ii]))


# plikHM_TTTEEE_lowl_lowE_BAO_Riess18_Pantheon18_lensing
g = plots.getSubplotPlotter(width_inch=8)
g.settings.axes_fontsize = 12
g.settings.legend_fontsize = 14
g.triangle_plot([chain_type['samples'],samples_DESI[0],samples_DESI[1],samples_DESI[2]],
                ['linP_DL2_star','linP_n_star','nrun','nnu','w','mnu'],
                legend_labels=["Planck 2018 + SDSS + SN",labels_DESI[0],labels_DESI[1],labels_DESI[2]])
#plt.savefig(plot_label+'.pdf')
#plt.savefig(plot_label+'.png')

plt.tight_layout()
plt.savefig("figs/import_nrun_nnu_w_mnu.png")
plt.savefig("figs/import_nrun_nnu_w_mnu.pdf")

# %% [markdown]
# ### base_r

# %%
samples_DESI=[]
true_DL2=[0.33, 0.35, 0.37]
true_neff=[-2.295,-2.30,-2.305]
DL2_err = 0.056
neff_err = 0.022
r=-0.134
coeff_mult = 2.3 # should it be 0.5?

chain_type = cmb_r


labels_DESI=[]
for ii in range(len(true_DL2)):
    new_samples=chain_type['samples'].copy()
    p=new_samples.getParams()
    new_loglike = coeff_mult * gaussian_chi2_mock_DESI(
        p.linP_n_star, 
        p.linP_DL2_star, 
        true_DL2=true_DL2[ii],
        true_neff=true_neff[ii],
        neff_err=neff_err,
        DL2_err=DL2_err,
        r=r
    )
    new_samples.reweightAddingLogLikes(new_loglike) #re-weight cut_samples to account for the new likelihood
    samples_DESI.append(new_samples)
    labels_DESI.append(r'+ DESI Ly$\alpha \,(\Delta^2_\ast = ${} $n_\ast = ${})'.format(true_DL2[ii], true_neff[ii]))


# plikHM_TTTEEE_lowl_lowE_BAO_Riess18_Pantheon18_lensing
g = plots.getSubplotPlotter(width_inch=8)
g.settings.axes_fontsize = 12
g.settings.legend_fontsize = 14
g.triangle_plot([chain_type['samples'],samples_DESI[0],samples_DESI[1],samples_DESI[2]],
                ['linP_DL2_star','linP_n_star','r'],
                legend_labels=["Planck 2018 + SDSS + SN",labels_DESI[0],labels_DESI[1],labels_DESI[2]])
#plt.savefig(plot_label+'.pdf')
#plt.savefig(plot_label+'.png')

plt.tight_layout()
plt.savefig("figs/import_r.png")
plt.savefig("figs/import_r.pdf")

# %%
