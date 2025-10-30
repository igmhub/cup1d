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
from getdist import plots,loadMCSamples
import matplotlib.pyplot as plt
from cup1d.planck import planck_chains
from cup1d.planck import add_linP_params
from cup1d.likelihood import marg_lya_like
# because of black magic, getdist needs this strange order of imports
# %matplotlib inline
from cup1d.utils.utils import get_path_repo

from matplotlib import rcParams

rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "STIXGeneral"

# %% [markdown]
# ### Read an extended Planck chains and plot linear power parameters
#
# Planck 2018 with free neutrino mass, already provided in cup1d

# %%
# add_BAO=True
# if add_BAO:
#     planck_data='plikHM_TTTEEE_lowl_lowE_BAO'
#     planck_label='Planck 2018 + BAO'
#     plot_label='Planck_BAO_LyaDESI_mnu'
# else:
#     planck_data='plikHM_TTTEEE_lowl_lowE'
#     planck_label='Planck 2018'
#     plot_label='Planck_LyaDESI_mnu'
# # model with massive neutrinos
# model='base_mnu'
# planck2018=planck_chains.get_planck_2018(model=model,data=planck_data)

# %%

# %%

root_dir=os.path.join(get_path_repo("cup1d"), "data", "planck_linP_chains")

cmb = planck_chains.get_planck_2018(
    model='base',
    data='plikHM_TTTEEE_lowl_lowE_linP',
    root_dir=root_dir,
    linP_tag=None
)

cmb_tau = planck_chains.get_planck_2018(
    model='base',
    data='plikHM_TTTEEE_lowl_linP',
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

# cmb_r = planck_chains.get_planck_2018(
#     model='base_r',
#     data='plikHM_TTTEEE_lowl_lowE_linP',
#     root_dir=root_dir,
#     linP_tag=None
# )

cmb_omega_k = planck_chains.get_planck_2018(
    model='base_omegak',
    data='plikHM_TTTEEE_lowl_lowE_linP',
    root_dir=root_dir,
    linP_tag=None
)

cmb_w_wa = planck_chains.get_planck_2018(
    model='base_w_wa',
    data='plikHM_TTTEEE_lowl_lowE_BAO_linP',
    root_dir=root_dir,
    linP_tag=None
)


# cmb_nrun_nnu_w_mnu = planck_chains.get_planck_2018(
#     model='base_nrun_nnu_w_mnu',
#     data='plikHM_TTTEEE_lowl_lowE_BAO_Riess18_Pantheon18_lensing_linP',
#     root_dir=root_dir,
#     linP_tag=None
# )

# %%
# cmb["samples"].getParamSampleDict(0).keys()

# %%
# fit_type = "global_opt"
# data_lab = "DESIY1_QMLE3"
# emu = "mpg"
# file = "../tutorials/out_pl/"+ data_lab + "_" + emu + "_" + fit_type + ".npy"
# out_dict = np.load(file, allow_pickle=True).item()

base = "/home/jchaves/Proyectos/projects/lya/data/out_DESI_DR1/"
folder = base + "DESIY1_QMLE3/global_opt/CH24_mpgcen_gpr/chain_1/"

dat_mpg = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()
sum_mpg = np.load(folder + "summary.npy", allow_pickle=True).item()
true_cosmo = {
    'Delta2_star': sum_mpg["delta2_star_16_50_84"][1]-np.median(cmb["samples"]["linP_DL2_star"]),
     'n_star': sum_mpg["n_star_16_50_84"][1]-np.median(cmb["samples"]["linP_n_star"]),
}
true_cosmo

# %%

blobs = np.load(folder + "blobs.npy")
corr = np.corrcoef(blobs["Delta2_star"].reshape(-1), blobs["n_star"].reshape(-1))
# per_ds = np.percentile(blobs["Delta2_star"].reshape(-1), [16, 50, 84])
# per_ns = np.percentile(blobs["n_star"].reshape(-1), [16, 50, 84])
r = corr[0, 1]

# %%
desi_dr1 = {
    "Delta2_star":sum_mpg["delta2_star_16_50_84"][1] - true_cosmo["Delta2_star"],
    "n_star":sum_mpg["n_star_16_50_84"][1] - true_cosmo["n_star"],
    "r":r,
    "Delta2_star_err":sum_mpg['delta2_star_err'],
    "n_star_err":sum_mpg["n_star_err"],
}

# %%

# %%
ftsize = 26
cmap = plt.colormaps["Blues"]
col = [0.7, 0.3]
lw = [3, 2]
cmb_all = [
    cmb, 
    # cmb_tau, 
    # cmb_mnu, 
    cmb_nnu, 
    cmb_nrun, 
    # cmb_nrun_nrunrun
]
cmb_labs = [
    r"Planck 2018: $\Lambda$CDM", 
    # r"Planck+18 $\Lambda$CDM (no lowE)",
    # r"Planck+18 $\sum m_\nu$",
    r"Planck 2018: $N_\mathrm{eff}$",
    r"Planck 2018: $\alpha_\mathrm{s}$",
    # r"Planck+18 $\alpha_\mathrm{s}$, $\mathrm{d} \alpha_\mathrm{s} / \mathrm{d}\log k$"
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
        lwu = 3
    g.plot_2d(
        new_samples, 
        ['linP_DL2_star', 'linP_n_star'], 
        colors=["C"+str(ii+1)], 
        lws=lwu, 
        alphas=[0.8, 0.5],
        filled=filled,
    )

ax = g.subplots[0,0]


for inum, num in enumerate([0.68, 0.95]):
    if inum == 0:
        label="This work"
    else:
        label=None
    for jj in range(len(dat_mpg[num])):
        x = dat_mpg[num][jj][0] - true_cosmo["Delta2_star"]
        y = dat_mpg[num][jj][1] - true_cosmo["n_star"]
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

plt.legend(fontsize=ftsize, loc="upper left", ncol=2)
# plt.tight_layout()

plt.savefig("figs/star_planck_mine.png", bbox_inches='tight')
plt.savefig("figs/star_planck_mine.pdf", bbox_inches='tight')


# %%

# %%

# %%
# from matplotlib import colormaps

# %%
# # plot also neutrino mass (for nuLCDM)
# g = plots.getSubplotPlotter(width_inch=10)
# g.settings.axes_fontsize = 10
# g.settings.legend_fontsize = 14
# g.triangle_plot(planck2018['samples'],
#                 ['linP_DL2_star','linP_n_star','linP_alpha_star','linP_f_star','linP_g_star','omegam','mnu'],
#                 legend_labels=[planck_label])

# %% [markdown]
# ### Add mock DESI Lya likelihood
#
# We will generate three fake DESI likelihoods, for different fiducial values, and see how they affect the cosmo params.

# %%
def gaussian_chi2_mock_DESI(neff, DL2, true_DL2=0.35, true_neff=-2.3, DL2_err=0.003, neff_err=0.002, r=0.55):
    """Compute Gaussian Delta chi^2 for a particular point(s) (neff,DL2),
    using a mock measurement from DESI (complete made up at this point).
    """
    # # DL2 = k^3 P(k) / (2 pi^2), at z=3
    # DL2_err=0.003
    # # neff = effective slope at kp = 0.009 s/km, i.e., d ln P / dln k
    # neff_err=0.002
    # # correlation coefficient
    # r=0.55
    return marg_lya_like.gaussian_chi2(neff, DL2, true_neff, true_DL2, neff_err, DL2_err, r)


# %%
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FormatStrFormatter

def plot_combine(chain_type, desi_dr1, param_name, fontsize=24):
    latex_param = {
        "nnu": r"$N_\mathrm{eff}$",
        "tau": r"$\tau$",
        "mnu": r"$\sum m_\nu$",
        "nrun": r"$\alpha_\mathrm{s}$",
        "nrunrun": r"$\mathrm{d} \alpha_\mathrm{s} / \mathrm{d}\log k$"
    }
    samples_DESI=[]
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
        labels_DESI.append(r"Planck 2018 + DESI DR1")
    
    g = plots.getSubplotPlotter(width_inch=8)
    g.settings.axes_fontsize = fontsize
    g.settings.legend_fontsize = fontsize

    if param_name == "nrunrun":
        arr_plot = ['linP_DL2_star','linP_n_star', "nrun", param_name]
    else:
        arr_plot = ['linP_DL2_star','linP_n_star', param_name]

    
    g.triangle_plot(
        [chain_type['samples'],samples_DESI[0]],
        arr_plot,
        legend_labels=["Planck 2018",labels_DESI[0]],
        colors=["C0", "C1"],
        filled=True,
        lws=[3,3],
        alphas=[0.8, 0.8],
        line_args=[
            {"color": "C0", "lw": 3, "alpha": 0.8},
            {"color": "C1", "lw": 3, "alpha": 0.8},
        ],
    )

    if param_name == "mnu":
        print("1 sigma", param_name, chain_type['samples'].getInlineLatex(param_name,limit=1))
        print("1 sigma", param_name, samples_DESI[0].getInlineLatex(param_name,limit=1))
        print("2 sigma", param_name, chain_type['samples'].getInlineLatex(param_name,limit=2))
        print("2 sigma", param_name, samples_DESI[0].getInlineLatex(param_name,limit=2))
    else:
        if (param_name != "nrunrun"): 
            print(chain_type['samples'].getInlineLatex(param_name, limit=1))
            print(samples_DESI[0].getInlineLatex(param_name, limit=1))
        else:
            print("nrun", chain_type['samples'].getInlineLatex("nrun", limit=1))
            print("nrun", samples_DESI[0].getInlineLatex("nrun", limit=1))
            print(param_name, chain_type['samples'].getInlineLatex(param_name, limit=1))
            print(param_name, samples_DESI[0].getInlineLatex(param_name, limit=1))
            

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

        
    plt.tight_layout()
    plt.savefig("figs/import_"+param_name+".png", bbox_inches="tight")
    plt.savefig("figs/import_"+param_name+".pdf", bbox_inches="tight")

# %%

plot_combine(cmb_nnu, desi_dr1, "nnu")

# %%

plot_combine(cmb_nrun, desi_dr1, "nrun")

# %%

plot_combine(cmb_mnu, desi_dr1, "mnu")

# %%
# plot_combine(cmb_tau, desi_dr1, "tau")
# plot_combine(cmb_mnu, desi_dr1, "mnu")
# plot_combine(cmb_nnu, desi_dr1, "nnu")
# plot_combine(cmb_nrun, desi_dr1, "nrun")
# plot_combine(cmb_nrun_nrunrun, desi_dr1, "nrunrun")

# %%
1-46/67

# %%
1-16/19

# %%

# %%

# %%
# true_DL2=0.35
# true_neff=-2.3

# DL2_err = 0.06
# neff_err = 0.02
# r=-0.134
# coeff_mult = 2.3

# chi2_desi = marg_lya_like.gaussian_chi2(neff, DL2, true_neff, true_DL2, neff_err, DL2_err, r)

# %%
# {'x0': 0.443151934816415,
#  'y0': -2.2830673979023928,
#  'a': 0.05671364141407182,
#  'b': 0.02081226136234671,
#  'theta': -0.13448802462450818}

# %%
# true_DL2=0.35
# true_neff=-2.3
# chi2_desi = marg_lya_like.gaussian_chi2(neff_grid, DL2_grid, true_neff, true_DL2, out_dict["err_n_star"], out_dict["err_Delta2_star"], out_dict["rho"])

# %%
new_loglike.shape

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
