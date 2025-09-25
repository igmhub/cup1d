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


# cmb_nrun_nnu_w_mnu = planck_chains.get_planck_2018(
#     model='base_nrun_nnu_w_mnu',
#     data='plikHM_TTTEEE_lowl_lowE_BAO_Riess18_Pantheon18_lensing_linP',
#     root_dir=root_dir,
#     linP_tag=None
# )

# %%
{'x0': 0.443151934816415,
 'y0': -2.2830673979023928,
 'a': 0.05671364141407182,
 'b': 0.02081226136234671,
 'theta': -0.13448802462450818}

# %%
cmb["samples"].getParamSampleDict(0).keys()

# %%
fit_type = "global_opt"
data_lab = "DESIY1_QMLE3"
emu = "mpg"
file = "../tutorials/out_pl/"+ data_lab + "_" + emu + "_" + fit_type + ".npy"
out_dict = np.load(file, allow_pickle=True).item()

# %%
ftsize = 22
lw = 3
cmb_all = [cmb, cmb_tau, cmb_mnu, cmb_nnu, cmb_nrun, cmb_nrun_nrunrun]
cmb_labs = [
    r"Planck $\Lambda$CDM", 
    r"Planck $\Lambda$CDM (no lowE)",
    r"Planck $\sum m_\nu$",
    r"Planck $N_\mathrm{eff}$",
    r"Planck $\mathrm{d}n_\mathrm{s} / \mathrm{d}\log k$",
    r"Planck $\mathrm{d}n_\mathrm{s} / \mathrm{d}\log k$, $\mathrm{d}^2 n_\mathrm{s} / \mathrm{d}\log k^2$"
]

g = plots.getSinglePlotter(width_inch=10)
g.settings.num_plot_contours = 1

for ii, icmb in enumerate(cmb_all):
    if ii == 0:
        filled = True
    else:
        filled = False
    g.plot_2d(
        icmb['samples'], 
        ['linP_DL2_star', 'linP_n_star'], 
        colors=["C"+str(ii)], 
        lws=lw, 
        alphas=0.8,
        filled=filled,
    )

ax = g.subplots[0,0]
ax.axhline(y=1,color='k',lw=lw,label=r"DESI-DR1 Ly$\alpha$ (this work)")
for ii, icmb in enumerate(cmb_all):
    ax.axhline(y=1,color="C"+str(ii),lw=lw,label=cmb_labs[ii])    



true_DL2=0.35
true_neff=-2.3

# DL2_err = 0.056
# neff_err = 0.022
# r=-0.134
# thresholds = [2.30, 6.18]
thresholds = [2.30]
neff_grid, DL2_grid = np.mgrid[-2.4:-2.2:200j, 0.2:0.65:200j]

# chi2_desi = coeff_mult * marg_lya_like.gaussian_chi2(neff_grid,DL2_grid, true_neff, true_DL2, neff_err, DL2_err, r)
chi2_desi = marg_lya_like.gaussian_chi2(neff_grid, DL2_grid, true_neff, true_DL2, out_dict["err_n_star"], out_dict["err_Delta2_star"], out_dict["rho"])
CS = ax.contour(DL2_grid, neff_grid, chi2_desi, levels=thresholds[:2], colors='k', linewidths=lw)



ax.set_xlim(0.23, 0.47)
ax.set_ylim(-2.37, -2.23)
ax.set_ylabel(r"$n_\star$", fontsize=ftsize)
ax.set_xlabel(r"$\Delta^2_\star$", fontsize=ftsize)
ax.tick_params(axis="both", which="major", labelsize=ftsize)

plt.legend(fontsize=ftsize-4, loc="upper left", ncol=2)
# plt.tight_layout()

plt.savefig("figs/star_planck_mine.png", bbox_inches='tight')
plt.savefig("figs/star_planck_mine.pdf", bbox_inches='tight')


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
DL2_err = out_dict["err_Delta2_star"]
neff_err = out_dict["err_n_star"]
r=out_dict["rho"]
coeff_mult = 1
# coeff_mult = 2.3 # should it be 0.5?

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
samples_DESI=[]
true_DL2=[0.33,0.35,0.37]
true_neff=[-2.295,-2.30,-2.305]
DL2_err = out_dict["err_Delta2_star"]
neff_err = out_dict["err_n_star"]
r=out_dict["rho"]
coeff_mult = 1 # should it be 0.5?

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
DL2_err = 0.056
neff_err = 0.022
r=-0.134
coeff_mult = 2.3 # should it be 0.5?

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
DL2_err = 0.056
neff_err = 0.022
r=-0.134
coeff_mult = 2.3 # should it be 0.5?

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
samples_DESI=[]
true_DL2=[0.33,0.35,0.37]
true_neff=[-2.295,-2.30,-2.305]
DL2_err = 0.056
neff_err = 0.022
r=-0.134
coeff_mult = 2.3 # should it be 0.5?

chain_type = cmb_mnu


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


# g = plots.getSubplotPlotter(width_inch=8)
# g.settings.axes_fontsize = 12
# g.settings.legend_fontsize = 14
# g.triangle_plot([chain_type['samples'],samples_DESI[0],samples_DESI[1],samples_DESI[2]],
#                 ['linP_DL2_star','linP_n_star', 'sigma8', 'mnu'],
#                 legend_labels=["Planck 2018",labels_DESI[0],labels_DESI[1],labels_DESI[2]])


g = plots.getSubplotPlotter(width_inch=8)
g.settings.axes_fontsize = 12
g.settings.legend_fontsize = 14
g.triangle_plot([chain_type['samples'],samples_DESI[0],samples_DESI[1],samples_DESI[2]],
                ['linP_DL2_star','linP_n_star','mnu'],
                legend_labels=["Planck 2018",labels_DESI[0],labels_DESI[1],labels_DESI[2]])


print(chain_type['samples'].getInlineLatex('mnu',limit=1))
# print(samples_DESI[0].getInlineLatex('mnu',limit=1))
print(samples_DESI[1].getInlineLatex('mnu',limit=1))
# print(samples_DESI[2].getInlineLatex('mnu',limit=1))
# print("")
print(chain_type['samples'].getInlineLatex('mnu',limit=2))
# print(samples_DESI[0].getInlineLatex('mnu',limit=2))
print(samples_DESI[1].getInlineLatex('mnu',limit=2))
# print(samples_DESI[2].getInlineLatex('mnu',limit=2))

plt.tight_layout()
plt.savefig("figs/import_neutrinos.png")
plt.savefig("figs/import_neutrinos.pdf")

# %%
cmb_nrun["samples"].getParamSampleDict(0).keys()

# %% [markdown]
# #### Constraints on nrun

# %%
samples_DESI=[]
true_DL2=[0.33,0.35,0.37]
true_neff=[-2.295,-2.30,-2.305]
DL2_err = out_dict["err_Delta2_star"]
neff_err = out_dict["err_n_star"]
r=out_dict["rho"]
coeff_mult = 1 # should it be 0.5?

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

print(chain_type['samples'].getInlineLatex('nrun',limit=1))
print(samples_DESI[1].getInlineLatex('nrun',limit=1))

plt.tight_layout()
plt.savefig("figs/import_nrun.png")
plt.savefig("figs/import_nrun.pdf")

# %% [markdown]
# #### Constraints on nnu

# %%
samples_DESI=[]
true_DL2=[0.33,0.35,0.37]
true_neff=[-2.295,-2.30,-2.305]
DL2_err = out_dict["err_Delta2_star"]
neff_err = out_dict["err_n_star"]
r=out_dict["rho"]
coeff_mult = 1 # should it be 0.5?

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
samples_DESI=[]
true_DL2=[0.33,0.35,0.37]
true_neff=[-2.295,-2.30,-2.305]
DL2_err = out_dict["err_Delta2_star"]
neff_err = out_dict["err_n_star"]
r=out_dict["rho"]
coeff_mult = 1 # should it be 0.5?

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
DL2_err = 0.056
neff_err = 0.022
r=-0.134
coeff_mult = 2.3 # should it be 0.5?

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
