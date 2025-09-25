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
# # Compare historical CMB chains
#
# Compare linear power parameters from WMAP9 and Planck (2013, 2015, 2018).
#
# All chains have free neutrino mass, CMB info only.

# %% jupyter={"outputs_hidden": false}
# %load_ext autoreload
# %autoreload 2
import numpy as np
import os
from getdist import plots,loadMCSamples
import matplotlib.pyplot as plt
from cup1d.planck import planck_chains
from cup1d.likelihood import marg_lya_like
# because of black magic, getdist needs this strange order of imports
# %matplotlib inline
from cup1d.utils.utils import get_path_repo


from matplotlib import rcParams

rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "STIXGeneral"

# %% [markdown]
# ### Read extended CMB chains from historical releases
#
# These chains are already provided in cup1d, form different CMB releases. All chains have free neutrino mass.

# %% jupyter={"outputs_hidden": false}
# model='base_mnu'
model='base'
# WMAP9
# wmap9=planck_chains.get_planck_2013(model=model,data='WMAP')
# wmap9['label']='WMAP 9'
# # Planck 2013
# planck2013=planck_chains.get_planck_2013(model=model)
# planck2013['label']='Planck 2013'
# # Planck 2015
# planck2015=planck_chains.get_planck_2015(model=model)
# planck2015['label']='Planck 2015'
# Planck 2018
# planck2018=planck_chains.get_planck_2018(model=model)
# planck2018['label']='Planck 2018'

root_dir=os.path.join(get_path_repo("cup1d"), "data", "planck_linP_chains")
cmb = planck_chains.get_planck_2018(
    model='base',
    data='plikHM_TTTEEE_lowl_lowE_linP',
    root_dir=root_dir,
    linP_tag=None
)


# %%
# # put everything together to pass to getdist
# analyses=[wmap9,planck2013,planck2015,planck2018]
# samples=[analysis['samples'] for analysis in analyses]
# labels=[analysis['label'] for analysis in analyses]

# %%

# params=['w', 'wa']
# g = plots.getSubplotPlotter(width_inch=14)
# g.settings.axes_fontsize = 12
# g.settings.legend_fontsize = 16
# g.triangle_plot(cmbw['samples'],params)

# %% jupyter={"outputs_hidden": false}
# # plot traditional parameters
# params=['omegach2','omegabh2','mnu','H0','tau','logA','ns']
# g = plots.getSubplotPlotter(width_inch=14)
# g.settings.axes_fontsize = 12
# g.settings.legend_fontsize = 16
# g.triangle_plot(samples,params,legend_labels=labels)

# %% [markdown]
# ### Plot linear power parameters

# %% jupyter={"outputs_hidden": false}
# # plot traditional parameters
# params=['linP_DL2_star','linP_n_star','linP_alpha_star','linP_f_star','linP_g_star']
# g = plots.getSubplotPlotter(width_inch=14)
# g.settings.axes_fontsize = 12
# g.settings.legend_fontsize = 16
# g.triangle_plot(samples,params,legend_labels=labels)

# %% jupyter={"outputs_hidden": false}
# # plot traditional parameters (including now Omega_m and m_nu)
# params=['linP_DL2_star','linP_n_star','linP_alpha_star','linP_f_star','linP_g_star','omegam','mnu']
# g = plots.getSubplotPlotter(width_inch=14)
# g.settings.axes_fontsize = 12
# g.settings.legend_fontsize = 16
# g.triangle_plot(samples,params,legend_labels=labels)

# %% [markdown]
# ### Plot linear power parameters from chain and from Lya likelihoods

# %%
fit_type = "global_opt"
data_lab = "DESIY1_QMLE3"
emu = "mpg"
file = "../tutorials/out_pl/"+ data_lab + "_" + emu + "_" + fit_type + ".npy"
out_dict = np.load(file, allow_pickle=True).item()

# %%
true_DL2=0.35
true_neff=-2.3

# DL2_err = 0.056
# neff_err = 0.022
# r=-0.134
coeff_mult = 1

# create grid (note j in number of elements, crazy python)
# thresholds = [2.30,6.17,11.8]
# thresholds = [2.30, 6.18]
thresholds = [2.30]
neff_grid, DL2_grid = np.mgrid[-2.4:-2.2:200j, 0.2:0.65:200j]
chi2_Mc2005=coeff_mult * marg_lya_like.gaussian_chi2_McDonald2005(neff_grid,DL2_grid)
chi2_PD2015=coeff_mult * marg_lya_like.gaussian_chi2_PalanqueDelabrouille2015(neff_grid,DL2_grid)
chi2_Ch2019=coeff_mult * marg_lya_like.gaussian_chi2_Chabanier2019(neff_grid,DL2_grid)
chi2_Wa2024=coeff_mult * marg_lya_like.gaussian_chi2_Walther2024(neff_grid,DL2_grid)


# chi2_desi = marg_lya_like.gaussian_chi2(neff_grid, DL2_grid, out_dict["ycen_2d"], out_dict["xcen_2d"], out_dict["err_n_star"], out_dict["err_Delta2_star"], out_dict["rho"])
chi2_desi = marg_lya_like.gaussian_chi2(neff_grid, DL2_grid, true_neff, true_DL2, out_dict["err_n_star"], out_dict["err_Delta2_star"], out_dict["rho"])


# %%
# plt.contour(DL2_grid, neff_grid,chi2_desi,levels=thresholds[:2],colors='C0')
# plt.xlim(0.4, 0.63)
# plt.ylim(-2.34, -2.2)

# %%
def print_err(ell):
    ver = ell.collections[0].get_paths()[0].vertices
    xcen = ver[:,0].mean()
    ycen = ver[:,1].mean()
    xerr = 0.5 * (ver[:,0].max() - ver[:,0].min())
    yerr = 0.5 * (ver[:,1].max() - ver[:,1].min())
    print(np.round(xerr, 3), np.round(yerr, 3))
    return xcen, ycen, xerr, yerr

# %%

# g = plots.getSinglePlotter(width_inch=8)
# g.plot_2d(cmb, ['linP_n_star', 'linP_DL2_star'],lims=[-2.4,-2.25, 0.2,0.5])


# g = plots.getSubplotPlotter(width_inch=8)
# g.settings.axes_fontsize = 10
# g.settings.legend_fontsize = 14
# g.triangle_plot([cmb['samples']],
#                 ['linP_n_star', 'linP_DL2_star'],
#                 legend_labels=[r'$\Lambda$CDM'])

# %%
import scipy.stats as stats
import matplotlib.lines as mlines

# %%
cmb.keys()

# %% jupyter={"outputs_hidden": false}
labs = ['McDonald+05', 'Palanque-Delabrouille+15', 'Chabanier+19', 'Walther+24', 'This work']

fontsize = 18

# specify CMB chain to plot
# cmb=wmap9
# g = plots.getSinglePlotter(width_inch=8)
# g.plot_2d(cmb, ['linP_n_star', 'linP_DL2_star'],lims=[-2.4,-2.25, 0.2,0.5])
# g.plot_2d(cmb['samples'], ['linP_n_star', 'linP_DL2_star'],lims=[-2.4,-2.25,0.2,0.5])
#g.plot_2d(planck2018_mnu_BAO['samples'], ['linP_n_star', 'linP_DL2_star'],lims=[-2.4,-2.25,0.2,0.5])

g = plots.getSubplotPlotter(width_inch=8)
g.settings.num_plot_contours = 1

g.settings.axes_fontsize = 10
g.settings.legend_fontsize = 14
g.triangle_plot(
    [cmb['samples']],
    ['linP_n_star', 'linP_DL2_star'],
    # legend_labels=[r'Planck+18 $\Lambda$CDM']
)

ax = g.subplots[1, 0]
ax_NS = g.subplots[0, 0]
ax_D2S = g.subplots[1, 1]
x_D2S = np.linspace(0.2, 0.5, 500)
x_NS = np.linspace(-2.4, -2.2, 500)

CS = ax.contour(neff_grid,DL2_grid,chi2_Mc2005,levels=thresholds[:2],colors='C0')
xcen, ycen, xerr, yerr = print_err(CS)
pdf = stats.norm.pdf(x_D2S, ycen, yerr)
ax_D2S.plot(x_D2S, pdf/pdf.max())
pdf = stats.norm.pdf(x_NS, xcen, xerr)
ax_NS.plot(x_NS, pdf/pdf.max())

CS = ax.contour(neff_grid,DL2_grid,chi2_PD2015,levels=thresholds[:2],colors='C1')
xcen, ycen, xerr, yerr = print_err(CS)
pdf = stats.norm.pdf(x_D2S, ycen, yerr)
ax_D2S.plot(x_D2S, pdf/pdf.max())
pdf = stats.norm.pdf(x_NS, xcen, xerr)
ax_NS.plot(x_NS, pdf/pdf.max())

CS = ax.contour(neff_grid,DL2_grid,chi2_Ch2019,levels=thresholds[:2],colors='C2')
xcen, ycen, xerr, yerr = print_err(CS)
pdf = stats.norm.pdf(x_D2S, ycen, yerr)
ax_D2S.plot(x_D2S, pdf/pdf.max())
pdf = stats.norm.pdf(x_NS, xcen, xerr)
ax_NS.plot(x_NS, pdf/pdf.max())

CS = ax.contour(neff_grid,DL2_grid,chi2_Wa2024,levels=thresholds[:2],colors='C3')
xcen, ycen, xerr, yerr = print_err(CS)
pdf = stats.norm.pdf(x_D2S, ycen, yerr)
ax_D2S.plot(x_D2S, pdf/pdf.max())
pdf = stats.norm.pdf(x_NS, xcen, xerr)
ax_NS.plot(x_NS, pdf/pdf.max())

CS = ax.contour(neff_grid,DL2_grid,chi2_desi,levels=thresholds[:2],colors='C4')
xcen, ycen, xerr, yerr = print_err(CS)
pdf = stats.norm.pdf(x_D2S, ycen, yerr)
ax_D2S.plot(x_D2S, pdf/pdf.max())
pdf = stats.norm.pdf(x_NS, xcen, xerr)
ax_NS.plot(x_NS, pdf/pdf.max())

# plt.axhline(y=1,color='C0',label=cmb['label'])
for ii in range(len(labs)):
    ax.axhline(y=1,color='C'+str(ii),label=labs[ii])

ax.set_xlim(-2.4, -2.2)
ax.set_ylim(0.2, 0.5)

# plt.title(r'Linear power constraints at ($z=3$, $k_p=0.009$ s/km)')
# plt.grid()  

# ax.legend()

 # $\Lambda$CDM
handles = []
handles.append(mlines.Line2D([], [], color='black', label=r'Planck+18', lw=2))
for ii in range(len(labs)):
    handles.append(mlines.Line2D([], [], color='C'+str(ii), label=labs[ii], lw=2))
    
g.subplots[1, 1].legend(
    handles=handles, 
    bbox_to_anchor=(1, 2),
    loc='upper right', 
    borderaxespad=0.,
    fontsize=fontsize-2
)

ax.tick_params(
    axis="both", which="major", labelsize=fontsize
)
ax_D2S.tick_params(
    axis="both", which="major", labelsize=fontsize
)


ax.set_ylabel(r"$\Delta_\star$", fontsize=fontsize)
ax.set_xlabel(r"$n_\star$", fontsize=fontsize)
ax_D2S.set_xlabel(r"$\Delta_\star$", fontsize=fontsize)


plt.tight_layout()
# plt.savefig("figs/star_literature.png")
# plt.savefig("figs/star_literature.pdf")

# %%

# CS = plt.contour(DL2_grid, neff_grid, chi2_Ch2019, levels=thresholds[:2],colors='C0')
# print_err(CS)

# # plt.ylim(-2.41, -2.29)
# # plt.ylim(-2.4, -2.25)
# plt.ylim(-2.35, -2.33)
# # plt.xlim(0.14, 0.44)
# # plt.xlim(0.20, 0.40)
# plt.xlim(0.28, 0.36)
# plt.grid()

# %%
