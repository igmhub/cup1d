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

# %% [markdown]
# ### Read an extended Planck chains and plot linear power parameters
#
# Planck 2018 with free neutrino mass, already provided in cup1d

# %%
add_BAO=True
if add_BAO:
    planck_data='plikHM_TTTEEE_lowl_lowE_BAO'
    planck_label='Planck 2018 + BAO'
    plot_label='Planck_BAO_LyaDESI_mnu'
else:
    planck_data='plikHM_TTTEEE_lowl_lowE'
    planck_label='Planck 2018'
    plot_label='Planck_LyaDESI_mnu'
# model with massive neutrinos
model='base_mnu'
planck2018=planck_chains.get_planck_2018(model=model,data=planck_data)

# %%

root_dir=os.path.join(get_path_repo("cup1d"), "data", "planck_linP_chains")

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

cmb_omega_k = planck_chains.get_planck_2018(
    model='base_omegak',
    data='plikHM_TTTEEE_lowl_lowE_BAO_linP',
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
#     data='plikHM_TTTEEE_lowl_lowE_BAO_Riess18_Pantheon18_lensing',
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
g = plots.getSinglePlotter(width_inch=6)
g.plot_2d(cmb['samples'], ['linP_DL2_star', 'linP_n_star'])
#g.plot_2d(planck2018['samples'], ['linP_n_star', 'linP_DL2_star'],lims=[-2.4,-2.25,0.2,0.5])

true_DL2=0.35
true_neff=-2.3

DL2_err = 0.056
neff_err = 0.022
r=-0.134
coeff_mult = 2.3
thresholds = [2.30, 6.18]
neff_grid, DL2_grid = np.mgrid[-2.4:-2.2:200j, 0.2:0.65:200j]

chi2_desi = coeff_mult * marg_lya_like.gaussian_chi2(neff_grid,DL2_grid, true_neff, true_DL2, neff_err, DL2_err, r)

CS = plt.contour(DL2_grid, neff_grid, chi2_desi, levels=thresholds[:2], colors='green')
plt.xlim(0.25, 0.45)
plt.ylim(-2.35, -2.25)


# %%
def print_err(ell):
    ver = ell.collections[0].get_paths()[0].vertices
    xerr = 0.5 * (ver[:,0].max() - ver[:,0].min())
    yerr = 0.5 * (ver[:,1].max() - ver[:,1].min())
    print(xerr, yerr)


# %%
print_err(CS)

# %%
# plot also neutrino mass (for nuLCDM)
g = plots.getSubplotPlotter(width_inch=10)
g.settings.axes_fontsize = 10
g.settings.legend_fontsize = 14
g.triangle_plot(planck2018['samples'],
                ['linP_DL2_star','linP_n_star','linP_alpha_star','linP_f_star','linP_g_star','omegam','mnu'],
                legend_labels=[planck_label])


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
true_DL2=0.35
true_neff=-2.3

DL2_err = 0.06
neff_err = 0.02
r=-0.134
coeff_mult = 2.3

chi2_desi = marg_lya_like.gaussian_chi2(neff, DL2, true_neff, true_DL2, neff_err, DL2_err, r)

# %%
# {'x0': 0.443151934816415,
#  'y0': -2.2830673979023928,
#  'a': 0.05671364141407182,
#  'b': 0.02081226136234671,
#  'theta': -0.13448802462450818}

# %%

# %% [markdown]
# #### Almost no constraints on As, ns

# %%
samples_DESI=[]
true_DL2=[0.33,0.35,0.37]
DL2_err = 0.056
neff_err = 0.022
r=-0.134
coeff_mult = 2.3 # should it be 0.5?

chain_type = cmb


# new_samples=cmb_w_wa['samples'].copy()
# new_samples=cmb_omega_k['samples'].copy()
# new_samples=cmb_mnu['samples'].copy()
# new_samples=cmb_nrun['samples'].copy()


labels_DESI=[]
for ii in range(len(true_DL2)):
    new_samples=chain_type['samples'].copy()
    p=new_samples.getParams()
    new_loglike = coeff_mult * gaussian_chi2_mock_DESI(
        p.linP_n_star, 
        p.linP_DL2_star, 
        true_DL2=true_DL2[ii],
        neff_err=neff_err,
        DL2_err=DL2_err,
        r=r
    )
    new_samples.reweightAddingLogLikes(new_loglike) #re-weight cut_samples to account for the new likelihood
    samples_DESI.append(new_samples)
    labels_DESI.append(r'+ DESI Ly$\alpha \, \Delta^2_\ast = ${}'.format(true_DL2[ii]))


g = plots.getSubplotPlotter(width_inch=16)
g.settings.axes_fontsize = 10
g.settings.legend_fontsize = 14
g.triangle_plot([chain_type['samples'],samples_DESI[0],samples_DESI[1],samples_DESI[2]],
                ['linP_DL2_star','linP_n_star','logA','ns'],
                legend_labels=['Planck 2018 + SDSS',labels_DESI[0],labels_DESI[1],labels_DESI[2]])

# %% [markdown]
# #### No constraints on w0, wa

# %%
samples_DESI=[]
true_DL2=[0.33,0.35,0.37]
DL2_err = 0.056
neff_err = 0.022
r=-0.134
coeff_mult = 2.3 # should it be 0.5?

chain_type = cmb_w_wa


# new_samples=cmb_w_wa['samples'].copy()
# new_samples=cmb_omega_k['samples'].copy()
# new_samples=cmb_mnu['samples'].copy()
# new_samples=cmb_nrun['samples'].copy()


labels_DESI=[]
for ii in range(len(true_DL2)):
    new_samples=chain_type['samples'].copy()
    p=new_samples.getParams()
    new_loglike = coeff_mult * gaussian_chi2_mock_DESI(
        p.linP_n_star, 
        p.linP_DL2_star, 
        true_DL2=true_DL2[ii],
        neff_err=neff_err,
        DL2_err=DL2_err,
        r=r
    )
    new_samples.reweightAddingLogLikes(new_loglike) #re-weight cut_samples to account for the new likelihood
    samples_DESI.append(new_samples)
    labels_DESI.append(r'+ DESI Ly$\alpha \, \Delta^2_\ast = ${}'.format(true_DL2[ii]))


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
DL2_err = 0.056
neff_err = 0.022
r=-0.134
coeff_mult = 2.3 # should it be 0.5?

chain_type = cmb_omega_k


# new_samples=cmb_w_wa['samples'].copy()
# new_samples=cmb_omega_k['samples'].copy()
# new_samples=cmb_mnu['samples'].copy()
# new_samples=cmb_nrun['samples'].copy()


labels_DESI=[]
for ii in range(len(true_DL2)):
    new_samples=chain_type['samples'].copy()
    p=new_samples.getParams()
    new_loglike = coeff_mult * gaussian_chi2_mock_DESI(
        p.linP_n_star, 
        p.linP_DL2_star, 
        true_DL2=true_DL2[ii],
        neff_err=neff_err,
        DL2_err=DL2_err,
        r=r
    )
    new_samples.reweightAddingLogLikes(new_loglike) #re-weight cut_samples to account for the new likelihood
    samples_DESI.append(new_samples)
    labels_DESI.append(r'+ DESI Ly$\alpha \, \Delta^2_\ast = ${}'.format(true_DL2[ii]))


g = plots.getSubplotPlotter(width_inch=16)
g.settings.axes_fontsize = 10
g.settings.legend_fontsize = 14
g.triangle_plot([chain_type['samples'],samples_DESI[0],samples_DESI[1],samples_DESI[2]],
                ['linP_DL2_star','linP_n_star','omegak'],
                legend_labels=['Planck 2018 + SDSS',labels_DESI[0],labels_DESI[1],labels_DESI[2]])

# %% [markdown]
# ### Constraints on mnu

# %%
samples_DESI=[]
true_DL2=[0.33,0.35,0.37]
DL2_err = 0.056
neff_err = 0.022
r=-0.134
coeff_mult = 2.3 # should it be 0.5?

chain_type = cmb_mnu


# new_samples=cmb_w_wa['samples'].copy()
# new_samples=cmb_omega_k['samples'].copy()
# new_samples=cmb_mnu['samples'].copy()
# new_samples=cmb_nrun['samples'].copy()


labels_DESI=[]
for ii in range(len(true_DL2)):
    new_samples=chain_type['samples'].copy()
    p=new_samples.getParams()
    new_loglike = coeff_mult * gaussian_chi2_mock_DESI(
        p.linP_n_star, 
        p.linP_DL2_star, 
        true_DL2=true_DL2[ii],
        neff_err=neff_err,
        DL2_err=DL2_err,
        r=r
    )
    new_samples.reweightAddingLogLikes(new_loglike) #re-weight cut_samples to account for the new likelihood
    samples_DESI.append(new_samples)
    labels_DESI.append(r'+ DESI Ly$\alpha \, \Delta^2_\ast = ${}'.format(true_DL2[ii]))


g = plots.getSubplotPlotter(width_inch=8)
g.settings.axes_fontsize = 12
g.settings.legend_fontsize = 14
g.triangle_plot([chain_type['samples'],samples_DESI[0],samples_DESI[1],samples_DESI[2]],
                ['linP_DL2_star','linP_n_star','mnu'],
                legend_labels=["Planck 2018",labels_DESI[0],labels_DESI[1],labels_DESI[2]])
#plt.savefig(plot_label+'.pdf')
#plt.savefig(plot_label+'.png')

print(chain_type['samples'].getInlineLatex('mnu',limit=1))
print(samples_DESI[0].getInlineLatex('mnu',limit=1))
print(samples_DESI[1].getInlineLatex('mnu',limit=1))
print(samples_DESI[2].getInlineLatex('mnu',limit=1))
print("")
print(chain_type['samples'].getInlineLatex('mnu',limit=2))
print(samples_DESI[0].getInlineLatex('mnu',limit=2))
print(samples_DESI[1].getInlineLatex('mnu',limit=2))
print(samples_DESI[2].getInlineLatex('mnu',limit=2))

# %% [markdown]
# #### Constraints on nrun

# %%
samples_DESI=[]
true_DL2=[0.33,0.35,0.37]
DL2_err = 0.056
neff_err = 0.022
r=-0.134
coeff_mult = 2.3 # should it be 0.5?

chain_type = cmb_nrun


# new_samples=cmb_w_wa['samples'].copy()
# new_samples=cmb_omega_k['samples'].copy()
# new_samples=cmb_mnu['samples'].copy()
# new_samples=cmb_nrun['samples'].copy()


labels_DESI=[]
for ii in range(len(true_DL2)):
    new_samples=chain_type['samples'].copy()
    p=new_samples.getParams()
    new_loglike = coeff_mult * gaussian_chi2_mock_DESI(
        p.linP_n_star, 
        p.linP_DL2_star, 
        true_DL2=true_DL2[ii],
        neff_err=neff_err,
        DL2_err=DL2_err,
        r=r
    )
    new_samples.reweightAddingLogLikes(new_loglike) #re-weight cut_samples to account for the new likelihood
    samples_DESI.append(new_samples)
    labels_DESI.append(r'+ DESI Ly$\alpha \, \Delta^2_\ast = ${}'.format(true_DL2[ii]))


g = plots.getSubplotPlotter(width_inch=8)
g.settings.axes_fontsize = 12
g.settings.legend_fontsize = 14
g.triangle_plot([chain_type['samples'],samples_DESI[0],samples_DESI[1],samples_DESI[2]],
                ['linP_DL2_star','linP_n_star','nrun'],
                legend_labels=["Planck 2018",labels_DESI[0],labels_DESI[1],labels_DESI[2]])
#plt.savefig(plot_label+'.pdf')
#plt.savefig(plot_label+'.png')

# %% [markdown]
# #### Constraints on nrun and nrunrun

# %%
samples_DESI=[]
true_DL2=[0.33,0.35,0.37]
DL2_err = 0.056
neff_err = 0.022
r=-0.134
coeff_mult = 2.3 # should it be 0.5?

chain_type = cmb_nrun_nrunrun


labels_DESI=[]
for ii in range(len(true_DL2)):
    new_samples=chain_type['samples'].copy()
    p=new_samples.getParams()
    new_loglike = coeff_mult * gaussian_chi2_mock_DESI(
        p.linP_n_star, 
        p.linP_DL2_star, 
        true_DL2=true_DL2[ii],
        neff_err=neff_err,
        DL2_err=DL2_err,
        r=r
    )
    new_samples.reweightAddingLogLikes(new_loglike) #re-weight cut_samples to account for the new likelihood
    samples_DESI.append(new_samples)
    labels_DESI.append(r'+ DESI Ly$\alpha \, \Delta^2_\ast = ${}'.format(true_DL2[ii]))


g = plots.getSubplotPlotter(width_inch=8)
g.settings.axes_fontsize = 12
g.settings.legend_fontsize = 14
g.triangle_plot([chain_type['samples'],samples_DESI[0],samples_DESI[1],samples_DESI[2]],
                ['linP_DL2_star','linP_n_star','nrun','nrunrun'],
                legend_labels=["Planck 2018",labels_DESI[0],labels_DESI[1],labels_DESI[2]])
#plt.savefig(plot_label+'.pdf')
#plt.savefig(plot_label+'.png')

# %% [markdown]
# ### Constraints on nrun_nnu_w_mnu

# %%
samples_DESI=[]
true_DL2=[0.33,0.35,0.37]
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
        neff_err=neff_err,
        DL2_err=DL2_err,
        r=r
    )
    new_samples.reweightAddingLogLikes(new_loglike) #re-weight cut_samples to account for the new likelihood
    samples_DESI.append(new_samples)
    labels_DESI.append(r'+ DESI Ly$\alpha \, \Delta^2_\ast = ${}'.format(true_DL2[ii]))


# plikHM_TTTEEE_lowl_lowE_BAO_Riess18_Pantheon18_lensing
g = plots.getSubplotPlotter(width_inch=8)
g.settings.axes_fontsize = 12
g.settings.legend_fontsize = 14
g.triangle_plot([chain_type['samples'],samples_DESI[0],samples_DESI[1],samples_DESI[2]],
                ['linP_DL2_star','linP_n_star','nrun','nnu','w','mnu'],
                legend_labels=["Planck 2018 + SDSS + SN",labels_DESI[0],labels_DESI[1],labels_DESI[2]])
#plt.savefig(plot_label+'.pdf')
#plt.savefig(plot_label+'.png')

