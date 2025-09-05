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
# # Plot linear power parameters from extended Planck chains
#
# This notebook shows how to read one of the extended Planck chains with linear power parameters, provided in cup1d.
#
# It also compares the contours with the likelihoods on these parameters from Lya P1D.

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

# %% [markdown]
# ### Read extended Planck chains and plot linear power parameters
#
# These chains are already provided in cup1d

# %% jupyter={"outputs_hidden": false}
# massless neutrinos, Planck only
planck2018=planck_chains.get_planck_2018(model='base',data='plikHM_TTTEEE_lowl_lowE',linP_tag='zlinP_10')
# massive neutrinos, Planck only
planck2018_mnu=planck_chains.get_planck_2018(model='base_mnu',data='plikHM_TTTEEE_lowl_lowE',linP_tag='zlinP_10')
# massive neutrinos, Planck + BAO
planck2018_mnu_BAO=planck_chains.get_planck_2018(model='base_mnu',data='plikHM_TTTEEE_lowl_lowE_BAO',linP_tag='zlinP_10')

# %% jupyter={"outputs_hidden": false}
# linear power parameters
g = plots.getSubplotPlotter(width_inch=14)
g.settings.axes_fontsize = 12
g.settings.legend_fontsize = 16
g.triangle_plot([planck2018['samples'],planck2018_mnu['samples'],planck2018_mnu_BAO['samples']],
                ['linP_DL2_star','linP_n_star','linP_alpha_star','linP_f_star','linP_g_star'],
                legend_labels=[r'$\Lambda$CDM',r'$\nu\Lambda$CDM',r'+ BAO'])

# %% jupyter={"outputs_hidden": false}
# plot also Omega_m and H_0
# linear power parameters
g = plots.getSubplotPlotter(width_inch=14)
g.settings.axes_fontsize = 12
g.settings.legend_fontsize = 16
g.triangle_plot([planck2018['samples'],planck2018_mnu['samples'],planck2018_mnu_BAO['samples']],
                ['linP_DL2_star','linP_n_star','linP_alpha_star','linP_f_star','linP_g_star','omegam','H0'],
                legend_labels=[r'$\Lambda$CDM',r'$\nu\Lambda$CDM',r'+ BAO'])

# %% jupyter={"outputs_hidden": false}
# plot also neutrino mass (for nuLCDM)
g = plots.getSubplotPlotter(width_inch=14)
g.settings.axes_fontsize = 12
g.settings.legend_fontsize = 16
g.triangle_plot([planck2018_mnu['samples'],planck2018_mnu_BAO['samples']],
                ['linP_DL2_star','linP_n_star','linP_alpha_star','linP_f_star','linP_g_star','omegam','mnu'],
                legend_labels=[r'$\nu\Lambda$CDM',r'+ BAO'])

# %% [markdown]
# ### Comparison with parameters in our Meadows20 suite
#
# All simulations in our suite had the following fixed parameters: ($f_\ast=0.981$, $g_\ast=0.968$, $\alpha_\ast=-0.215$). These are all within the 2-sigma contours from Planck+BAO in $\nu\Lambda$CDM.
#
# The other two parameters are varied in the suite of simulations, with ranges ($0.25 < \Delta^2_\ast < 0.45$) and ($-2.35 < n_\ast < -2.25$). These extend well beyond the 3-sigma contours from Planck alone.

# %% [markdown]
# ### Plot linear power parameters from chain and from Lya likelihoods

# %%
# create grid (note j in number of elements, crazy python)
thresholds = [2.30,6.17,11.8]
neff_grid,DL2_grid = np.mgrid[-2.4:-2.2:100j, 0.2:0.5:100j]
chi2_Mc2005=marg_lya_like.gaussian_chi2_McDonald2005(neff_grid,DL2_grid)
chi2_PD2015=marg_lya_like.gaussian_chi2_PalanqueDelabrouille2015(neff_grid,DL2_grid)
chi2_Ch2019=marg_lya_like.gaussian_chi2_Chabanier2019(neff_grid,DL2_grid)

# %% jupyter={"outputs_hidden": false}
g = plots.getSinglePlotter(width_inch=8)
#g.plot_2d(planck2018['samples'], ['linP_n_star', 'linP_DL2_star'],lims=[-2.4,-2.25,0.2,0.5])
g.plot_2d(planck2018_mnu['samples'], ['linP_n_star', 'linP_DL2_star'],lims=[-2.4,-2.25,0.2,0.5])
#g.plot_2d(planck2018_mnu_BAO['samples'], ['linP_n_star', 'linP_DL2_star'],lims=[-2.4,-2.25,0.2,0.5])
plt.contour(neff_grid,DL2_grid,chi2_Mc2005,levels=thresholds[:2],colors='green')
plt.contour(neff_grid,DL2_grid,chi2_PD2015,levels=thresholds[:2],colors='red')
plt.contour(neff_grid,DL2_grid,chi2_Ch2019,levels=thresholds[:2],colors='blue')
plt.axhline(y=1,color='green',label='McDonald 2005')
plt.axhline(y=1,color='red',label='Palanque-Delabrouille 2015')
plt.axhline(y=1,color='blue',label='Chabanier 2019')
plt.axhline(y=1,color='black',label=r'$\nu \Lambda$CDM Planck 2018')
plt.title(r'Linear power constraints at ($z=3$, $k_p=0.009$ s/km)')
plt.grid()  
plt.legend(loc=4)

# %%
