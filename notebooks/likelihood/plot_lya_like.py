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
#     display_name: cup1d
#     language: python
#     name: cup1d
# ---

# %% [markdown]
# # User interface: the compressed likelihood
#
# This notebook focuses on the final product, and how we expect most (non-experts) users to use our compressed likelihood. 
#
# By compressed likelihood we mean a likelihood that has already marginalized over nuisance (astro) parameters, and that uses a reduced set of parameters to describe the cosmological model. 
#
# The compressed likelihood doesn't know about redshift bins, or band powers, it doesn't know about mean flux, temperature or redshift or reionization.

# %% [markdown]
# Summary:
#     - Given an input cosmological model, it computes the parameters describing the linear power spectrum (linP).
#     - Given a set of linP parameters, it calls a precomputed object with the likelihood for these parameters. For this notebook, the precomputed object is just a Gaussian fit to ($\Delta_p^2$,$n_{\rm eff}$), i.e., amplitude and slope of the linaer power at $z=3$ and $k_p= 0.009 s/km$.

# %% jupyter={"outputs_hidden": false}
# %matplotlib inline
import numpy as np
import os
import matplotlib.pyplot as plt
from cup1d.likelihood import marg_lya_like
from lace.cosmo import camb_cosmo
from lace.cosmo import fit_linP

# %% [markdown]
# ## Plot marginalised likelihoods
#
# This is the likelihood that has already been marginalized over nuisance parameters, and is of course, experiment specific. At this point it has lost all information about nuisance parameters, or data points. 
#
# Here we use Gaussian approximations to some published likelihoods.

# %% [markdown]
# ### Compare McDonald et al. (2005) vs Chabanier et al. (2019)
#
# Note that we approximate the likelihood from McDonald as a Gaussian, not a good approximation!

# %%
# create grid (note j in number of elements, crazy python)
neff_grid,DL2_grid = np.mgrid[-2.5:-2.1:200j, 0.2:0.7:200j]
chi2_McDonald2005=marg_lya_like.gaussian_chi2_McDonald2005(neff_grid,DL2_grid)
chi2_PalanqueDelabrouille2015=marg_lya_like.gaussian_chi2_PalanqueDelabrouille2015(neff_grid,DL2_grid)
chi2_Chabanier2019=marg_lya_like.gaussian_chi2_Chabanier2019(neff_grid,DL2_grid)

# %% jupyter={"outputs_hidden": false}
thresholds = [2.30,6.17,11.8]
plt.figure(figsize=[10,8])
plt.contour(neff_grid,DL2_grid,chi2_McDonald2005,levels=thresholds,colors='green')
plt.contour(neff_grid,DL2_grid,chi2_PalanqueDelabrouille2015,levels=thresholds,colors='red')
plt.contour(neff_grid,DL2_grid,chi2_Chabanier2019,levels=thresholds,colors='blue')
# hack to get legend entry for contours above
plt.axhline(y=0.8,color='green',label='McDonald 2005')
plt.axhline(y=0.8,color='red',label='Palanque-Delabrouille 2019')
plt.axhline(y=0.8,color='blue',label='Chabanier 2019')
plt.ylim(np.min(DL2_grid),np.max(DL2_grid))
plt.grid()                 
plt.legend(loc=2)
plt.title(r'Linear power constraints at ($z=3$, $k_p=0.009$ s/km)')
plt.xlabel(r'$n_p$')
plt.ylabel(r'$\Delta_p^2$')

# %% [markdown]
# ### Replicate Figure 11 in Palanque-Delabrouille et al. (2015)
#

# %% jupyter={"outputs_hidden": false}
plt.figure(figsize=[8,8])
plt.contour(DL2_grid,neff_grid,chi2_PalanqueDelabrouille2015,levels=thresholds,colors='green')
# hack to get legend entry for contours above
plt.axhline(y=0.8,color='green',label='Palanque-Delabrouille et al. 2015')
plt.xlim(0.14,0.44)
plt.ylim(-2.41,-2.29)
plt.grid()                 
plt.legend(loc=2)
plt.title(r'Linear power constraints at ($z=3$, $k_p=0.009$ s/km)')
plt.xlabel(r'$\Delta_p^2$')
plt.ylabel(r'$n_p$')

# %% [markdown]
# ### Replicate Figure 20 in Chabanier et al. (2019)
#

# %% jupyter={"outputs_hidden": false}
plt.figure(figsize=[10,6])
plt.contour(DL2_grid,neff_grid,chi2_Chabanier2019,levels=thresholds[:2],colors='blue')
# hack to get legend entry for contours above
plt.axhline(y=0.8,color='blue',label='Chabanier et al. 2019')
plt.xlim(0.24,0.42)
plt.ylim(-2.36,-2.3)
plt.grid()                 
plt.legend(loc=2)
plt.title(r'Linear power constraints at ($z=3$, $k_p=0.009$ s/km)')
plt.xlabel(r'$\Delta_p^2$')
plt.ylabel(r'$n_p$')

# %% [markdown]
# ## Compute predictions from Planck model

# %% jupyter={"outputs_hidden": false}
# setup cosmology, roughly inspired by Planck 2018
cosmo = camb_cosmo.get_cosmology()
# print relevant information about the cosmology object
camb_cosmo.print_info(cosmo)

# %% [markdown]
# ### Compute parameters describing the linear power spectrum around $z_\star=3$, $k_p=0.009$ s/km

# %% jupyter={"outputs_hidden": false}
z_star=3.0
kp_kms=0.009
params=fit_linP.parameterize_cosmology_kms(cosmo=cosmo,camb_results=None,z_star=z_star,kp_kms=kp_kms)
print('Lya parameters for massless cosmology',params)

# %% jupyter={"outputs_hidden": false}
plt.figure(figsize=[10,8])
plt.contour(DL2_grid,neff_grid,chi2_PalanqueDelabrouille2015,levels=thresholds,colors='green')
plt.contour(DL2_grid,neff_grid,chi2_Chabanier2019,levels=thresholds,colors='blue')
# hack to get legend entry for contours above
plt.axhline(y=0.8,color='blue',label='Chabanier 2019')
plt.axhline(y=0.8,color='green',label='Palanque-Delabrouille 2015')
# add point from fiducial cosmology
plt.plot(params['Delta2_star'],params['n_star'],'o',color='red',label='Planck 2018')
plt.xlim(0.2,0.45)
plt.ylim(-2.4,-2.28)
plt.grid()                 
plt.legend(loc=2)
plt.title(r'Linear power constraints at ($z=3$, $k_p=0.009$ s/km)')
plt.xlabel(r'$\Delta_p^2$')
plt.ylabel(r'$n_p$')

# %%
