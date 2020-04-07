from cup1d.cosmology import camb_cosmo
from cup1d.cosmology import fit_linP

def get_linP_params(isample,samples,print_every=1):
    """Given point in getdist MCMC chain, compute linear power parameters.
        The setup is such that it can be used via multiprocessing."""

    verbose=(isample%print_every==0)
    if verbose: print('sample',isample)
    params=samples.getParamSampleDict(isample)
 
    linP_params=_get_linP_params(params,verbose=verbose)
    if verbose: print('linP parameters',linP_params)
    return linP_params


def _get_linP_params(params,z_star=3.0,kp_kms=0.009,verbose=False):
	"""Given point in getdist MCMC chain, compute linear power parameters"""

	# create CAMB cosmology object from input params dictionary
    cosmo=camb_cosmo.get_cosmology(params)
    if verbose: 
		camb_cosmo.print_info(cosmo,simulation=True)
	# compute linear power and fit power law at pivot point
    linP_model=fit_linP.LinearPowerModel(cosmo=cosmo,z_star=z_star,
												k_units='kms',kp=kp_kms)
    linP_params=linP_model.get_params()
    return linP_params

