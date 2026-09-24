from lace.cosmo.cosmology import Cosmology


def get_linP_params(
    params, z_star=3.0, kp_kms=0.009, verbose=False, camb_kmax_Mpc_fast=1.5
):
    """Given point in getdist MCMC chain, compute linear power parameters.
    - z_star, kp_kms set the pivot point"""

    # create CAMB cosmology object from input params dictionary
    cosmo = Cosmology(cosmo_params_dict=params)
    if verbose:
        cosmo.print_info()

    # compute linear power and fit power law at pivot point
    linP_params = cosmo.get_linP_kms_params(z_star, kp_kms)

    return linP_params
