from lace.cosmo import camb_cosmo
from lace.cosmo import fit_linP


def get_linP_params(
    params, z_star=3.0, kp_kms=0.009, verbose=False, camb_kmax_Mpc_fast=1.5
):
    """Given point in getdist MCMC chain, compute linear power parameters.
    - z_star, kp_kms set the pivot point"""

    # create CAMB cosmology object from input params dictionary
    cosmo = camb_cosmo.get_cosmology_from_dictionary(params)
    if verbose:
        camb_cosmo.print_info(cosmo)

    # compute linear power and fit power law at pivot point
    linP_params = fit_linP.parameterize_cosmology_kms(
        cosmo,
        camb_results=None,
        z_star=z_star,
        kp_kms=kp_kms,
        camb_kmax_Mpc_fast=camb_kmax_Mpc_fast,
    )

    return linP_params
