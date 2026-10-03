from lace.cosmo.cosmology import Cosmology


def get_linP_params(
    params, z_star=3.0, kp_kms=0.009, verbose=False, camb_kmax_Mpc_fast=1.5
):
    """Compute compressed linear-power parameters for cosmological inputs.

    Parameters
    ----------
    params : mapping
        Cosmological parameters accepted by :class:`lace.cosmo.cosmology.Cosmology`.
    z_star : float, default: 3.0
        Redshift of the linear-power compression.
    kp_kms : float, default: 0.009
        Velocity-space pivot wavenumber in ``s / km``.
    verbose : bool, default: False
        Print the constructed cosmology information.
    camb_kmax_Mpc_fast : float, default: 1.5
        Retained compatibility argument; it is not used by this implementation.

    Returns
    -------
    dict
        Linear-power amplitude, slope, and running at the requested pivot.
    """

    # create CAMB cosmology object from input params dictionary
    cosmo = Cosmology(cosmo_params_dict=params)
    if verbose:
        cosmo.print_info()

    # compute linear power and fit power law at pivot point
    linP_params = cosmo.get_linP_kms_params(z_star, kp_kms)

    return linP_params
