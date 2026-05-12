"""Helpers for adding linear-power parameters to cosmological samples."""

from lace.cosmo import camb_cosmo, fit_linP


def get_linP_params(
    params, z_star=3.0, kp_kms=0.009, verbose=False, camb_kmax_Mpc_fast=1.5
):
    """Compute linear-power parameters for one cosmological sample.

    Parameters
    ----------
    params : dict
        Cosmological parameters accepted by
        :func:`lace.cosmo.camb_cosmo.get_cosmology_from_dictionary`.
    z_star : float, optional
        Redshift pivot.
    kp_kms : float, optional
        Velocity-space wavenumber pivot in s/km.
    verbose : bool, optional
        If true, print CAMB cosmology information.
    camb_kmax_Mpc_fast : float, optional
        Maximum CAMB wavenumber used by the fast linear-power calculation.
    """

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
