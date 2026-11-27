"""Emulator inputs from a cosmology and the IGM histories.

The single mapping from sampled parameters to emulator inputs, used by
:meth:`cup1d.theory.theory.Theory.get_emulator_calls` and by the
lya_interface Cobaya theory (whose cosmology comes from Cobaya): compressed
linear power at the emulator pivot, mean flux, gamma, and the thermal and
pressure scales converted from km/s to Mpc with ``M(z) = dkms/dMpc``.
"""

import numpy as np

LINEAR_POWER_INPUTS = ("Delta2_p", "n_p", "alpha_p")


def emulator_inputs(emu_params, zs, linP_Mpc_params, M_of_z, model_igm, like_params=None):
    """Emulator inputs at every redshift.

    Parameters
    ----------
    emu_params : sequence of str
        Emulator input names (``Delta2_p``, ``n_p``, ``alpha_p``, ``mF``,
        ``gamma``, ``sigT_Mpc``, ``kF_Mpc``, ``lambda_P``).
    zs : array_like
        Redshifts.
    linP_Mpc_params : sequence of mapping
        Compressed linear power at the emulator pivot, one mapping per
        redshift (``cosmology.get_linP_Mpc_params(z, kp_Mpc)``).
    M_of_z : array_like
        ``dkms/dMpc`` at every redshift (km/s per Mpc).
    model_igm : cup1d.models.igm.model_igm.IGM
        IGM histories (``F_model``, ``T_model``, ``P_model``).
    like_params : mapping, optional
        Scalar IGM parameter values (physical, native names); fiducial
        histories if omitted.

    Returns
    -------
    dict of str to ndarray
        One value per redshift for every input; with ``mF`` also ``mF_fid``,
        the fiducial mean flux.

    Raises
    ------
    ValueError
        For an input without a model.
    """
    zs = np.atleast_1d(np.asarray(zs, dtype=float))
    M_of_z = np.asarray(M_of_z, dtype=float)
    models = model_igm.models
    inputs = {}
    for key in emu_params:
        if key in LINEAR_POWER_INPUTS:
            inputs[key] = np.asarray([params[key] for params in linP_Mpc_params], dtype=float)
        elif key == "mF":
            inputs[key] = models["F_model"].get_mean_flux(zs, like_params=like_params)
            inputs["mF_fid"] = models["F_model"].get_mean_flux(zs)
        elif key == "gamma":
            inputs[key] = models["T_model"].get_gamma(zs, like_params=like_params)
        elif key == "sigT_Mpc":
            inputs[key] = models["T_model"].get_sigT_kms(zs, like_params=like_params) / M_of_z
        elif key == "kF_Mpc":
            inputs[key] = models["P_model"].get_kF_kms(zs, like_params=like_params) * M_of_z
        elif key == "lambda_P":
            inputs[key] = 1000 / (models["P_model"].get_kF_kms(zs, like_params=like_params) * M_of_z)
        else:
            raise ValueError(f"no theory model for emulator input {key!r}")
    return inputs
