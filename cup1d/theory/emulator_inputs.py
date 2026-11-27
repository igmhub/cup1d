"""Emulator inputs from a cosmology and the IGM histories.

The single mapping from sampled parameters to emulator inputs, used by
:meth:`cup1d.theory.theory.Theory.get_emulator_calls` and by the
lya_interface Cobaya theory (whose cosmology comes from Cobaya), for one
parameter point or a batch: compressed
linear power at the emulator pivot, mean flux, gamma, and the thermal and
pressure scales converted from km/s to Mpc with ``M(z) = dkms/dMpc``.
"""

from collections.abc import Mapping

import numpy as np

from cup1d.utils.batch import as_columns

LINEAR_POWER_INPUTS = ("Delta2_p", "n_p", "alpha_p")


def emulator_inputs(emu_params, zs, linP_Mpc_params, M_of_z, model_igm, like_params=None):
    """Emulator inputs at every redshift, for one parameter point or a batch.

    Parameters
    ----------
    emu_params : sequence of str
        Emulator input names (``Delta2_p``, ``n_p``, ``alpha_p``, ``mF``,
        ``gamma``, ``sigT_Mpc``, ``kF_Mpc``, ``lambda_P``).
    zs : array_like
        Redshifts.
    linP_Mpc_params : mapping or sequence of mapping
        Compressed linear power at the emulator pivot: ``{name: array}`` with
        shape ``(n_z,)`` or ``(n_batch, n_z)``, or one mapping per redshift
        (``cosmology.get_linP_Mpc_params(z, kp_Mpc)``) for one point.
    M_of_z : array_like
        ``dkms/dMpc`` (km/s per Mpc), ``(n_z,)`` or ``(n_batch, n_z)``.
    model_igm : cup1d.models.igm.model_igm.IGM
        IGM histories (``F_model``, ``T_model``, ``P_model``).
    like_params : mapping, optional
        IGM parameter values (physical, native names): scalars for one point
        or columns of shape ``(n_batch,)``; fiducial histories if omitted.

    Returns
    -------
    dict of str to ndarray
        Every input, ``(n_z,)`` for one point and ``(n_batch, n_z)`` for a
        batch; with ``mF`` also ``mF_fid``, the fiducial mean flux.

    Raises
    ------
    ValueError
        For an input without a model.
    """
    zs = np.atleast_1d(np.asarray(zs, dtype=float))
    columns, n_batch, scalar = as_columns(like_params)
    shape = (n_batch, len(zs))
    if not isinstance(linP_Mpc_params, Mapping):
        linP_Mpc_params = {
            name: np.asarray([params[name] for params in linP_Mpc_params], dtype=float)
            for name in linP_Mpc_params[0]
        }
    M_of_z = np.broadcast_to(np.asarray(M_of_z, dtype=float), shape)
    models = model_igm.models
    inputs = {}
    for key in emu_params:
        if key in LINEAR_POWER_INPUTS:
            inputs[key] = np.broadcast_to(np.asarray(linP_Mpc_params[key], dtype=float), shape)
        elif key == "mF":
            inputs[key] = models["F_model"].get_mean_flux(zs, like_params=columns)
            inputs["mF_fid"] = np.broadcast_to(models["F_model"].get_mean_flux(zs), shape)
        elif key == "gamma":
            inputs[key] = models["T_model"].get_gamma(zs, like_params=columns)
        elif key == "sigT_Mpc":
            inputs[key] = models["T_model"].get_sigT_kms(zs, like_params=columns) / M_of_z
        elif key == "kF_Mpc":
            inputs[key] = models["P_model"].get_kF_kms(zs, like_params=columns) * M_of_z
        elif key == "lambda_P":
            inputs[key] = 1000 / (models["P_model"].get_kF_kms(zs, like_params=columns) * M_of_z)
        else:
            raise ValueError(f"no theory model for emulator input {key!r}")
    # (n_batch, n_z) arrays of their own (no broadcast views)
    inputs = {key: np.array(np.broadcast_to(value, shape), dtype=float) for key, value in inputs.items()}
    if scalar:
        return {key: value[0] for key, value in inputs.items()}
    return inputs
