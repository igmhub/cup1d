"""Observation model: contaminants and resolution applied to the Lya P1D.

``P_obs = (C_HCD C_mul IC P_Lya + C_add) C_res`` for one parameter point or a
batch, shared by the native theory (:meth:`cup1d.theory.theory.Theory.get_p1d_kms`)
and the external likelihood (:class:`cup1d.likelihood.external.ExternalP1DLikelihood`).
"""

import numpy as np


def dataset_nuisance(key, like_params, syst_data_labels=None):
    """Return the nuisance parameters that apply to one data set.

    Parameters
    ----------
    key : str
        Data-set label.
    like_params : mapping
        Scalar or columnar nuisance values.
    syst_data_labels : sequence of str, optional
        Data sets that receive the resolution correction (``R_coeff_*``);
        ``None`` applies it to every data set (``args.syst_data_labels``).

    Returns
    -------
    mapping
        ``like_params`` unchanged when ``syst_data_labels`` is ``None`` or
        contains ``key``; otherwise a copy without ``R_coeff_*`` entries, so
        no resolution correction is applied to this data set.
    """
    if syst_data_labels is None or key in syst_data_labels:
        return like_params
    return {name: value for name, value in like_params.items()
            if not name.startswith("R_coeff")}


def without_resolution(like_params):
    """Return ``like_params`` without the resolution-correction ``R_coeff_*`` entries."""
    return {name: value for name, value in like_params.items() if not name.startswith("R_coeff")}


def apply_resolution(model_syst, zs, k_kms, powers, like_params):
    """Multiply contaminated P1D rows by one data set's resolution correction.

    With a shared fine grid, the contaminated P1D is computed once without
    ``R_coeff_*`` and each data set applies its own (smooth) correction, as
    the last factor of :func:`apply_observation_model`.

    Parameters
    ----------
    model_syst : object
        cup1d resolution-systematics model.
    zs, k_kms : sequence
        Redshifts and fine wavenumber rows in s/km.
    powers : sequence of ndarray
        Contaminated P1D rows, scalar ``(n_k,)`` or columnar ``(n_batch, n_k)``.
    like_params : mapping
        The data set's nuisance values (:func:`dataset_nuisance`).

    Returns
    -------
    list of ndarray
        ``powers`` times the correction, or ``powers`` itself without
        ``R_coeff_*`` parameters.
    """
    if not any(name.startswith("R_coeff") for name in like_params):
        return list(powers)
    syst = model_syst.get_contamination(zs, k_kms, like_params=like_params)
    return [power * correction for power, correction in zip(powers, syst)]


def apply_observation_model(model_cont, model_syst, zs, k_kms, p1d_kms, mean_flux, M_of_z, like_params=None, remove=None, metal_moments=None):
    """Apply contaminant, resolution, and IC responses to raw P1D arrays.

    Parameters
    ----------
    model_cont, model_syst : object
        cup1d contaminant and systematic response models.
    zs, k_kms, p1d_kms : sequence
        Redshifts, matching wavenumber rows in s/km, and uncontaminated P1D
        rows: ``(n_k,)`` for one point or ``(n_batch, n_k)`` for a batch.
    mean_flux, M_of_z : array-like
        Mean flux and km/s-to-Mpc conversion at each redshift, ``(n_z,)`` or
        ``(n_batch, n_z)``.
    like_params : mapping, optional
        Nuisance values used by response models: scalars for one point or
        columns of shape ``(n_batch,)``.
    remove : sequence of str, optional
        Contaminant terms to omit.
    metal_moments : sequence of mapping, optional
        Per-redshift moments from :func:`project_metal_moments`, required by
        the Kaiser metal model.

    Returns
    -------
    powers : list of ndarray
        Observed-model P1D rows.
    terms : list of dict
        Per-redshift response components for diagnostics.
    """
    like_params = {} if like_params is None else like_params
    syst = model_syst.get_contamination(zs, k_kms, like_params=like_params) if any(name.startswith("R_coeff") for name in like_params) else np.ones(len(zs))
    cont = model_cont.get_contamination(zs, k_kms, mean_flux, M_of_z, like_params=like_params, remove=remove, metal_moments=metal_moments)
    powers, terms = [], []
    for index, z in enumerate(zs):
        powers.append((cont["cont_HCD"][index] * cont["cont_mul_metals"][index] * cont["IC_corr"][index] * p1d_kms[index] + cont["cont_add_metals"][index]) * syst[index])
        terms.append(dict(z=z, k_kms=k_kms[index], p1d_emu_kms=p1d_kms[index], C_res=syst[index], C_mul_metals=cont["cont_mul_metals"][index], C_add_metals=cont["cont_add_metals"][index], C_HCD=cont["cont_HCD"][index], IC_corr=cont["IC_corr"][index], p1d_tot_kms="[(C_mul_metals * C_HCD * IC_corr * p1d_emu_kms + C_add_metals) * C_res]"))
    return powers, terms
