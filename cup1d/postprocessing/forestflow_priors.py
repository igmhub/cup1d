"""
Decode cup1d P1D posterior chains into ForestFlow emulator inputs.

This is the cup1d side of the ForestFlow priors analysis
(:mod:`forestflow.priors`, ``ForestFlow/notebooks/Figures_CM2026``): decoding a
chain needs the cup1d likelihood (IGM models and parameter definitions), so it
lives here, while ForestFlow only reads the products written by
:func:`set_process_p1d_chain` (``<lab_sample>_priors_cosmo_IGM_from_p1d.npy``,
see :func:`forestflow.priors.loaders.load_p1d_chain_for_forestflow`).

Moved from ``forestflow.priors.paper.set_samples`` (ForestFlow must not import
cup1d).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from lace.cosmo import cosmology, rescale_cosmology

__all__ = ["P1D_CHAINS", "set_input_process_p1d_chain", "set_process_p1d_chain"]

# cup1d P1D chains mapped to ForestFlow inputs. The chains store unit-cube
# coordinates; they are decoded with the parameter definitions saved next to
# them (fitter_results.npy), not with the bounds of the current cup1d config.
# Folders are relative to the external Lyman-alpha data root
# (forestflow.paths.get_lya_data_path; on NERSC: /global/cfs/cdirs/desi/users/jjchaves/p1d).
P1D_CHAINS = {
    "desi": {
        "folder": "out_DESI_DR1/DESIY1_QMLE3/global_opt/CH24_mpgcen_gpr/chain_7",
        "zs": [2.2, None, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2],
    },
    "accel2": {
        "folder": "accel2/chains/chain_1",
        "zs": [2.2, None, 2.6, 3.0, 3.6, 4.0],
    },
}


def _chain_folder(lab_sample: str, chain_folder=None, lya_data_path=None) -> Path:
    """Return the folder of a P1D chain, explicit or from ``P1D_CHAINS``."""
    if chain_folder is not None:
        return Path(chain_folder).expanduser()
    from forestflow.paths import get_lya_data_path

    return get_lya_data_path(lya_data_path) / P1D_CHAINS[lab_sample]["folder"]


def set_input_process_p1d_chain(
    lab_sample: str,
    zeff: float | None = 2.33,
    chain_folder: str | os.PathLike | None = None,
    lya_data_path: str | os.PathLike | None = None,
) -> tuple[Any, ...]:
    """
    Load a cup1d analysis and a P1D posterior chain with its parameters.

    Parameters
    ----------
    lab_sample : {"desi", "accel2"}
        Source chain, see ``P1D_CHAINS``.
    zeff : float, default=2.33
        Effective redshift inserted into the configuration's redshift grid.
    chain_folder : str or path-like, optional
        Folder with ``fitter_results.npy``, ``chain.npy`` and ``blobs.npy``;
        by default ``P1D_CHAINS[lab_sample]["folder"]`` below ``lya_data_path``.
    lya_data_path : str or path-like, optional
        Root of the external data (``forestflow.paths.get_lya_data_path``).

    Returns
    -------
    pipeline : cup1d.inference.analysis.Analysis
        CM2026 baseline analysis (``lace_mpg``, DESI DR1). Only its IGM models
        are used to decode the chains; for ``"accel2"`` too, since both chains
        share the baseline IGM parameterisation (checked against the saved
        parameter names).
    chain : numpy.memmap, shape (n_samples, n_params)
        Flattened unit-cube posterior chain, memory-mapped.
    chain_params : dict
        Parameter definitions saved with the chain, in chain order.
    d2star, nstar : ndarray, shape (n_samples,)
        Blob-derived linear-power summaries.
    zs : ndarray
        Analysis redshift grid.
    zeff : float
        Effective redshift used by downstream conversions.
    """
    from cup1d.inference.analysis import Analysis

    folder = _chain_folder(lab_sample, chain_folder, lya_data_path)
    zs = np.array([zeff if z is None else z for z in P1D_CHAINS[lab_sample]["zs"]])

    saved = np.load(folder / "fitter_results.npy", allow_pickle=True).item()
    names = saved["like"]["free_param_names"]
    chain_params = {name: saved["like"]["free_params"][name] for name in names}

    pip = Analysis(create_output=False)
    if list(pip.like.free_params) != list(names):
        raise ValueError(
            "free parameters of the cup1d baseline differ from those of the chain"
        )

    # memory-mapped: only the selected samples are read
    chain = np.load(folder / "chain.npy", mmap_mode="r")
    chain = chain.reshape(-1, len(names))

    blobs = np.load(folder / "blobs.npy", allow_pickle=True)
    d2star = np.array(blobs["Delta2_star"]).reshape(-1)
    nstar = np.array(blobs["n_star"]).reshape(-1)
    del blobs

    return pip, chain, chain_params, d2star, nstar, zs, zeff


def set_process_p1d_chain(
    lab_sample: str,
    nn: int | None = 10000,
    seed: int | None = 12345,
    store_p1d: bool | None = False,
    out_folder: str | os.PathLike = "int_data_figs",
    chain_folder: str | os.PathLike | None = None,
    lya_data_path: str | os.PathLike | None = None,
) -> None:
    """
    Subsample a cup1d P1D chain and write the ForestFlow inputs of each sample.

    Parameters
    ----------
    lab_sample : {"desi", "accel2"}
        Source-chain configuration, see ``P1D_CHAINS``.
    nn : int, default=10000
        Number of posterior samples selected without replacement.
    seed : int, default=12345
        Random-number generator seed.
    store_p1d : bool, default=False
        Also store the P1D predicted by cup1d (contaminants included) for each
        sample. Only implemented for ``"desi"``.
    out_folder : str or path-like, default="int_data_figs"
        Folder for the output products.
    chain_folder, lya_data_path : str or path-like, optional
        Location of the chain, see :func:`set_input_process_p1d_chain`.

    Notes
    -----
    Writes ``<lab_sample>_inter_chain.npy`` and
    ``<lab_sample>_priors_cosmo_IGM_from_p1d.npy`` in ``out_folder``; the
    latter is read by
    :func:`forestflow.priors.loaders.load_p1d_chain_for_forestflow`.
    IGM histories are converted from km/s to Mpc with the Planck18 cosmology.
    """
    from cup1d.likelihood.parameter import values_from_cube_batch

    pip, chain, chain_params, d2star, nstar, zs, zeff = set_input_process_p1d_chain(
        lab_sample, chain_folder=chain_folder, lya_data_path=lya_data_path
    )
    if store_p1d and lab_sample != "desi":
        raise NotImplementedError("store_p1d only implemented for lab_sample='desi'")

    # get a random sample
    rng = np.random.default_rng(seed=seed)
    ind = rng.permutation(np.arange(chain.shape[0]))[:nn]
    # physical values of the selected samples, shape (nn,) per parameter
    values = values_from_cube_batch(chain_params, np.asarray(chain[ind]))
    pars_chain = {}

    ## create holder for p1d chain
    pars_chain["z"] = zs

    # cosmo (for Arinyo model)
    pars_chain["As"] = values["As"]
    pars_chain["ns"] = values["ns"]
    pars_chain["Delta2star"] = d2star[ind]
    pars_chain["nstar"] = nstar[ind]

    # cosmo for this project
    pars_chain["sig_8"] = np.zeros((ind.shape[0]))
    pars_chain["sig_8_z0"] = np.zeros((ind.shape[0]))
    pars_chain["f_sig_8"] = np.zeros((ind.shape[0]))

    # input ForestFlow
    for par in ["Delta2_p", "n_p", "mF", "gamma", "sigT_Mpc", "kF_Mpc"]:
        pars_chain[par] = np.zeros((ind.shape[0], zs.shape[0]))

    igm_models = pip.like.theory.model_igm.models

    if store_p1d:
        # p1d at zeff is that of the first data redshift (2.2), as the k grid
        data_label = pip.args.data_label[0]
        k_kms = pip.data[data_label].k_kms
        iz_data = [0, 0] + list(range(1, zs.shape[0] - 1))
        nk = max(len(k_kms[iz]) for iz in iz_data)
        pars_chain["k_kms"] = np.zeros((zs.shape[0], nk))
        pars_chain["p1d"] = np.zeros((ind.shape[0], zs.shape[0], nk))
        for jj, iz in enumerate(iz_data):
            pars_chain["k_kms"][jj, : len(k_kms[iz])] = k_kms[iz]

    ## fill the holder
    # set cosmology
    class_planck = cosmology.Cosmology(cosmo_label="Planck18_wonu")
    planck_f = class_planck.get_growth_rate(zeff)
    dkms_dMpc_zs = class_planck.get_dkms_dMpc(pars_chain["z"])

    for ii in range(ind.shape[0]):
        if ii % 100 == 0:
            print(ii)

        like_params = {name: values[name][ii] for name in values}

        pars_chain["mF"][ii] = igm_models["F_model"].get_mean_flux(
            zs, like_params=like_params
        )
        pars_chain["gamma"][ii] = igm_models["T_model"].get_gamma(
            zs, like_params=like_params
        )
        sigT_kms = igm_models["T_model"].get_sigT_kms(zs, like_params=like_params)
        pars_chain["sigT_Mpc"][ii] = sigT_kms / dkms_dMpc_zs
        kF_kms = igm_models["P_model"].get_kF_kms(zs, like_params=like_params)
        pars_chain["kF_Mpc"][ii] = kF_kms * dkms_dMpc_zs

        class_new = rescale_cosmology.RescaledCosmology(
            fid_cosmo=class_planck,
            new_params_dict={"As": pars_chain["As"][ii], "ns": pars_chain["ns"][ii]},
        )

        pars_chain["sig_8"][ii] = class_new.get_sigma8(zeff)
        pars_chain["sig_8_z0"][ii] = class_new.get_sigma8(0)
        pars_chain["f_sig_8"][ii] = planck_f * pars_chain["sig_8"][ii]

        for jj in range(zs.shape[0]):
            linP_params = class_new.get_linP_Mpc_params(pars_chain["z"][jj], kp_Mpc=0.7)
            pars_chain["Delta2_p"][ii, jj] = linP_params["Delta2_p"]
            pars_chain["n_p"][ii, jj] = linP_params["n_p"]

        if store_p1d:
            p1d = pip.like.get_p1d_kms(parameters=like_params)[0][data_label]
            for jj, iz in enumerate(iz_data):
                pars_chain["p1d"][ii, jj, : len(p1d[iz])] = p1d[iz]

    np.save(os.path.join(out_folder, lab_sample + "_inter_chain.npy"), pars_chain)

    dict_save_file = {"zs": pars_chain["z"]}
    for par in pars_chain.keys():
        if par not in ["k_kms", "p1d", "z"]:
            dict_save_file[par] = pars_chain[par]

    np.save(
        os.path.join(out_folder, lab_sample + "_priors_cosmo_IGM_from_p1d.npy"),
        dict_save_file,
    )
