import os
import numpy as np
from lace.configuration import get_nyx_path
from cup1d.utils.utils import is_number_string
from cup1d.utils.utils import get_path_repo
from cup1d.utils.various_dicts import get_blob_value


def get_training_hc(
    sim_suite,
    emu_params=None,
    nyx_version="models_Nyx_Mar2025_with_CGAN_val_3axes",
):
    """Load simulator cosmology and IGM histories used for emulator domains.

    Parameters
    ----------
    sim_suite : {"mpg", "nyx"}
        Simulation suite whose stored histories are loaded.
    emu_params : sequence of str, optional
        Cosmology-parameter selection. Its inclusion of ``"alpha_p"`` selects
        the three-parameter Nyx-style cosmology input.
    nyx_version : str, default: "models_Nyx_Mar2025_with_CGAN_val_3axes"
        Nyx cosmology catalogue version.

    Returns
    -------
    hc_params : list of str
        Ordered emulator-domain parameter names.
    hc_points : ndarray, shape (n_training_rows, n_parameters)
        Concatenated cosmology and IGM training inputs.
    cosmo_all, igm_all : dict
        Raw stored cosmology and IGM-history mappings.

    Raises
    ------
    ValueError
        If the suite is unsupported or a required stored product is missing.
    """

    # get name of files storing cosmo and igm
    if sim_suite == "mpg":
        repo = get_path_repo("lace")
        cosmo_fname = os.path.join(
            repo, "data", "sim_suites", "Australia20", "mpg_emu_cosmo.npy"
        )
        igm_fname = os.path.join(
            repo, "data", "sim_suites", "Australia20", "IGM_histories.npy"
        )
    elif sim_suite == "nyx":
        nyx_path = get_nyx_path()
        cosmo_fname = os.path.join(
            nyx_path, "nyx_emu_cosmo_" + nyx_version + ".npy"
        )
        igm_fname = os.path.join(nyx_path, "IGM_histories.npy")
    else:
        raise ValueError(f"sim_suite {sim_suite} not recognized")

    # read cosmo
    try:
        cosmo_all = np.load(cosmo_fname, allow_pickle=True).item()
    except:
        script_fname = os.path.join(
            get_path_repo("lace"),
            "script",
            "developers",
            "save_" + sim_suite + "_emu_cosmo.py",
        )
        raise ValueError(
            f"{cosmo_fname} not found. You can produce it using {script_fname}"
        )

    # read igm
    try:
        igm_all = np.load(igm_fname, allow_pickle=True).item()
    except:
        script_fname = os.path.join(
            get_path_repo("lace"),
            "script",
            "developers",
            "save_" + sim_suite + "_IGM.py",
        )
        raise ValueError(
            f"{igm_fname} not found. You can produce it using {script_fname}"
        )

    # get input parameters to emulator
    if emu_params is None:
        if sim_suite == "mpg":
            pars_cosmo = ["Delta2_p", "n_p"]
        else:
            pars_cosmo = ["Delta2_p", "n_p", "alpha_p"]
    else:
        if "alpha_p" in emu_params:
            pars_cosmo = ["Delta2_p", "n_p", "alpha_p"]
        else:
            pars_cosmo = ["Delta2_p", "n_p"]
    pars_igm = ["mF", "sigT_Mpc", "gamma", "kF_Mpc"]
    hc_params = pars_cosmo + pars_igm

    # extract data
    dict_out = {}
    for par in hc_params:
        dict_out[par] = []

    sim_label_cosmo = ["_".join(s.split("_")[:2]) for s in igm_all.keys()]
    for ii, sim_label in enumerate(igm_all):
        # only use simulations in the training set
        if (is_number_string(sim_label[-1]) == False) | (
            sim_label_cosmo[ii] == "accel2"
        ):
            continue

        mask = igm_all[sim_label]["z"] != 0
        for par in pars_igm:
            mask = mask & (igm_all[sim_label][par] != 0)
        for par in pars_igm:
            dict_out[par].append(igm_all[sim_label][par][mask])

        for par in pars_cosmo:
            dict_out[par].append(
                cosmo_all[sim_label_cosmo[ii]]["linP_params"][par][mask]
            )

    for par in hc_params:
        dict_out[par] = np.concatenate(np.array(dict_out[par], dtype=object))

    hc_points = np.vstack(list(dict_out.values())).T

    return hc_params, hc_points, cosmo_all, igm_all


def load_chains_for_cosmopower(fname):
    """Load cup1d fitter samples into a CosmoPower-compatible table.

    Parameters
    ----------
    fname : str or path-like
        Saved fitter-result ``.npy`` file.

    Returns
    -------
    pandas.DataFrame
        Physical sampled parameters, compressed power summaries, and fixed
        fiducial background quantities.
    """

    import pandas as pd

    data = np.load(fname, allow_pickle=True).item()
    sampling_params = data["fitter"]["chain_names"]  # to chain
    star_params = data["fitter"]["blobs_names"]  # to blob
    _chain = data["fitter"]["chain"].reshape(
        -1, data["fitter"]["chain"].shape[-1]
    )
    _blobs = np.asarray(data["fitter"]["blobs"])
    if _blobs.dtype.names is None:
        _blobs = _blobs.reshape(-1, _blobs.shape[-1])
    else:
        _blobs = _blobs.reshape(-1)
    if "nrun" in sampling_params:
        nstar = 3
    else:
        nstar = 2
    all_params = np.zeros((_chain.shape[0], _chain.shape[1] + nstar))
    all_params_names = []
    for ii in range(_chain.shape[-1]):
        prange = data["fitter"]["chain_from_cube"][sampling_params[ii]]
        # print(sampling_params[ii], prange)
        all_params[:, ii] = _chain[:, ii] * (prange[1] - prange[0]) + prange[0]
        all_params_names.append(sampling_params[ii])

    for ii in range(nstar):
        all_params[:, -nstar + ii] = get_blob_value(_blobs, star_params[ii])
        all_params_names.append(star_params[ii])

    df = pd.DataFrame(all_params, columns=all_params_names)
    h = data["like"]["cosmo_fid_label"]["cosmo"]["H0"] / 100
    omch2 = data["like"]["cosmo_fid_label"]["cosmo"]["omch2"]
    ombh2 = data["like"]["cosmo_fid_label"]["cosmo"]["ombh2"]
    mnu = data["like"]["cosmo_fid_label"]["cosmo"]["mnu"]
    # next two lines to be updated when using with neutrinos
    omnuh2 = mnu / 94.07  # this is more complicated, need CAMB or CLASS
    Omega_m = (omch2 + ombh2) / h**2  # should I include omnuh2 here?

    if "nrun" not in sampling_params:
        df["nrun"] = 0

    df["ln_A_s_1e10"] = np.log(df.As * 1e10)
    df["h"] = h
    df["m_ncdm"] = mnu
    df["omch2"] = omch2
    df["ombh2"] = ombh2
    df["omnuh2"] = omnuh2
    df["Omega_m"] = Omega_m
    df["Omega_Lambda"] = 1 - Omega_m

    return df
