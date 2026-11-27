import os
import numpy as np
from lace.configuration import get_nyx_path
from cup1d.utils.utils import is_number_string
from cup1d.utils.utils import get_path_repo


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
