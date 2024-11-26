import os
import numpy as np
from cup1d.utils.utils import is_number_string
from cup1d.utils.utils import get_path_repo


def get_training_hc(sim_suite, emu_params=None, nyx_version="Jul2024"):
    """
    Loads and processes the training data for the emulator, including cosmological and IGM parameters.

    This function reads the relevant cosmological and IGM history files for the specified simulation suite
    (`mpg` or `nyx`), extracts the parameters needed for the emulator, and organizes them into a structure
    suitable for training. It returns the parameters used for training, the associated data points, and the raw
    cosmological and IGM data.

    Parameters:
    -----------
    sim_suite : str
        The simulation suite to use, either "mpg" or "nyx". Determines which files are loaded and processed.

    emu_params : list of str, optional, default=None
        A list of parameters to use for the cosmological emulator. If not provided, default parameters are
        selected based on the simulation suite. Possible values are `["Delta2_p", "n_p"]` for "mpg" and
        `["Delta2_p", "n_p", "alpha_p"]` for "nyx".

    nyx_version : str, optional, default="Jul2024"
        The version of the NYX simulation to use. Only used if `sim_suite` is "nyx".

    Returns:
    --------
    hc_params : list of str
        The list of parameters used for training the emulator, combining both cosmological and IGM parameters.

    hc_points : numpy.ndarray
        A 2D array where each row represents a set of values for the cosmological and IGM parameters used for training.

    cosmo_all : list of dict
        The raw cosmological data loaded from the emulator files. This includes the simulation parameters and labels.

    igm_all : dict
        The raw IGM history data loaded from the IGM history files. This includes the IGM parameters for each simulation.

    Raises:
    -------
    ValueError
        If the simulation suite is not recognized or if any of the required files are missing.

    Notes:
    -----
    - The function expects specific files for "mpg" and "nyx" simulations (cosmological and IGM history data).
      If any of these files are not found, it will raise a `ValueError` with a suggestion on how to generate them.
    - The cosmological parameters and IGM parameters are extracted and returned in a format suitable for training an emulator.
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
        cosmo_fname = os.path.join(
            os.environ["NYX_PATH"], "nyx_emu_cosmo_" + nyx_version + ".npy"
        )
        igm_fname = os.path.join(os.environ["NYX_PATH"], "IGM_histories.npy")
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
        if is_number_string(sim_label_cosmo[ii][-1]) == False:
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
