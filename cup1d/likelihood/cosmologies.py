import os
import numpy as np

import lace
from lace.cosmo import camb_cosmo


def get_cosmology_from_label(cosmo_label="default"):
    if cosmo_label == "default":
        return camb_cosmo.get_cosmology()
    elif cosmo_label == "low_omch2":
        return camb_cosmo.get_cosmology(omch2=0.11)
    elif cosmo_label == "high_omch2":
        return camb_cosmo.get_cosmology(omch2=0.13)
    elif cosmo_label == "omch2_0115":
        return camb_cosmo.get_cosmology(omch2=0.115)
    elif cosmo_label == "omch2_0125":
        return camb_cosmo.get_cosmology(omch2=0.125)
    elif cosmo_label == "mnu_03":
        return camb_cosmo.get_cosmology(mnu=0.3)
    elif cosmo_label == "mnu_06":
        return camb_cosmo.get_cosmology(mnu=0.6)
    elif cosmo_label == "SHOES":
        return camb_cosmo.get_cosmology(H0=73.0)
    else:
        raise ValueError("implement cosmo_label " + cosmo_label)


def set_cosmo(
    cosmo_label="mpg_central", return_all=False, nyx_version="Jul2024"
):
    """Set fiducial cosmology

    Parameters
    ----------
    cosmo_label : str

    Returns
    -------
    cosmo : object
    """
    if (cosmo_label[:3] == "mpg") | (cosmo_label[:3] == "nyx"):
        if cosmo_label[:3] == "mpg":
            repo = os.path.dirname(lace.__path__[0]) + "/"
            fname = repo + ("data/sim_suites/Australia20/mpg_emu_cosmo.npy")
            get_cosmo = camb_cosmo.get_cosmology_from_dictionary
        elif cosmo_label[:3] == "nyx":
            fname = (
                os.environ["NYX_PATH"] + "nyx_emu_cosmo_" + nyx_version + ".npy"
            )
            get_cosmo = camb_cosmo.get_Nyx_cosmology

        try:
            data_cosmo = np.load(fname, allow_pickle=True).item()
        except:
            raise ValueError(f"{fname} not found")

        if cosmo_label in data_cosmo.keys():
            cosmo = get_cosmo(data_cosmo[cosmo_label]["cosmo_params"])
        else:
            raise ValueError(f"Cosmo not found in {fname} for {cosmo_label}")

    elif cosmo_label == "Planck18":
        cosmo = camb_cosmo.get_cosmology(
            H0=67.66,
            mnu=0.0,
            omch2=0.119,
            ombh2=0.0224,
            omk=0.0,
            As=2.105e-09,
            ns=0.9665,
            nrun=0.0,
            pivot_scalar=0.05,
            w=-1,
        )
    else:
        raise ValueError(f"cosmo_label {cosmo_label} not implemented")

    if return_all:
        return data_cosmo
    else:
        return cosmo
