"""Named cosmology helpers used by the likelihood pipeline."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from lace.cosmo import camb_cosmo

from cup1d.utils.utils import get_path_repo


def get_cosmology_from_label(cosmo_label: str = "default") -> Any:
    """Return a small set of hard-coded CAMB cosmology variations.

    Parameters
    ----------
    cosmo_label : str, optional
        Label for the desired cosmology variation. Default is "default".

    Returns
    -------
    Any
        CAMB cosmology object.

    Raises
    ------
    ValueError
        If the cosmo_label is not recognized.
    """
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
    cosmo_label: str = "mpg_central",
    return_all: bool = False,
    nyx_version: str = "models_Nyx_Mar2025_with_CGAN_val_3axes",
) -> Any:
    """Return a CAMB cosmology for a simulation or named analysis label.

    Parameters
    ----------
    cosmo_label : str
        Simulation label or named cosmology variation.
    return_all : bool, optional
        If supported by a branch, return all loaded cosmology metadata.
    nyx_version : str, optional
        Nyx cosmology file suffix used for Nyx simulation labels.

    Returns
    -------
    Any
        CAMB cosmology object.

    Raises
    ------
    ValueError
        If the cosmology file is not found or the label is not in the file.
    """
    if (cosmo_label[:3] == "mpg") | (cosmo_label[:3] == "nyx"):
        if cosmo_label[:3] == "mpg":
            fname = os.path.join(
                get_path_repo("lace"),
                "data",
                "sim_suites",
                "Australia20",
                "mpg_emu_cosmo.npy",
            )
            get_cosmo = camb_cosmo.get_cosmology_from_dictionary
        elif cosmo_label[:3] == "nyx":
            fname = os.path.join(
                os.environ["NYX_PATH"], "nyx_emu_cosmo_" + nyx_version + ".npy"
            )
            get_cosmo = camb_cosmo.get_Nyx_cosmology

        try:
            data_cosmo = np.load(fname, allow_pickle=True).item()
        except Exception:
            raise ValueError(f"{fname} not found") from None

        if cosmo_label in data_cosmo.keys():
            # print(data_cosmo[cosmo_label]["cosmo_params"])
            cosmo = get_cosmo(data_cosmo[cosmo_label]["cosmo_params"])
        else:
            raise ValueError(f"Cosmo not found in {fname} for {cosmo_label}")

    elif cosmo_label == "Planck18":
        # Tab 2 of https://arxiv.org/abs/1807.06209, TT,TE,EE+lowE+lensing+BAO
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
    elif cosmo_label == "Planck18_high_omh2":
        cosmo = camb_cosmo.get_cosmology(
            H0=67.66,
            mnu=0.0,
            omch2=0.1309,  # 10% higher
            ombh2=0.0224,
            omk=0.0,
            As=2.105e-09,
            ns=0.9665,
            nrun=0.0,
            pivot_scalar=0.05,
            w=-1,
        )
    elif cosmo_label == "Planck18_low_omh2":
        cosmo = camb_cosmo.get_cosmology(
            H0=67.66,
            mnu=0.0,
            omch2=0.1071,  # 10% lower
            ombh2=0.0224,
            omk=0.0,
            As=2.105e-09,
            ns=0.9665,
            nrun=0.0,
            pivot_scalar=0.05,
            w=-1,
        )
    elif cosmo_label == "Planck15":
        # Tab 4 of https://arxiv.org/abs/1502.01589, TT,TE,EE+lowP+lensing+ext
        cosmo = camb_cosmo.get_cosmology(
            H0=67.74,
            mnu=0.06,
            omch2=0.1188,
            ombh2=0.0223,
            omk=0.0,
            As=2.142e-09,
            ns=0.9667,
            nrun=0.0,
            pivot_scalar=0.05,
            w=-1,
        )
    else:
        raise ValueError("cosmo_label " + cosmo_label + " not implemented")

    if return_all:
        return cosmo, data_cosmo[cosmo_label]
    else:
        return cosmo
