import os
import numpy as np
from lace.cosmo.cosmology import Cosmology
from lace.configuration import get_nyx_path
from cup1d.utils.utils import get_path_repo


def get_cosmology_from_label(cosmo_label="default"):
    if cosmo_label == "default":
        return Cosmology()
    elif cosmo_label == "low_omch2":
        return Cosmology(cosmo_params_dict=dict(omch2=0.11))
    elif cosmo_label == "high_omch2":
        return Cosmology(cosmo_params_dict=dict(omch2=0.13))
    elif cosmo_label == "omch2_0115":
        return Cosmology(cosmo_params_dict=dict(omch2=0.115))
    elif cosmo_label == "omch2_0125":
        return Cosmology(cosmo_params_dict=dict(omch2=0.125))
    elif cosmo_label == "mnu_03":
        return Cosmology(cosmo_params_dict=dict(mnu=0.3))
    elif cosmo_label == "mnu_06":
        return Cosmology(cosmo_params_dict=dict(mnu=0.6))
    elif cosmo_label == "SHOES":
        return Cosmology(cosmo_params_dict=dict(H0=73.0))
    else:
        raise ValueError("implement cosmo_label " + cosmo_label)


def set_cosmo(
    cosmo_label="mpg_central",
    return_all=False,
    nyx_version="models_Nyx_Mar2025_with_CGAN_val_3axes",
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
            fname = os.path.join(
                get_path_repo("lace"),
                "data",
                "sim_suites",
                "Australia20",
                "mpg_emu_cosmo.npy",
            )
            get_cosmo = lambda params: Cosmology(cosmo_params_dict=params)
        elif cosmo_label[:3] == "nyx":
            fname = os.path.join(
                get_nyx_path(), "nyx_emu_cosmo_" + nyx_version + ".npy"
            )
            get_cosmo = lambda params: Cosmology(
                cosmo_params_dict={
                    "H0": params["H_0"],
                    "ombh2": 0.02233,
                    "omch2": params["omega_m"] - 0.02233,
                    "mnu": 0.0,
                    "As": params["A_s"],
                    "ns": params["n_s"],
                    "nrun": params.get("nrun", 0.0),
                }
            )

        try:
            data_cosmo = np.load(fname, allow_pickle=True).item()
        except:
            raise ValueError(f"{fname} not found")

        if cosmo_label in data_cosmo.keys():
            # print(data_cosmo[cosmo_label]["cosmo_params"])
            cosmo = get_cosmo(data_cosmo[cosmo_label]["cosmo_params"])
        else:
            raise ValueError(f"Cosmo not found in {fname} for {cosmo_label}")

    elif cosmo_label == "Planck18":
        # Tab 2 of https://arxiv.org/abs/1807.06209, TT,TE,EE+lowE+lensing+BAO
        cosmo = Cosmology(cosmo_params_dict=dict(
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
        ))
    elif cosmo_label == "Planck18_high_omh2":
        cosmo = Cosmology(cosmo_params_dict=dict(
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
        ))
    elif cosmo_label == "Planck18_high3s_omh2":
        err_omch2 = 0.0009
        cosmo = Cosmology(cosmo_params_dict=dict(
            H0=67.66,
            mnu=0.0,
            omch2=0.119 + err_omch2 * 3,  # 3 sigma higher
            ombh2=0.0224,
            omk=0.0,
            As=2.105e-09,
            ns=0.9665,
            nrun=0.0,
            pivot_scalar=0.05,
            w=-1,
        ))
    elif cosmo_label == "Planck18_high1s_omh2":
        err_omch2 = 0.0009
        cosmo = Cosmology(cosmo_params_dict=dict(
            H0=67.66,
            mnu=0.0,
            omch2=0.119 + err_omch2,  # 1 sigma higher
            ombh2=0.0224,
            omk=0.0,
            As=2.105e-09,
            ns=0.9665,
            nrun=0.0,
            pivot_scalar=0.05,
            w=-1,
        ))
    elif cosmo_label == "Planck18_low_omh2":
        cosmo = Cosmology(cosmo_params_dict=dict(
            H0=67.66,
            mnu=0.0,
            omch2=0.1071,  # 10% smaller
            ombh2=0.0224,
            omk=0.0,
            As=2.105e-09,
            ns=0.9665,
            nrun=0.0,
            pivot_scalar=0.05,
            w=-1,
        ))
    elif cosmo_label == "Planck18_low3s_omh2":
        err_omch2 = 0.0009
        cosmo = Cosmology(cosmo_params_dict=dict(
            H0=67.66,
            mnu=0.0,
            omch2=0.119 - err_omch2 * 3,  # 3 sigma lower
            ombh2=0.0224,
            omk=0.0,
            As=2.105e-09,
            ns=0.9665,
            nrun=0.0,
            pivot_scalar=0.05,
            w=-1,
        ))
    elif cosmo_label == "Planck18_low1s_omh2":
        err_omch2 = 0.0009
        cosmo = Cosmology(cosmo_params_dict=dict(
            H0=67.66,
            mnu=0.0,
            omch2=0.119 - err_omch2,  # 1 sigma lower
            ombh2=0.0224,
            omk=0.0,
            As=2.105e-09,
            ns=0.9665,
            nrun=0.0,
            pivot_scalar=0.05,
            w=-1,
        ))
    elif cosmo_label == "Planck18_h74":
        cosmo = Cosmology(cosmo_params_dict=dict(
            H0=74.00,
            mnu=0.0,
            omch2=0.119,
            ombh2=0.0224,
            omk=0.0,
            As=2.105e-09,
            ns=0.9665,
            nrun=0.0,
            pivot_scalar=0.05,
            w=-1,
        ))
    elif cosmo_label == "Planck18_mnu03":
        # at fixed omh2, vary omch2
        omnuh2 = 0.00322433285312557  # for mnu 0.3 eV
        cosmo = Cosmology(cosmo_params_dict=dict(
            H0=67.66,
            mnu=0.3,
            omch2=0.119 - omnuh2,
            ombh2=0.0224,
            omk=0.0,
            As=2.105e-09,
            ns=0.9665,
            nrun=0.0,
            pivot_scalar=0.05,
            w=-1,
        ))
    elif cosmo_label == "Planck18_mnu03_varh":
        ## at fixed Om, for that, vary h
        # define cosmology first to get omnuh2
        cosmo = Cosmology(cosmo_params_dict=dict(
            H0=67.66,
            mnu=0.3,
            omch2=0.119,
            ombh2=0.0224,
            omk=0.0,
            As=2.105e-09,
            ns=0.9665,
            nrun=0.0,
            pivot_scalar=0.05,
            w=-1,
        ))
        background = cosmo.get_background_params()
        OmegaM_planck = (background["omch2"] + background["ombh2"]) / cosmo.get_h()**2
        omh2_nu = background["omch2"] + background["ombh2"] + background["omnuh2"]
        h_nu = np.sqrt(omh2_nu / OmegaM_planck)

        cosmo = Cosmology(cosmo_params_dict=dict(
            H0=h_nu * 100,
            mnu=0.3,
            omch2=0.119,
            ombh2=0.0224,
            omk=0.0,
            As=2.105e-09,
            ns=0.9665,
            nrun=0.0,
            pivot_scalar=0.05,
            w=-1,
        ))
    elif cosmo_label == "DESIDR2_ACT":
        # ACT https://arxiv.org/pdf/2503.14452, Table 5 (P-ACT)
        # omch2 = 0.1193
        # hact = 0.6762
        ombh2 = 0.0225
        ns = 0.9709
        As = np.exp(3.056) / 1e10

        # DESI https://arxiv.org/pdf/2504.18464, Table 3 (DESI+P-ACT+DESY5)
        h = 0.6685
        om = 0.3175
        omch2 = om * h**2 - ombh2
        w0 = -0.764
        wa = -0.77

        cosmo = Cosmology(cosmo_params_dict=dict(
            H0=h * 100,
            mnu=0.0,
            omch2=omch2,
            ombh2=ombh2,
            omk=0.0,
            As=As,
            ns=ns,
            nrun=0.0,
            pivot_scalar=0.05,
            w=w0,
            wa=wa,
        ))
    elif cosmo_label == "Planck18_nyx":
        cosmo = Cosmology(cosmo_params_dict=dict(
            H0=67.66,
            mnu=0.0,
            omch2=0.119,
            ombh2=0.0224,
            omk=0.0,
            As=2.24e-09,
            ns=0.937,
            nrun=0.0,
            pivot_scalar=0.05,
            w=-1,
        ))
    elif cosmo_label == "Planck18_mpg":
        cosmo = Cosmology(cosmo_params_dict=dict(
            H0=67.0,
            mnu=0.0,
            omch2=0.119,
            ombh2=0.022,
            omk=0.0,
            As=2.26e-09,
            ns=0.982,
            nrun=0.0,
            pivot_scalar=0.05,
            w=-1,
        ))
    elif (cosmo_label == "ACCEL2_6144_160") | (cosmo_label == "accel2"):
        # https://arxiv.org/pdf/2407.04473
        # Planck15 ΛCDM Planck TT,TE,EE+lowP (approx...)
        Omegam = 0.31
        Omegab = 0.0487
        h = 0.675
        omch2 = (Omegam - Omegab) * h**2
        ombh2 = Omegab * h**2
        cosmo = Cosmology(cosmo_params_dict=dict(
            H0=h * 100,
            mnu=0.0,
            omch2=omch2,
            ombh2=ombh2,
            omk=0.0,
            As=np.exp(3.094) / 1e10,  # Planck15 ΛCDM Planck TT,TE,EE+lowP
            ns=0.96,
            nrun=0.0,
            pivot_scalar=0.05,
            w=-1,
        ))
    elif (cosmo_label == "Sherwood_2048_40") | (cosmo_label == "sherwood"):
        # https://academic.oup.com/mnras/article/464/1/897/2236089
        # Planck13 ΛCDM Planck+WP+highL+BAO
        Omegam = 0.308
        Omegab = 0.0482
        h = 0.678
        omch2 = (Omegam - Omegab) * h**2
        ombh2 = Omegab * h**2
        cosmo = Cosmology(cosmo_params_dict=dict(
            H0=h * 100,
            mnu=0.0,
            omch2=omch2,
            ombh2=ombh2,
            omk=0.0,
            As=np.exp(3.0973) / 1e10,  # Planck13 ΛCDM Planck+WP+highL+BAO
            ns=0.961,
            nrun=0.0,
            pivot_scalar=0.05,
            w=-1,
        ))
    else:
        raise ValueError(f"cosmo_label {cosmo_label} not implemented")

    if return_all:
        return data_cosmo
    else:
        return cosmo
