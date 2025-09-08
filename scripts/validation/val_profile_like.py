import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=4
import numpy as np
from cup1d.likelihood.input_pipeline import Args
from cup1d.likelihood.pipeline import Pipeline
from cup1d.utils.utils import get_path_repo


def main():
    # dataset
    # data_label = "mpg_central"
    data_label = "nyx_central"
    # data_label = "sherwood"

    if data_label == "mpg_central":
        zmin = 2.25
        zmax = 4.25
    else:
        zmin = 2.2
        zmax = 4.2

    emu = "mpg"
    fit_type = "global_opt"
    cov_label = "DESIY1_QMLE3"

    name_variation = "sim_" + data_label

    # prof_type = "prof_2d"
    # nsig = 8
    # nelem = 8
    # mle_cosmo_cen = None

    prof_type = "prof_2d_deep"
    nsig = 5
    nelem = 30

    # if data_label == "mpg_central":
    #     mle_cosmo_cen = {"Delta2_star": 0.35, "n_star": -2.30}  # all mpg
    # elif data_label == "nyx_central":
    #     mle_cosmo_cen = {"Delta2_star": 0.36, "n_star": -2.31}  # nyx
    # elif data_label == "sherwood":
    #     mle_cosmo_cen = {"Delta2_star": 0.34, "n_star": -2.30}  # sherwood
    # else:
    #     raise ValueError("Wrong data_label")

    mle_cosmo_cen = {"Delta2_star": 0.35, "n_star": -2.30}

    # baseline
    fit_type = "global_opt"
    args = Args(
        data_label=data_label,
        cov_label=cov_label,
        emulator_label="CH24_" + emu + "cen_gpr",
        true_cosmo_label=data_label,
        fid_cosmo_label=data_label,
        apply_smoothing=True,
    )
    args.set_baseline(
        fit_type=fit_type,
        fix_cosmo=True,
        name_variation=name_variation,
        zmin=zmin,
        zmax=zmax,
    )
    out_folder = os.path.join(args.out_folder, prof_type)
    pip = Pipeline(args, out_folder=out_folder)

    if name_variation == "cov":
        pip.fitter.like.full_icov_Pk_kms /= 1.1**2

    sigma_cosmo = {"Delta2_star": 0.027, "n_star": 0.017}

    # sigma_cosmo = {"Delta2_star": 0.02}
    # out_folder = os.path.join(args.out_folder, "prof_dstar")

    # sigma_cosmo = {"n_star": 0.01}
    # out_folder = os.path.join(args.out_folder, "prof_nstar")

    pip.run_profile(
        sigma_cosmo, nelem=nelem, nsig=nsig, mle_cosmo_cen=mle_cosmo_cen
    )


if __name__ == "__main__":
    main()
