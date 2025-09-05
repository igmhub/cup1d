import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=4
import numpy as np
from cup1d.likelihood.input_pipeline import Args
from cup1d.likelihood.pipeline import Pipeline
from cup1d.utils.utils import get_path_repo


def main():
    emu = "mpg"
    # emu = "nyx"
    # data_label = "DESIY1_QMLE3"
    data_label = "DESIY1_QMLE"
    # data_label = "DESIY1_FFT_dir"
    # data_label = "DESIY1_FFT"

    name_variation = None
    # name_variation = "cov"

    # prof_type = "prof_2d"
    # nsig = 8
    # nelem = 8
    # mle_cosmo_cen = None

    prof_type = "prof_2d_deep"
    nsig = 5
    nelem = 30

    mle_cosmo_cen = {"Delta2_star": 0.44, "n_star": -2.28}  # all mpg
    # mle_cosmo_cen = {"Delta2_star": 0.42, "n_star": -2.22}  # mpg qmle done
    # mle_cosmo_cen = {"Delta2_star": 0.45, "n_star": -2.27}  # mpg fft_dir done
    # mle_cosmo_cen = {"Delta2_star": 0.45, "n_star": -2.27}  # mpg fft done
    # mle_cosmo_cen = {"Delta2_star": 0.36, "n_star": -2.26}  # nyx qmle3 done

    # baseline
    fit_type = "global_opt"
    args = Args(data_label=data_label, emulator_label="CH24_" + emu + "cen_gpr")
    args.set_baseline(
        fit_type=fit_type,
        fix_cosmo=True,
        P1D_type=data_label,
        name_variation=name_variation,
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
