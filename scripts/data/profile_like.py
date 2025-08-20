import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=4
import numpy as np
from cup1d.likelihood.input_pipeline import Args
from cup1d.likelihood.pipeline import Pipeline


def main():
    type_minimizer = "NM"
    # type_minimizer = "DA"
    emu = "mpg"
    # emu = "nyx"

    # baseline
    fit_type = "global_opt"
    args = Args(
        data_label="DESIY1_QMLE3", emulator_label="CH24_" + emu + "cen_gpr"
    )

    # # baseline
    # fit_type = "andreu2"
    # args = Args(data_label="DESIY1_QMLE3", emulator_label="CH24_nyxcen_gpr")

    # nuisance
    # fit_type = "global"
    # args = Args(data_label="DESIY1_QMLE3", emulator_label="CH24_nyxcen_gpr")

    # emulator
    # fit_type = "andreu2"
    # args = Args(data_label="DESIY1_QMLE3", emulator_label="CH24_mpgcen_gpr")

    # QMLE
    # fit_type = "andreu2"
    # args = Args(data_label="DESIY1_QMLE", emulator_label="CH24_nyxcen_gpr")

    # FFT
    # fit_type = "andreu2"
    # args = Args(data_label="DESIY1_FFT_dir", emulator_label="CH24_nyxcen_gpr")

    args.set_baseline(fit_type=fit_type, fix_cosmo=True)
    pip = Pipeline(args, out_folder=args.out_folder)

    fname = emu + "_ic_global_red.npy"
    fullpath = os.path.join(
        os.path.dirname(get_path_repo("cup1d")), "data", "ics", fname
    )
    pip.fitter.like.set_ic_global(fullpath, verbose=True)

    sigma_cosmo = {"Delta2_star": 0.02, "n_star": 0.01}

    # sigma_cosmo = {"Delta2_star": 0.02}
    # out_folder = os.path.join(args.out_folder, "prof_dstar")

    # sigma_cosmo = {"n_star": 0.01}
    # out_folder = os.path.join(args.out_folder, "prof_nstar")

    out_folder = os.path.join(args.out_folder, "prof_2d")

    pip = Pipeline(args, out_folder=out_folder)
    # p0 = pip.fitter.like.sampling_point_from_parameters().copy()
    pip.run_profile(args, sigma_cosmo, nelem=4, type_minimizer=type_minimizer)


if __name__ == "__main__":
    main()
