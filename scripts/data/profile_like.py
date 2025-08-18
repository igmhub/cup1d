import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
import numpy as np
from cup1d.likelihood.input_pipeline import Args
from cup1d.likelihood.pipeline import Pipeline
from cup1d.utils.utils import get_path_repo


def main():
    type_minimizer = "NM"
    # type_minimizer = "DA"

    # baseline
    fit_type = "andreu2"
    args = Args(data_label="DESIY1_QMLE3", emulator_label="CH24_nyxcen_gpr")

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

    args.set_baseline(fit_type=fit_type, fix_cosmo=False)
    path_out = os.path.join(
        os.path.dirname(get_path_repo("cup1d")),
        "data",
        "out_DESI_DR1",
        args.P1D_type,
        args.fit_type,
        args.emulator_label,
        type_minimizer,
    )

    if fit_type == "global":
        sigma_cosmo = {"Delta2_star": 0.02, "n_star": 0.01}
    elif fit_type == "andreu2":
        sigma_cosmo = {"Delta2_star": 0.02, "n_star": 0.01}
    else:
        raise ValueError("fit_type must be 'global' or 'andreu2'")

    sigma_cosmo = {"Delta2_star": 0.02}

    pip = Pipeline(args, out_folder=path_out)
    # p0 = pip.fitter.like.sampling_point_from_parameters().copy()
    pip.run_profile(args, sigma_cosmo, nelem=10, type_minimizer=type_minimizer)


if __name__ == "__main__":
    main()
