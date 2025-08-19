import os

# os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=4
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
    pip = Pipeline(args, out_folder=args.out_folder)

    input_pars = pip.fitter.like.sampling_point_from_parameters().copy()

    try:
        file_in = os.path.join(
            os.path.dirname(pip.fitter.save_directory),
            "best_dircosmo.npy",
        )
    except:
        input_pars = pip.fitter.like.sampling_point_from_parameters().copy()
    else:
        print("loading IC from:", file_in)
        out_dict = np.load(file_in, allow_pickle=True).item()
        input_pars = out_dict["mle_cube"]

    # input_pars[:] = 0.5

    print("starting minimization")
    if type_minimizer == "NM":
        pip.fitter.run_minimizer(
            pip.fitter.like.minus_log_prob,
            p0=input_pars,
            restart=True,
            burn_in=True,
        )
    else:
        pip.fitter.run_minimizer_da(
            pip.fitter.like.minus_log_prob, p0=input_pars, restart=True
        )

    out_dict = {
        "best_chi2": pip.fitter.mle_chi2,
        "mle_cosmo_cen": pip.fitter.mle_cosmo,
        "mle_cube": pip.fitter.mle_cube,
        "mle": pip.fitter.mle,
    }

    file_out = os.path.join(args.out_folder, "best_dircosmo.npy")

    print("saving IC to:", file_out)
    np.save(file_out, out_dict)

    pip.fitter.save_fitter()


if __name__ == "__main__":
    main()
