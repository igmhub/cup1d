import os

# os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "25"  # export OMP_NUM_THREADS=4
import numpy as np
from cup1d.likelihood.input_pipeline import Args
from cup1d.likelihood.pipeline import Pipeline
from cup1d.utils.utils import get_path_repo


def main():
    fit_type = "global"
    # fit_type = "andreu2"
    type_minimizer = "NM"
    # type_minimizer = "DA"
    args = Args(data_label="DESIY1_QMLE3", emulator_label="CH24_mpgcen_gpr")
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
    os.makedirs(path_out, exist_ok=True)

    pip = Pipeline(args, out_folder=path_out)

    input_pars = pip.fitter.like.sampling_point_from_parameters().copy()

    print("starting minimization")
    if type_minimizer == "NM":
        pip.fitter.run_minimizer(
            pip.fitter.like.minus_log_prob,
            p0=input_pars,
            restart=True,
            nsamples=10,
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

    file_out = os.path.join(path_out, "best_dircosmo.npy")
    # file_out = os.path.join(path_out, "best_dircosmo_NM.npy")
    # file_out = os.path.join(path_out, "best_dircosmo_DA.npy")
    # file_out = os.path.join(path_out, "best_dircosmo_DE.npy")
    # file_out = os.path.join(path_out, "best_dircosmo_DI.npy")

    print("saving IC to:", file_out)
    np.save(file_out, out_dict)

    pip.fitter.save_fitter()


if __name__ == "__main__":
    main()
