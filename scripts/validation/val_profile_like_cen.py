import os

# os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=4
import numpy as np
from cup1d.likelihood.input_pipeline import Args
from cup1d.likelihood.pipeline import Pipeline
from cup1d.utils.utils import get_path_repo


def main():
    emu = "mpg"
    # emu = "nyx"

    # baseline
    # data_label = "mpg_central"
    # data_label = "nyx_central"
    data_label = "sherwood"

    if data_label == "mpg_central":
        zmin = 2.25
        zmax = 4.25
    else:
        zmin = 2.2
        zmax = 4.2

    fit_type = "global_opt"
    cov_label = "DESIY1_QMLE3"

    name_variation = "sim_" + data_label

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
        fix_cosmo=False,
        name_variation=name_variation,
        zmin=zmin,
        zmax=zmax,
    )
    pip = Pipeline(args, out_folder=args.out_folder)

    if name_variation == "cov":
        pip.fitter.like.full_icov_Pk_kms /= 1.1**2

    input_pars = pip.fitter.like.sampling_point_from_parameters().copy()

    print("starting minimization")
    type_minimizer = "NM"
    if type_minimizer == "NM":
        pip.fitter.run_minimizer(
            pip.fitter.like.minus_log_prob,
            p0=input_pars,
            restart=True,
            # burn_in=True,
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

    print("saving output to:", file_out)
    np.save(file_out, out_dict)

    pip.fitter.save_fitter()


if __name__ == "__main__":
    main()
