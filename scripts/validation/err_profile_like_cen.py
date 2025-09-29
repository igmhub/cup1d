import os

# os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=4
import numpy as np
from cup1d.likelihood.input_pipeline import Args
from cup1d.likelihood.pipeline import Pipeline
from cup1d.utils.utils import get_path_repo
from cup1d.pipeline.set_archive import set_archive


def main():
    nseed = 1000
    emu = "mpg"
    # emu = "nyx"

    # baseline
    # data_label = "mpg_central"
    data_label = "nyx_central"
    # data_label = "sherwood"

    cov_label = "DESIY1_QMLE3"
    name_variation = "sim_" + data_label

    args = Args(
        data_label=data_label,
        cov_label=cov_label,
        emulator_label="CH24_" + emu + "cen_gpr",
        true_cosmo_label=data_label,
        fid_cosmo_label=data_label,
        apply_smoothing=True,
        add_noise=True,
        seed_noise=0,
    )

    if data_label == "mpg_central":
        zmin = 2.25
        zmax = 4.25
        archive_mock = set_archive(training_set="Cabayol23")
    else:
        zmin = 2.2
        zmax = 4.2
        archive_mock = set_archive(training_set=args.nyx_training_set)

    args.set_baseline(
        fit_type="global_opt",
        fix_cosmo=False,
        P1D_type=cov_label,
        name_variation=name_variation,
        z_min=zmin,
        z_max=zmax,
    )

    for iseed in range(1, nseed):
        for ii in range(5):
            print("")
        print("seed:", iseed)
        for ii in range(5):
            print("")

        args.seed_noise = iseed
        out_folder = os.path.join(
            args.out_folder,
            "seed_" + str(args.seed_noise),
        )
        pip = Pipeline(args, out_folder=out_folder)

        input_pars = pip.fitter.like.sampling_point_from_parameters().copy()

        pip.fitter.run_minimizer(
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

        # pip.fitter.save_fitter()


if __name__ == "__main__":
    main()
