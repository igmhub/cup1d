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
    fit_type = "global_opt"
    data_label = "DESIY1_QMLE3"
    # data_label = "DESIY1_QMLE"
    # data_label = "DESIY1_FFT_dir"
    # data_label = "DESIY1_FFT"

    variations = [
        # None,
        # "no_inflate",  # no increase errors for 3, 3.6, and 4
        "no_emu_cov",  # no emu error
        "no_inflate_no_emu_cov",  # no emu error, no increase errors for 3, 3.6, and 4
        # "cosmo",  # different fiducial cosmo
        # "metal_trad",  # 2 params for metals like eBOSS
        # "metal_si2",  # no SiII-SiII cont
        # "metal_deco",  # no decorrelation metals
        # "metal_thin",  # no desviation from optically-thin limit
        # "no_res",  # no resolution correction
        # "Turner24",  # mF from Turner24 with 1 free param to scale
        # "more_igm",  # 8 params for IGM evolution
        # "less_igm",  # 4 params for IGM evolution
        # "metals_z",  # 2 params for z ev metals
        # "hcd_z",  # 2 params for z ev hcd
    ]

    for ivar in range(len(variations)):
        name_variation = variations[ivar]

        for ii in range(5):
            print("")
        print("VARIATION:", name_variation)
        for ii in range(5):
            print("")

        args = Args(
            data_label=data_label, emulator_label="CH24_" + emu + "cen_gpr"
        )
        args.set_baseline(
            fit_type=fit_type,
            fix_cosmo=False,
            P1D_type=data_label,
            name_variation=name_variation,
            inflate_err=True,
        )
        pip = Pipeline(args, out_folder=args.out_folder)

        input_pars = pip.fitter.like.sampling_point_from_parameters().copy()

        # print("starting minimization")
        # type_minimizer = "NM"
        # if type_minimizer == "NM":
        pip.fitter.run_minimizer(
            pip.fitter.like.minus_log_prob,
            p0=input_pars,
            restart=True,
            # burn_in=True,
        )
        # else:
        #     pip.fitter.run_minimizer_da(
        #         pip.fitter.like.minus_log_prob, p0=input_pars, restart=True
        #     )

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
