import numpy as np
import time, os, sys
import glob
import matplotlib.pyplot as plt

# our own modules
from lace.cosmo import camb_cosmo
from lace.emulator.emulator_manager import set_emulator
from cup1d.likelihood import lya_theory, likelihood
from cup1d.likelihood.fitter import Fitter

from cup1d.likelihood.pipeline import (
    set_archive,
    set_P1D,
    set_cosmo,
    set_free_like_parameters,
    set_like,
)

from cup1d.p1ds.data_DESIY1 import P1D_DESIY1
from cup1d.likelihood.input_pipeline import Args


def main():
    folder_in = "/home/jchaves/Proyectos/projects/lya/data/mock_challenge/MockChallengeSnapshot/mockchallenge-0.2/"
    folder_out = (
        "/home/jchaves/Proyectos/projects/lya/data/mock_challenge/oct21/"
    )
    files = np.sort(glob.glob(folder_in + "*.fits"))
    for ii in range(len(files)):
        print(ii, files[ii])
    # sys.exit()

    ## set emulator

    # args = Args(emulator_label="Pedersen23_ext", training_set="Cabayol23")
    # args = Args(emulator_label="Cabayol23+", training_set="Cabayol23")
    args = Args(emulator_label="Nyx_alphap", training_set="Nyx23_Oct2023")

    archive = set_archive(args.training_set)

    emulator = set_emulator(
        emulator_label=args.emulator_label,
        archive=archive,
    )

    if emulator.emulator_label == "Nyx_alphap":
        emulator.list_sim_cube = archive.list_sim_cube
        emulator.list_sim_cube.remove("nyx_14")
    else:
        emulator.list_sim_cube = archive.list_sim_cube

    if len(sys.argv) == 1:
        niter = len(files)
    else:
        niter = 1

    # for isim in range(niter):
    for isim in range(2, 8):
        if isim == 4:
            continue
        if len(sys.argv) == 2:
            fname = files[int(sys.argv[1])]
        else:
            fname = files[isim]

        print("Analyzing:", fname)
        dir_out = folder_out + os.path.basename(fname)[:-5]
        os.makedirs(dir_out, exist_ok=True)
        print("Output in:", dir_out)

        if "fiducial" in fname:
            true_sim_label = "nyx_central"
        elif "bar_ic_grid" in fname:
            true_sim_label = "nyx_3"
        elif "cosmo_grid_3" in fname:
            true_sim_label = "nyx_3"
        elif "CGAN" in fname:
            # this is a temporary hack
            true_sim_label = "nyx_central"
        else:
            raise ValueError("true sim label not found")

        if "bar_ic_grid" in fname:
            args.ic_correction = True
        else:
            args.ic_correction = False

        ## set data
        data = {"P1Ds": None, "extra_P1Ds": None}
        data["P1Ds"] = P1D_DESIY1(
            fname=fname, true_sim_label=true_sim_label, emu_error=0.02
        )
        # data["P1Ds"].plot_p1d()

        ## set likelihood
        # cosmology
        # args.fid_cosmo_label="mpg_central"
        args.fid_cosmo_label = "nyx_central"
        # args.fid_cosmo_label = "nyx_3"
        # args.fid_cosmo_label="Planck18"
        fid_cosmo = set_cosmo(cosmo_label=args.fid_cosmo_label)

        # IGM
        args.fid_igm_label = "nyx_central"
        # args.fid_igm_label = "nyx_3"
        args.type_priors = "hc"

        # contaminants
        args.fid_SiIII = [0, -10]
        args.fid_SiII = [0, -10]
        args.fid_HCD = [0, -6]
        args.fid_SN = [0, -4]

        # parameters
        args.vary_alphas = True
        args.fix_cosmo = False
        args.n_tau = 2
        args.n_sigT = 2
        args.n_gamma = 2
        args.n_kF = 2
        if "fsiiii" in fname:
            args.n_SiIII = 1
            args.fid_SiIII = [0, -3]
        else:
            args.n_SiIII = 0
        args.n_SiII = 0
        args.n_dla = 0
        args.n_sn = 0

        free_parameters = set_free_like_parameters(args)

        like = set_like(
            data["P1Ds"],
            emulator,
            fid_cosmo,
            free_parameters,
            args,
            data_hires=data["extra_P1Ds"],
        )

        for p in like.free_params:
            print(p.name, p.value, p.min_value, p.max_value)

        # like.plot_igm()

        # for sampler, no real fit, just test
        args.n_steps = 50
        args.n_burn_in = 10
        args.parallel = False
        args.explore = True

        fitter = Fitter(
            like=like,
            rootdir=dir_out,
            save_chain=False,
            nburnin=args.n_burn_in,
            nsteps=args.n_steps,
            parallel=args.parallel,
            explore=args.explore,
            fix_cosmology=args.fix_cosmo,
        )
        run_sampler = False
        if run_sampler:
            _emcee_sam = sampler.run_sampler(log_func=fitter.like.get_chi2)

        p0 = np.array(list(like.fid["fit_cube"].values()))
        # p0[:] = 0.5
        fitter.run_minimizer(log_func_minimize=fitter.like.get_chi2, p0=p0)

        ## save results

        dict_out = {}
        dict_out["best"] = {}
        dict_out["true"] = {}
        dict_out["rel diff [%]"] = {}

        dict_out["best"]["Delta2_star"] = fitter.mle_cosmo["Delta2_star"]
        dict_out["best"]["n_star"] = fitter.mle_cosmo["n_star"]
        dict_out["best"]["alpha_star"] = fitter.mle_cosmo["alpha_star"]

        dict_out["true"]["Delta2_star"] = fitter.truth["$\\Delta^2_\\star$"]
        dict_out["true"]["n_star"] = fitter.truth["$n_\\star$"]
        dict_out["true"]["alpha_star"] = fitter.truth["$\\alpha_\\star$"]

        dict_out["rel diff [%]"]["Delta2_star"] = (
            dict_out["best"]["Delta2_star"] / dict_out["true"]["Delta2_star"]
            - 1
        ) * 100
        dict_out["rel diff [%]"]["n_star"] = (
            dict_out["best"]["n_star"] / dict_out["true"]["n_star"] - 1
        ) * 100
        dict_out["rel diff [%]"]["alpha_star"] = (
            dict_out["best"]["alpha_star"] / dict_out["true"]["alpha_star"] - 1
        ) * 100

        np.save(dir_out + "/results.npy", dict_out)

        fitter.plot_p1d(
            residuals=False, plot_every_iz=1, save_directory=dir_out
        )
        fitter.plot_p1d(residuals=True, plot_every_iz=2, save_directory=dir_out)
        fitter.plot_igm(cloud=True, save_directory=dir_out)

        for ii in range(10):
            plt.close()


if __name__ == "__main__":
    main()
