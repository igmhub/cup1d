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

# MPI stuff
from mpi4py import MPI
from cup1d.utils.utils import create_print_function


def main():
    ## MPI stuff
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    ##

    folder_in = "/home/jchaves/Proyectos/projects/lya/data/mock_challenge/MockChallengeSnapshot/mockchallenge-0.2/"
    folder_out = (
        "/home/jchaves/Proyectos/projects/lya/data/mock_challenge/oct21/"
    )
    files = np.sort(glob.glob(folder_in + "*.fits"))
    if rank == 0:
        for ii in range(len(files)):
            print(ii, files[ii])
    # sys.exit()

    ## set archive and emulator

    # args = Args(emulator_label="Pedersen23_ext", training_set="Cabayol23")
    # args = Args(emulator_label="Cabayol23+", training_set="Cabayol23")
    args = Args(emulator_label="Nyx_alphap", training_set="Nyx23_Oct2023")
    os.environ["OMP_NUM_THREADS"] = "1"
    args.parallel = True
    # create print function (only for rank 0)
    fprint = create_print_function(verbose=args.verbose)
    if rank == 0:
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

        # distribute emulator to all ranks
        for irank in range(1, size):
            comm.send(emulator, dest=irank, tag=(irank + 1) * 7)
    else:
        # receive emulator from ranks 0
        emulator = comm.recv(source=0, tag=(rank + 1) * 7)

    if len(sys.argv) == 1:
        niter = len(files)
    else:
        niter = 1

    # for isim in range(niter):
    for isim in range(20, 21):
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
        if rank == 0:
            data = {"P1Ds": None, "extra_P1Ds": None}
            data["P1Ds"] = P1D_DESIY1(
                fname=fname, true_sim_label=true_sim_label, emu_error=0.02
            )
            for irank in range(1, size):
                comm.send(data, dest=irank, tag=(irank + 1) * 11)
        else:
            # get testing_data from task 0
            data = comm.recv(source=0, tag=(rank + 1) * 11)

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
            fprint(p.name, p.value, p.min_value, p.max_value)

        # for sampler, no real fit, just test
        # args.n_steps = 5
        # args.n_burn_in = 1
        # args.explore = True
        args.n_steps = 1000
        args.n_burn_in = 1500
        args.explore = False

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

        # run minimizer
        if rank == 0:
            p0 = np.array(list(like.fid["fit_cube"].values()))
            p0[:] = 0.5
            fitter.run_minimizer(log_func_minimize=fitter.like.get_chi2, p0=p0)
            for irank in range(1, size):
                comm.send(fitter.mle_cube, dest=irank, tag=(irank + 1) * 7)
        else:
            # get testing_data from task 0
            fitter.mle_cube = comm.recv(source=0, tag=(rank + 1) * 7)
        comm.Barrier()

        # run sampler
        # function for sampler
        def func_for_sampler(p0):
            res = fitter.like.get_log_like(values=p0, return_blob=True)
            return res[0], *res[2]

        _emcee_sam = fitter.run_sampler(
            pini=fitter.mle_cube, log_func=func_for_sampler
        )

        if rank == 0:
            ind = np.argmax(fitter.lnprob.reshape(-1))
            nparam = fitter.chain.shape[-1]
            p0 = fitter.chain.reshape(-1, nparam)[ind, :]
            fitter.run_minimizer(log_func_minimize=fitter.like.get_chi2, p0=p0)

            fitter.write_chain_to_file()

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
                dict_out["best"]["Delta2_star"]
                / dict_out["true"]["Delta2_star"]
                - 1
            ) * 100
            dict_out["rel diff [%]"]["n_star"] = (
                dict_out["best"]["n_star"] / dict_out["true"]["n_star"] - 1
            ) * 100
            dict_out["rel diff [%]"]["alpha_star"] = (
                dict_out["best"]["alpha_star"] / dict_out["true"]["alpha_star"]
                - 1
            ) * 100

            np.save(dir_out + "/results.npy", dict_out)

            fitter.plot_p1d(
                residuals=False, plot_every_iz=1, save_directory=dir_out
            )
            fitter.plot_p1d(
                residuals=True, plot_every_iz=2, save_directory=dir_out
            )
            fitter.plot_igm(cloud=True, save_directory=dir_out)

            for ii in range(10):
                plt.close()


if __name__ == "__main__":
    main()
