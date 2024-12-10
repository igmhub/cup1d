import os, sys, time
import numpy as np
from mpi4py import MPI

# our own modules
from cup1d.utils.utils import create_print_function
from cup1d.likelihood.cosmologies import set_cosmo
from lace.emulator.emulator_manager import set_emulator
from cup1d.likelihood.fitter import Fitter
from cup1d.likelihood.plotter import Plotter

from cup1d.likelihood.pipeline import (
    set_free_like_parameters,
    set_archive,
    set_P1D,
    set_like,
)


class Pipeline_z(object):
    """Full pipeline for extracting cosmology from P1D using sampler one z at a time"""

    def __init__(self, args, make_plots=False, out_folder=None):
        """Set pipeline_z"""

        self.out_folder = out_folder

        ## MPI stuff
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        # create print function (only for rank 0)
        fprint = create_print_function(verbose=args.verbose)
        self.fprint = fprint
        self.explore = args.explore

        # when reusing archive and emulator, these must be None for
        # rank != 0 to prevent a very large memory footprint
        if rank != 0:
            args.archive = None
            args.emulator = None

        ###################

        ## set training set (only for rank 0)
        if rank == 0:
            start_all = time.time()
            start = time.time()
            fprint("----------")
            fprint("Setting training set " + args.training_set)

            # only when reusing archive
            if args.archive is None:
                archive = set_archive(args.training_set)
            else:
                archive = args.archive
            end = time.time()
            multi_time = str(np.round(end - start, 2))
            fprint("Training set loaded in " + multi_time + " s")
        #######################

        ## set emulator
        if rank == 0:
            fprint("----------")
            fprint("Setting emulator")
            start = time.time()

            _drop_sim = None
            if args.drop_sim & (args.data_label in archive.list_sim_cube):
                _drop_sim = args.data_label

            if args.emulator is None:
                emulator = set_emulator(
                    emulator_label=args.emulator_label,
                    archive=archive,
                    drop_sim=_drop_sim,
                )

                if "Nyx" in emulator.emulator_label:
                    emulator.list_sim_cube = archive.list_sim_cube
                    if "nyx_14" in emulator.list_sim_cube:
                        emulator.list_sim_cube.remove("nyx_14")
                else:
                    emulator.list_sim_cube = archive.list_sim_cube
            else:
                emulator = args.emulator

            multi_time = str(np.round(time.time() - start, 2))
            fprint("Emulator set in " + multi_time + " s")

            # distribute emulator to all ranks
            for irank in range(1, size):
                comm.send(emulator, dest=irank, tag=(irank + 1) * 7)
        else:
            # receive emulator from ranks 0
            emulator = comm.recv(source=0, tag=(rank + 1) * 7)

        #######################

        ## set P1D
        if rank == 0:
            fprint("----------")
            fprint("Setting P1D")
            start = time.time()

            # set fiducial cosmology
            if args.true_cosmo_label is not None:
                true_cosmo = set_cosmo(cosmo_label=args.true_cosmo_label)
            else:
                true_cosmo = None

            data = {"P1Ds": None, "extra_P1Ds": None}

            # set P1D
            data["P1Ds"] = set_P1D(
                args,
                archive=archive,
                true_cosmo=true_cosmo,
                emulator=emulator,
            )
            fprint(
                "Set " + str(len(data["P1Ds"].z)) + " P1Ds at z = ",
                data["P1Ds"].z,
            )

            # set hires P1D
            if args.data_label_hires is not None:
                data["extra_P1Ds"] = set_P1D_hires(
                    args,
                    archive=archive,
                    true_cosmo=true_cosmo,
                    emulator=emulator,
                )
                fprint(
                    "Set " + str(len(data["extra_P1Ds"].z)) + " P1Ds at z = ",
                    data["extra_P1Ds"].z,
                )
            # distribute data to all tasks
            for irank in range(1, size):
                comm.send(data, dest=irank, tag=(irank + 1) * 11)
        else:
            # get testing_data from task 0
            data = comm.recv(source=0, tag=(rank + 1) * 11)

        if rank == 0:
            multi_time = str(np.round(time.time() - start, 2))
            fprint("P1D set in " + multi_time + " s")

        #######################

        ## Checking data

        # check if data is blinded
        fprint("----------")
        fprint("Is the data blinded: ", data["P1Ds"].apply_blinding)
        if data["P1Ds"].apply_blinding:
            fprint("Type of blinding: ", data["P1Ds"].blinding)

        if rank == 0:
            # TBD save to file!
            if make_plots:
                data["P1Ds"].plot_p1d()
                if args.data_label_hires is not None:
                    data["extra_P1Ds"].plot_p1d()

                try:
                    data["P1Ds"].plot_igm()
                except:
                    print("Real data, no true IGM history")

        #######################

        # XXX one at a time here!

        ## set likelihood
        fprint("----------")
        fprint("Setting likelihood")

        like = set_like(
            data["P1Ds"],
            emulator,
            args,
            data_hires=data["extra_P1Ds"],
        )

        ## Validating likelihood

        if rank == 0:
            # TBD save to file!
            if make_plots:
                like.plot_p1d(residuals=False)
                like.plot_p1d(residuals=True)
                like.plot_igm()

        # print parameters
        for p in like.free_params:
            fprint(p.name, p.value, p.min_value, p.max_value)

        #######################

        # self.set_emcee_options(
        #     args.data_label,
        #     args.cov_label,
        #     args.n_igm,
        #     n_steps=args.n_steps,
        #     n_burn_in=args.n_burn_in,
        #     test=args.test,
        # )

        ## set fitter

        self.fitter = Fitter(
            like=like,
            rootdir=self.out_folder,
            nburnin=args.n_burn_in,
            nsteps=args.n_steps,
            parallel=args.parallel,
            explore=args.explore,
            fix_cosmology=args.fix_cosmo,
        )

        #######################

        if rank == 0:
            multi_time = str(np.round(time.time() - start_all, 2))
            fprint("Setting the sampler took " + multi_time + " s \n\n")

    def set_emcee_options(
        self,
        data_label,
        cov_label,
        n_igm,
        n_steps=0,
        n_burn_in=0,
        test=False,
    ):
        # set steps
        if test == True:
            self.n_steps = 10
        else:
            if n_steps != 0:
                self.n_steps = n_steps
            else:
                if data_label == "Chabanier2019":
                    self.n_steps = 2000
                else:
                    self.n_steps = 1250

        # set burn-in
        if test == True:
            self.n_burn_in = 0
        else:
            if n_burn_in != 0:
                self.n_burn_in = n_burn_in
            else:
                if data_label == "Chabanier2019":
                    self.n_burn_in = 2000
                else:
                    if cov_label == "Chabanier2019":
                        self.n_burn_in = 1500
                    elif cov_label == "QMLE_Ohio":
                        self.n_burn_in = 1500
                    else:
                        self.n_burn_in = 1500

    def run_minimizer(self):
        """
        Run the minimizer (only rank 0)
        """

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        if rank == 0:
            start = time.time()
            self.fprint("----------")
            self.fprint("Running minimizer")
            # start fit from initial values
            p0 = np.array(list(self.fitter.like.fid["fit_cube"].values()))
            self.fitter.run_minimizer(
                log_func_minimize=self.fitter.like.get_chi2, p0=p0
            )

            # save fit
            self.fitter.save_minimizer()

            # plot fit
            plotter = Plotter(
                self.fitter, save_directory=self.fitter.save_directory
            )
            plotter.plots_minimizer()

            # distribute best_fit to all tasks
            for irank in range(1, size):
                comm.send(
                    self.fitter.mle_cube, dest=irank, tag=(irank + 1) * 13
                )
        else:
            # get testing_data from task 0
            self.fitter.mle_cube = comm.recv(source=0, tag=(rank + 1) * 13)

    def run_sampler(self):
        """
        Run the sampler (after minimizer)
        """

        def func_for_sampler(p0):
            res = self.fitter.like.get_log_like(values=p0, return_blob=True)
            return res[0], *res[2]

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        if rank == 0:
            start = time.time()
            self.fprint("----------")
            self.fprint("Running sampler")

        # make sure all tasks start at the same time
        self.fitter.run_sampler(
            pini=self.fitter.mle_cube, log_func=func_for_sampler
        )

        if rank == 0:
            end = time.time()
            multi_time = str(np.round(end - start, 2))
            self.fprint("Sampler run in " + multi_time + " s")

            self.fprint("----------")
            self.fprint("Saving data")
            self.fitter.write_chain_to_file()

            # plot fit
            plotter = Plotter(
                self.fitter, save_directory=self.fitter.save_directory
            )
            plotter.plots_sampler()
