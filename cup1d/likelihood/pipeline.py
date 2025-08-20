import os
import time
import numpy as np
from mpi4py import MPI

from cup1d.pipeline.set_theory import set_theory
from cup1d.pipeline.set_emulator import set_emulator
from cup1d.pipeline.set_like_params import set_free_like_parameters
from cup1d.pipeline.set_p1d import set_P1D

from cup1d.likelihood.likelihood import Likelihood
from cup1d.likelihood.fitter import Fitter
from cup1d.likelihood.plotter import Plotter
from cup1d.utils.utils import get_path_repo
from cup1d.utils.utils import create_print_function
from cup1d.utils.utils import split_string


def get_grid_large(nelem):
    """Need to be moved somewhere else"""
    fname = os.path.join(
        get_path_repo("lace"),
        "data",
        "sim_suites",
        "Australia20",
        "mpg_emu_cosmo.npy",
    )

    data_cosmo = np.load(fname, allow_pickle=True).item()

    pars = np.zeros((30, 2))
    for ii, key in enumerate(data_cosmo):
        try:
            int(key[-1])
        except:
            continue

        pars[ii, 0] = data_cosmo[key]["star_params"]["Delta2_star"]
        pars[ii, 1] = data_cosmo[key]["star_params"]["n_star"]

    x = np.linspace(pars[:, 0].min(), pars[:, 0].max(), nelem)
    y = np.linspace(pars[:, 1].min(), pars[:, 1].max(), nelem)
    xgrid, ygrid = np.meshgrid(x, y)

    return xgrid, ygrid


class Pipeline(object):
    """Full pipeline for extracting cosmology from P1D using sampler"""

    def __init__(self, args, make_plots=False, out_folder=None):
        """Set pipeline"""

        self.out_folder = out_folder

        ## MPI stuff
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        # create print function (only for rank 0)
        fprint = create_print_function(verbose=args.verbose)
        self.fprint = fprint
        self.explore = args.explore

        if rank == 0:
            self.fprint("----------")
            self.fprint("Setting emulator")
            emulator = set_emulator(
                emulator_label=args.emulator_label,
                drop_sim=args.drop_sim,
                training_set=args.training_set,
            )
            self.fprint("Done setting emulator")
            self.fprint("----------")
            # distribute emulator to all ranks
            for irank in range(1, size):
                comm.send(emulator, dest=irank, tag=(irank + 1) * 3)
        else:
            # receive emulator from ranks 0
            emulator = comm.recv(source=0, tag=(rank + 1) * 3)

        free_parameters = set_free_like_parameters(
            args, emulator_label=emulator.emulator_label
        )

        # Set true theory to create mocks P1D measurements.
        # Ignored if setting P1D measurements from observations
        true_theory = set_theory(
            args, emulator, free_parameters, fid_or_true="true", use_hull=False
        )

        if rank == 0:
            data = {"P1Ds": None, "extra_P1Ds": None}
            fprint("----------")
            fprint("Setting P1Ds")
            data["P1Ds"] = set_P1D(args, theory=true_theory)

            if args.data_label_hires is not None:
                data["extra_P1Ds"] = set_P1D(args, theory=true_theory)

            fprint("Done setting P1Ds")
            fprint("----------")
            # distribute data to all tasks
            for irank in range(1, size):
                comm.send(data, dest=irank, tag=(irank + 1) * 5)
        else:
            # get testing_data from task 0
            data = comm.recv(source=0, tag=(rank + 1) * 5)

        if args.data_label_hires is not None:
            zs = np.concatenate([data["P1Ds"].z, data_hires["extra_P1Ds"].z])
        else:
            zs = data["P1Ds"].z

        theory = set_theory(
            args,
            emulator,
            free_parameters,
            fid_or_true="fid",
            use_hull=False,
            zs=zs,
        )

        like = Likelihood(
            data["P1Ds"],
            theory,
            extra_data=data["extra_P1Ds"],
            free_param_names=free_parameters,
            cov_factor=args.cov_factor,
            emu_cov_factor=args.emu_cov_factor,
            emu_cov_type=args.emu_cov_type,
            args=args,
        )

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

    def run_minimizer(
        self,
        p0,
        make_plots=True,
        mask_pars=False,
        save_chains=False,
        zmask=None,
        restart=False,
        type_minimizer="NM",
    ):
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

            if type_minimizer == "NM":
                self.fitter.run_minimizer(
                    log_func_minimize=self.fitter.like.minus_log_prob,
                    p0=p0,
                    zmask=zmask,
                    mask_pars=mask_pars,
                    restart=restart,
                )
            elif type_minimizer == "DA":
                self.fitter.run_minimizer_da(
                    log_func_minimize=self.fitter.like.minus_log_prob,
                    p0=p0,
                    zmask=zmask,
                    restart=restart,
                )
            else:
                raise ValueError("type_minimizer must be 'NM' or 'DA'")

            # save fit
            self.fitter.save_fitter(save_chains=save_chains)

            if make_plots:
                # plot fit
                self.plotter = Plotter(
                    self.fitter,
                    save_directory=self.fitter.save_directory,
                    zmask=zmask,
                )
                self.plotter.plots_minimizer()

            # distribute best_fit to all tasks
            for irank in range(1, size):
                comm.send(
                    self.fitter.mle_cube, dest=irank, tag=(irank + 1) * 13
                )
        else:
            # get testing_data from task 0
            self.fitter.mle_cube = comm.recv(source=0, tag=(rank + 1) * 13)

    def run_sampler(self, pini=None, make_plots=True, zmask=None):
        """
        Run the sampler (after minimizer)
        """

        # def func_for_sampler(p0):
        #     res = self.fitter.like.get_log_like(values=p0, return_blob=True)
        #     return res[0], *res[2]

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        if rank == 0:
            start = time.time()
            self.fprint("----------")
            self.fprint("Running sampler")

        # make sure all tasks start at the same time
        if pini is None:
            pini = self.fitter.mle_cube

        self.fitter.run_sampler(pini=pini, zmask=zmask)

        if rank == 0:
            end = time.time()
            multi_time = str(np.round(end - start, 2))
            self.fprint("Sampler run in " + multi_time + " s")

            self.fprint("----------")
            self.fprint("Saving data")
            self.fitter.save_fitter(save_chains=True)

            # plot fit
            if make_plots:
                self.plotter = Plotter(
                    self.fitter,
                    save_directory=self.fitter.save_directory,
                    zmask=zmask,
                )
                self.plotter.plots_sampler()

    def run_profile(
        self, args, sigma_cosmo, nelem=10, nsig=4, type_minimizer="NM"
    ):
        """
        Run profile likelihood

        First minimize with varying cosmology, then optimize while fixing the
        cosmology for different fiducial values
        """

        # if grid_type == "large":
        # xran, yran = get_grid_large(nelem)

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        dim = len(sigma_cosmo)
        x = np.linspace(-nsig, nsig, nelem)
        if dim == 1:
            if "Delta2_star" in sigma_cosmo:
                xgrid = sigma_cosmo["Delta2_star"] * x
            else:
                xgrid = x[:] * 0
            if "n_star" in sigma_cosmo:
                ygrid = sigma_cosmo["n_star"] * x
            else:
                ygrid = x[:] * 0
        elif dim == 2:
            xgrid, ygrid = np.meshgrid(x, x)
            xgrid = xgrid.reshape(-1) * sigma_cosmo["Delta2_star"]
            ygrid = ygrid.reshape(-1) * sigma_cosmo["n_star"]
        else:
            raise ValueError("dim must be 1 or 2")

        ind_ranks = np.array_split(np.arange(len(xgrid)), size)
        if rank == 0:
            print("IDs to each rank:", ind_ranks)

        if rank == 0:
            # read ini data and redistribute (from scripts/data/profile_like_cen.py)
            file_out = os.path.join(
                os.path.dirname(os.path.dirname(self.fitter.save_directory)),
                "best_dircosmo.npy",
            )
            print("Loading IC from", file_out)
            print("")
            out_dict = np.load(file_out, allow_pickle=True).item()
            # pini = out_dict["mle_cube"][2:]
            mle_cosmo_cen = out_dict["mle_cosmo_cen"]

            # distribute emulator to all ranks
            for irank in range(1, size):
                # comm.send(pini, dest=irank, tag=(irank + 1) * 3)
                comm.send(mle_cosmo_cen, dest=irank, tag=(irank + 1) * 5)
        else:
            # receive emulator from ranks 0
            # pini = comm.recv(source=0, tag=(rank + 1) * 3)
            mle_cosmo_cen = comm.recv(source=0, tag=(rank + 1) * 5)

        pini = self.fitter.like.sampling_point_from_parameters().copy()

        if rank == 0:
            start = time.time()
            self.fprint("----------")
            self.fprint("Running like profile")

        for irank in ind_ranks[rank]:
            if rank == 0:
                self.fprint(irank, max(ind_ranks[rank]))
            shift_cosmo = {
                "Delta2_star": xgrid[irank],
                "n_star": ygrid[irank],
            }
            self.fitter.run_profile(
                irank,
                mle_cosmo_cen,
                shift_cosmo,
                pini,
                type_minimizer=type_minimizer,
            )

        if rank == 0:
            end = time.time()
            multi_time = str(np.round(end - start, 2))
            self.fprint("Profile run in " + multi_time + " s")
            self.fprint("----------")

    def save_global_ic(self, fname):
        out_dict = {}
        vals = np.array(list(self.fitter.mle.values()))
        for jj, p in enumerate(self.fitter.like.free_params):
            pname, iistr = split_string(p.name)
            ii = int(iistr)
            try:
                znode = self.fitter.like.args.fid_igm[pname + "_znodes"][ii]
            except:
                znode = self.fitter.like.args.fid_cont[pname + "_znodes"][ii]
            # print(pname, znode, vals[jj])

            if pname not in out_dict:
                out_dict[pname] = {"z": [], "val": []}
            out_dict[pname]["z"].append(znode)
            out_dict[pname]["val"].append(vals[jj])

        for key in out_dict:
            out_dict[key]["z"] = np.array(out_dict[key]["z"])
            ind = np.argsort(out_dict[key]["z"])
            out_dict[key]["z"] = out_dict[key]["z"][ind]
            out_dict[key]["val"] = np.array(out_dict[key]["val"])[ind]

            print(key, out_dict[key]["val"])

        np.save(fname, out_dict)
