"""High-level MPI pipeline for fitting P1D likelihoods."""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np
from mpi4py import MPI

from cup1d.likelihood.cosmologies import set_cosmo
from cup1d.likelihood.fitter import Fitter
from cup1d.likelihood.input_pipeline import Args
from cup1d.likelihood.likelihood import Likelihood
from cup1d.likelihood.plotter import Plotter
from cup1d.pipeline.set_archive import set_archive
from cup1d.pipeline.set_emulator import set_emulator
from cup1d.pipeline.set_like_params import set_free_like_parameters
from cup1d.pipeline.set_p1d import set_P1D
from cup1d.pipeline.set_theory import set_theory
from cup1d.utils.utils import create_print_function, get_path_repo, split_string

__all__ = [
    "set_like",
    "set_archive",
    "set_cosmo",
    "set_emulator",
    "set_free_like_parameters",
    "set_P1D",
    "set_theory",
    "Pipeline",
]


def set_like(
    data: Any,
    emulator: Any,
    args: Args,
    data_hires: Any | None = None,
) -> Likelihood:
    """Set the likelihood object for a given data and emulator.

    This function sets up the free parameters, the theory model, and
    initializes the Likelihood object.

    Parameters
    ----------
    data : Any
        The primary P1D data to be fitted.
    emulator : Any
        The emulator used to provide fast model predictions.
    args : Args
        Configuration object containing analysis settings.
    data_hires : Any, optional
        Additional high-redshift or high-resolution data. Default is None.

    Returns
    -------
    Likelihood
        The initialized likelihood object ready for fitting.
    """
    free_parameters = set_free_like_parameters(
        args, emulator_label=emulator.emulator_label
    )

    if data_hires is not None:
        zs = np.concatenate([data.z, data_hires.z])
    else:
        zs = data.z

    theory = set_theory(
        args,
        emulator,
        free_parameters,
        fid_or_true="fid",
        use_hull=False,
        zs=zs,
    )

    like = Likelihood(
        data,
        theory,
        extra_data=data_hires,
        free_param_names=free_parameters,
        cov_factor=args.cov_factor,
        emu_cov_type=args.emu_cov_type,
        args=args,
    )
    return like


def get_grid_large(nelem: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a regular grid spanning the large Australia20 emulator domain.

    Parameters
    ----------
    nelem : int
        Number of elements in each dimension of the grid.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        X and Y grid arrays.
    """
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
        except Exception:
            continue

        pars[ii, 0] = data_cosmo[key]["star_params"]["Delta2_star"]
        pars[ii, 1] = data_cosmo[key]["star_params"]["n_star"]

    x = np.linspace(pars[:, 0].min(), pars[:, 0].max(), nelem)
    y = np.linspace(pars[:, 1].min(), pars[:, 1].max(), nelem)
    xgrid, ygrid = np.meshgrid(x, y)

    return xgrid, ygrid


class Pipeline:
    """Coordinate emulator setup, data loading, fitting, and plotting.

    Parameters
    ----------
    args : Args, optional
        Pipeline configuration. If omitted, the CM2026 defaults are used.
    make_plots : bool, optional
        Kept for API compatibility; plotting is controlled by run methods.
    out_folder : str, optional
        Output folder overriding ``args.out_folder``.
    archive : Any, optional
        Optional preloaded simulation archive.
    system : str, optional
        System label used when constructing default arguments. Default is "local".

    Attributes
    ----------
    out_folder : str
        Output folder for results.
    fprint : Callable
        Print function for rank 0.
    fitter : Fitter
        MCMC sampler wrapper.
    plotter : Plotter
        Plotting utility.
    """

    def __init__(
        self,
        args: Args | None = None,
        make_plots: bool = False,
        out_folder: str | None = None,
        archive: Any | None = None,
        system: str = "local",
    ):
        """Initialize the full likelihood pipeline."""

        if args is None:
            # set default args to Chaves-Montero+26 analysis
            args = Args(pre_defined="CM2026", system=system)

        if out_folder is None:
            self.out_folder = args.out_folder
        else:
            self.out_folder = out_folder

        ## MPI stuff
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        # create print function (only for rank 0)
        fprint = create_print_function(verbose=args.verbose)
        self.fprint = fprint

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
            data["P1Ds"] = set_P1D(args, theory=true_theory, archive=archive)

            if args.data_label_hires is not None:
                data["extra_P1Ds"] = set_P1D(
                    args, theory=true_theory, archive=archive
                )

            fprint("Done setting P1Ds")
            fprint("----------")
            # distribute data to all tasks
            for irank in range(1, size):
                comm.send(data, dest=irank, tag=(irank + 1) * 5)
        else:
            # get testing_data from task 0
            data = comm.recv(source=0, tag=(rank + 1) * 5)

        if args.data_label_hires is not None:
            zs = np.concatenate([data["P1Ds"].z, data["extra_P1Ds"].z])
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
            emu_cov_type=args.emu_cov_type,
            args=args,
        )

        self.fitter = Fitter(
            like=like,
            rootdir=self.out_folder,
            nwalkers=args.mcmc["n_walkers"],
            nburn=args.mcmc["n_burn_in"],
            nsteps=args.mcmc["n_steps"],
            thin=args.mcmc["thin"],
            parallel=args.mcmc["parallel"],
            explore=args.mcmc["explore"],
            fix_cosmology=args.fix_cosmo,
        )

        #######################

    def set_emcee_options(
        self,
        data_label: str,
        cov_label: str,
        n_igm: int,
        n_steps: int = 0,
        n_burn_in: int = 0,
        test: bool = False,
    ) -> None:
        """Set default emcee step counts for selected data/covariance labels.

        Parameters
        ----------
        data_label : str
            Data label.
        cov_label : str
            Covariance label.
        n_igm : int
            Number of IGM parameters.
        n_steps : int, optional
            Number of steps. Default is 0.
        n_burn_in : int, optional
            Number of burn-in steps. Default is 0.
        test : bool, optional
            Whether this is a test run. Default is False.
        """
        # set steps
        if test:
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
        if test:
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
        p0: np.ndarray | None = None,
        make_plots: bool = False,
        mask_pars: bool = False,
        save_chains: bool = False,
        zmask: np.ndarray | None = None,
        restart: bool = False,
        type_minimizer: str = "NM",
    ) -> None:
        """Run the selected minimizer on rank 0 and broadcast the best fit.

        Parameters
        ----------
        p0 : np.ndarray, optional
            Initial parameter values.
        make_plots : bool, optional
            Whether to make plots. Default is False.
        mask_pars : bool, optional
            Whether to mask parameters. Default is False.
        save_chains : bool, optional
            Whether to save chains. Default is False.
        zmask : np.ndarray, optional
            Redshift mask.
        restart : bool, optional
            Whether to restart. Default is False.
        type_minimizer : str, optional
            Type of minimizer ('NM' or 'DA'). Default is 'NM'.
        """

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        if rank == 0:
            time.time()
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

    def run_sampler(
        self,
        pini: np.ndarray | None = None,
        make_plots: bool = False,
        zmask: np.ndarray | None = None,
    ) -> None:
        """Run the MCMC sampler after a minimizer pass.

        Parameters
        ----------
        pini : np.ndarray, optional
            Initial parameter values.
        make_plots : bool, optional
            Whether to make plots. Default is False.
        zmask : np.ndarray, optional
            Redshift mask.
        """

        # def func_for_sampler(p0):
        #     res = self.fitter.like.get_log_like(values=p0, return_blob=True)
        #     return res[0], *res[2]

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        comm.Get_size()

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
        self,
        sigma_cosmo: dict[str, float],
        mle_cosmo_cen: dict[str, float] | None = None,
        nelem: int = 10,
        nsig: int = 10,
        type_minimizer: str = "NM",
        folder_ic: str | None = None,
    ) -> None:
        """Run a profile likelihood scan.

        First minimize with varying cosmology, then optimize while fixing the
        cosmology for different fiducial values.

        Parameters
        ----------
        sigma_cosmo : dict[str, float]
            Cosmological parameter uncertainties.
        mle_cosmo_cen : dict[str, float], optional
            Central cosmological parameter values.
        nelem : int, optional
            Number of elements in the grid. Default is 10.
        nsig : int, optional
            Number of sigma to scan. Default is 10.
        type_minimizer : str, optional
            Type of minimizer. Default is 'NM'.
        folder_ic : str, optional
            Folder for initial conditions.
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

        if mle_cosmo_cen is None:
            if rank == 0:
                # read ini data and redistribute (from scripts/data/profile_like_cen.py)
                if folder_ic is None:
                    folder_ic = os.path.dirname(
                        os.path.dirname(self.fitter.save_directory)
                    )
                file_out = os.path.join(folder_ic, "best_dircosmo.npy")
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

    def save_global_ic(self, fname: str) -> None:
        """Save best-fit redshift-dependent nuisance values for later reuse.

        Parameters
        ----------
        fname : str
            Filename to save the initial conditions.
        """
        out_dict = {}
        vals = np.array(list(self.fitter.mle.values()))
        for jj, p in enumerate(self.fitter.like.free_params):
            if (p.name == "As") or (p.name == "ns"):
                continue
            pname, iistr = split_string(p.name)
            ii = int(iistr)
            if pname in self.fitter.like.args.fid_igm:
                znode = self.fitter.like.args.fid_igm[pname + "_znodes"][ii]
            elif pname in self.fitter.like.args.fid_cont:
                znode = self.fitter.like.args.fid_cont[pname + "_znodes"][ii]
            elif pname in self.fitter.like.args.fid_syst:
                znode = self.fitter.like.args.fid_syst[pname + "_znodes"][ii]
            else:
                raise ValueError("pname not found:", pname)
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
