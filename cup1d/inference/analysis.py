import os
import time
from pathlib import Path
import numpy as np
from mpi4py import MPI

from cup1d.theory.factory import set_theory
from cup1d.emulator.factory import set_emulator
from cup1d.likelihood.parameters import set_free_likelihood_parameters
from cup1d.p1ds.factory import is_synthetic_data_label, set_p1d
from cup1d.configuration.args import Args
from cup1d.likelihood.likelihood import Likelihood
from cup1d.inference.fitter import Fitter
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


class Analysis(object):
    """Full analysis for extracting cosmology from P1D using sampler"""

    def __init__(
        self,
        args=None,
        data=None,
        archive=None,
        emulator=None,
        out_folder=None,
        system="local",
        create_output=True,
    ):
        """Set analysis."""

        if args is None:
            # set default args to Chaves-Montero+26 analysis
            self.args = Args.from_baseline()
            self.args.system = system
        else:
            self.args = args

        if out_folder is None:
            self.out_folder = self.args.out_folder
        else:
            self.out_folder = out_folder

        ## MPI stuff
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        # create print function (only for rank 0)
        fprint = create_print_function(verbose=self.args.verbose)
        self.fprint = fprint

        if emulator is None:
            if rank == 0:
                self.fprint("----------")
                self.fprint("Setting emulator")
                self.emulator = set_emulator(
                    emulator_label=self.args.emulator_label,
                )
                self.fprint("Done setting emulator")
                self.fprint("----------")
                # distribute emulator to all ranks
                for irank in range(1, size):
                    comm.send(self.emulator, dest=irank, tag=(irank + 1) * 3)
            else:
                # receive emulator from ranks 0
                self.emulator = comm.recv(source=0, tag=(rank + 1) * 3)
        else:
            self.emulator = emulator

        free_parameters = set_free_likelihood_parameters(
            self.args, emulator_label=self.args.emulator_label
        )

        # A true theory is only needed to construct synthetic P1D data.
        needs_true_theory = data is None and any(
            is_synthetic_data_label(label) for label in self.args.data_label
        )
        if needs_true_theory:
            true_theory = set_theory(
                self.args,
                self.emulator,
                free_parameters,
                fid_or_true="true",
                use_hull=False,
            )
        else:
            true_theory = None

        if data is None:
            if rank == 0:
                self.data = {}
                fprint("----------")
                fprint("Setting P1Ds")
                for data_label in self.args.data_label:
                    fprint("Setting P1D for", data_label)
                    self.data[data_label] = set_p1d(
                        self.args, data_label, theory=true_theory, archive=archive
                    )

                fprint("Done setting P1Ds")
                fprint("----------")
                # distribute data to all tasks
                for irank in range(1, size):
                    comm.send(self.data, dest=irank, tag=(irank + 1) * 5)
            else:
                # get testing_data from task 0
                self.data = comm.recv(source=0, tag=(rank + 1) * 5)
        else:
            self.data = data

        zs = []
        for data_label in self.args.data_label:
            zs.append(self.data[data_label].z)
        zs = np.unique(np.concatenate(zs))

        self.theory = set_theory(
            self.args,
            self.emulator,
            free_parameters,
            fid_or_true="fid",
            use_hull=False,
            zs=zs,
        )

        self.like = Likelihood(
            self.data,
            self.theory,
            free_param_names=free_parameters,
            cov_factor=self.args.cov_factor,
            emu_cov_type=self.args.emu_cov_type,
            args=self.args,
        )
        # Backward-compatible descriptive alias. Public analysis code should
        # use ``analysis.like`` rather than reaching through the fitter.
        self.likelihood = self.like

        self.fitter = Fitter(
            like=self.like,
            rootdir=self.out_folder,
            nwalkers=self.args.mcmc["n_walkers"],
            nburn=self.args.mcmc["n_burn_in"],
            nsteps=self.args.mcmc["n_steps"],
            thin=self.args.mcmc["thin"],
            parallel=self.args.mcmc["parallel"],
            explore=self.args.mcmc["explore"],
            fix_cosmology=self.args.fix_cosmo,
            random_seed=self.args.mcmc["seed"],
            verbose=self.args.verbose,
            create_output=create_output,
        )

        #######################

    @classmethod
    def from_results(cls, filename, **analysis_options):
        """Reconstruct an analysis and restore a saved fit or sampler run.

        The likelihood is rebuilt from the YAML recorded in the result file.
        Loading does not create a new output directory.
        """

        result_path = Path(filename).expanduser().resolve()
        payload = np.load(result_path, allow_pickle=True).item()
        if not isinstance(payload, dict) or payload.get("format_version") != 1:
            raise ValueError(f"Unsupported result file: {result_path}")
        result_type = payload.get("result_type")
        if result_type not in {"minimizer", "sampler"}:
            raise ValueError(f"Unknown result type: {result_type!r}")

        config_path = Path(payload["config_path"]).expanduser()
        if not config_path.exists():
            raise FileNotFoundError(
                f"YAML configuration recorded by the result does not exist: "
                f"{config_path}"
            )
        synthetic = bool(payload.get("synthetic", False))
        if payload.get("config_loader") == "variation":
            args = Args.from_variation(
                config_path, verbose=False, synthetic=synthetic
            )
        else:
            args = Args.from_yaml(
                config_path, verbose=False, synthetic=synthetic
            )
        if "args" in analysis_options or "create_output" in analysis_options:
            raise TypeError(
                "from_results reconstructs args from YAML and controls "
                "create_output"
            )
        analysis = cls(
            args=args, create_output=False, **analysis_options
        )
        analysis.fitter.restore_results(payload, result_path)
        analysis.results_path = str(result_path)
        return analysis

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
        make_plots=False,
        mask_pars=False,
        save_chains=False,
        zmask=None,
        restart=False,
        type_minimizer="NM",
        estimate_errors=False,
        hessian_step=1.0e-4,
        error_method="finite_difference",
        vectorize=True,
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
                    log_func_minimize=self.fitter.minus_log_prob,
                    p0=p0,
                    zmask=zmask,
                    mask_pars=mask_pars,
                    restart=restart,
                    estimate_errors=estimate_errors,
                    hessian_step=hessian_step,
                    error_method=error_method,
                )
            elif type_minimizer == "PSO":
                self.fitter.run_minimizer_pso(
                    p0=p0, zmask=zmask, vectorize=vectorize,
                    estimate_errors=estimate_errors, hessian_step=hessian_step,
                    error_method=error_method,
                )
            elif type_minimizer == "DA":
                self.fitter.run_minimizer_da(
                    log_func_minimize=self.fitter.minus_log_prob,
                    p0=p0,
                    zmask=zmask,
                    restart=restart,
                    estimate_errors=estimate_errors,
                    hessian_step=hessian_step,
                    error_method=error_method,
                )
            else:
                raise ValueError("type_minimizer must be 'NM', 'DA', or 'PSO'")

            # save fit
            if save_chains or hasattr(self.fitter, "chain"):
                self.fitter.save_sampler_results()
            else:
                self.fitter.save_minimizer_results()

            if make_plots:
                from cup1d.postprocessing.plotter import Plotter

                # plot fit
                self.plotter = Plotter(
                    self.fitter,
                    save_directory=self.fitter.save_directory,
                    zmask=zmask,
                )
                self.plotter.plots_minimizer()

            # distribute best_fit to all tasks
            for irank in range(1, size):
                comm.send(self.fitter.mle_cube, dest=irank, tag=(irank + 1) * 13)
        else:
            # get testing_data from task 0
            self.fitter.mle_cube = comm.recv(source=0, tag=(rank + 1) * 13)

    def run_sampler(
        self, pini=None, make_plots=False, zmask=None, vectorize=None
    ):
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

        self.fitter.run_sampler(pini=pini, zmask=zmask, vectorize=vectorize)

        if rank == 0:
            end = time.time()
            multi_time = str(np.round(end - start, 2))
            self.fprint("Sampler run in " + multi_time + " s")

            self.fprint("----------")
            self.fprint("Saving data")
            self.fitter.save_sampler_results()

            # plot fit
            if make_plots:
                from cup1d.postprocessing.plotter import Plotter

                self.plotter = Plotter(
                    self.fitter,
                    save_directory=self.fitter.save_directory,
                    zmask=zmask,
                )
                self.plotter.plots_sampler()

    def save_global_ic(self, fname):
        out_dict = {}
        for name in self.fitter.like.free_params:
            if name in ["As", "ns"]:
                continue
            pname, iistr = split_string(name)
            ii = int(iistr)
            if pname in self.fitter.like.args.fid_igm:
                znode = self.fitter.like.args.fid_igm[pname + "_znodes"][ii]
            elif pname in self.fitter.like.args.fid_cont:
                znode = self.fitter.like.args.fid_cont[pname + "_znodes"][ii]
            elif pname in self.fitter.like.args.fid_syst:
                znode = self.fitter.like.args.fid_syst[pname + "_znodes"][ii]
            else:
                raise ValueError("pname not found:", pname)
            # print(pname, znode, self.fitter.mle[name])

            if pname not in out_dict:
                out_dict[pname] = {"z": [], "val": []}
            out_dict[pname]["z"].append(znode)
            out_dict[pname]["val"].append(self.fitter.mle[name])

        for key in out_dict:
            out_dict[key]["z"] = np.array(out_dict[key]["z"])
            ind = np.argsort(out_dict[key]["z"])
            out_dict[key]["z"] = out_dict[key]["z"][ind]
            out_dict[key]["val"] = np.array(out_dict[key]["val"])[ind]

            print(key, out_dict[key]["val"])

        np.save(fname, out_dict)
