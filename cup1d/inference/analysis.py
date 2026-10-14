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
    """Build a rectangular grid spanning the MPG cosmology training set.

    Parameters
    ----------
    nelem : int
        Number of samples along each of the ``Delta2_star`` and ``n_star``
        axes.

    Returns
    -------
    xgrid, ygrid : ndarray
        Two arrays of shape ``(nelem, nelem)`` returned by
        :func:`numpy.meshgrid`.  Together they cover the extrema stored in
        LaCE's ``Australia20/mpg_emu_cosmo.npy`` metadata.

    Notes
    -----
    This helper depends on the installed LaCE repository data rather than an
    archive supplied to :class:`Analysis`.
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
        except:
            continue

        pars[ii, 0] = data_cosmo[key]["star_params"]["Delta2_star"]
        pars[ii, 1] = data_cosmo[key]["star_params"]["n_star"]

    x = np.linspace(pars[:, 0].min(), pars[:, 0].max(), nelem)
    y = np.linspace(pars[:, 1].min(), pars[:, 1].max(), nelem)
    xgrid, ygrid = np.meshgrid(x, y)

    return xgrid, ygrid


class Analysis(object):
    """Assemble the P1D data, theory, likelihood, and inference driver.

    The constructor is MPI-aware: rank zero builds default emulators and data
    and sends the resulting objects to worker ranks.  A caller may instead
    inject already constructed data, archive, or emulator objects.

    Attributes
    ----------
    args : cup1d.configuration.args.Args
        Resolved analysis configuration.
    data : dict
        P1D data sets indexed by their configured labels.
    emulator : object
        Emulator selected by ``args.emulator_label`` or supplied explicitly.
    theory : object
        Fiducial theory object evaluated by the likelihood.
    like : cup1d.likelihood.likelihood.Likelihood
        Likelihood used for minimization and sampling.  ``likelihood`` is a
        backward-compatible alias for this attribute.
    fitter : cup1d.inference.fitter.Fitter
        Object managing minimizer and sampler runs.
    """

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
        """Initialize an analysis from configuration and optional components.

        Parameters
        ----------
        args : cup1d.configuration.args.Args, optional
            Resolved configuration.  When omitted, construct the CM2026
            baseline configuration and apply ``system`` to it.
        data : dict, optional
            Pre-built P1D data sets keyed by every label in
            ``args.data_label``.  If omitted, construct them with
            :func:`cup1d.p1ds.factory.set_p1d`.
        archive : object, optional
            Simulation archive passed while constructing P1D data.  It is not
            used when ``data`` is supplied.
        emulator : object, optional
            Pre-built emulator.  If omitted, resolve
            ``args.emulator_label`` through the emulator factory.
        out_folder : str or path-like, optional
            Directory used by :class:`Fitter` for result products.  Defaults
            to the output folder in ``args``.
        system : str, default='local'
            Execution-system label applied only when ``args`` is omitted.
        create_output : bool, default=True
            Whether the fitter may create its output directory.  Result
            restoration disables this to avoid modifying a saved run.
        """

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
            covariance_method=self.args.covariance_method,
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

        Parameters
        ----------
        filename : str or path-like
            NumPy result file produced by :class:`Fitter` with format version
            1.  Its recorded YAML path determines the reconstructed
            configuration.
        **analysis_options
            Keyword arguments forwarded to :class:`Analysis`, except ``args``
            and ``create_output``.  Those are controlled by the saved result.

        Returns
        -------
        Analysis
            Rebuilt analysis whose fitter has restored the saved state and
            whose ``results_path`` identifies ``filename``.

        Raises
        ------
        ValueError
            If the file does not contain a supported minimizer or sampler
            result.
        FileNotFoundError
            If the YAML configuration recorded in the result is unavailable.
        TypeError
            If ``analysis_options`` attempts to override the reconstructed
            configuration or output-creation policy.
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
        """Choose legacy emcee step and burn-in counts from data labels.

        Parameters
        ----------
        data_label : str
            Label selecting the default production step count.
        cov_label : str
            Covariance label selecting the default burn-in count.
        n_igm : int
            Retained for compatibility with older callers.  It does not alter
            the counts chosen by this implementation.
        n_steps, n_burn_in : int, default=0
            Explicit positive counts.  A value of zero requests the
            label-dependent defaults.
        test : bool, default=False
            If true, use ten production steps and no burn-in.

        Notes
        -----
        The values are assigned to ``self.n_steps`` and ``self.n_burn_in``;
        they do not modify the already-created fitter configuration.
        """
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
        pso_type="global",
    ):
        """Run a minimizer on rank zero and broadcast its best-fit cube.

        Parameters
        ----------
        p0 : array-like
            Initial point in the fitter's sampling-coordinate convention.
        make_plots : bool, default=False
            Create minimizer diagnostic plots after a successful root-rank
            run.
        mask_pars : bool, default=False
            Request the Nelder--Mead parameter-masking behavior implemented
            by :meth:`Fitter.run_minimizer`.
        save_chains : bool, default=False
            Save sampler-style results even when the minimizer did not attach
            a chain.
        zmask : array-like, optional
            Redshift mask forwarded to the selected minimizer and plotter.
        restart : bool, default=False
            Forward restart handling to the selected fitter method.
        type_minimizer : {'NM', 'DA', 'PSO'}, default='NM'
            Select Nelder--Mead, differential annealing, or particle swarm
            optimization.
        estimate_errors : bool, default=False
            Estimate parameter errors after minimization.
        hessian_step : float, default=1e-4
            Finite-difference displacement used for error estimation.
        error_method : str, default='finite_difference'
            Error-estimation method understood by :class:`Fitter`.
        vectorize : bool, default=True
            Enable vectorized likelihood evaluation for particle swarm runs.
        pso_type : str, default='global'
            Particle-swarm variant forwarded when ``type_minimizer`` is
            ``'PSO'``.

        Raises
        ------
        ValueError
            If ``type_minimizer`` is not ``'NM'``, ``'DA'``, or ``'PSO'``.

        Notes
        -----
        The method returns ``None``.  The root rank saves results and every
        rank receives the resulting ``fitter.mle_cube``.
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
                    p0=p0, zmask=zmask, vectorize=vectorize, pso_type=pso_type,
                    restart=restart,
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
        """Run the configured sampler, optionally starting at a supplied point.

        Parameters
        ----------
        pini : array-like, optional
            Initial sampling-coordinate point.  Defaults to the current
            ``fitter.mle_cube``, normally produced by :meth:`run_minimizer`.
        make_plots : bool, default=False
            Create sampler diagnostic plots on rank zero after saving results.
        zmask : array-like, optional
            Redshift mask forwarded to the fitter and optional plotter.
        vectorize : bool, optional
            Override the fitter's vectorized likelihood-evaluation setting.

        Notes
        -----
        All MPI ranks participate in sampling; only rank zero saves result
        products and creates plots.  The method returns ``None``.
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
        """Save the best-fit non-cosmological parameters by redshift node.

        Parameters
        ----------
        fname : str or path-like
            Target filename for :func:`numpy.save`.  The saved dictionary is
            keyed by parameter family and each value has sorted ``'z'`` and
            ``'val'`` arrays.

        Raises
        ------
        ValueError
            If a free parameter cannot be associated with an IGM,
            contaminant, or systematic fiducial-node definition.

        Notes
        -----
        ``As`` and ``ns`` are deliberately omitted because this helper is for
        global IGM, contaminant, and systematic initial conditions.
        """
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
            out_dict[pname]["val"].append(self.fitter.get_mle_value(name))

        for key in out_dict:
            out_dict[key]["z"] = np.array(out_dict[key]["z"])
            ind = np.argsort(out_dict[key]["z"])
            out_dict[key]["z"] = out_dict[key]["z"][ind]
            out_dict[key]["val"] = np.array(out_dict[key]["val"])[ind]

            print(key, out_dict[key]["val"])

        np.save(fname, out_dict)
