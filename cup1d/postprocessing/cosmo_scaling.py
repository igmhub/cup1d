"""Residual cosmological dependence after matching compressed star parameters.

P1D is a Gaussian-damped projection of linear CDM+baryon power, not a flux
emulator prediction. ForestFlow is required only for computing the projection.
"""

from copy import deepcopy
from pathlib import Path

import numpy as np
from lace.cosmo.cosmology import Cosmology
from lace.cosmo.rescale_cosmology import RescaledCosmology

FIDUCIAL_PARAMETERS = dict(
    H0=67.66,
    mnu=0.0,
    omch2=0.119,
    ombh2=0.0224,
    omk=0.0,
    As=2.105e-9,
    ns=0.9665,
    nrun=0.0,
    pivot_scalar=0.05,
    w=-1.0,
)
CASES = ("unscaled", "match_amplitude_slope", "match_amplitude_slope_running")
TITLES = (
    "No scaling",
    r"Scaling $\Delta^2_\star$, $n_\star$",
    r"Scaling $\Delta^2_\star$, $n_\star$, $\alpha_\star$",
)


def match_star_parameters(background, target, z_star, k_star_kms, match_running=False):
    """Rescale primordial parameters to match compressed linear-power values.

    Parameters
    ----------
    background : lace.cosmo.cosmology.Cosmology
        Cosmology to rescale without changing its background parameters.
    target : mapping
        Target ``Delta2_star``, ``n_star``, and, when requested,
        ``alpha_star`` values.
    z_star : float
        Redshift of the compressed linear-power pivot.
    k_star_kms : float
        Pivot wavenumber in ``s / km``.
    match_running : bool, default: False
        Also match ``alpha_star`` by changing primordial running.

    Returns
    -------
    lace.cosmo.rescale_cosmology.RescaledCosmology
        Cosmology with adjusted ``As``, ``ns``, and optionally ``nrun``.

    Raises
    ------
    AssertionError
        If the rescaled spectrum does not reproduce the requested star
        parameters to the stated numerical tolerance.
    """
    current = background.get_linP_kms_params(z_star, k_star_kms)
    primordial = background.get_primordial_params()
    log_pivot = np.log(
        k_star_kms * background.get_dkms_dMpc(z_star)
        / primordial["pivot_scalar"]
    )
    dnrun = target["alpha_star"] - current["alpha_star"] if match_running else 0.0
    dns = target["n_star"] - current["n_star"] - dnrun * log_pivot
    dlog_as = (
        np.log(target["Delta2_star"] / current["Delta2_star"])
        - dns * log_pivot
        - 0.5 * dnrun * log_pivot**2
    )
    result = RescaledCosmology(
        background,
        dict(
            As=primordial["As"] * np.exp(dlog_as),
            ns=primordial["ns"] + dns,
            nrun=primordial["nrun"] + dnrun,
        ),
    )
    fitted = result.get_linP_kms_params(z_star, k_star_kms)
    keys = (
        ("Delta2_star", "n_star", "alpha_star")
        if match_running
        else ("Delta2_star", "n_star")
    )
    np.testing.assert_allclose(
        [fitted[k] for k in keys], [target[k] for k in keys], rtol=1e-6, atol=1e-8
    )
    return result


def damped_linear_model(cosmology, kpressure_kms):
    """Create a Gaussian-damped linear P3D callable for P1D projection.

    Parameters
    ----------
    cosmology : lace.cosmo.cosmology.Cosmology
        Cosmology used to evaluate linear power and velocity conversion.
    kpressure_kms : float
        Gaussian damping scale in ``s / km``.

    Returns
    -------
    callable
        ForestFlow-compatible ``p3d(linear, z, k, mu, parameters, ...)``
        function returning P3D in ``Mpc**3`` for ``k`` in ``1 / Mpc``.
    """

    def p3d(linear, z, k, mu, parameters, new_cosmo_params=None):
        """Evaluate the damped linear P3D on a scalar or array integration grid.

        Parameters
        ----------
        linear, mu, parameters
            ForestFlow callback compatibility arguments; they do not alter the
            damped linear calculation.
        z : float
            Evaluation redshift.
        k : array_like
            Total comoving wavenumbers in ``1 / Mpc``.
        new_cosmo_params : mapping, optional
            Unsupported cosmology overrides.

        Returns
        -------
        ndarray
            Damped linear P3D with the same shape as ``k`` in ``Mpc**3``.

        Raises
        ------
        ValueError
            If cosmology overrides are supplied after construction.
        """
        if new_cosmo_params is not None:
            raise ValueError("Supply cosmology overrides when constructing the model")
        unique, inverse = np.unique(k, return_inverse=True)
        power = cosmology.get_linP_Mpc(z, unique)
        conversion = cosmology.get_dkms_dMpc(z)
        power *= np.exp(-((unique / conversion / kpressure_kms) ** 2))
        return power[inverse].reshape(np.shape(k))

    p3d.coordinates = "k_mu"
    return p3d


class CosmoScalingPlotter:
    """Compute and plot cached omch2, H0, and neutrino-mass scaling scans.

    Parameters use CAMB conventions. Wavenumber grids, the star pivot, pressure
    scale, and transverse limits are in s/km. Figures/data are saved only via
    explicit save arguments. Treat calculation settings as fixed after creation.
    """

    def __init__(
        self,
        fiducial_parameters=None,
        z_star=3.0,
        k_star_kms=0.009,
        scans=None,
        k_parallel_kms=None,
        k_linear_kms=None,
        kpressure_kms=0.4,
        k_perp_min=1e-6,
        k_perp_max=5.0,
        n_k_perp=397,
        camb_kmax_mpc=400.0,
    ):
        """Initialize a cached cosmological-scaling experiment.

        Parameters
        ----------
        fiducial_parameters : mapping, optional
            CAMB cosmological parameters overriding :data:`FIDUCIAL_PARAMETERS`.
        z_star : float, default: 3
            Redshift of the compressed linear-power pivot.
        k_star_kms : float, default: 0.009
            Pivot wavenumber in ``s / km``.
        scans : mapping[str, array_like], optional
            Scan values for ``omch2``, ``H0``, and ``mnu``.
        k_parallel_kms, k_linear_kms : array_like, optional
            Positive one-dimensional P1D and linear-power grids in ``s / km``.
        kpressure_kms : float, default: 0.4
            Positive Gaussian pressure-damping scale in ``s / km``.
        k_perp_min, k_perp_max : float
            Transverse integration bounds in ``s / km``.
        n_k_perp : int
            Number of transverse integration samples.
        camb_kmax_mpc : float, default: 400
            Minimum CAMB maximum wavenumber in ``1 / Mpc``.

        Raises
        ------
        ValueError
            If either wavenumber grid or ``kpressure_kms`` is non-finite,
            non-positive, or not one-dimensional where required.
        """
        self.parameters = {**FIDUCIAL_PARAMETERS, **(fiducial_parameters or {})}
        self.z_star, self.k_star_kms = z_star, k_star_kms
        self.scans = {
            key: np.array(values, dtype=float, copy=True)
            for key, values in (
                scans
                if scans is not None
                else dict(
                    omch2=np.linspace(0.1071, 0.1309, 6),
                    H0=np.linspace(57.66, 77.66, 6),
                    mnu=np.linspace(0.06, 0.3, 6),
                )
            ).items()
        }
        self.k_parallel = np.array(
            np.linspace(0.001, 0.04, 40) if k_parallel_kms is None else k_parallel_kms,
            copy=True,
        )
        self.k_linear = np.array(
            np.geomspace(0.001, 0.1, 300) if k_linear_kms is None else k_linear_kms,
            copy=True,
        )
        for grid in (self.k_parallel, self.k_linear):
            if (
                grid.ndim != 1
                or grid.size == 0
                or np.any(grid <= 0)
                or not np.all(np.isfinite(grid))
            ):
                raise ValueError("Wavenumber grids must be finite positive 1D arrays")
        if not np.isfinite(kpressure_kms) or kpressure_kms <= 0:
            raise ValueError("kpressure_kms must be finite and positive")
        self.kpressure_kms = kpressure_kms
        self.integration = dict(
            k_perp_min=k_perp_min, k_perp_max=k_perp_max, n_k_perp=n_k_perp
        )
        self.camb_kmax_mpc = camb_kmax_mpc
        self._backgrounds = {}
        self.results = {}
        self.fiducial = self._make_cosmology(self.parameters)
        self.target = self.fiducial.get_linP_kms_params(z_star, k_star_kms)
        self._fiducial_spectra = None

    def _make_cosmology(self, parameters):
        """Construct a cosmology with CAMB coverage for all projection grids.

        Parameters
        ----------
        parameters : mapping
            CAMB cosmological parameters.

        Returns
        -------
        lace.cosmo.cosmology.Cosmology
            Cosmology whose CAMB maximum is enlarged for the requested grids.
        """
        cosmo = Cosmology(
            cosmo_params_dict=dict(parameters), camb_kmax_Mpc=self.camb_kmax_mpc
        )
        conversion = cosmo.get_dkms_dMpc(self.z_star)
        maximum = (
            max(
                np.hypot(self.k_parallel.max(), self.integration["k_perp_max"]),
                self.k_linear.max(),
                2 * self.k_star_kms,
            )
            * conversion
        )
        # Set before any full CAMB calculation; include a margin at the endpoint.
        cosmo.camb_kmax_Mpc = max(self.camb_kmax_mpc, maximum * 1.01)
        return cosmo

    def project(self, cosmology, n_k_perp=None):
        """Project the damped linear spectrum to velocity-space P1D.

        Parameters
        ----------
        cosmology : lace.cosmo.cosmology.Cosmology
            Cosmology whose damped linear P3D is projected.
        n_k_perp : int, optional
            Override the configured number of transverse integration samples.

        Returns
        -------
        ndarray
            P1D with shape ``(len(k_parallel),)`` and units ``km / s``.
        """
        from forestflow.statistics.p1d import P1D_kms

        options = dict(self.integration)
        if n_k_perp is not None:
            options["n_k_perp"] = n_k_perp
        return P1D_kms(
            None,
            self.z_star,
            self.k_parallel,
            damped_linear_model(cosmology, self.kpressure_kms),
            cosmology.get_dkms_dMpc(self.z_star),
            **options,
        )

    def _spectra(self, cosmology):
        """Evaluate the linear and projected spectra for one cosmology.

        Parameters
        ----------
        cosmology : lace.cosmo.cosmology.Cosmology
            Cosmology to evaluate at :attr:`z_star`.

        Returns
        -------
        dict[str, ndarray]
            ``linear`` power in ``(km / s)**3`` and ``p1d`` in ``km / s``.

        Raises
        ------
        ValueError
            If either calculated spectrum contains non-finite or non-positive
            values.
        """
        spectra = dict(
            linear=cosmology.get_linP_kms(self.z_star, self.k_linear),
            p1d=self.project(cosmology),
        )
        for power in spectra.values():
            if not np.all(np.isfinite(power)) or np.any(power <= 0):
                raise ValueError("Non-finite or non-positive spectrum")
        return spectra

    def compute_scan(self, parameter):
        """Calculate and cache all matching cases for one parameter scan.

        Parameters
        ----------
        parameter : {"omch2", "H0", "mnu"}
            Cosmological parameter varied over :attr:`scans`.

        Returns
        -------
        list of dict
            One entry per scan value containing background parameters, models,
            star parameters, and linear/P1D spectra for every matching case.

        Raises
        ------
        ValueError
            If ``parameter`` is unsupported.

        Notes
        -----
        The neutrino-mass scan compensates CDM density to preserve the former
        experiment's fixed ``omch2 + omnuh2`` convention.
        """
        if parameter in self.results:
            return self.results[parameter]
        if parameter not in ("omch2", "H0", "mnu"):
            raise ValueError("Supported scans: omch2, H0, mnu")
        if self._fiducial_spectra is None:
            self._fiducial_spectra = self._spectra(self.fiducial)
        rows = []
        for value in self.scans[parameter]:
            parameters = {**self.parameters, parameter: float(value)}
            if parameter == "mnu":
                probe = Cosmology(cosmo_params_dict=parameters)
                parameters["omch2"] += (
                    self.fiducial.get_background_params()["omnuh2"] - probe.get_background_params()["omnuh2"]
                )
            key = tuple(sorted(parameters.items()))
            if key not in self._backgrounds:
                self._backgrounds[key] = self._make_cosmology(parameters)
            background = self._backgrounds[key]
            models = {CASES[0]: background}
            for case, running in zip(CASES[1:], (False, True)):
                models[case] = match_star_parameters(
                    background, self.target, self.z_star, self.k_star_kms, running
                )
            rows.append(
                dict(
                    value=value,
                    parameters=parameters,
                    models=models,
                    spectra={
                        case: self._spectra(model) for case, model in models.items()
                    },
                    star={
                        case: model.get_linP_kms_params(self.z_star, self.k_star_kms)
                        for case, model in models.items()
                    },
                )
            )
        self.results[parameter] = rows
        return rows

    def get_figure_data(self, parameter, spectrum):
        """Return plotted power ratios and complete reconstruction metadata.

        Parameters
        ----------
        parameter : {"omch2", "H0", "mnu"}
            Cosmological parameter scan to evaluate.
        spectrum : {"linear", "p1d"}
            Spectrum whose fractional ratio to the fiducial is returned.

        Returns
        -------
        dict
            ``x`` in ``s / km``, panel/scan ratio arrays defined as
            ``P / P_fid - 1``, and metadata with units and calculation inputs.

        Raises
        ------
        ValueError
            If ``spectrum`` is not ``"linear"`` or ``"p1d"``.
        """
        if spectrum not in ("linear", "p1d"):
            raise ValueError("spectrum must be 'linear' or 'p1d'")
        rows = self.compute_scan(parameter)
        data = {
            "x": (self.k_linear if spectrum == "linear" else self.k_parallel).copy()
        }
        for panel, case in enumerate(CASES):
            for index, row in enumerate(rows):
                data[f"y{panel}_{index}"] = (
                    row["spectra"][case][spectrum] / self._fiducial_spectra[spectrum]
                    - 1
                )
        data["metadata"] = deepcopy(
            dict(
                parameter=parameter,
                values=self.scans[parameter],
                spectrum=spectrum,
                x_units="s/km",
                y_units="dimensionless",
                y_definition="P/P_fid - 1",
                power_units="(km/s)^3" if spectrum == "linear" else "km/s",
                cases=CASES,
                titles=TITLES,
                fiducial=self.parameters,
                z_star=self.z_star,
                k_star_kms=self.k_star_kms,
                kpressure_kms=self.kpressure_kms,
                integration=self.integration,
                backgrounds=[row["parameters"] for row in rows],
                matched_parameters=[
                    {case: row["models"][case].new_params for case in CASES[1:]}
                    for row in rows
                ],
                star_parameters=[row["star"] for row in rows],
            )
        )
        return data

    @staticmethod
    def _label(parameter, value):
        """Format one scan value for a plot legend.

        Parameters
        ----------
        parameter : str
            Scanned parameter name.
        value : float
            Parameter value in its native CAMB convention.

        Returns
        -------
        str
            LaTeX-formatted legend label.
        """
        if parameter == "H0":
            return rf"$h={value/100:.3f}$"
        if parameter == "mnu":
            return rf"$\sum m_\nu={value:.2f}$ eV"
        return rf"$\Omega_\mathrm{{cdm}}h^2={value:.3f}$"

    def plot(
        self,
        parameter,
        spectrum="p1d",
        *,
        fontsize=20,
        figsize=(8, 8),
        cmap="turbo",
        ylims=None,
        save_path=None,
        save_data=False,
        output_directory=None,
    ):
        """Plot scaling residuals for the three compressed-parameter matches.

        Parameters
        ----------
        parameter : {"omch2", "H0", "mnu"}
            Cosmological parameter scan to plot.
        spectrum : {"linear", "p1d"}, default: "p1d"
            Select linear or projected P1D residuals.
        fontsize : float, default: 20
            Base figure font size.
        figsize : tuple, default: (8, 8)
            Matplotlib figure dimensions in inches.
        cmap : str, default: "turbo"
            Matplotlib colormap for scan values.
        ylims : sequence of tuple, optional
            Explicit limits for the three panels.
        save_path : str or path-like, optional
            Figure output path.
        save_data : bool, default: False
            Save the plotted arrays using :meth:`save_data_to_zenodo`.
        output_directory : str or path-like, optional
            Directory for saved arrays.

        Returns
        -------
        tuple
            Matplotlib figure, three axes, and the plotted data dictionary.
        """
        import matplotlib.pyplot as plt

        data = self.get_figure_data(parameter, spectrum)
        values = self.scans[parameter]
        fig, axes = plt.subplots(3, figsize=figsize, sharex=True)
        colors = plt.get_cmap(cmap)(np.arange(len(values)) / len(values))
        for panel, axis in enumerate(axes):
            for i, value in enumerate(values):
                label = (
                    self._label(parameter, value)
                    if min(3 * i // len(values), 2) == panel
                    else None
                )
                axis.plot(
                    data["x"], data[f"y{panel}_{i}"], color=colors[i], label=label
                )
            axis.axhline(0, color="k", ls=":")
            axis.set_title(TITLES[panel], fontsize=fontsize)
            if axis.get_legend_handles_labels()[0]:
                axis.legend(loc="upper left", fontsize=fontsize - 5, ncol=3)
            axis.tick_params(labelsize=fontsize)
            if spectrum == "linear":
                axis.set(xscale="log", xlim=(0.001, 0.1))
                axis.set_ylim(
                    (-0.01, 0.01)
                    if parameter == "H0"
                    else ((-0.25, 0.25) if panel == 0 else (-0.02, 0.02))
                )
                if panel:
                    axis.axvline(self.k_star_kms, color="k", ls=":")
            elif parameter == "H0":
                axis.set_ylim(-0.003, 0.003)
            elif panel:
                limit = 0.008 if parameter == "mnu" else 0.02
                axis.set_ylim(-limit, limit)
            if ylims is not None:
                axis.set_ylim(ylims[panel])
        symbol = r"P_\mathrm{lin}" if spectrum == "linear" else r"P_\mathrm{1D}"
        fig.supylabel(rf"${symbol}/{symbol}^\mathrm{{fid}}-1$", fontsize=fontsize)
        fig.supxlabel(
            r"$k$ [s/km]" if spectrum == "linear" else r"$k_\parallel$ [s/km]",
            fontsize=fontsize,
        )
        fig.tight_layout()
        data["metadata"].update(
            colors=colors,
            labels=[self._label(parameter, v) for v in values],
            xlim=axes[0].get_xlim(),
            ylims=[a.get_ylim() for a in axes],
        )
        if save_path is not None:
            fig.savefig(save_path, bbox_inches="tight")
        if save_data:
            self.save_data_to_zenodo(data, output_directory=output_directory)
        return fig, axes, data

    @staticmethod
    def save_data_to_zenodo(data, filename=None, output_directory=None):
        """Write figure arrays locally in NumPy format without uploading.

        Parameters
        ----------
        data : dict
            Payload produced by :meth:`get_figure_data`.
        filename : str, optional
            Output filename.  Defaults to the appendix-figure convention.
        output_directory : str or path-like, optional
            Output directory.  Defaults to Cup1D's local Zenodo-data folder.

        Returns
        -------
        pathlib.Path
            Written ``.npy`` file path.
        """
        from cup1d.utils.utils import get_path_repo

        metadata = data["metadata"]
        if filename is None:
            number = {"H0": 1, "mnu": 2, "omch2": 3}[metadata["parameter"]]
            suffix = "a" if metadata["spectrum"] == "linear" else "b"
            filename = f"fig_A{number}{suffix}.npy"
        directory = (
            Path(output_directory)
            if output_directory is not None
            else Path(get_path_repo("cup1d")) / "data" / "zenodo"
        )
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / filename
        np.save(path, data)
        return path

    def plot_fiducial(self):
        """Plot the fiducial dimensionless velocity-space P1D.

        Returns
        -------
        tuple
            Matplotlib figure and axes, with ``k_parallel P1D / pi`` plotted
            against ``k_parallel`` in ``s / km``.
        """
        import matplotlib.pyplot as plt

        if self._fiducial_spectra is None:
            self._fiducial_spectra = self._spectra(self.fiducial)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(
            self.k_parallel,
            self.k_parallel * self._fiducial_spectra["p1d"] / np.pi,
            ".-",
        )
        ax.set(
            xlabel=r"$k_\parallel$ [s/km]", ylabel=r"$k_\parallel P_\mathrm{1D}/\pi$"
        )
        return fig, ax

    def plot_matched_detail(self, parameter):
        """Compare two- and three-parameter linear matching residuals.

        Parameters
        ----------
        parameter : {"omch2", "H0", "mnu"}
            Cosmological parameter scan to compare.

        Returns
        -------
        tuple
            Matplotlib figure, axes, and underlying plotted data dictionary.
        """
        import matplotlib.pyplot as plt

        data = self.get_figure_data(parameter, "linear")
        fig, ax = plt.subplots(figsize=(8, 6))
        for i, value in enumerate(self.scans[parameter]):
            ax.plot(
                data["x"],
                data[f"y2_{i}"],
                color=f"C{i}",
                label=self._label(parameter, value),
            )
            ax.plot(data["x"], data[f"y1_{i}"], color=f"C{i}", ls="--")
        ax.set(
            xscale="log",
            xlabel="k [s/km]",
            ylabel=r"$P_\mathrm{lin}/P_\mathrm{lin}^\mathrm{fid}-1$",
            title="Solid: match three parameters; dashed: match two",
        )
        ax.set_ylim((-0.0005, 0.0005) if parameter == "H0" else (-0.01, 0.01))
        if parameter == "H0":
            ax.set_xlim(0.004, 0.02)
        ax.legend(loc="upper right")
        fig.tight_layout()
        return fig, ax, data
