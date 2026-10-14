"""Plot linear-power ratios for CMB chains and a DESI P1D constraint."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cup1d.utils.various_dicts import get_blob_value


class PowerRatioPlotter:
    """Plot linear-power ratios relative to a reference CMB cosmology.

    The class keeps data loading separate from plotting.  Pass already-loaded
    chain specifications and file paths to :meth:`load_data`. The class then
    loads the Planck chains, DESI blobs, blinding offsets, and precomputed
    linear-power samples itself.
    """

    def __init__(
        self,
        pivot_kms=0.009,
        n_desi_samples=20000,
        n_cmb_samples=500,
        redshift=3.0,
        random_seed=None,
    ):
        """Initialize Figure-22 power-ratio calculation settings.

        Parameters
        ----------
        pivot_kms : float, default: 0.009
            Common compressed-linear-power pivot in ``s / km``.
        n_desi_samples : int, default: 20000
            Maximum randomly selected DESI P1D samples for its band.
        n_cmb_samples : int, default: 500
            Maximum CMB chain samples used to compute a missing power cache.
        redshift : float, default: 3
            Redshift at which velocity-space linear power is evaluated.
        random_seed : int, optional
            Seed for reproducible chain subsampling.
        """
        self.pivot_kms = pivot_kms
        self.n_desi_samples = n_desi_samples
        self.n_cmb_samples = n_cmb_samples
        self.redshift = redshift
        self.random_generator = np.random.default_rng(random_seed)
        self._is_loaded = False

    @staticmethod
    def load_power_samples(file_paths):
        """Load cached linear-power arrays.

        Parameters
        ----------
        file_paths : sequence of str or path-like
            NumPy files containing one ``(n_samples, n_k)`` array each.

        Returns
        -------
        list of ndarray
            Cached velocity-space linear-power samples in file-path order.
        """
        return [np.load(Path(file_path)) for file_path in file_paths]

    def load_data(
        self,
        chain_specs,
        desi_blobs_path,
        blinding_path,
        power_file_paths,
        planck_root_dir=None,
        k_kms=None,
    ):
        """Load all Figure-22 chains, DESI samples, and power-cache inputs.

        Parameters
        ----------
        chain_specs : sequence of mapping
            Planck chain specifications with ``model``, ``data``, and ``label``;
            optional ``name`` and ``linP_tag`` select the stored chain.
        desi_blobs_path, blinding_path : str or path-like
            DESI ``blobs.npy`` and additive compressed-parameter offsets.
        power_file_paths : sequence of str or path-like
            Cache paths aligned with ``chain_specs``.  Missing files are
            calculated and saved.
        planck_root_dir : str or path-like, optional
            Parent directory containing Planck chain releases.
        k_kms : array_like, optional
            Velocity-space wavenumber grid in ``s / km``; defaults to the
            Figure-22 cache grid.

        Returns
        -------
        PowerRatioPlotter
            This loaded instance.
        """
        from cup1d.postprocessing.chains import planck

        chains = planck.load_planck_2018_chains(
            chain_specs, root_dir=planck_root_dir
        )
        cmb_chains = [
            chains[spec.get("name", spec["model"])] for spec in chain_specs
        ]
        labels = [spec["label"] for spec in chain_specs]

        blobs = np.load(desi_blobs_path)
        blinding = np.load(blinding_path, allow_pickle=True).item()
        desi_delta2_star = get_blob_value(blobs, "Delta2_star").reshape(-1) - blinding["Delta2_star"]
        desi_n_star = get_blob_value(blobs, "n_star").reshape(-1) - blinding["n_star"]
        if k_kms is None:
            k_kms = self.default_power_kms_grid()
        return self.set_data(
            cmb_chains=cmb_chains,
            labels=labels,
            desi_delta2_star=desi_delta2_star,
            desi_n_star=desi_n_star,
            power_samples=self.load_or_create_power_samples(
                cmb_chains, power_file_paths, k_kms
            ),
            k_kms=k_kms,
        )

    @staticmethod
    def default_power_kms_grid():
        """Return the Figure-22 velocity-space linear-power grid.

        Returns
        -------
        ndarray
            Logarithmically spaced wavenumbers in ``s / km`` used by the
            distributed ``P_kms_*.npy`` caches.
        """
        log10_k_min = -5.888706504390846
        log10_k_max = -0.41158524967118454
        log10_k_step = 0.0054826
        return 10 ** np.arange(log10_k_min, log10_k_max, log10_k_step)

    def load_or_create_power_samples(self, cmb_chains, file_paths, k_kms):
        """Load cached power samples or calculate missing arrays from CMB chains.

        Parameters
        ----------
        cmb_chains : sequence of dict
            Chain dictionaries containing GetDist ``samples`` objects.
        file_paths : sequence of str or path-like
            Cache paths aligned with ``cmb_chains``.
        k_kms : array_like
            Evaluation wavenumber grid in ``s / km``.

        Returns
        -------
        list of ndarray
            One linear-power array with shape ``(n_samples, len(k_kms))`` per
            CMB chain, in velocity-space power units ``(km / s)**3``.

        Raises
        ------
        ValueError
            If chain and cache-path sequence lengths differ.
        """
        if len(cmb_chains) != len(file_paths):
            raise ValueError("cmb_chains and file_paths must have equal length")

        power_samples = []
        for chain, file_path in zip(cmb_chains, file_paths):
            path = Path(file_path)
            if path.exists():
                power_samples.append(np.load(path))
                continue

            print(f"creating missing linear-power cache: {path}")
            samples = self._compute_power_samples(chain["samples"], k_kms)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, samples)
            power_samples.append(samples)
        return power_samples

    def _compute_power_samples(self, chain_samples, k_kms):
        """Compute velocity-space linear power for a random CMB-chain subset.

        Parameters
        ----------
        chain_samples : getdist.MCSamples
            CMB posterior samples convertible to LaCE cosmology parameters.
        k_kms : array_like
            Wavenumbers in ``s / km``.

        Returns
        -------
        ndarray
            Linear power with shape ``(n_selected, len(k_kms))`` and units
            ``(km / s)**3`` at :attr:`redshift`.
        """
        from lace.cosmo.cosmology import Cosmology

        n_chain_samples = chain_samples.samples.shape[0]
        n_samples = min(self.n_cmb_samples, n_chain_samples)
        indices = self.random_generator.choice(
            n_chain_samples, size=n_samples, replace=False
        )
        power_samples = np.empty((n_samples, len(k_kms)))
        for output_index, chain_index in enumerate(indices):
            if output_index % 25 == 0:
                print(f"computing linear power: {output_index + 1}/{n_samples}")
            parameters = chain_samples.getParamSampleDict(chain_index)
            cosmology = Cosmology(cosmo_params_dict=parameters)
            power_samples[output_index] = cosmology.get_linP_kms(
                self.redshift, k_kms
            )
        return power_samples

    def set_data(
        self,
        cmb_chains,
        labels,
        desi_delta2_star,
        desi_n_star,
        power_samples,
        k_kms,
        delta2_parameter="linP_DL2_star",
        n_parameter="linP_n_star",
    ):
        """Store already-loaded inputs needed for a power-ratio figure.

        Parameters
        ----------
        cmb_chains : sequence of dict
            Chain dictionaries containing a GetDist ``samples`` object.  The
            first element defines the reference LCDM cosmology.
        labels : sequence of str
            Labels for the CMB chains, in the same order as ``cmb_chains``.
        desi_delta2_star, desi_n_star : array-like
            DESI P1D samples at the common linear-power pivot.
        power_samples : sequence of ndarray
            Precomputed ``(n_samples, n_k)`` linear-power arrays, one per CMB
            chain and in the same order.
        k_kms : array-like
            Wavenumber grid of the precomputed linear-power arrays, in s/km.
        delta2_parameter, n_parameter : str
            Names of the amplitude and slope columns in the CMB chains.

        Returns
        -------
        PowerRatioPlotter
            This instance, marked ready for :meth:`plot`.

        Raises
        ------
        ValueError
            If aligned input sequences have inconsistent lengths, DESI sample
            arrays have different shapes, or a power array does not match
            ``k_kms``.
        KeyError
            If the reference CMB chain lacks either requested compressed
            linear-power parameter.
        """
        if not (len(cmb_chains) == len(labels) == len(power_samples)):
            raise ValueError("cmb_chains, labels, and power_samples must have equal length")
        if not cmb_chains:
            raise ValueError("at least one reference CMB chain is required")

        self.cmb_chains = list(cmb_chains)
        self.labels = list(labels)
        self.desi_delta2_star = np.asarray(desi_delta2_star)
        self.desi_n_star = np.asarray(desi_n_star)
        self.power_samples = [np.asarray(samples) for samples in power_samples]
        self.k_kms = np.asarray(k_kms)
        self.delta2_parameter = delta2_parameter
        self.n_parameter = n_parameter

        if self.desi_delta2_star.shape != self.desi_n_star.shape:
            raise ValueError("DESI amplitude and slope samples must have the same shape")
        for samples in self.power_samples:
            if samples.ndim != 2 or samples.shape[1] != self.k_kms.size:
                raise ValueError("each power-sample array must have shape (n_samples, len(k_kms))")

        reference_samples = self.cmb_chains[0]["samples"]
        try:
            self.reference_delta2_star = np.asarray(
                reference_samples[reference_samples.index[delta2_parameter]]
            )
            self.reference_n_star = np.asarray(
                reference_samples[reference_samples.index[n_parameter]]
            )
        except KeyError as error:
            raise KeyError(f"reference chain does not contain {error.args[0]!r}") from error

        self._is_loaded = True
        return self

    def plot(self, panel_indices=None, figsize=(10, 12), fontsize=22, save_path=None):
        """Create the CMB power-ratio figure and return figure, axes, and data.

        Parameters
        ----------
        panel_indices : sequence of sequence of int, optional
            CMB-chain indices grouped into vertical panels. By default the
            reference occupies the first panel, the next two extensions the
            second, and remaining extensions the third.
        figsize : tuple of float, default: (10, 12)
            Matplotlib figure size in inches.
        fontsize : float, default: 22
            Font size used for axis labels, ticks, and legends.
        save_path : str or path-like, optional
            Output filename for a copy of the figure. No file is written when
            omitted.

        Returns
        -------
        matplotlib.figure.Figure
            Created power-ratio figure.
        ndarray of matplotlib.axes.Axes
            One axis per requested panel.
        dict
            Plotted CMB and DESI bands, indexed by component name.

        Raises
        ------
        RuntimeError
            If :meth:`load_data` or :meth:`set_data` has not been called.
        """
        if not self._is_loaded:
            raise RuntimeError("call load_data before plot")

        if panel_indices is None:
            panel_indices = [[0], list(range(1, min(3, len(self.labels)))), list(range(3, len(self.labels)))]
            panel_indices = [indices for indices in panel_indices if indices]

        figure, axes = plt.subplots(
            len(panel_indices), 1, figsize=figsize, sharex=True, sharey=True
        )
        axes = np.atleast_1d(axes)

        reference_power = np.percentile(self.power_samples[0], [16, 50, 84], axis=0)
        reference_low, reference_median, reference_high = reference_power
        figure_data = {}
        self._add_desi_constraint(axes, figure_data)

        # Preserve the Figure-22 color and line-style convention.
        colors = ["C2", "C6", "C5", "C7"]
        line_styles = ["--", "-", "--", "-"]
        # Hatch neutrino-mass and running-spectrum extensions, as in Fig. 22.
        hatches = ["/", "", "/", ""]
        for panel_index, chain_indices in enumerate(panel_indices):
            axis = axes[panel_index]
            for chain_index in chain_indices:
                low, median, high = np.percentile(
                    self.power_samples[chain_index], [16, 50, 84], axis=0
                )
                if chain_index == 0:
                    color, line_style, hatch = "C1", "-", ""
                else:
                    color = colors[(chain_index - 1) % len(colors)]
                    line_style = line_styles[(chain_index - 1) % len(line_styles)]
                    hatch = hatches[(chain_index - 1) % len(hatches)]
                key = f"cmb_{chain_index}"
                axis.fill_between(
                    self.k_kms,
                    low / reference_median,
                    high / reference_median,
                    color=color,
                    alpha=0.2,
                    hatch=hatch,
                    label=self.labels[chain_index],
                )
                axis.plot(
                    self.k_kms,
                    median / reference_median,
                    color=color,
                    linestyle=line_style,
                    linewidth=2,
                )
                figure_data[key] = {
                    "k_kms": self.k_kms,
                    "median": median / reference_median,
                    "lower": low / reference_median,
                    "upper": high / reference_median,
                    "label": self.labels[chain_index],
                }

            axis.axhline(1, linestyle=":", color="k", alpha=0.5)
            axis.set(xlim=(1.5e-5, 0.06), ylim=(0.85, 1.2), xscale="log")
            axis.legend(fontsize=fontsize, loc="upper left")
            axis.tick_params(axis="both", which="major", labelsize=fontsize)
            self._add_scale_annotations(axis, fontsize)

        figure.supylabel(
            r"$P_\mathrm{lin}(k,z=3)/P_\mathrm{lin}^{\Lambda\mathrm{CDM}}(k,z=3)$",
            fontsize=fontsize,
        )
        axes[-1].set_xlabel(r"$k$ [s/km]", fontsize=fontsize)
        figure.tight_layout()

        if save_path is not None:
            figure.savefig(save_path, bbox_inches="tight")
        return figure, axes, figure_data

    @staticmethod
    def save_data_to_zenodo(figure_data, filename="fig_22.npy"):
        """Save a figure-data dictionary under cup1d's Zenodo data directory.

        Parameters
        ----------
        figure_data : dict
            Data returned by :meth:`plot`.
        filename : str
            Name of the NumPy file, defaulting to the Figure 22 convention.

        Returns
        -------
        pathlib.Path
            Saved ``.npy`` file under ``data/zenodo``.
        """
        from cup1d.utils.utils import get_path_repo

        output_path = Path(get_path_repo("cup1d")) / "data" / "zenodo" / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, figure_data)
        return output_path

    def _add_desi_constraint(self, axes, figure_data):
        """Add the DESI amplitude-and-slope band to every panel.

        Parameters
        ----------
        axes : sequence of matplotlib.axes.Axes
            Axes receiving the DESI pivot error bar and ratio band.
        figure_data : dict
            Mutable figure-data dictionary updated with a ``"desi_p1d"``
            entry. Wavenumbers are in ``s / km`` and all ratios are
            dimensionless.
        """
        n_samples = min(self.n_desi_samples, self.desi_delta2_star.size)
        sample_indices = self.random_generator.choice(
            self.desi_delta2_star.size, size=n_samples, replace=False
        )
        desi_delta2 = self.desi_delta2_star[sample_indices]
        desi_n = self.desi_n_star[sample_indices]
        conversion = self.pivot_kms**3 / (2 * np.pi**2)
        k_band = np.geomspace(0.5 * self.pivot_kms, 2 * self.pivot_kms, 200)

        reference_log_power = np.median(
            np.log(self.reference_delta2_star / conversion)[:, None]
            + self.reference_n_star[:, None] * np.log(k_band[None, :] / self.pivot_kms),
            axis=0,
        )
        desi_log_power = (
            np.median(np.log(desi_delta2 / conversion))
            + desi_n[:, None] * np.log(k_band[None, :] / self.pivot_kms)
        )
        desi_ratio = np.exp(desi_log_power - reference_log_power)
        lower, median, upper = np.percentile(desi_ratio, [16, 50, 84], axis=0)
        amplitude_ratio = np.exp(np.log(desi_delta2 / conversion) - np.median(
            np.log(self.reference_delta2_star / conversion)
        ))

        for index, axis in enumerate(axes):
            axis.errorbar(
                self.pivot_kms,
                np.median(amplitude_ratio),
                np.std(amplitude_ratio),
                marker="o",
                color="C0",
                elinewidth=2,
            )
            axis.fill_between(
                k_band,
                lower,
                upper,
                alpha=0.3,
                color="C0",
                label=r"DESI $P_\mathrm{1D}$" if index == 0 else None,
            )

        figure_data["desi_p1d"] = {
            "k_kms": k_band,
            "median": median,
            "lower": lower,
            "upper": upper,
            "pivot_kms": self.pivot_kms,
            "pivot_ratio": np.median(amplitude_ratio),
            "pivot_error": np.std(amplitude_ratio),
        }

    @staticmethod
    def _add_scale_annotations(axis, fontsize):
        """Mark the approximate Ly-alpha P1D and CMB scale ranges.

        Parameters
        ----------
        axis : matplotlib.axes.Axes
            Axis on which the scale-range lines and labels are drawn.
        fontsize : float
            Font size for the annotations.
        """
        axis.plot([0.00125, 0.04], [0.92, 0.92], linewidth=2, color="k")
        axis.text(5e-3, 0.87, r"Ly$\alpha$ $P_\mathrm{1D}$", fontsize=fontsize)
        axis.plot([2.85e-5, 0.0025], [0.93, 0.93], linewidth=2, color="k")
        axis.text(1.2e-4, 0.87, "CMB T&E", fontsize=fontsize)
