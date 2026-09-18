"""Plot linear-power ratios for CMB chains and a DESI P1D constraint."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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
        self.pivot_kms = pivot_kms
        self.n_desi_samples = n_desi_samples
        self.n_cmb_samples = n_cmb_samples
        self.redshift = redshift
        self.random_generator = np.random.default_rng(random_seed)
        self._is_loaded = False

    @staticmethod
    def load_power_samples(file_paths):
        """Load one two-dimensional linear-power sample array per file."""
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
        """Load all Figure-22 inputs directly from chains and files.

        Each element of ``chain_specs`` must define ``model``, ``data``, and
        ``label``; it may optionally define ``linP_tag``.  The first chain is
        the LCDM reference. ``desi_blobs_path`` points to the DESI ``blobs.npy``
        file and ``blinding_path`` to its ``blinding.npy`` offsets.
        """
        from cup1d.postprocessing.chains import planck

        cmb_chains = []
        labels = []
        for spec in chain_specs:
            cmb_chains.append(
                planck.get_planck_2018(
                    model=spec["model"],
                    data=spec["data"],
                    root_dir=planck_root_dir,
                    linP_tag=spec.get("linP_tag"),
                )
            )
            labels.append(spec["label"])

        blobs = np.load(desi_blobs_path)
        blinding = np.load(blinding_path, allow_pickle=True).item()
        desi_delta2_star = blobs["Delta2_star"].reshape(-1) - blinding["Delta2_star"]
        desi_n_star = blobs["n_star"].reshape(-1) - blinding["n_star"]
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
        """Return the k grid used to generate the stored ``P_kms_*.npy`` files."""
        log10_k_min = -5.888706504390846
        log10_k_max = -0.41158524967118454
        log10_k_step = 0.0054826
        return 10 ** np.arange(log10_k_min, log10_k_max, log10_k_step)

    def load_or_create_power_samples(self, cmb_chains, file_paths, k_kms):
        """Load cached power samples, computing and caching missing arrays.

        Missing files are generated from random samples of their corresponding
        GetDist chain. ``n_cmb_samples`` controls the number of CAMB calls.
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
        """Compute linear power at the configured redshift for chain samples."""
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

        ``panel_indices`` groups CMB-chain indices into panels.  By default the
        reference occupies the first panel, the next two extensions the second,
        and all remaining extensions the third.
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
        """
        from cup1d.utils.utils import get_path_repo

        output_path = Path(get_path_repo("cup1d")) / "data" / "zenodo" / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, figure_data)
        return output_path

    def _add_desi_constraint(self, axes, figure_data):
        """Add the DESI amplitude-and-slope band to every panel."""
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
        """Mark the approximate Ly-alpha P1D and CMB scale ranges."""
        axis.plot([0.00125, 0.04], [0.92, 0.92], linewidth=2, color="k")
        axis.text(5e-3, 0.87, r"Ly$\alpha$ $P_\mathrm{1D}$", fontsize=fontsize)
        axis.plot([2.85e-5, 0.0025], [0.93, 0.93], linewidth=2, color="k")
        axis.text(1.2e-4, 0.87, "CMB T&E", fontsize=fontsize)
