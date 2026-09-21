"""Historical linear-power constraints from CMB and Lyman-alpha analyses."""

from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from getdist import plots
from scipy.stats import gaussian_kde, norm

from cup1d.likelihood import marginal
from cup1d.postprocessing.chains import planck
from cup1d.utils.utils import get_path_repo


class HistoricalLinearPowerPlotter:
    """Create the CMB, DESI DR1, and historical P1D comparison figure."""

    def __init__(self, fontsize=26):
        self.fontsize = fontsize
        self._is_loaded = False

    def load_data(
        self,
        planck_root_dir=None,
        desi_chain_directory=None,
        blinding_path=None,
    ):
        """Load the Planck LCDM chain, DESI DR1 contours, and literature data.

        All paths are optional. By default the method uses the repository's
        Planck chains, the DESI DR1 ``chain_7`` output next to the repository,
        and the tutorial blinding offsets.
        """
        repository_path = Path(get_path_repo("cup1d"))
        if planck_root_dir is None:
            planck_root_dir = repository_path / "data" / "planck_linP_chains"
        if desi_chain_directory is None:
            desi_chain_directory = (
                repository_path.parent
                / "data"
                / "out_DESI_DR1"
                / "DESIY1_QMLE3"
                / "global_opt"
                / "CH24_mpgcen_gpr"
                / "chain_7"
            )
        if blinding_path is None:
            blinding_path = repository_path / "data" / "blinding_dr1.npy"

        desi_chain_directory = Path(desi_chain_directory)
        self.cmb = planck.get_planck_2018(
            model="base",
            data="plikHM_TTTEEE_lowl_lowE_linP",
            root_dir=planck_root_dir,
            linP_tag=None,
        )
        self.desi_contours = np.load(
            desi_chain_directory / "line_sigmas.npy", allow_pickle=True
        ).item()
        self.desi_blobs = np.load(desi_chain_directory / "blobs.npy")
        self.desi_summary = np.load(
            desi_chain_directory / "summary.npy", allow_pickle=True
        ).item()
        self.blinding = np.load(blinding_path, allow_pickle=True).item()
        self._is_loaded = True
        return self

    def plot(self, include_simulations=True):
        """Return the historical constraint figure without writing files.

        Set ``include_simulations=False`` to omit the MPG simulation points.
        """
        self._require_loaded()
        cmb_samples = self.cmb["samples"].copy()
        self._ensure_ranges_periodic(cmb_samples)

        plotter = plots.getSubplotPlotter(width_inch=10)
        plotter.settings.num_plot_contours = 2
        plotter.settings.axes_fontsize = self.fontsize - 6
        plotter.settings.legend_fontsize = self.fontsize - 6
        plotter.triangle_plot(
            [cmb_samples],
            ["linP_DL2_star", "linP_n_star"],
            lws=[3, 2],
            line_args={"color": "black", "lw": 2, "alpha": 0.7},
        )

        joint_axis = plotter.subplots[1, 0]
        slope_axis = plotter.subplots[1, 1]
        amplitude_axis = plotter.subplots[0, 0]
        self._plot_literature(joint_axis, amplitude_axis, slope_axis)
        self._plot_desi(joint_axis, amplitude_axis, slope_axis)
        if include_simulations:
            self._plot_simulations(joint_axis)
        self._format_axes(joint_axis, amplitude_axis, slope_axis)
        self._add_legend(slope_axis)
        return plotter.fig, plotter.subplots

    def get_zenodo_data(self):
        """Return the compact Figure-18 summary data dictionary."""
        self._require_loaded()
        samples = self.cmb["samples"]
        delta2_star = np.asarray(samples["linP_DL2_star"])
        n_star = np.asarray(samples["linP_n_star"])
        data = {
            "black": self._constraint_from_samples(delta2_star, n_star),
            "blue": self._desi_constraint(),
        }
        for color, constraint in zip(
            ["orange", "green", "red", "purple"], self._literature_constraints()
        ):
            data[color] = {
                "x": constraint["Delta2_star"],
                "xerr": constraint["Delta2_star_err"],
                "y": constraint["n_star"],
                "yerr": constraint["n_star_err"],
                "r": constraint["r"],
            }
        return data

    def save_data_to_zenodo(self, filename="fig_18.npy"):
        """Save Figure-18 summary data under cup1d's Zenodo directory."""
        output_path = Path(get_path_repo("cup1d")) / "data" / "zenodo" / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, self.get_zenodo_data())
        return output_path

    def _plot_literature(self, joint_axis, amplitude_axis, slope_axis):
        thresholds = [2.30, 6.18]
        n_star_grid, delta2_grid = np.mgrid[-2.4:-2.2:200j, 0.2:0.65:200j]
        x_delta2 = np.linspace(0.2, 0.55, 500)
        x_n_star = np.linspace(-2.4, -2.2, 500)
        colors = ["C1", "C2", "C3", "C4"]
        line_styles = [":", "-.", "--", "--"]
        for constraint, color, line_style in zip(
            self._literature_constraints(), colors, line_styles
        ):
            joint_axis.contour(
                delta2_grid,
                n_star_grid,
                constraint["chi2"],
                levels=thresholds,
                colors=color,
                linewidths=[3, 2],
                alpha=0.7,
                linestyles=line_style,
            )
            amplitude_pdf = norm.pdf(
                x_delta2,
                constraint["Delta2_star"],
                constraint["Delta2_star_err"],
            )
            amplitude_axis.plot(
                x_delta2,
                amplitude_pdf / amplitude_pdf.max(),
                color=color,
                lw=3,
                alpha=0.7,
                ls=line_style,
            )
            slope_pdf = norm.pdf(
                x_n_star, constraint["n_star"], constraint["n_star_err"]
            )
            slope_axis.plot(
                x_n_star,
                slope_pdf / slope_pdf.max(),
                color=color,
                lw=3,
                alpha=0.7,
                ls=line_style,
            )

    def _plot_desi(self, joint_axis, amplitude_axis, slope_axis):
        blues = plt.colormaps["Blues"]
        for index, probability in enumerate([0.68, 0.95]):
            color = blues([0.7, 0.3][index])
            for polygon in self.desi_contours[probability]:
                delta2_star = polygon[0] - self.blinding["Delta2_star"]
                n_star = polygon[1] - self.blinding["n_star"]
                joint_axis.plot(delta2_star, n_star, color=color, lw=[3, 2][index], alpha=0.7)
                joint_axis.fill(delta2_star, n_star, color=color, alpha=0.7)

        delta2_star = self.desi_blobs["Delta2_star"].reshape(-1) - self.blinding["Delta2_star"]
        n_star = self.desi_blobs["n_star"].reshape(-1) - self.blinding["n_star"]
        self._plot_kde(amplitude_axis, delta2_star, blues(0.7))
        self._plot_kde(slope_axis, n_star, blues(0.7))

    @staticmethod
    def _plot_kde(axis, samples, color):
        kde = gaussian_kde(samples)
        values = np.linspace(samples.min(), samples.max(), 200)
        density = kde(values)
        axis.plot(values, density / density.max(), color=color, lw=3)

    @staticmethod
    def _plot_simulations(axis):
        from cup1d.theory.cosmology import set_cosmo

        for label, cosmology in set_cosmo("mpg_0", return_all=True).items():
            if label[-1].isdigit() or label == "mpg_central":
                star_params = cosmology["star_params"]
                axis.scatter(star_params["Delta2_star"], star_params["n_star"], color="C0")

    def _format_axes(self, joint_axis, amplitude_axis, slope_axis):
        joint_axis.set(xlim=(0.23, 0.52), ylim=(-2.39, -2.24))
        joint_axis.set_xticks([0.3, 0.4, 0.5])
        joint_axis.set_yticks([-2.35, -2.30, -2.25])
        slope_axis.set_xticks([-2.35, -2.30, -2.25])
        for axis in [joint_axis, slope_axis]:
            axis.tick_params(axis="both", which="major", labelsize=self.fontsize)
        joint_axis.set_xlabel(r"$\Delta^2_\star$", fontsize=self.fontsize)
        joint_axis.set_ylabel(r"$n_\star$", fontsize=self.fontsize)
        slope_axis.set_xlabel(r"$n_\star$", fontsize=self.fontsize)
        for axis in joint_axis.figure.axes:
            for label in axis.get_xticklabels():
                label.set_rotation(45)
                label.set_ha("right")

    def _add_legend(self, axis):
        labels = [
            r"DESI DR1 (this work)",
            r"SDSS (McDonald+05)",
            "BOSS\n(Palanque-Delabrouille+15)",
            r"eBOSS (Chabanier+19)",
            "eBOSS + priors\n(Walther+24)",
        ]
        colors = ["C1", "C2", "C3", "C4"]
        line_styles = [":", "-.", "--", "--"]
        handles = [
            mlines.Line2D(
                [], [], color="black", label=r"$\mathit{Planck}$ T&E: $\Lambda$CDM", lw=3
            ),
            mlines.Line2D([], [], color="C0", label=labels[0], lw=3),
        ]
        handles.extend(
            mlines.Line2D(
                [], [], color=color, label=label, lw=3, ls=line_style
            )
            for color, label, line_style in zip(colors, labels[1:], line_styles)
        )
        axis.legend(
            handles=handles,
            bbox_to_anchor=(1, 2),
            loc="upper right",
            borderaxespad=0.0,
            fontsize=self.fontsize - 6,
        )

    @staticmethod
    def _literature_constraints():
        n_star_grid, delta2_grid = np.mgrid[-2.4:-2.2:200j, 0.2:0.65:200j]
        return [
            marginal.gaussian_chi2_McDonald2005(n_star_grid, delta2_grid),
            marginal.gaussian_chi2_PalanqueDelabrouille2015(n_star_grid, delta2_grid),
            marginal.gaussian_chi2_Chabanier2019(n_star_grid, delta2_grid),
            marginal.gaussian_chi2_Walther2024(
                n_star_grid, delta2_grid, ana_type="priors"
            ),
        ]

    def _desi_constraint(self):
        return {
            "x": self.desi_summary["delta2_star_16_50_84"][1]
            - self.blinding["Delta2_star"],
            "xerr": self.desi_summary["delta2_star_err"],
            "y": self.desi_summary["n_star_16_50_84"][1]
            - self.blinding["n_star"],
            "yerr": self.desi_summary["n_star_err"],
            "r": np.corrcoef(
                self.desi_blobs["Delta2_star"].reshape(-1),
                self.desi_blobs["n_star"].reshape(-1),
            )[0, 1],
        }

    @staticmethod
    def _constraint_from_samples(delta2_star, n_star):
        return {
            "x": np.mean(delta2_star),
            "xerr": np.std(delta2_star),
            "y": np.mean(n_star),
            "yerr": np.std(n_star),
            "r": np.corrcoef(delta2_star, n_star)[0, 1],
        }

    @staticmethod
    def _ensure_ranges_periodic(samples):
        if not hasattr(samples.ranges, "periodic"):
            samples.ranges.periodic = set()

    def _require_loaded(self):
        if not self._is_loaded:
            raise RuntimeError("call load_data before plotting or exporting data")
