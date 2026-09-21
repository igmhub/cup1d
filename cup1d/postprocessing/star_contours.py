"""Plot DESI and CMB contours in the linear-power amplitude--slope plane."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from getdist import plots


class StarContourPlotter:
    """Create cumulative CMB and DESI contours in ``Delta2_star``--``n_star``."""

    def __init__(self, fontsize=26):
        self.fontsize = fontsize
        self._is_loaded = False

    def load_data(self, chain_specs, desi_contours_path, planck_root_dir=None):
        """Load Planck chains and the DESI ``line_sigmas.npy`` contour data.

        ``chain_specs`` contains dictionaries with ``model``, ``data``, and
        ``label``. The contours are plotted cumulatively in that order.
        """
        from cup1d.postprocessing.chains import planck

        chains = planck.load_planck_2018_chains(
            chain_specs, root_dir=planck_root_dir
        )
        self.chains = [
            chains[spec.get("name", spec["model"])] for spec in chain_specs
        ]
        self.labels = [spec["label"] for spec in chain_specs]
        self.desi_contours = np.load(Path(desi_contours_path), allow_pickle=True).item()
        self._is_loaded = True
        return self

    def plot_progressive(self, xlim=(0.25, 0.45), ylim=(-2.35, -2.23), save_directory=None):
        """Plot DESI contours plus progressively more CMB extensions.

        Returns a list of ``(figure, axis)`` pairs, one for each cumulative
        subset of the configured CMB chains.
        """
        if not self._is_loaded:
            raise RuntimeError("call load_data before plot_progressive")

        outputs = []
        colors = ["C1", "C2", "C6", "C5", "C7", "black"]
        line_styles = ["--", "-", "--", "-", "--", ":"]
        if save_directory is not None:
            save_directory = Path(save_directory)
            save_directory.mkdir(parents=True, exist_ok=True)

        for number_of_chains in range(1, len(self.chains) + 1):
            plotter = plots.getSinglePlotter(width_inch=10)
            plotter.settings.num_plot_contours = 2
            for index, chain in enumerate(self.chains[:number_of_chains]):
                plotter.plot_2d(
                    chain["samples"].copy(),
                    ["linP_DL2_star", "linP_n_star"],
                    colors=[colors[index]],
                    lws=[3, 2],
                    alphas=[0.8, 0.5],
                    filled=False,
                )

            axis = plotter.subplots[0, 0]
            for collection, line_style in zip(axis.collections, line_styles):
                collection.set_linestyle(line_style)
            self._plot_desi_contours(axis)
            axis.set(xlim=xlim, ylim=ylim)
            axis.set_xlabel(r"$\Delta^2_\star$", fontsize=self.fontsize)
            axis.set_ylabel(r"$n_\star$", fontsize=self.fontsize)
            axis.tick_params(axis="both", which="major", labelsize=self.fontsize)
            self._add_legend(axis, colors, line_styles, number_of_chains)
            if save_directory is not None:
                stem = save_directory / f"star_planck_mine{number_of_chains - 1}"
                axis.figure.savefig(stem.with_suffix(".png"), bbox_inches="tight")
                axis.figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
            outputs.append((axis.figure, axis))
        return outputs

    def get_zenodo_data(self, desi_constraint=None):
        """Return the compact Figure-21 data dictionary for a Zenodo release."""
        if not self._is_loaded:
            raise RuntimeError("call load_data before get_zenodo_data")
        if desi_constraint is None:
            desi_constraint = {
                "x": 0.379,
                "xerr": 0.032,
                "y": -2.309,
                "yerr": 0.019,
                "r": -0.1738,
            }

        figure_data = {"blue": desi_constraint.copy()}
        color_names = ["orange", "green", "pink", "brown", "gray", "black"]
        for color_name, chain in zip(color_names, self.chains):
            samples = chain["samples"]
            delta2_star = np.asarray(samples["linP_DL2_star"])
            n_star = np.asarray(samples["linP_n_star"])
            figure_data[color_name] = {
                "x": np.mean(delta2_star),
                "xerr": np.std(delta2_star),
                "y": np.mean(n_star),
                "yerr": np.std(n_star),
                "r": np.corrcoef(delta2_star, n_star)[0, 1],
            }
        return figure_data

    def save_data_to_zenodo(self, filename="fig_21.npy", desi_constraint=None):
        """Save the Figure-21 summary data under cup1d's Zenodo directory."""
        from cup1d.utils.utils import get_path_repo

        output_path = Path(get_path_repo("cup1d")) / "data" / "zenodo" / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, self.get_zenodo_data(desi_constraint))
        return output_path

    def _plot_desi_contours(self, axis):
        """Draw the 68% and 95% DESI contour polygons."""
        blues = plt.colormaps["Blues"]
        for contour_index, probability in enumerate([0.68, 0.95]):
            label = r"DESI $P_\mathrm{1D}$" if contour_index == 0 else None
            color = blues([0.7, 0.3][contour_index])
            linewidth = [3, 2][contour_index]
            for polygon in self.desi_contours[probability]:
                axis.plot(polygon[0], polygon[1], color=color, label=label, lw=linewidth, alpha=0.75)
                axis.fill(polygon[0], polygon[1], color=color, alpha=0.5)
                label = None

    def _add_legend(self, axis, colors, line_styles, number_of_chains):
        """Add a compact legend matching the plotted contour styles."""
        from matplotlib.lines import Line2D

        handles = [Line2D([], [], color="C0", label=r"DESI $P_\mathrm{1D}$", lw=3)]
        handles.extend(
            Line2D([], [], color=colors[index], label=self.labels[index], lw=3, ls=line_styles[index])
            for index in range(number_of_chains)
        )
        axis.legend(handles=handles, fontsize=self.fontsize - 3, loc="upper left", ncol=2)
