"""Load and compare predefined CUP1D variation chains."""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cup1d.utils.utils import get_path_repo


@dataclass(frozen=True)
class _ChainSpec:
    """Location and display information for one variation result."""

    label: str
    relative_path: str
    display_label: str
    simulation_label: str | None = None


class VariationPlotter:
    """Load and plot standard DR1 analysis-variation contours.

    Parameters
    ----------
    output_root : path-like, optional
        Root directory containing ``out_DESI_DR1``. By default it uses the
        shared project data directory.
    prior_plotter : EmulatorPriorPlotter, optional
        Precomputed emulator prior domains. It is only needed when displaying
        the shaded prior region for observational comparisons.
    """

    _SPECS = {
        "baseline": _ChainSpec(
            "baseline", "DESIY1_QMLE3/global_opt/CH24_mpgcen_gpr/chain_7", "Baseline"
        ),
        "nyx": _ChainSpec(
            "nyx",
            "DESIY1_QMLE3/global_opt/CH24_nyxcen_gpr/chain_3",
            "Emulator: lace-lyssa",
        ),
        "qmle": _ChainSpec(
            "qmle", "DESIY1_QMLE/global_opt/CH24_mpgcen_gpr/chain_2", "Data: w/ low SNR"
        ),
        "fft3": _ChainSpec(
            "fft3",
            "DESIY1_FFT3_dir/global_opt/CH24_mpgcen_gpr/chain_2",
            "Chaves-Montero+26",
        ),
        "fft3_tan": _ChainSpec(
            "fft3_tan", "DESIY1_FFT3_dir/DLA_TAN/CH24_mpgcen_gpr/chain_2", "This work"
        ),
        "zmin": _ChainSpec(
            "zmin", "DESIY1_QMLE3/zmin/CH24_mpgcen_gpr/chain_2", r"Data: $z \geq 2.6$"
        ),
        "zmax": _ChainSpec(
            "zmax", "DESIY1_QMLE3/zmax/CH24_mpgcen_gpr/chain_2", r"Data: $z \leq 3.4$"
        ),
        "no_inflate": _ChainSpec(
            "no_inflate",
            "DESIY1_QMLE3/no_inflate/CH24_mpgcen_gpr/chain_2",
            "Cov: w/o 5% err",
        ),
        "no_emu_cov": _ChainSpec(
            "no_emu_cov",
            "DESIY1_QMLE3/no_emu_cov/CH24_mpgcen_gpr/chain_2",
            "Cov: w/o emu",
        ),
        "emu_diag": _ChainSpec(
            "emu_diag", "DESIY1_QMLE3/emu_diag/CH24_mpgcen_gpr/chain_3", "Cov: emu diag"
        ),
        "emu_block": _ChainSpec(
            "emu_block",
            "DESIY1_QMLE3/emu_block/CH24_mpgcen_gpr/chain_3",
            "Cov: emu block diag",
        ),
        "data_syst_diag": _ChainSpec(
            "data_syst_diag",
            "DESIY1_QMLE3/data_syst_diag/CH24_mpgcen_gpr/chain_2",
            "Cov: uncorr syst",
        ),
        "more_igm": _ChainSpec(
            "more_igm", "DESIY1_QMLE3/more_igm/CH24_mpgcen_gpr/chain_2", "IGM: $n_z=8$"
        ),
        "cosmo": _ChainSpec(
            "cosmo", "DESIY1_QMLE3/cosmo/CH24_mpgcen_gpr/chain_3", r"Cosmo: $w_0w_a$CDM"
        ),
        "cosmo_low": _ChainSpec(
            "cosmo_low",
            "DESIY1_QMLE3/cosmo_low_3sig/CH24_mpgcen_gpr/chain_1",
            r"Cosmo: low $\Omega_{\rm cdm}h^2$",
        ),
        "cosmo_high": _ChainSpec(
            "cosmo_high",
            "DESIY1_QMLE3/cosmo_high_3sig/CH24_mpgcen_gpr/chain_1",
            r"Cosmo: high $\Omega_{\rm cdm}h^2$",
        ),
        "cosmo_h74": _ChainSpec(
            "cosmo_h74",
            "DESIY1_QMLE3/cosmo_h74/CH24_mpgcen_gpr/chain_1",
            "Cosmo: $h=0.74$",
        ),
        "cosmo_mnu": _ChainSpec(
            "cosmo_mnu",
            "DESIY1_QMLE3/cosmo_mnu_varh/CH24_mpgcen_gpr/chain_2",
            r"Cosmo: $\sum m_\nu=0.3$ eV",
        ),
        "dlas": _ChainSpec(
            "dlas", "DESIY1_QMLE3/DLAs/CH24_mpgcen_gpr/chain_2", "HCD: only DLAs"
        ),
        "hcd0": _ChainSpec(
            "hcd0",
            "DESIY1_QMLE3/HCD0/CH24_mpgcen_gpr/chain_2",
            r"HCD: w/ $f_{\rm const}^{\rm HCD}$",
        ),
        "hcd_boss": _ChainSpec(
            "hcd_boss", "DESIY1_QMLE3/HCD_BOSS/CH24_mpgcen_gpr/chain_2", "HCD: simple"
        ),
        "metal_deco": _ChainSpec(
            "metal_deco",
            "DESIY1_QMLE3/metal_deco/CH24_mpgcen_gpr/chain_2",
            "Metals: no H-Si decorr",
        ),
        "metal_si2": _ChainSpec(
            "metal_si2",
            "DESIY1_QMLE3/metal_si2/CH24_mpgcen_gpr/chain_2",
            "Metals: no SiII-SiII",
        ),
        "metal_trad": _ChainSpec(
            "metal_trad",
            "DESIY1_QMLE3/metal_trad/CH24_mpgcen_gpr/chain_2",
            "Metals: simple",
        ),
        "metal_thin": _ChainSpec(
            "metal_thin",
            "DESIY1_QMLE3/metal_thin/CH24_mpgcen_gpr/chain_2",
            "Metals: opt thin",
        ),
        "metals_ma2025": _ChainSpec(
            "metals_ma2025",
            "DESIY1_QMLE3/Metals_Ma2025/CH24_mpgcen_gpr/chain_5",
            "Metals: Ma+2026",
        ),
        "sim_mpg_central": _ChainSpec(
            "sim_mpg_central",
            "DESIY1_QMLE3/sim_mpg_central/CH24_mpgcen_gpr/chain_3",
            "mpg-central",
            "mpg_central",
        ),
        "sim_mpg_seed": _ChainSpec(
            "sim_mpg_seed",
            "DESIY1_QMLE3/sim_mpg_seed/CH24_mpgcen_gpr/chain_3",
            "mpg-seed",
            "mpg_seed",
        ),
        "sim_nyx_central": _ChainSpec(
            "sim_nyx_central",
            "DESIY1_QMLE3/sim_nyx_central/CH24_mpgcen_gpr/chain_2",
            "lyssa-central",
            "nyx_central",
        ),
        "sim_sherwood": _ChainSpec(
            "sim_sherwood",
            "DESIY1_QMLE3/sim_sherwood/CH24_mpgcen_gpr/chain_1",
            "sherwood",
            "sherwood",
        ),
        "sim_mpg_central_igm": _ChainSpec(
            "sim_mpg_central_igm",
            "DESIY1_QMLE3/sim_mpg_central_igm/CH24_mpgcen_gpr/chain_2",
            "Model: cosmo, IGM",
            "mpg_central",
        ),
        "sim_mpg_central_igm0": _ChainSpec(
            "sim_mpg_central_igm0",
            "DESIY1_QMLE3/sim_mpg_central_igm0/CH24_mpgcen_gpr/chain_2",
            "Model: cosmo",
            "mpg_central",
        ),
    }
    GROUPS = {
        "data": ("baseline", "qmle", "fft3"),
        "tan": ("fft3", "fft3_tan"),
        "redshift": ("baseline", "zmin", "zmax"),
        "data_covariance": ("baseline", "no_inflate", "data_syst_diag"),
        "emulator_covariance": ("baseline", "emu_block", "emu_diag", "no_emu_cov"),
        "cosmology": ("baseline", "cosmo", "cosmo_h74", "cosmo_mnu"),
        "cosmology_asns": ("baseline", "cosmo", "cosmo_h74", "cosmo_mnu"),
        "metals": ("baseline", "metal_deco", "metal_thin", "metal_si2"),
        "metal_models": ("baseline", "metal_trad", "metals_ma2025"),
        "hcd": ("baseline", "hcd0", "dlas", "hcd_boss"),
        "igm": ("baseline", "more_igm"),
        "emulator": ("baseline", "nyx"),
        "simulations": (
            "sim_mpg_central",
            "sim_mpg_seed",
            "sim_nyx_central",
            "sim_sherwood",
        ),
        "simulation_model": (
            "sim_mpg_central",
            "sim_mpg_central_igm",
            "sim_mpg_central_igm0",
        ),
        "cosmology_bounds": ("baseline", "cosmo_low", "cosmo_high"),
        "cosmology_bounds_asns": ("baseline", "cosmo_low", "cosmo_high"),
    }

    def __init__(self, output_root=None, prior_plotter=None, blinding=None):
        if output_root is None:
            output_root = Path(get_path_repo("cup1d")).parent / "data" / "out_DESI_DR1"
        self.output_root = Path(output_root)
        self.prior_plotter = prior_plotter
        if blinding is None:
            blinding_path = Path(get_path_repo("cup1d")) / "data" / "blinding_dr1.npy"
            blinding = np.load(blinding_path, allow_pickle=True).item()
        self.blinding = blinding
        self._contours = {}

    @classmethod
    def available_groups(cls):
        """Return a dictionary of named comparison groups and their members."""

        return {name: list(members) for name, members in cls.GROUPS.items()}

    def plot(
        self,
        groups,
        axes=None,
        fontsize=22,
        save_figures=False,
        save_directory=None,
    ):
        """Plot one or more named comparison groups.

        ``groups`` may be one group name or a list. When multiple names are
        supplied, one figure is created for each group. Set ``save_figures``
        to ``True`` to write PDF and PNG versions. By default no files are
        written. Without an explicit ``save_directory``, files are written to
        ``figs/variations`` below the current working directory.
        """

        if isinstance(groups, str):
            groups = [groups]
        unknown = set(groups).difference(self.GROUPS)
        if unknown:
            raise ValueError(f"Unknown variation groups: {sorted(unknown)}")

        if axes is None:
            import matplotlib.pyplot as plt

            _, axes = plt.subplots(
                1, len(groups), figsize=(8 * len(groups), 6), squeeze=False
            )
            axes = axes[0]
        elif len(groups) == 1:
            axes = [axes]

        for group, axis in zip(groups, axes, strict=True):
            self._plot_group(group, axis, fontsize)
        if save_figures:
            if save_directory is None:
                save_directory = Path.cwd() / "figs" / "variations"
            self._save_figures(groups, axes, save_directory)
        return axes[0] if len(groups) == 1 else axes

    def _load_contour(self, name, as_ns=False):
        key = (name, as_ns)
        if key not in self._contours:
            spec = self._SPECS[name]
            filename = "line_sigmas_Asns.npy" if as_ns else "line_sigmas.npy"
            path = self.output_root / spec.relative_path / filename
            if not path.is_file():
                raise FileNotFoundError(f"Missing variation contour: {path}")
            self._contours[key] = np.load(path, allow_pickle=True).item()
        contour = deepcopy(self._contours[key])
        if name == "baseline" and not as_ns:
            self._reintroduce_blinding(contour)
        return contour

    def _plot_group(self, group, ax, fontsize):
        import matplotlib.pyplot as plt

        names = self.GROUPS[group]
        as_ns = group.endswith("_asns")
        contours = [self._load_contour(name, as_ns=as_ns) for name in names]
        reference_x, reference_y = self._reference_point(contours[0], names[0])
        if not as_ns:
            self._add_prior_region(ax, names, reference_x, reference_y)

        x_limits, y_limits = [np.inf, -np.inf], [np.inf, -np.inf]
        line_styles = ["-", "--", ":", "-."]
        hatches = ["", "/", "\\", "|"]
        colormaps = ["Blues", "Oranges", "Greens", "Purples"]
        for index, (name, contour) in enumerate(zip(names, contours, strict=True)):
            x_offset, y_offset = self._reference_point(
                contour, name, default=(reference_x, reference_y)
            )
            cmap = plt.colormaps[colormaps[index]]
            for level_index, level in enumerate((0.68, 0.95)):
                for contour_index, values in enumerate(contour[level]):
                    x = values[0] - x_offset
                    if as_ns:
                        x *= 1e9
                    y = values[1] - y_offset
                    label = (
                        self._SPECS[name].display_label
                        if level_index == 0 and contour_index == 0
                        else None
                    )
                    ax.plot(
                        x,
                        y,
                        color=cmap((0.7, 0.3)[level_index]),
                        lw=(3, 2)[level_index],
                        ls=line_styles[index],
                        alpha=0.75,
                        label=label,
                    )
                    ax.fill(
                        x,
                        y,
                        color=cmap((0.7, 0.3)[level_index]),
                        alpha=0.3,
                        hatch=hatches[index],
                    )
                    x_limits = [min(x_limits[0], x.min()), max(x_limits[1], x.max())]
                    y_limits = [min(y_limits[0], y.min()), max(y_limits[1], y.max())]

        self._set_axes(ax, x_limits, y_limits, as_ns, fontsize)
        ax.legend(
            fontsize=fontsize - 6,
            loc="lower right" if group == "data" else "upper right",
            frameon=False,
            ncol=2 if group == "simulations" else 1,
        )

    def _reference_point(self, contour, name, default=None):
        spec = self._SPECS[name]
        if spec.simulation_label is None:
            if default is not None:
                return default
            return tuple(np.median(contour[0.68][0][axis]) for axis in (0, 1))

        from cup1d.theory.camb import CAMBModel
        from cup1d.theory.cosmology import set_cosmo

        star = CAMBModel(
            np.asarray([3.0]), cosmo=set_cosmo(spec.simulation_label)
        ).get_linP_params()
        return star["Delta2_star"], star["n_star"]

    def _add_prior_region(self, ax, names, reference_x, reference_y):
        if self.prior_plotter is None or any(
            self._SPECS[name].simulation_label for name in names
        ):
            return
        if not self.prior_plotter.boundaries:
            self.prior_plotter.compute_priors()
        from matplotlib.path import Path
        from matplotlib.patches import PathPatch

        for emulator, color in (("mpg", "0.5"), ("nyx", "0.3")):
            if emulator == "nyx" and "nyx" not in names:
                continue
            boundary = self.prior_plotter.boundaries[emulator].copy()
            boundary[:, 0] += self.blinding["Delta2_star"] - reference_x
            boundary[:, 1] += self.blinding["n_star"] - reference_y
            xmin, ymin = boundary.min(axis=0)
            xmax, ymax = boundary.max(axis=0)
            outer = np.array(
                [[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin], [xmin, ymin]]
            )
            vertices = np.concatenate((outer, boundary))
            codes = (
                [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
                + [Path.MOVETO]
                + [Path.LINETO] * (len(boundary) - 2)
                + [Path.CLOSEPOLY]
            )
            ax.add_patch(
                PathPatch(
                    Path(vertices, codes), facecolor=color, edgecolor="none", alpha=0.5
                )
            )
            ax.fill(boundary[:, 0], boundary[:, 1], "white")

    def _reintroduce_blinding(self, contour):
        """Restore the DR1 offsets in the baseline star-parameter contours."""

        for level in (0.68, 0.95):
            values = list(contour[level][0])
            values[0] = values[0] + self.blinding["Delta2_star"]
            values[1] = values[1] + self.blinding["n_star"]
            contour[level][0] = tuple(values)

    @staticmethod
    def _set_axes(ax, x_limits, y_limits, as_ns, fontsize):
        x_range, y_range = x_limits[1] - x_limits[0], y_limits[1] - y_limits[0]
        ax.set_xlim(x_limits[0] - 0.05 * x_range, x_limits[1] + 0.05 * x_range)
        ax.set_ylim(y_limits[0] - 0.05 * y_range, y_limits[1] + 0.05 * y_range)
        ax.set_xlabel(
            r"$\Delta A_s\,[10^{-9}]$" if as_ns else r"$\Delta(\Delta^2_\star)$",
            fontsize=fontsize + 2,
        )
        ax.set_ylabel(
            r"$\Delta n_s$" if as_ns else r"$\Delta n_\star$", fontsize=fontsize + 2
        )
        ax.tick_params(axis="both", which="major", labelsize=fontsize - 2)
        ax.axhline(0, color="k", ls=":")
        ax.axvline(0, color="k", ls=":")

    @staticmethod
    def _save_figures(groups, axes, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for group, axis in zip(groups, axes, strict=True):
            axis.figure.savefig(directory / f"variations_{group}.pdf")
            axis.figure.savefig(directory / f"variations_{group}.png")
