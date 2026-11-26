"""Figures of a minimizer result (the posterior counterpart is
:mod:`cup1d.postprocessing.posterior`).

Modules
-------
star
    Best-fitting Delta2_star--n_star with Gauss--Newton errors against the
    emulator training simulations.
plotter
    :class:`MinimizerPlotter`: figures and summary of ``minimizer_results.npy``,
    optionally compared with other best fits.
"""

from matplotlib import rcParams

from cup1d.postprocessing.minimizer.plotter import MinimizerPlotter, legacy_fit_entry
from cup1d.postprocessing.minimizer.star import plot_star_simulations, simulation_star_params

rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "STIXGeneral"

__all__ = ["MinimizerPlotter", "legacy_fit_entry", "plot_star_simulations", "simulation_star_params"]
