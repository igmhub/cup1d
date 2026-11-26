"""Figures and summary of a minimizer result."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cup1d.postprocessing.minimizer.star import STAR, plot_star_simulations, simulation_star_params


class MinimizerPlotter:
    """Figures of one best fit (``minimizer_results.npy``), optionally with others.

    Parameters
    ----------
    analysis : cup1d.inference.analysis.Analysis
        Analysis whose fitter holds the best fit (e.g. from
        :meth:`from_results`, or right after ``run_minimizer``).
    out_folder : str or path-like, optional
        Where :meth:`plot_all` writes the figures (default: the chain folder).
    label : str, optional
        Legend label (default: the variation, or the chain folder name).
    unblind : bool, default: False
        Remove the blinding offsets of the star parameters (only for
        unblinded data, e.g. DESI DR1). Blinded values of different runs of
        the same data share the offsets, so their differences are meaningful.
    """

    FIGURES = ("star_simulations", "best_fit_p1d")

    def __init__(self, analysis, out_folder=None, label=None, unblind=False):
        self.analysis = analysis
        self.fitter = analysis.fitter
        self.unblind = unblind
        folder = self.fitter.save_directory
        self.out_folder = Path(out_folder or folder or ".")
        variation = getattr(analysis.args, "name_variation", None)
        self.label = label or (variation if variation not in (None, "None") else
                               (Path(folder).name if folder else "best fit"))

    @classmethod
    def from_results(cls, path, **kwargs):
        """Load ``minimizer_results.npy`` (or the ``chain_N`` folder holding it)."""
        from cup1d.inference.analysis import Analysis

        path = Path(path)
        if path.is_dir():
            path = path / "minimizer_results.npy"
        return cls(Analysis.from_results(path), **kwargs)

    @property
    def chi2(self):
        return float(self.fitter.mle_chi2)

    @property
    def star(self):
        """Best-fitting star parameters (blinded unless ``unblind``)."""
        star = {name: float(self.fitter.mle_cosmo[name]) for name in STAR}
        if self.unblind:
            star = {name: star[name] - self.fitter.like.blind[name] for name in STAR}
        return star

    @property
    def star_covariance(self):
        """Gauss--Newton covariance of (Delta2_star, n_star, alpha_star)."""
        if not hasattr(self.fitter, "mle_cosmo_covariance"):
            self.fitter.estimate_mle_errors(method="gauss_newton")
        return np.asarray(self.fitter.mle_cosmo_covariance)

    def fit_entry(self, color=None, marker="X"):
        """This best fit in the format of :func:`plot_star_simulations`."""
        entry = dict(label=f"{self.label} ($\\chi^2$ = {self.chi2:.1f})", star=self.star,
                     covariance=self.star_covariance, marker=marker)
        if color is not None:
            entry["color"] = color
        return entry

    def near_bounds(self, fraction=0.02):
        """Parameters within ``fraction`` of a flat prior edge (unit cube)."""
        cube = np.asarray(self.fitter.mle_cube)
        names = list(self.fitter.like.free_param_names)
        return [(n, c) for n, c in zip(names, cube) if c < fraction or c > 1 - fraction]

    def summary(self):
        """Text: chi2, star parameters with Gauss--Newton errors, prior edges."""
        errors = np.sqrt(np.diag(self.star_covariance))
        lines = [f"{self.label}: chi2 = {self.chi2:.2f}, "
                 f"{'unblinded' if self.unblind else 'blinded'} star parameters"]
        lines += [f"  {name:12s} {self.star[name]: .5f} +- {err:.5f}" for name, err in zip(STAR, errors)]
        edges = self.near_bounds()
        lines.append("  near prior edges: " + (", ".join(f"{n} ({c:.3f})" for n, c in edges) if edges else "none"))
        return "\n".join(lines)

    def star_simulations(self, others=(), nsigma=(1, 2), annotate=True):
        """Delta2_star--n_star: training simulations, this and other best fits.

        ``others``: further :class:`MinimizerPlotter` (e.g. previous runs),
        drawn with their own errors, or entries of :func:`plot_star_simulations`
        (e.g. :func:`legacy_fit_entry` of a DR1 paper run).
        """
        simulations = simulation_star_params(self.fitter.like.theory.emulator)
        fits = [self.fit_entry(color="C1")]
        for i, other in enumerate(others):
            entry = other.fit_entry(color=f"C{i + 2}", marker="P") if hasattr(other, "fit_entry") else dict(other)
            entry.setdefault("color", f"C{i + 2}")
            fits.append(entry)
        return plot_star_simulations(simulations, fits, nsigma=nsigma, annotate=annotate)

    def best_fit_p1d(self):
        """Best-fitting model against the data, one row per data set."""
        from cup1d.postprocessing.p1d import P1DPlotter
        from cup1d.postprocessing.posterior.p1d import plot_best_fit

        point = self.fitter.parameters_from_sampling_point(self.fitter.mle_cube)
        p1d = P1DPlotter(self.fitter.like)
        bins, chi2, dof = p1d.prepare(point)
        return plot_best_fit(bins, chi2, dof, ndata=p1d.ndata, title=self.label)

    def plot_all(self, others=(), skip=()):
        """Write every figure (png and pdf) and ``minimizer_summary.txt``."""
        self.out_folder.mkdir(parents=True, exist_ok=True)
        for name in self.FIGURES:
            if name in skip:
                continue
            fig = self.star_simulations(others=others) if name == "star_simulations" else getattr(self, name)()
            for ext in ("png", "pdf"):
                fig.savefig(self.out_folder / f"minimizer_{name}.{ext}", bbox_inches="tight")
            plt.close(fig)
        text = "\n\n".join(p.summary() for p in (self, *others) if hasattr(p, "summary"))
        (self.out_folder / "minimizer_summary.txt").write_text(text + "\n")
        print(text)
        print(f"figures in {self.out_folder}")


def legacy_fit_entry(folder, label=None, unblind=False):
    """Best fit of a DR1-paper run (``fitter_results.npy``), without errors.

    Its star parameters are blinded with the DR1 offsets
    (``data/blinding_dr1.npy``), removed with ``unblind``.
    """
    import os

    from cup1d.utils.utils import get_path_repo

    fit = np.load(Path(folder) / "fitter_results.npy", allow_pickle=True).item()["fitter"]
    star = {name: float(fit["mle_cosmo"][name]) for name in STAR}
    if unblind:
        blind = np.load(os.path.join(get_path_repo("cup1d"), "data", "blinding_dr1.npy"),
                        allow_pickle=True).item()
        star = {name: star[name] - blind[name] for name in STAR}
    chi2 = -2 * float(fit["lnprob_mle"])
    return dict(label=f"{label or Path(folder).name} ($\\chi^2$ = {chi2:.1f})", star=star, marker="s")
