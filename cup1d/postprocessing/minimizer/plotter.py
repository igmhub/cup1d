"""Figures and summary of a minimizer result."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cup1d.postprocessing.minimizer.star import STAR, plot_star_simulations, simulation_star_params
from cup1d.postprocessing.style import apply_house_style


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

    FIGURES = ("star_simulations", "best_fit_p1d", "igm_histories")

    def __init__(self, analysis, out_folder=None, label=None, unblind=False):
        self.analysis = analysis
        self.fitter = analysis.fitter
        self.unblind = unblind
        folder = self.fitter.save_directory
        self.out_folder = Path(out_folder or folder or ".")
        variation = getattr(analysis.args, "name_variation", None)
        self.label = label or (variation if variation not in (None, "None") else
                               (Path(folder).name if folder else "best fit"))
        apply_house_style()

    @classmethod
    def from_results(cls, path, **kwargs):
        """Load ``minimizer_results.npy`` (or the ``chain_N`` folder holding it)."""
        from cup1d.inference.analysis import Analysis

        path = Path(path).expanduser()
        if path.is_dir():
            path = path / "minimizer_results.npy"
        if not path.is_file():
            raise FileNotFoundError(f"no minimizer result: {path} does not exist")
        return cls(Analysis.from_results(path), **kwargs)

    @property
    def chi2(self):
        return float(self.fitter.mle_chi2)

    @property
    def star(self):
        """Best-fitting star parameters (blinded unless ``unblind``)."""
        star = {name: float(self.fitter.mle_cosmo[name]) for name in STAR}
        if self.unblind:
            star = self.fitter.like.blind.unblind(star)
        return star

    @property
    def star_covariance(self):
        """Gauss--Newton covariance of (Delta2_star, n_star, alpha_star).

        With the flat priors as Gaussians (the default of
        ``Fitter.estimate_mle_errors``); results saved without them are
        recomputed.
        """
        if not (hasattr(self.fitter, "mle_cosmo_covariance")
                and getattr(self.fitter, "mle_flat_prior_curvature", False)):
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
        """Parameters within ``fraction`` of a flat-prior edge at the best fit
        (:meth:`cup1d.inference.fitter.Fitter.near_prior_bounds`)."""
        return self.fitter.near_prior_bounds(fraction=fraction)

    def summary(self):
        """Text: chi2, star parameters with Gauss--Newton errors, prior edges."""
        errors = np.sqrt(np.diag(self.star_covariance))
        lines = [f"{self.label}: chi2 = {self.chi2:.2f}, "
                 f"{'unblinded' if self.unblind else 'blinded'} star parameters, "
                 "Gauss-Newton errors with flat priors as Gaussians"]
        lines += [f"  {name:12s} {self.star[name]: .5f} +- {err:.5f}" for name, err in zip(STAR, errors)]
        edges = self.near_bounds()
        lines.append("  near prior edges: " + (", ".join(f"{r['name']} ({r['side']}, {r['distance']:.1%})"
                                                         for r in edges) if edges else "none"))
        return "\n".join(lines)

    @property
    def point(self):
        """Best-fitting physical point ``{name: value}``."""
        return self.fitter.parameters_from_sampling_point(self.fitter.mle_cube)

    def truth_entry(self, marker="*"):
        """True star parameters of a mock or forecast (entry of :func:`plot_star_simulations`), or None."""
        truth = getattr(self.fitter.like, "truth", None)
        if not truth or "linP" not in truth:
            return None
        star = {name: float(truth["linP"][name]) for name in STAR if name in truth["linP"]}
        return dict(label="truth", star=star, marker=marker, color="C3")

    def star_simulations(self, others=(), nsigma=(1, 2), annotate=True, truth=True):
        """Delta2_star--n_star: training simulations, this and other best fits.

        ``others``: further :class:`MinimizerPlotter` (e.g. previous runs),
        drawn with their own errors, or entries of :func:`plot_star_simulations`
        (e.g. :func:`legacy_fit_entry` of a DR1 paper run).
        """
        simulations = simulation_star_params(self.fitter.like.theory.emulator)
        fits = [self.fit_entry(color="C1")]
        if truth and self.truth_entry() is not None:
            fits.append(self.truth_entry())
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

    def igm_histories(self, quantities=("tau_eff", "gamma", "sigT_kms", "kF_kms"), truth=True,
                      literature=None, prior_envelope=False, training=False, n_z=100):
        """IGM histories of the best fit (with nodes), optionally with the truth of a
        mock, the literature, the prior envelope and the training simulations.

        ``literature`` defaults to showing it only when there is no truth.
        """
        from cup1d.postprocessing.fit.igm import igm_at_point, igm_nodes, igm_prior_envelope
        from cup1d.postprocessing.posterior.histories import plot_igm_histories

        like = self.fitter.like
        data = next(iter(like.data.values()))
        z = np.linspace(min(data.z), max(data.z), n_z)
        best = igm_at_point(like.theory.model_igm, self.point, z)
        mock_truth = getattr(like, "truth", None) or {}
        truth_igm = mock_truth.get("igm") if truth else None
        if literature is None:
            literature = truth_igm is None
        return plot_igm_histories(
            None, z, best_fit=best, best_fit_label=self.label, quantities=quantities,
            best_fit_nodes=igm_nodes(best, z, self.analysis.args.fid_igm, quantities),
            literature=literature, truth=truth_igm,
            prior_envelope=igm_prior_envelope(like, z) if prior_envelope else None,
            training=like.theory.emu_igm_all if training else None,
        )

    def plot_all(self, others=(), skip=()):
        """Write every figure (png and pdf) and ``minimizer_summary.txt``; return the summary text."""
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
        return text


def legacy_fit_entry(folder, label=None, unblind=False):
    """Best fit of a DR1-paper run (``fitter_results.npy``), without errors.

    Its star parameters are blinded with the DR1 offsets
    (``data/blinding_dr1.npy``), removed with ``unblind``.
    """
    fit = np.load(Path(folder) / "fitter_results.npy", allow_pickle=True).item()["fitter"]
    star = {name: float(fit["mle_cosmo"][name]) for name in STAR}
    if unblind:
        from cup1d.postprocessing.literature import load_blinding_dr1
        from cup1d.utils.blinding import Blinding

        star = Blinding(load_blinding_dr1()).unblind(star)
    chi2 = -2 * float(fit["lnprob_mle"])
    return dict(label=f"{label or Path(folder).name} ($\\chi^2$ = {chi2:.1f})", star=star, marker="s")
