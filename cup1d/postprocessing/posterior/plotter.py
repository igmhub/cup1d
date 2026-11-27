"""One object that writes every posterior figure of a run."""

from pathlib import Path
import traceback

import matplotlib.pyplot as plt
import numpy as np

from cup1d.postprocessing.style import apply_house_style

from cup1d.postprocessing.posterior import contaminants as cont
from cup1d.postprocessing.posterior import histories, parameters, star
from cup1d.postprocessing.posterior.p1d import plot_best_fit


class PosteriorPlotter:
    """Write the standard figures of a posterior to a folder.

    The figures only need weighted samples and, for the derived histories,
    the static cup1d IGM and contaminant models; no emulator or likelihood is
    evaluated, except for the best-fit P1D (whose bins the caller computes).

    Parameters
    ----------
    samples : PosteriorSamples
        Posterior in native cup1d names.
    out_folder : str or path-like
        Output folder (created if needed).
    igm : object, optional
        cup1d IGM model (IGM histories, mean flux of the metal terms).
    contaminants : Contaminants, optional
        cup1d contaminant model (metal and HCD figures).
    syst_znodes : mapping, optional
        Systematics family (e.g. ``"R_coeff"``) to its node redshifts.
    z_range : tuple of float, default=(2.2, 4.2)
        Redshift range of the IGM histories.
    k_range : tuple of float, default=(1e-3, 0.04)
        Wavenumber range (s/km) of the contamination figures.
    M_of_z : callable, optional
        ``H(z)/(1+z)`` in km/s/Mpc, to show the DESI DR1 temperature and
        pressure scale (whose samples are stored in Mpc units).
    best_fit : mapping, optional
        Best-fitting point in native names, drawn on the histories.
    n_model_samples : int, default=4000
        Posterior draws used for the derived histories and contamination.
    seed : int, default=0
        Seed of those draws.
    fontsize : float, default=16
        Base font size.
    formats : sequence of str, default=("png", "pdf")
        File formats written.
    log : callable, default=print
        Progress messages.
    """

    def __init__(self, samples, out_folder, igm=None, contaminants=None, syst_znodes=None,
                 z_range=(2.2, 4.2), k_range=(1e-3, 0.04), M_of_z=None, best_fit=None,
                 n_model_samples=4000, seed=0, fontsize=16, formats=("png", "pdf"), log=print):
        self.samples = samples
        self.out = Path(out_folder)
        self.out.mkdir(parents=True, exist_ok=True)
        self.igm = igm
        self.contaminants = contaminants
        self.syst_znodes = syst_znodes or {}
        self.z_range = z_range
        self.k_range = k_range
        self.M_of_z = M_of_z
        self.best_fit = best_fit
        self.n_model_samples = n_model_samples
        self.seed = seed
        self.fontsize = fontsize
        self.formats = tuple(formats)
        self.log = log
        self._draws = None
        apply_house_style()

    @classmethod
    def from_analysis(cls, analysis, out_folder=None, samples=None, thin=1, **kwargs):
        """Plotter of a native cup1d run, with the models of its likelihood.

        Parameters
        ----------
        analysis : cup1d.inference.analysis.Analysis
            Analysis whose sampler wrote ``analysis.fitter.save_directory``
            (or rebuilt with :meth:`~cup1d.inference.analysis.Analysis.from_results`).
        out_folder : str or path-like, optional
            Output folder (default: the chain folder).
        samples : PosteriorSamples, optional
            Samples (default: read from the chain folder, star parameters as
            stored).
        thin : int, default=1
            Thinning when reading the samples.
        **kwargs
            Further :class:`PosteriorPlotter` options.
        """
        from cup1d.postprocessing.posterior.samples import PosteriorSamples

        folder = analysis.fitter.save_directory
        if samples is None:
            samples = PosteriorSamples.from_cup1d_folder(folder, thin=thin,
                                                         blinded=analysis.like.blind.is_blinded)
        like = analysis.like
        data = next(iter(like.data.values()))
        znodes = {name[: -len("_znodes")]: values for name, values in analysis.args.fid_syst.items()
                  if name.endswith("_znodes")}
        kwargs.setdefault("z_range", (float(min(data.z)), float(max(data.z))))
        kwargs.setdefault("syst_znodes", znodes)
        return cls(samples, out_folder or folder, igm=like.theory.model_igm,
                   contaminants=like.theory.model_cont, **kwargs)

    @property
    def draws(self):
        """Unit-weight posterior draws for the derived quantities (cached)."""
        if self._draws is None:
            self._draws = self.samples.subsample(self.n_model_samples, self.seed)
        return self._draws

    def save(self, fig, name):
        """Save a figure in every format, close it, and return the paths."""
        paths = []
        for fmt in self.formats:
            path = self.out / f"{name}.{fmt}"
            fig.savefig(path, bbox_inches="tight", dpi=150 if fmt == "png" else None)
            paths.append(path)
        plt.close(fig)
        self.log(f"wrote {self.out / name}.{{{','.join(self.formats)}}}")
        return paths

    # ----- individual figures -------------------------------------------------

    def star_posterior(self):
        """``star_posterior``: Delta2_star and n_star with 16/50/84 titles."""
        return self.save(star.plot_star_posterior(self.samples, fontsize=self.fontsize), "star_posterior")

    def star_literature(self):
        """``star_literature``: comparison with SDSS/BOSS/eBOSS and DESI DR1 (Gaussian)."""
        if self.samples.blinded:
            self.log("star_literature skipped: the star parameters are blinded (unblind to compare)")
            return []
        return self.save(star.plot_star_literature(self.samples, fontsize=self.fontsize),
                         "star_literature")

    def parameters_vs_star(self, max_columns=6, include_cosmology=False):
        """``params_vs_star_<group>``: every parameter group against the star parameters."""
        paths = []
        for group, names in parameters.group_parameters(self.samples, include_cosmology=include_cosmology).items():
            chunks = [names[i:i + max_columns] for i in range(0, len(names), max_columns)]
            for n, chunk in enumerate(chunks):
                suffix = f"_{n}" if len(chunks) > 1 else ""
                slug = "".join(c if c.isalnum() else "_" for c in group.replace("$", "")).strip("_").lower()
                fig = parameters.plot_parameters_vs_star(self.samples, chunk, title=group, fontsize=self.fontsize)
                paths += self.save(fig, f"params_vs_star_{slug}{suffix}")
        return paths

    def correlation(self):
        """``correlation_matrix``: weighted correlations of all parameters."""
        return self.save(parameters.plot_correlation_matrix(self.samples), "correlation_matrix")

    def systematics(self):
        """``systematics_vs_z``: systematics coefficients at their node redshifts."""
        families = {f: z for f, z in self.syst_znodes.items() if self.samples.family(f)}
        if not families:
            self.log("systematics_vs_z skipped: no systematics with redshift nodes")
            return []
        fig = histories.plot_parameters_vs_z(self.samples, families, best_fit=self.best_fit,
                                             fontsize=self.fontsize)
        return self.save(fig, "systematics_vs_z")

    def igm_histories(self, n_z=60, include_dr1=True):
        """``igm_histories``: mean flux, T0, gamma and k_F against redshift."""
        if self.igm is None:
            self.log("igm_histories skipped: no IGM model given")
            return []
        z = np.linspace(*self.z_range, n_z)
        bands = histories.igm_histories(self.igm, self.draws.columns(), z)
        best = None
        if self.best_fit is not None:
            best = histories.igm_histories(self.igm, {k: [v] for k, v in self.best_fit.items()}, z)
        dr1 = None
        if include_dr1:
            try:
                dr1 = histories.load_dr1_igm(self.M_of_z)
            except ImportError as error:
                self.log(f"igm_histories: DESI DR1 IGM not shown ({error})")
        fig = histories.plot_igm_histories(bands, z, best_fit=best, dr1=dr1,
                                           label=self.samples.label or "This work",
                                           fontsize=self.fontsize)
        return self.save(fig, "igm_histories")

    def _k_grid(self, n_k=300):
        return np.linspace(*self.k_range, n_k)

    def metals(self, z=3.0):
        """``metals_z<z>``: multiplicative (and additive) silicon contamination."""
        if self.contaminants is None or self.igm is None:
            self.log("metals skipped: no contaminant or IGM model given")
            return []
        k = self._k_grid()
        components = cont.metal_contamination(self.contaminants, self.igm, self.draws.columns(), z, k)
        ylabel = {label: r"$C_\mathrm{metal}$" for label in components}
        ylabel["SiII-SiII (additive)"] = r"$P_\mathrm{SiII-SiII}\,[\mathrm{km\,s^{-1}}]$"
        fig = cont.plot_contamination(components, k, ylabel=ylabel, stacked=True,
                                      title=rf"$z={z:g}$", fontsize=self.fontsize)
        return self.save(fig, f"metals_z{z:g}")

    def hcd(self, z=3.0):
        """``hcd_z<z>``: HCD correction by column-density class."""
        if self.contaminants is None:
            self.log("hcd skipped: no contaminant model given")
            return []
        k = self._k_grid()
        components = cont.hcd_contamination(self.contaminants, self.draws.columns(), z, k)
        fig = cont.plot_contamination(components, k, ylabel=r"$C_\mathrm{HCD}$",
                                      title=rf"$z={z:g}$", fontsize=self.fontsize)
        return self.save(fig, f"hcd_z{z:g}")

    def best_fit_p1d(self, entries, chi2, dof, ndata=None, title=None):
        """``best_fit_residuals`` and ``best_fit_p1d``: one row per data set.

        Parameters are those of :func:`cup1d.postprocessing.posterior.p1d.plot_best_fit`.
        """
        paths = []
        for residuals, name in ((True, "best_fit_residuals"), (False, "best_fit_p1d")):
            fig = plot_best_fit(entries, chi2, dof, ndata=ndata, residuals=residuals, title=title)
            paths += self.save(fig, name)
        return paths

    # ----- everything -----------------------------------------------------------

    FIGURES = ("star_posterior", "star_literature", "parameters_vs_star", "correlation",
               "systematics", "igm_histories", "metals", "hcd")

    def plot_all(self, skip=()):
        """Write every figure of :attr:`FIGURES` not in ``skip``.

        A figure that fails is reported with its traceback and the others are
        still written.

        Returns
        -------
        dict
            Figure method name to written paths, or to the error message.
        """
        out = {}
        for name in self.FIGURES:
            if name in skip:
                continue
            try:
                out[name] = getattr(self, name)()
            except Exception as error:  # keep going: one figure should not stop the others
                self.log(f"{name} failed: {error!r}\n{traceback.format_exc()}")
                plt.close("all")
                out[name] = repr(error)
        return out
