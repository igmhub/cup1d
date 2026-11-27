"""Figures, chain products and result tables.

Computation and drawing are kept apart: ``fit`` and the ``compute`` functions
of each module return arrays, and the plotting functions take them and
return the figure (nothing is shown, printed or styled at import time; the
plotters call :func:`cup1d.postprocessing.style.apply_house_style`).

Subpackages and modules
-----------------------
style, labels, io, primitives, literature
    House style, label registry, saving, drawing primitives, external data.
p1d
    P1D data, data against model, residuals, errors and covariance.
posterior
    Figures of a posterior (any sampler): :class:`PosteriorPlotter`.
minimizer
    Figures of a best fit: :class:`MinimizerPlotter`, training hull.
fit
    Quantities derived from a point or chain (IGM, contaminant scenarios).
chains
    Products written next to a chain; external (Planck) chains.
tables
    Result tables.
"""

__all__ = ["EmulatorPriorPlotter", "VariationPlotter", "P1DPlotter", "plot_p1d", "MinimizerPlotter",
           "PosteriorSamples", "PosteriorPlotter", "apply_house_style"]


def __getattr__(name):
    """Lazily import a public post-processing export.

    Parameters
    ----------
    name : str
        Name listed in :data:`__all__`.

    Returns
    -------
    object
        Requested plotting class or function.

    Raises
    ------
    AttributeError
        If ``name`` is not a supported lazy export.
    """

    if name in {"P1DPlotter", "plot_p1d"}:
        from cup1d.postprocessing import p1d

        return getattr(p1d, name)
    if name == "EmulatorPriorPlotter":
        from cup1d.postprocessing.emulator_priors import EmulatorPriorPlotter

        return EmulatorPriorPlotter
    if name == "VariationPlotter":
        from cup1d.postprocessing.variation_plotter import VariationPlotter

        return VariationPlotter
    if name == "MinimizerPlotter":
        from cup1d.postprocessing.minimizer import MinimizerPlotter

        return MinimizerPlotter
    if name in {"PosteriorSamples", "PosteriorPlotter"}:
        from cup1d.postprocessing import posterior

        return getattr(posterior, name)
    if name == "apply_house_style":
        from cup1d.postprocessing.style import apply_house_style

        return apply_house_style
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
