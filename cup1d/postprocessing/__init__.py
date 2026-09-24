"""Plotting, chain inspection, and result tables."""

__all__ = ["EmulatorPriorPlotter", "Plotter", "VariationPlotter", "P1DPlotter", "plot_p1d", "plot_cosmo_sampler_and_fit"]


def __getattr__(name):
    """Import optional plotting support only when it is requested."""

    if name in {"P1DPlotter", "plot_p1d"}:
        from cup1d.postprocessing import p1d

        return getattr(p1d, name)
    if name == "Plotter":
        from cup1d.postprocessing.plotter import Plotter

        return Plotter
    if name == "EmulatorPriorPlotter":
        from cup1d.postprocessing.emulator_priors import EmulatorPriorPlotter

        return EmulatorPriorPlotter
    if name == "VariationPlotter":
        from cup1d.postprocessing.variation_plotter import VariationPlotter

        return VariationPlotter
    if name == "plot_cosmo_sampler_and_fit":
        from cup1d.postprocessing.cosmology import plot_cosmo_sampler_and_fit

        return plot_cosmo_sampler_and_fit
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
