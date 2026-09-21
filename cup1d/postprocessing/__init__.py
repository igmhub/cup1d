"""Plotting, chain inspection, and result tables."""

__all__ = ["EmulatorPriorPlotter", "Plotter", "VariationPlotter"]


def __getattr__(name):
    """Import optional plotting support only when it is requested."""

    if name == "Plotter":
        from cup1d.postprocessing.plotter import Plotter

        return Plotter
    if name == "EmulatorPriorPlotter":
        from cup1d.postprocessing.emulator_priors import EmulatorPriorPlotter

        return EmulatorPriorPlotter
    if name == "VariationPlotter":
        from cup1d.postprocessing.variation_plotter import VariationPlotter

        return VariationPlotter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
