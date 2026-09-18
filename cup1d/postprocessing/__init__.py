"""Plotting, chain inspection, and result tables."""

__all__ = ["Plotter"]


def __getattr__(name):
    """Import optional plotting support only when it is requested."""

    if name == "Plotter":
        from cup1d.postprocessing.plotter import Plotter

        return Plotter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
