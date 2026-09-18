from ._version import __version__

__all__ = ["Analysis", "Args", "__version__"]


def __getattr__(name):
    """Load the public analysis interface only when requested."""

    if name == "Analysis":
        from cup1d.inference import Analysis

        return Analysis
    if name == "Args":
        from cup1d.configuration import Args

        return Args
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
