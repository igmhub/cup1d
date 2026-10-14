from ._version import __version__

__all__ = ["Analysis", "Args", "__version__"]


def __getattr__(name):
    """Lazily resolve a public top-level cup1d symbol.

    Parameters
    ----------
    name : str
        Requested module attribute.

    Returns
    -------
    type
        ``Analysis`` or ``Args`` when either public symbol is requested.

    Raises
    ------
    AttributeError
        If ``name`` is not a lazily exported public symbol.
    """

    if name == "Analysis":
        from cup1d.inference import Analysis

        return Analysis
    if name == "Args":
        from cup1d.configuration import Args

        return Args
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
