"""Statistical likelihood and its parameter definitions."""

__all__ = ["Likelihood", "make_parameter"]


def __getattr__(name):
    """Lazily expose likelihood symbols without initializing MPI at import time.

    Parameters
    ----------
    name : str
        Requested module attribute.

    Returns
    -------
    object
        ``Likelihood`` or the parameter factory.

    Raises
    ------
    AttributeError
        If ``name`` is not a supported lazy module attribute.
    """

    if name == "Likelihood":
        from cup1d.likelihood.likelihood import Likelihood

        return Likelihood
    if name == "make_parameter":
        from cup1d.likelihood.parameter import make_parameter

        return make_parameter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
