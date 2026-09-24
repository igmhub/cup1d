"""Statistical likelihood and its parameter definitions."""

__all__ = ["Likelihood", "make_parameter", "LikelihoodParameter"]


def __getattr__(name):
    """Avoid initializing MPI unless the full likelihood is requested."""

    if name == "Likelihood":
        from cup1d.likelihood.likelihood import Likelihood

        return Likelihood
    if name in {"make_parameter", "LikelihoodParameter"}:
        from cup1d.likelihood.parameter import make_parameter

        return make_parameter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
