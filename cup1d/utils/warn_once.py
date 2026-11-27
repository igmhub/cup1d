"""Warnings that are shown once per process, and only on MPI rank 0.

Python's built-in "default"/"once" filters are unreliable for this: the
per-module registry is reset whenever any code touches the warning filters
(e.g. ``warnings.catch_warnings``), and every MPI rank prints its own copy.
These helpers keep their own registry and silence non-root ranks.

Examples
--------
>>> from cup1d.utils.warn_once import warn_once, numpy_errors_once
>>> warn_once("emulator extrapolating in kF", key="emu-kF")
>>> with numpy_errors_once("chi2"):
...     chi2 = diff @ icov @ diff
"""

import os
import sys
import warnings
from contextlib import contextmanager

import numpy as np

_SEEN = set()

# environment variables set by common MPI launchers
_RANK_ENV_VARS = (
    "OMPI_COMM_WORLD_RANK",
    "PMI_RANK",
    "PMIX_RANK",
    "MPI_LOCALRANKID",
    "SLURM_PROCID",
)


def mpi_rank():
    """Return the MPI rank of this process, or 0 when not running under MPI.

    Uses ``mpi4py`` only if it is already imported, so calling this never
    initialises MPI as a side effect. Falls back on launcher environment
    variables.
    """
    if "mpi4py.MPI" in sys.modules:
        return sys.modules["mpi4py.MPI"].COMM_WORLD.Get_rank()
    for var in _RANK_ENV_VARS:
        if var in os.environ:
            try:
                return int(os.environ[var])
            except ValueError:
                pass
    return 0


def warn_once(message, category=UserWarning, key=None, stacklevel=1):
    """Issue a warning the first time ``key`` is seen, on rank 0 only.

    Parameters
    ----------
    message : str
        Warning text.
    category : type, default: UserWarning
        Warning class.
    key : hashable, optional
        Deduplication key; defaults to ``(category, message)``. Use an explicit
        key when the message contains varying numbers.
    stacklevel : int, default: 1
        As in :func:`warnings.warn`, counted from the caller of this function.
    """
    if key is None:
        key = (category, message)
    if key in _SEEN:
        return
    _SEEN.add(key)
    if mpi_rank() != 0:
        return
    warnings.warn(
        f"{message} (shown once; further occurrences silenced)",
        category,
        stacklevel=stacklevel + 1,
    )


@contextmanager
def numpy_errors_once(label, errors=("over", "invalid", "divide")):
    """Report NumPy floating-point errors in this block once per label.

    Replaces NumPy's repeated ``RuntimeWarning`` with a single
    :func:`warn_once` per ``(label, error type)``. Results are unchanged:
    the block still produces inf/nan as usual.

    Parameters
    ----------
    label : str
        Short name of the computation, used in the message and the key.
    errors : tuple of str
        Error types to intercept (any of over, under, invalid, divide).
    """

    def _handler(err, flag):
        # frames: warn_once <- _handler <- code inside the ``with`` block
        warn_once(
            f"{label}: floating-point error ({err}) encountered",
            RuntimeWarning,
            key=("numpy", label, err),
            stacklevel=2,
        )

    with np.errstate(call=_handler, **{e: "call" for e in errors}):
        yield
