"""General-purpose helpers used across :mod:`cup1d`."""

import os
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


def purge_chains(ln_prop_chains, nsplit=4, abs_diff=15):
    """Return walker indices that pass simple log-probability stability cuts."""
    minval = np.median(ln_prop_chains) - abs_diff
    print(minval)
    # split each walker in nsplit chunks
    split_arr = np.array_split(ln_prop_chains, nsplit, axis=0)
    # compute median of each chunck
    split_med = []
    for ii in range(nsplit):
        split_med.append(split_arr[ii].mean(axis=0))
    # (nwalkers, nchucks)
    split_res = np.array(split_med).T
    # compute median of chunks for each walker ()
    split_res_med = split_res.mean(axis=1)

    # step-dependence convergence
    # check that average logprob does not vary much with step
    # compute difference between chunks and median of each chain
    keep1 = (np.abs(split_res - split_res_med[:, np.newaxis]) < abs_diff).all(
        axis=1
    )
    # total-dependence convergence
    # check that average logprob is close to minimum logprob of all chains
    # check that all chunks are above a target minimum value
    keep2 = (split_res > minval).all(axis=1)

    # combine both criteria
    both = keep1 & keep2
    keep = np.argwhere(both)[:, 0]
    keep_not = np.argwhere(~both)[:, 0]

    return keep, keep_not


def is_number_string(value):
    """Return whether ``value`` can be parsed as a number."""
    try:
        float(value)  # Try to convert to a float
        return True
    except ValueError:
        return False


def split_string(s):
    """Split a trailing ``_<integer>`` suffix from a parameter name."""
    match = re.match(r"^(.*)_(\d+)$", s)
    if match:
        return match.group(1), match.group(2)
    else:
        return s, None


def get_discrete_cmap(n, base_cmap="jet"):
    """Return a colormap with ``n`` colors sampled from ``base_cmap``."""
    cmap = plt.cm.get_cmap(
        base_cmap, n
    )  # Sample n colors from the base colormap
    return ListedColormap(cmap(np.linspace(0, 1, n)))


def mpi_hello_world():
    """Print a short MPI rank/size diagnostic from every process."""
    from mpi4py import MPI

    # Get the MPI communicator
    comm = MPI.COMM_WORLD

    # Get the rank and size of the MPI process
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Print a "Hello, World!" message from each MPI process
    print(f"Hello from rank {rank} out of {size} processes.", flush=True)


def create_print_function(verbose=True):
    """Create a rank-zero-only print function."""

    from mpi4py import MPI

    mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI.COMM_WORLD.Get_size() > 1 else 0

    def print_new(*args, verbose=verbose):
        if verbose and mpi_rank == 0:
            print(*args, flush=True)
        else:
            pass

    return print_new


def get_path_repo(name_repo):
    """Return the installed root directory for a known repository.

    Parameters
    ----------
    name_repo : str
        Repository name. Supported values are ``"cup1d"`` and ``"lace"``.

    Returns
    -------
    str
        Path to the repository root.

    Raises
    ------
    ImportError
        If ``name_repo`` is not supported.
    """
    if name_repo == "cup1d":
        import cup1d

        path = os.path.dirname(cup1d.__path__[0])
    elif name_repo == "lace":
        import lace

        path = os.path.dirname(lace.__path__[0])
    else:
        raise ImportError(
            name_repo
            + " is not a valid repository name. Expected values are 'cup1d' or 'lace'."
        )

    # if name_repo in path:
    #     pass
    # else:
    #     path = os.path.join(path, name_repo)
    return path
