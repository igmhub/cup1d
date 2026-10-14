import os
import re
import numpy as np


def purge_chains(ln_prop_chains, nsplit=4, abs_diff=15):
    """Select walkers with stable, sufficiently high log-probability histories.

    Parameters
    ----------
    ln_prop_chains : ndarray
        Chain log-probabilities with sampling steps on axis zero.
    nsplit : int, default: 4
        Number of temporal chunks used for stability checks.
    abs_diff : float, default: 15
        Allowed chunk variation and distance below the global median.

    Returns
    -------
    keep, keep_not : ndarray
        Walker indices satisfying or failing both convergence heuristics.
    """
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
    keep_not = np.argwhere(both == False)[:, 0]

    return keep, keep_not


def is_number_string(value):
    """Return whether a value can be converted to a floating-point number.

    Parameters
    ----------
    value : object
        Candidate numeric representation.

    Returns
    -------
    bool
        Whether ``float(value)`` succeeds.
    """
    try:
        float(value)  # Try to convert to a float
        return True
    except ValueError:
        return False


def split_string(s):
    """Split a trailing underscore-index from a parameter name.

    Parameters
    ----------
    s : str
        Name optionally ending in ``_<integer>``.

    Returns
    -------
    tuple of str and str or None
        Base name and trailing index when present.
    """
    match = re.match(r"^(.*)_(\d+)$", s)
    if match:
        return match.group(1), match.group(2)
    else:
        return s, None


# Function to generate n discrete colors from any continuous colormap
def get_discrete_cmap(n, base_cmap="jet"):
    """Construct a discrete Matplotlib colormap through the style helper.

    Parameters
    ----------
    n : int
        Number of colors.
    base_cmap : str, default: "jet"
        Matplotlib colormap name.

    Returns
    -------
    matplotlib.colors.ListedColormap
        Discrete sampled colormap.
    """
    from cup1d.postprocessing.style import get_discrete_cmap as _plot

    return _plot(n, base_cmap)


def mpi_hello_world():
    """Print an MPI communicator greeting from every rank.

    Requires
    --------
    mpi4py
        Available MPI Python bindings.
    """
    from mpi4py import MPI

    # Get the MPI communicator
    comm = MPI.COMM_WORLD

    # Get the rank and size of the MPI process
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Print a "Hello, World!" message from each MPI process
    print(f"Hello from rank {rank} out of {size} processes.", flush=True)


def create_print_function(verbose=True):
    """Create a rank-zero MPI-aware print function.

    Parameters
    ----------
    verbose : bool, default: True
        Retained compatibility argument; per-call verbosity controls output.

    Returns
    -------
    callable
        Function accepting ``*args`` and a ``verbose`` keyword. It prints only
        from rank zero.
    """

    from mpi4py import MPI

    mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI.COMM_WORLD.Get_size() > 1 else 0

    def print_new(*args, verbose=True):
        """Print arguments from rank zero when per-call verbosity is enabled."""
        if verbose and mpi_rank == 0:
            print(*args, flush=True)
        else:
            pass

    return print_new


def get_path_repo(name_repo):
    """Return the installed source root of a supported IGMHub package.

    Parameters
    ----------
    name_repo : {"cup1d", "lace"}
        Package whose repository root is requested.

    Returns
    -------
    str
        Absolute source-root path.

    Raises
    ------
    ImportError
        If the repository name is unsupported.
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
