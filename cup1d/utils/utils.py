from mpi4py import MPI
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def is_number_string(value):
    """
    Check if the input string represents a valid number (integer or float).
    """
    try:
        float(value)  # Try to convert to a float
        return True
    except ValueError:
        return False


# Function to generate n discrete colors from any continuous colormap
def get_discrete_cmap(n, base_cmap="jet"):
    """Returns a colormap with n discrete colors."""
    cmap = plt.cm.get_cmap(
        base_cmap, n
    )  # Sample n colors from the base colormap
    return ListedColormap(cmap(np.linspace(0, 1, n)))


def mpi_hello_world():
    # Get the MPI communicator
    comm = MPI.COMM_WORLD

    # Get the rank and size of the MPI process
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Print a "Hello, World!" message from each MPI process
    print(f"Hello from rank {rank} out of {size} processes.", flush=True)


def create_print_function(verbose=True):
    """Create a function to print messages"""
    mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI.COMM_WORLD.Get_size() > 1 else 0

    def print_new(*args, verbose=True):
        if verbose and mpi_rank == 0:
            print(*args, flush=True)
        else:
            pass

    return print_new


def get_path_cup1d():
    path_cup1d = os.path.dirname(cup1d.__path__[0])
    if "cup1d" in path_cup1d:
        pass
    else:
        path_cup1d += "/cup1d"
    return path_cup1d


def get_path_repo(name_repo):
    """
    Returns the file path to the root directory of a specified repository.

    This function checks the name of the repository and imports the corresponding module
    (`cup1d` or `lace`) to obtain the directory path. If the repository name matches a part
    of the path, it returns the path directly; otherwise, it appends the repository name to the
    path and returns the resulting full path.

    Parameters:
    ----------
    name_repo : str
        The name of the repository. Expected values are "cup1d" or "lace".

    Returns:
    -------
    str
        The file path to the root directory of the specified repository.

    Raises:
    ------
    ImportError
        If the specified repository name is not recognized, this function will raise an ImportError.

    Notes:
    -----
    - The function uses the `__path__` attribute of the imported repository modules to determine the root directory.
    - The repository name should exactly match one of the recognized values ("cup1d" or "lace").
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
