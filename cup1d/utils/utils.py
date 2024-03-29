from mpi4py import MPI
import os
import cup1d


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
