import os, sys, time
import numpy as np
from mpi4py import MPI

# our own modules
from cup1d.utils.utils import create_print_function
from cup1d.likelihood.cosmologies import set_cosmo
from lace.emulator.emulator_manager import set_emulator
from cup1d.likelihood.fitter import Fitter
from cup1d.likelihood.plotter import Plotter

from cup1d.likelihood.pipeline import Pipeline


class Pipeline_z(object):
    """Full pipeline for extracting cosmology from P1D using sampler one z at a time"""

    def __init__(self, args, out_folder=None):
        """Set pipeline_z"""

        # set archive and emulator here. Call iteratively pipeline using different z_min and z_max

        self.out_folder = out_folder

        ## MPI stuff
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        if rank == 0:
            if args.archive is None:
                args.archive = set_archive(args.training_set)
        else:
            args.archive = None

        #######################

        pip = Pipeline(args)

        list_z = pip.like.data.z

        for z in list_z:
            out_folder = os.path.join(self.out_folder, "z{}".format(z))
            args.z_min = z - 0.01
            args.z_max = z + 0.01
            pip2 = Pipeline(args, out_folder=out_folder)
            pip2.run_minimizer()
