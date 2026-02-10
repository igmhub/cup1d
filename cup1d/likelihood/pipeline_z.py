import os
import numpy as np
from mpi4py import MPI

# our own modules
from lace.emulator.emulator_manager import set_emulator
from cup1d.likelihood.pipeline import set_archive, Pipeline


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

        # set archive and emulator
        if rank == 0:
            if args.archive is None:
                args.archive = set_archive(args.training_set)
            if args.emulator is None:
                args.emulator = set_emulator(
                    emulator_label=args.emulator_label,
                    archive=args.archive,
                )

            if "Nyx" in args.emulator_label:
                args.emulator.list_sim_cube = args.archive.list_sim_cube
                if "nyx_14" in args.emulator.list_sim_cube:
                    args.emulator.list_sim_cube.remove("nyx_14")
            else:
                args.emulator.list_sim_cube = args.archive.list_sim_cube
        else:
            args.archive = None
            args.emulator = None

        #######################

        pip = Pipeline(args)

        list_z = pip.fitter.like.data.z
        print("list_z = {}".format(list_z))

        # only minimizer for now, need to implement sampler
        for z in list_z:
            if rank == 0:
                print("Analyzing z = {}".format(z))
            out_folder = os.path.join(self.out_folder, "z{}".format(z))
            args.z_min = z - 0.01
            args.z_max = z + 0.01
            self.pip2 = Pipeline(args, out_folder=out_folder)
            p0 = np.array(list(self.pip2.fitter.like.fid["fit_cube"].values()))
            self.pip2.run_minimizer(p0)
