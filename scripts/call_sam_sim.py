import os, sys

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from mpi4py import MPI
from itertools import product

# our own modules
from lace.archive import gadget_archive, nyx_archive
from cup1d.likelihood.sampler_pipeline import path_sampler, SamplerPipeline
from cup1d.likelihood.input_pipeline import Args
from cup1d.utils.utils import create_print_function


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # print_2_file

    fprint = create_print_function(verbose=True)

    # general info
    dict_training_set = {
        "Pedersen21": "Pedersen21",
        "Pedersen23": "Pedersen21",
        "Pedersen21_ext": "Cabayol23",
        "Pedersen23_ext": "Cabayol23",
        "CH24": "Cabayol23",
        "Cabayol23": "Cabayol23",
        "Cabayol23+": "Cabayol23",
        "Cabayol23_extended": "Cabayol23",
        "Nyx_v0": "Nyx23_Oct2023",
        "Nyx_v0_extended": "Nyx23_Oct2023",
    }
    dict_high_res = {
        "Pedersen21": False,
        "Pedersen23": False,
        "Pedersen21_ext": False,
        "Pedersen23_ext": False,
        "CH24": False,
        "Cabayol23": False,
        "Cabayol23+": False,
    }
    dict_apply_smoothing = {
        "Pedersen21": False,
        "Pedersen23": True,
        "Pedersen21_ext": False,
        "Pedersen23_ext": True,
        "CH24": True,
        "Cabayol23": True,
        "Cabayol23+": True,
        "Cabayol23_extended": True,
        "Nyx_v0": True,
        "Nyx_v0_extended": True,
    }
    sim_avoid = [
        "nyx_14",
        "nyx_15",
        "nyx_16",
        "nyx_17",
        "nyx_seed",
        "nyx_wdm",
    ]

    #############################################################################
    #############################################################################
    # list of options to set
    version = "v3"
    arr_emulator_label = [
        # "Pedersen21",
        # "Pedersen23_ext",
        # "CH24",
        # "Cabayol23",
        "Cabayol23+",
    ]
    # use l1O or test sim to set mock (True) or whatever sim is specified
    arr_mock_own = [True]
    # use own IGM history or that of central simulation (True for own history)
    # arr_igm_own = [True, False]
    arr_igm_own = [True]
    # use own cosmo or that of central simulation (True for own cosmo)
    # arr_cosmo_own = [True, False]
    arr_cosmo_own = [True]
    cov_label = "Chabanier2019"
    cov_label_hires = "Karacayli2022"
    # None for whatever is best for emulator
    do_apply_smoothing = None
    arr_drop_sim = [True]
    arr_n_igm = [2]
    # add noise to mock data (False for no noise, any int for noise seed)
    # for l1O
    arr_add_noise = [False]
    # for MC mocks
    # arr_add_noise = np.arange(100)
    # Fix cosmological parameters while sampling
    fix_cosmo = False
    override = False
    z_min = 0
    z_max = 10

    for emulator_label in arr_emulator_label:
        # read archive
        training_set = dict_training_set[emulator_label]
        if rank == 0:
            if training_set[:5] == "Nyx23":
                archive = nyx_archive.NyxArchive(nyx_version=training_set[6:])
                # 11 snaps between 2.2 and 4.2
                # z_min = 2.1
                # z_max = 4.3
            else:
                archive = gadget_archive.GadgetArchive(postproc=training_set)
                # 11 snaps between 2 and 4.5
                # z_min = 1.9
                # z_max = 4.6
            # all
            list_sim = archive.list_sim
            # training sims
            # for l1O
            # list_sim = archive.list_sim_cube
            # list_sim = ["mpg_central"]
            # testing sims
            list_sim_test = archive.list_sim_test
            # list_sim_test = ["mpg_central"]
            for irank in range(1, size):
                comm.send(list_sim, dest=irank, tag=(irank + 1) * 3)
                comm.send(list_sim_test, dest=irank, tag=(irank + 1) * 5)
        else:
            archive = None
            list_sim = comm.recv(source=0, tag=(rank + 1) * 3)
            list_sim_test = comm.recv(source=0, tag=(rank + 1) * 5)

        combined_loop = product(
            arr_mock_own,
            arr_igm_own,
            arr_cosmo_own,
            arr_drop_sim,
            arr_n_igm,
            arr_add_noise,
            list_sim,
        )

        for ind in combined_loop:
            (
                mock_own,
                igm_own,
                cosmo_own,
                drop_sim,
                n_igm,
                _add_noise,
                sim_label,
            ) = ind

            if sim_label in sim_avoid:
                continue

            fprint("")
            fprint("External loop: ", emulator_label, " ", sim_label)
            fprint("")

            # see sam_sim to see what each parameter means
            if mock_own:
                data_label = sim_label
            else:
                data_label = sim_label[:3] + "_central"

            if igm_own:
                igm_label = sim_label
            else:
                igm_label = sim_label[:3] + "_central"

            if cosmo_own:
                cosmo_label = sim_label
            else:
                cosmo_label = sim_label[:3] + "_central"

            if sim_label in list_sim_test:
                drop_sim = False
            else:
                pass

            add_hires = dict_high_res[emulator_label]
            if do_apply_smoothing is None:
                apply_smoothing = dict_apply_smoothing[emulator_label]
            else:
                apply_smoothing = do_apply_smoothing

            if _add_noise is not False:
                add_noise = True
                seed_noise = _add_noise
            else:
                add_noise = False
                seed_noise = 0

            args = Args(
                archive=archive,
                training_set=training_set,
                emulator_label=emulator_label,
                data_label=data_label,
                z_min=z_min,
                z_max=z_max,
                igm_label=igm_label,
                drop_sim=drop_sim,
                n_igm=n_igm,
                cosmo_label=cosmo_label,
                cov_label=cov_label,
                fix_cosmo=fix_cosmo,
                add_hires=add_hires,
                apply_smoothing=apply_smoothing,
                cov_label_hires=cov_label_hires,
                add_noise=add_noise,
                seed_noise=seed_noise,
                version=version,
            )
            args.check_emulator_label()

            if rank == 0:
                path = path_sampler(
                    emulator_label=args.emulator_label,
                    data_label=args.data_label,
                    igm_label=args.igm_label,
                    n_igm=args.n_igm,
                    cosmo_label=args.cosmo_label,
                    cov_label=args.cov_label,
                    version=args.version,
                    drop_sim=args.drop_sim,
                    apply_smoothing=args.apply_smoothing,
                    add_hires=args.add_hires,
                    add_noise=args.add_noise,
                    seed_noise=args.seed_noise,
                    fix_cosmo=args.fix_cosmo,
                )
                # check if run already done
                if (override == False) & os.path.isfile(
                    path + "/chain_1/results.npy"
                ):
                    fprint("Skipping: ", path)
                    run = False
                else:
                    fprint("Running: ", path)
                    run = True

                for irank in range(1, size):
                    comm.send(run, dest=irank, tag=(irank + 1) * 17)
            else:
                run = comm.recv(source=0, tag=(rank + 1) * 17)

            comm.Barrier()
            if run:
                pip = SamplerPipeline(args)
                pip.run_sampler()
                comm.Barrier()

    fprint("End of the program")


if __name__ == "__main__":
    main()
