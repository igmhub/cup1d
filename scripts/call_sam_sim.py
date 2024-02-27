import os, sys

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from mpi4py import MPI
from itertools import product

# our own modules
from lace.archive import gadget_archive, nyx_archive
from cup1d.likelihood.sampler_pipeline import path_sampler, SamplerPipeline
from cup1d.utils.utils import create_print_function


class Args:
    def __init__(
        self,
        archive=None,
        training_set="Pedersen21",
        emulator_label="Pedersen21",
        data_label="mpg_central",
        z_min=2,
        z_max=4.5,
        igm_label="mpg_central",
        n_igm=2,
        cosmo_label="mpg_central",
        drop_sim=False,
        add_hires=False,
        apply_smoothing=None,
        cov_label="Chabanier2019",
        cov_label_hires="Karacayli2022",
        add_noise=False,
        seed_noise=0,
        fix_cosmo=False,
        version="v3",
        prior_Gauss_rms=None,
        emu_cov_factor=0,
        verbose=True,
        test=False,
        parallel=False,
        n_burn_in=0,
        n_steps=0,
    ):
        # see sam_sim to see what each parameter means
        self.archive = archive
        self.training_set = training_set
        self.emulator_label = emulator_label
        self.data_label = data_label
        self.z_min = z_min
        self.z_max = z_max
        self.igm_label = igm_label
        self.n_igm = n_igm
        self.cosmo_label = cosmo_label
        self.drop_sim = drop_sim
        self.add_hires = add_hires
        self.apply_smoothing = apply_smoothing
        self.cov_label = cov_label
        self.cov_label_hires = cov_label_hires
        self.add_noise = add_noise
        self.seed_noise = seed_noise
        self.fix_cosmo = fix_cosmo
        self.version = version
        self.prior_Gauss_rms = prior_Gauss_rms
        self.emu_cov_factor = emu_cov_factor
        self.verbose = verbose
        self.test = test
        self.parallel = parallel
        self.n_burn_in = n_burn_in
        self.n_steps = n_steps

        self.par2save = [
            "emulator_label",
            "data_label",
            "z_min",
            "z_max",
            "igm_label",
            "n_igm",
            "cosmo_label",
            "drop_sim",
            "add_hires",
            "apply_smoothing",
            "add_noise",
            "fix_cosmo",
            "cov_label",
            "cov_label_hires",
        ]

    def save(self):
        out = {}
        for par in self.par2save:
            out[par] = getattr(self, par)
        return out

    def check_emulator_label(self):
        avail_emulator_label = [
            "Pedersen21",
            "Pedersen23",
            "Pedersen21_ext",
            "Pedersen23_ext",
            "CH24",
            "Cabayol23",
            "Cabayol23_extended",
            "Nyx_v0",
            "Nyx_v0_extended",
        ]
        if self.emulator_label not in avail_emulator_label:
            raise ValueError(
                "emulator_label " + self.emulator_label + " not implemented"
            )


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    fprint = create_print_function(verbose=True)

    # general info
    dict_training_set = {
        "Pedersen21": "Pedersen21",
        "Pedersen23": "Cabayol23",
        "Pedersen21_ext": "Cabayol23",
        "Pedersen23_ext": "Cabayol23",
        "CH24": "Cabayol23",
        "Cabayol23": "Cabayol23",
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
    }
    dict_apply_smoothing = {
        "Pedersen21": False,
        "Pedersen23": True,
        "Pedersen21_ext": False,
        "Pedersen23_ext": True,
        "CH24": True,
        "Cabayol23": True,
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
        "Pedersen21",
        "Pedersen23_ext",
        # "CH24",
        # "Cabayol23",
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
    arr_add_noise = [False]
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
            list_sim = archive.list_sim
            list_sim_test = archive.list_sim_test
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
                add_noise,
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

            if add_noise is not False:
                add_noise = True
                seed_noise = add_noise
            else:
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
            else:
                fprint("Running: ", path)
                comm.Barrier()
                pip = SamplerPipeline(args)
                pip.run_sampler()

    fprint("End of the program")


if __name__ == "__main__":
    main()
