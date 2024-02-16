import os, sys
import numpy as np
from itertools import product

# our own modules
from lace.archive import gadget_archive, nyx_archive
from cup1d.data import data_gadget, data_nyx
from cup1d.scripts.sam_sim import sam_sim, path_sampler
from lace.cosmo import camb_cosmo


class Args:
    def __init__(self):
        # see sam_sim to see what each parameter means
        self.version = "v1"
        self.training_set = "Pedersen21"
        self.emulator_label = "Pedersen21"
        self.mock_label = "mpg_central"
        self.img_sim_label = "mpg_central"
        self.cosmo_sim_label = "mpg_central"
        self.drop_sim = True
        self.add_hires = False
        self.add_noise = False
        self.apply_smoothing = True
        self.cov_label = "Chabanier2019"
        self.cov_label_hires = "Karacayli2022"
        self.emu_cov_factor = 0
        self.n_igm = 2
        self.z_min = 2
        self.z_max = 4.5
        self.n_steps = None
        self.n_burn_in = None
        self.fix_cosmo = False
        self.prior_Gauss_rms = None
        self.test = False
        self.parallel = True
        self.verbose = True
        self.par2save = [
            "training_set",
            "emulator_label",
            "mock_label",
            "igm_sim_label",
            "cosmo_sim_label",
            "drop_sim",
            "add_hires",
            "add_noise",
            "apply_smoothing",
            "cov_label",
            "cov_label_hires",
            "fix_cosmo",
            "emu_cov_factor",
            "n_igm",
            "prior_Gauss_rms",
        ]

    def save(self):
        out = {}
        for par in self.par2save:
            out[par] = getattr(self, par)
        return out


def main():
    # general info
    avail_training_set = ["Pedersen21", "Cabayol23", "Nyx23_Oct2023"]
    avail_emulator_label = [
        "Pedersen21",
        "Pedersen23",
        "Pedersen21_ext",
        "Pedersen23_ext",
        "k_bin_sm",
        "Cabayol23",
        "Cabayol23_extended",
        "Nyx_v0",
        "Nyx_v0_extended",
    ]
    dict_training_set = {
        "Pedersen21_ext": "Cabayol23",
        "Pedersen23_ext": "Cabayol23",
        "k_bin_sm": "Cabayol23",
        "Cabayol23": "Cabayol23",
    }
    dict_high_res = {
        "Pedersen21_ext": False,
        "Pedersen23_ext": False,
        "k_bin_sm": False,
        "Cabayol23": False,
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
        "Pedersen21_ext",
        "Pedersen23_ext",
        "k_bin_sm",
        "Cabayol23",
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
    # apply smoothing if the emulator uses it
    apply_smoothing = True
    arr_drop_sim = [True]
    arr_n_igm = [2]
    # add noise to mock data (False for no noise, any int for noise seed)
    arr_add_noise = [False]
    # Fix cosmological parameters while sampling
    fix_cosmo = False
    override = False

    combined_loop = product(
        arr_mock_own,
        arr_igm_own,
        arr_cosmo_own,
        arr_drop_sim,
        arr_n_igm,
        arr_add_noise,
        archive.list_sim,
    )

    for emulator_label in arr_emulator_label:
        # read archive
        training_set = dict_training_set[emulator_label]
        if training_set == "Pedersen21":
            set_P1D = data_gadget.Gadget_P1D
            get_cosmo = camb_cosmo.get_cosmology_from_dictionary
            archive = gadget_archive.GadgetArchive(postproc=training_set)
            # 11 snaps between 2 and 4.5
            z_min = 1.9
            z_max = 4.6
        elif training_set == "Cabayol23":
            set_P1D = data_gadget.Gadget_P1D
            get_cosmo = camb_cosmo.get_cosmology_from_dictionary
            archive = gadget_archive.GadgetArchive(postproc=training_set)
            # 11 snaps between 2 and 4.5
            z_min = 1.9
            z_max = 4.6
        elif training_set[:5] == "Nyx23":
            set_P1D = data_nyx.Nyx_P1D
            get_cosmo = camb_cosmo.get_Nyx_cosmology
            archive = nyx_archive.NyxArchive(nyx_version=training_set[6:])
            # 11 snaps between 2.2 and 4.2
            z_min = 2.1
            z_max = 4.3
        else:
            raise ValueError("Training_set not implemented")

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

            if sim_label not in sim_avoid:
                print("")
                print("external loop")
                print("")

                # see sam_sim to see what each parameter means
                args = Args()

                args.emulator_label = emulator_label

                if mock_own:
                    args.mock_label = sim_label
                else:
                    args.mock_label = mock_own
                args.z_min = z_min
                args.z_max = z_max

                if igm_own:
                    args.igm_sim_label = sim_label
                else:
                    args.img_sim_label = sim_label[:3] + "_central"
                args.n_igm = n_igm

                if cosmo_own:
                    args.cosmo_sim_label = sim_label
                else:
                    args.cosmo_sim_label = sim_label[:3] + "_central"
                args.fix_cosmo = fix_cosmo

                args.add_hires = dict_high_res[emulator_label]
                args.apply_smoothing = apply_smoothing
                if sim_label in archive.list_sim_test:
                    args.drop_sim = False
                else:
                    args.drop_sim = drop_sim

                args.cov_label = cov_label
                args.cov_label_hires = cov_label_hires
                if add_noise is False:
                    args.add_noise = add_noise
                else:
                    args.add_noise = True
                    args.seed_noise = add_noise

                args.version = version

                args.archive = archive
                args.set_P1D = set_P1D
                args.get_cosmo = get_cosmo

                path = path_sampler(args)
                # check if run already done
                if (override == False) & os.path.isfile(
                    path + "/chain_1/results.npy"
                ):
                    print("Skipping: ", path)
                else:
                    print("Running: ", path)
                    sam_sim(args)
                print("bye!")
                break


if __name__ == "__main__":
    main()
