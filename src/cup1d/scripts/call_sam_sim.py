import os, sys
import numpy as np
from itertools import product

# our own modules
from lace.archive import gadget_archive, nyx_archive
from cup1d.data import data_gadget, data_nyx
from cup1d.scripts.sam_like_sim import sam_like_sim, path_sampler
from lace.cosmo import camb_cosmo


class Args:
    def __init__(self):
        self.version = "v1"
        self.training_set = "Pedersen21"
        self.emulator_label = "Pedersen21"
        self.mock_sim_label = "mpg_central"
        self.img_sim_label = "mpg_central"
        self.cosmo_sim_label = "mpg_central"
        self.drop_sim = True
        self.add_hires = False
        self.use_polyfit = True
        self.cov_label = "Chabanier2019"
        self.emu_cov_factor = 0
        self.n_igm = 2
        self.prior_Gauss_rms = None
        self.test = False
        self.no_parallel = True
        self.no_verbose = True  # reverse
        self.par2save = [
            "training_set",
            "emulator_label",
            "mock_sim_label",
            "igm_sim_label",
            "cosmo_sim_label",
            "drop_sim",
            "add_hires",
            "use_polyfit",
            "cov_label",
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
    list_training_set = ["Pedersen21", "Cabayol23", "Nyx23_Oct2023"]
    emulator_label = [
        "Pedersen21",
        "Cabayol23",
        "Cabayol23_extended",
        "Nyx_v0",
        "Nyx_v0_extended",
    ]

    # list of options to set
    version = "v1"
    training_set = "Cabayol23"
    emulator_label = "Cabayol23"
    add_hires = False
    # use own IGM history or that of central simulation
    arr_igm_own = [True, False]
    # use own cosmo or that of central simulation
    arr_cosmo_own = [True, False]
    # emulator_label = "Cabayol23_extended"
    # add_hires = True
    use_polyfit = True
    cov_label = "Chabanier2019"
    arr_drop_sim = [True]
    arr_n_igm = [2]
    override = False

    # read archive from outside
    if training_set == "Pedersen21":
        set_P1D = data_gadget.Gadget_P1D
        get_cosmo = camb_cosmo.get_cosmology_from_dictionary
        archive = gadget_archive.GadgetArchive(postproc=training_set)
        z_min = 2
        z_max = np.max(archive.list_sim_redshifts)
    elif training_set == "Cabayol23":
        set_P1D = data_gadget.Gadget_P1D
        get_cosmo = camb_cosmo.get_cosmology_from_dictionary
        archive = gadget_archive.GadgetArchive(postproc=training_set)
        z_min = 2
        z_max = np.max(archive.list_sim_redshifts)
    elif training_set[:5] == "Nyx23":
        set_P1D = data_nyx.Nyx_P1D
        get_cosmo = camb_cosmo.get_Nyx_cosmology
        archive = nyx_archive.NyxArchive(nyx_version=training_set[6:])
        z_min = 2.2
        z_max = np.max(archive.list_sim_redshifts)
    else:
        raise ValueError("Training_set not implemented")

    sim_avoid = [
        "nyx_14",
        "nyx_15",
        "nyx_16",
        "nyx_17",
        "nyx_seed",
        "nyx_wdm",
    ]

    combined_loop = product(
        arr_igm_own,
        arr_cosmo_own,
        arr_drop_sim,
        arr_n_igm,
        archive.list_sim,
    )

    for ind in combined_loop:
        igm_own, cosmo_own, drop_sim, n_igm, sim_label = ind
        if sim_label not in sim_avoid:
            print("")
            print("external loop")
            print("")

            args = Args()
            args.version = version
            args.archive = archive
            args.z_min = z_min
            args.z_max = z_max
            if igm_own:
                args.igm_sim_label = sim_label
            else:
                args.img_sim_label = sim_label[:3] + "_central"
            if cosmo_own:
                args.cosmo_sim_label = sim_label
            else:
                args.cosmo_sim_label = sim_label[:3] + "_central"

            args.set_P1D = set_P1D
            args.get_cosmo = get_cosmo

            args.training_set = training_set
            args.emulator_label = emulator_label
            args.add_hires = add_hires
            args.use_polyfit = use_polyfit
            args.cov_label = cov_label

            args.drop_sim = drop_sim
            args.n_igm = n_igm
            args.mock_sim_label = sim_label

            path = path_sampler(args)
            # check if run already done
            if (override == False) & os.path.isfile(
                path + "/chain_1/results.npy"
            ):
                print("Skipping: ", path)
            else:
                print("Running: ", path)
                sam_like_sim(args)
            print("bye!")
            break


if __name__ == "__main__":
    main()
