import os, sys
import numpy as np

# our own modules
from lace.archive import gadget_archive, nyx_archive
from cup1d.data import data_gadget, data_nyx
from cup1d.scripts.max_like_sim import max_like_sim, fname_minimize


class Args:
    def __init__(self):
        self.training_set = "Pedersen21"
        self.emulator_label = "Pedersen21"
        self.test_sim_label = "mpg_central"
        self.drop_sim = True
        self.add_hires = False
        self.use_polyfit = True
        self.cov_label = "Chabanier2019"
        self.emu_cov_factor = 0
        self.n_igm = 2
        self.prior_Gauss_rms = None
        self.par2save = [
            "training_set",
            "emulator_label",
            "test_sim_label",
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
        "Nyx_v1",
        "Nyx_v1_extended",
    ]

    # list of options to set
    training_set = "Cabayol23"
    emulator_label = "Cabayol23"
    add_hires = False
    # emulator_label = "Cabayol23_extended"
    # add_hires = True
    use_polyfit = True
    cov_label = "Chabanier2019"
    arr_drop_sim = [True, False]
    arr_n_igm = [0, 1, 2, 3]
    override = False

    if (training_set == "Pedersen21") | (training_set == "Cabayol23"):
        list_sims = [
            "mpg_0",
            "mpg_1",
            "mpg_2",
            "mpg_3",
            "mpg_4",
            "mpg_5",
            "mpg_6",
            "mpg_7",
            "mpg_8",
            "mpg_9",
            "mpg_10",
            "mpg_11",
            "mpg_12",
            "mpg_13",
            "mpg_14",
            "mpg_15",
            "mpg_16",
            "mpg_17",
            "mpg_18",
            "mpg_19",
            "mpg_20",
            "mpg_21",
            "mpg_22",
            "mpg_23",
            "mpg_24",
            "mpg_25",
            "mpg_26",
            "mpg_27",
            "mpg_28",
            "mpg_29",
            "mpg_central",
            "mpg_seed",
            "mpg_growth",
            "mpg_neutrinos",
            "mpg_curved",
            "mpg_running",
            "mpg_reio",
        ]
    elif training_set[:5] == "Nyx23":
        list_sims = [
            "nyx_0",
            "nyx_1",
            "nyx_2",
            "nyx_3",
            "nyx_4",
            "nyx_5",
            "nyx_6",
            "nyx_7",
            "nyx_8",
            "nyx_9",
            "nyx_10",
            "nyx_11",
            "nyx_12",
            "nyx_13",
            "nyx_14",
            "nyx_15",
            "nyx_16",
            "nyx_17",
            "nyx_central",
            "nyx_seed",
            "nyx_wdm",
        ]

    # read archive from outside
    if training_set == "Pedersen21":
        archive = gadget_archive.GadgetArchive(postproc=training_set)
        set_P1D = data_gadget.Gadget_P1D
        z_min = 2
        z_max = np.max(archive.list_sim_redshifts)
        sim_igm = "mpg"
    elif training_set == "Cabayol23":
        archive = gadget_archive.GadgetArchive(postproc=training_set)
        set_P1D = data_gadget.Gadget_P1D
        z_min = 2
        z_max = np.max(archive.list_sim_redshifts)
        sim_igm = "mpg"
    elif training_set[:5] == "Nyx23":
        archive = nyx_archive.NyxArchive(nyx_version=training_set[6:])
        set_P1D = data_nyx.Nyx_P1D
        z_min = 2.2
        z_max = np.max(archive.list_sim_redshifts)
        sim_igm = "nyx"
    else:
        raise ValueError("Training_set not implemented")

    for drop_sim in arr_drop_sim:
        for n_igm in arr_n_igm:
            for sim_label in list_sims:
                print("")
                print("external loop")
                print("")

                args = Args()
                args.archive = archive
                args.z_min = z_min
                args.z_max = z_max
                args.sim_igm = sim_igm
                args.set_P1D = set_P1D

                args.training_set = training_set
                args.emulator_label = emulator_label
                args.add_hires = add_hires
                args.use_polyfit = use_polyfit
                args.cov_label = cov_label

                args.drop_sim = drop_sim
                args.n_igm = n_igm
                args.test_sim_label = sim_label

                fname = fname_minimize(args)
                # check if run already done
                if (override == False) & os.path.isfile(fname):
                    print("Skipping: ", fname)
                else:
                    print("Running: ", fname)
                    max_like_sim(args)


if __name__ == "__main__":
    main()
