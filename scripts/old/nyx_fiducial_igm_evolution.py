from lace.archive.nyx_archive import NyxArchive
import numpy as np
import os


def main():
    """Compute fiducial IGM evolution from Nyx"""

    # Read full post-processing from archive
    nyx_version = "Oct2023"
    nyx_archive = NyxArchive(nyx_version=nyx_version, verbose=True)

    # We cannot use the central simulation because some values of lambda_P are missing
    # only simulation with all redshifts
    sim_to_use = "nyx_1"
    prop_list = np.zeros((16, 5))
    jj = 0
    for ii in range(len(nyx_archive.data)):
        if nyx_archive.data[ii]["sim_label"] == sim_to_use:
            if nyx_archive.data[ii]["ind_rescaling"] == 0:
                if nyx_archive.data[ii]["ind_axis"] == 0:
                    prop_list[jj, 0] = nyx_archive.data[ii]["z"]
                    # Mean flux to tau_eff
                    prop_list[jj, 1] = -np.log(nyx_archive.data[ii]["mF"])
                    prop_list[jj, 2] = nyx_archive.data[ii]["gamma"]
                    # Mpc to kms
                    prop_list[jj, 3] = (
                        nyx_archive.data[ii]["sigT_Mpc"]
                        * nyx_archive.data[ii]["dkms_dMpc"]
                    )
                    # kpc to kms
                    prop_list[jj, 4] = 1 / (
                        1e-3
                        * nyx_archive.data[ii]["lambda_P"]
                        * nyx_archive.data[ii]["dkms_dMpc"]
                    )
                    jj += 1

    # Write to file using the same format as LaCE file

    with open(os.environ["NYX_PATH"] + "/fiducial_igm_evolution.txt", "w") as f:
        f.write("# list[z, tau_eff, gamma, sigt_kms, kF_kms]")
        f.write("\n")
        for ii in range(5):
            print(prop_list[::-1, ii])
            f.write(" ".join(list(prop_list[::-1, ii].astype("str"))))
            f.write("\n")


if __name__ == "__main__":
    main()
