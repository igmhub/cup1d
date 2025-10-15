import numpy as np
import matplotlib.pyplot as plt
from cup1d.pipeline.set_archive import set_archive


def plot_cosmic_variance():
    nyx_training_set = "models_Nyx_Sept2025_include_Nyx_fid_rseed"
    archive_mock = set_archive(training_set=nyx_training_set)
    central = archive_mock.get_testing_data("nyx_central")
    seed = archive_mock.get_testing_data("nyx_seed")

    archive_mock = set_archive(training_set="Cabayol23")
    mpg_central = archive_mock.get_testing_data("mpg_central")
    mpg_seed = archive_mock.get_testing_data("mpg_seed")

    _ = (central[0]["k_Mpc"] < 2) & (central[0]["k_Mpc"] > 0.1)
    mean1 = np.median(central[0]["p1d_Mpc"][_] / seed[1]["p1d_Mpc"][_])
    plt.plot(
        central[0]["k_Mpc"][_],
        (central[0]["p1d_Mpc"] / seed[1]["p1d_Mpc"])[_] - mean1,
        label="nyx-central/nyx-seed-1",
    )
    std1 = np.std((central[0]["p1d_Mpc"] / seed[1]["p1d_Mpc"])[_] - mean1)

    _ = (mpg_central[-2]["k_Mpc"] < 2) & (mpg_central[-2]["k_Mpc"] > 0.1)
    mean2 = np.median(
        mpg_central[-2]["p1d_Mpc"][_] / mpg_seed[-2]["p1d_Mpc"][_]
    )
    plt.plot(
        mpg_central[-2]["k_Mpc"][_],
        (mpg_central[-2]["p1d_Mpc"] / mpg_seed[-2]["p1d_Mpc"])[_] - mean2,
        label="mpg-central/mpg-seed-1",
    )
    std2 = np.std(
        (mpg_central[-2]["p1d_Mpc"] / mpg_seed[-2]["p1d_Mpc"])[_] - mean2
    )

    plt.legend()
    plt.axhline(0, color="k")
    plt.xlim(0.1, 4)
    plt.ylim(-0.04, 0.04)
    plt.xscale("log")
    plt.xlabel("k[1/Mpc]")
    plt.ylabel("Relative difference")
    plt.savefig("cosmic_variance_nyx_mpg.png")
