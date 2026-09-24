"""Data.simulations plotting implementations; scientific objects retain compatibility wrappers."""

import numpy as np
import matplotlib.pyplot as plt


def plot_p1d_z(self, out_dict):
    for ii in range(out_dict["p1d_Mpc"].shape[0]):
        plt.plot(
            out_dict["k1d_Mpc"],
            out_dict["k1d_Mpc"] * out_dict["p1d_Mpc"][ii] / np.pi,
            label=str(out_dict["z"][ii]),
        )
    plt.legend()
    plt.yscale("log")
    plt.xscale("log")


def plot_p1d_axes(self, out_dict):
    labs_dirs = ["x", "y", "z"]
    iz = 0
    for ii in range(3):
        plt.plot(
            out_dict["k1d_Mpc"],
            out_dict["p1d_Mpc_axes"][iz, :, ii] / out_dict["p1d_Mpc"][iz],
            label=labs_dirs[ii],
        )
    plt.axhline(1, ls=":", color="k")
    plt.axhline(1.01, ls=":", color="k")
    plt.axhline(0.99, ls=":", color="k")
    plt.xscale("log")
    # plt.ylim(0.98, 1.02)
    plt.legend()
    plt.ylabel("P1D_direction/P1D_average-1")


def plot_p3d_z(self, out_dict):
    for iz in range(0, 5, 2):
        for ii in range(out_dict["k3d_Mpc"].shape[1]):
            col = "C" + str(ii)
            plt.loglog(
                out_dict["k3d_Mpc"][:, ii],
                out_dict["k3d_Mpc"][:, ii] ** 3
                * out_dict["p3d_Mpc"][iz, :, ii]
                / 2
                / np.pi**2,
                col,
            )
