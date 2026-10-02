"""Data.simulations plotting implementations; scientific objects retain compatibility wrappers."""

import numpy as np
import matplotlib.pyplot as plt


def plot_p1d_z(self, out_dict):
    """Plot dimensionless ACCEL2 P1D for every available redshift.

    Parameters
    ----------
    self : object
        Compatibility receiver; no instance state is read.
    out_dict : dict
        ACCEL2 payload with ``z``, ``k1d_Mpc`` in ``1 / Mpc``, and ``p1d_Mpc``
        in ``Mpc`` as returned by :func:`cup1d.p1ds.simulations.data_accel2.load_data`.
    """
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
    """Plot directional-to-mean P1D ratios for the first ACCEL2 redshift.

    Parameters
    ----------
    self : object
        Compatibility receiver; no instance state is read.
    out_dict : dict
        ACCEL2 payload containing ``k1d_Mpc``, ``p1d_Mpc``, and
        ``p1d_Mpc_axes`` with final axis ordered as x, y, z.
    """
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
    """Plot dimensionless ACCEL2 P3D for selected redshifts and mu bins.

    Parameters
    ----------
    self : object
        Compatibility receiver; no instance state is read.
    out_dict : dict
        ACCEL2 payload containing ``k3d_Mpc`` in ``1 / Mpc`` and ``p3d_Mpc``
        in ``Mpc**3`` on ``(redshift, k, mu)`` grids.
    """
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
