"""Likelihood plotting implementations; scientific objects retain compatibility wrappers."""

import numpy as np
import matplotlib.pyplot as plt
import os
from cup1d.postprocessing.p1d import (  # noqa: F401 - public legacy re-exports
    P1DPlotter,
    old_plot_p1d,
    plot_p1d,
    plot_p1d_errors,
    plot_p1d_residuals,
    plot_p1d_spectra,
)


def plot_cov_terms(self, save_directory=None):
    """Plot fractional statistical, systematic, and emulator variances.

    Parameters
    ----------
    self : object
        Likelihood object exposing per-redshift total, emulator, and data
        covariance blocks in ``(km / s)**2``.
    save_directory : str or path-like, optional
        Directory for ``cov_terms.pdf`` and ``cov_terms.png``.

    Notes
    -----
    This helper creates and displays the figure in place and returns None.
    """
    npanels = int(np.round(np.sqrt(len(self.cov_Pk_kms))))
    fig, ax = plt.subplots(
        npanels + 1, npanels, sharex=True, sharey=True, figsize=(10, 8)
    )
    ax = ax.reshape(-1)
    for ii in range(len(self.cov_Pk_kms)):
        cov_stat = np.diag(self.data.covstat_Pk_kms[ii])
        cov_syst = np.diag(self.data.cov_Pk_kms[ii]) - cov_stat
        cov_emu = np.diag(self.cov_emu_Pk_kms[ii])
        cov_tot = np.diag(self.cov_Pk_kms[ii])
        ax[ii].plot(self.data.k_kms[ii], cov_stat / cov_tot, label=r"$x$ = Stat")
        ax[ii].plot(self.data.k_kms[ii], cov_syst / cov_tot, label=r"$x$ = Syst")
        ax[ii].plot(self.data.k_kms[ii], cov_emu / cov_tot, label=r"$x$ = Emu")
        ax[ii].text(0.0, 0.1, "z=" + str(self.data.z[ii]))
    if len(ax) > len(self.cov_Pk_kms):
        for ii in range(len(self.cov_Pk_kms), len(ax)):
            ax[ii].axis("off")
    ax[0].legend()
    fig.supxlabel(r"$k\,[\mathrm{km}^{-1}\mathrm{s}]$")
    fig.supylabel(r"$\sigma^2_x/\sigma^2_\mathrm{total}$")
    plt.tight_layout()

    if save_directory is not None:
        name = os.path.join(save_directory, "cov_terms")
        plt.savefig(name + ".pdf")
        plt.savefig(name + ".png")
    else:
        plt.show()


def plot_cov_to_pk(
    self, use_pk_smooth=True, fname=None, ftsize=18, store_data=False
):
    """Plot diagonal covariance components relative to P1D.

    Parameters
    ----------
    self : object
        Likelihood object with named P1D data, covariance scaling, and emulator
        covariance blocks.
    use_pk_smooth : bool, default: True
        Normalize errors by smooth P1D rather than measured P1D.
    fname : str or path-like, optional
        File root for PDF and PNG outputs.
    ftsize : float, default: 18
        Base figure font size.
    store_data : bool, default: False
        Return plotted wavenumber and fractional-error arrays.

    Returns
    -------
    dict, optional
        Per-redshift ``xN`` and colored fractional-error arrays when requested.
    """
    key = list(self.data.keys())[0]
    nz = len(self.data[key].z)
    npanels = int(np.round(np.sqrt(nz)))

    fig, ax = plt.subplots(
        npanels + 1, npanels, sharex=True, sharey="row", figsize=(10, 8)
    )
    ax = ax.reshape(-1)

    if store_data:
        out_data = {}
    for ii in range(nz):
        cov_stat = np.diag(self.data[key].covstat_Pk_kms[ii])
        cov_syst = np.diag(self.data[key].cov_Pk_kms[ii]) - cov_stat

        ind = np.argmin(np.abs(self.cov_factor["z"] - self.data[key].z[ii]))
        # inflate errors stat
        cov_stat = cov_stat * self.cov_factor["val_stat"][ind] ** 2
        # inflate errors syst
        cov_syst = cov_syst * self.cov_factor["val_syst"][ind] ** 2

        cov_emu = np.diag(self.cov_emu_Pk_kms[key][ii])
        cov_tot = np.diag(self.cov_Pk_kms[key][ii])
        if use_pk_smooth:
            pk = self.data[key].Pksmooth_kms[ii].copy()
        else:
            pk = self.data[key].Pk_kms[ii].copy()

        if store_data:
            out_data["x" + str(ii)] = self.data[key].k_kms[ii]
            out_data["y" + str(ii) + "_blue"] = np.sqrt(cov_stat) / pk
            out_data["y" + str(ii) + "_orange"] = np.sqrt(cov_syst) / pk
            out_data["y" + str(ii) + "_green"] = np.sqrt(cov_emu) / pk
            out_data["y" + str(ii) + "_red"] = np.sqrt(cov_tot) / pk

        ax[ii].plot(
            self.data[key].k_kms[ii],
            np.sqrt(cov_stat) / pk,
            ls="-",
            lw=3,
        )
        ax[ii].plot(
            self.data[key].k_kms[ii],
            np.sqrt(cov_syst) / pk,
            ls=":",
            lw=3,
        )
        ax[ii].plot(
            self.data[key].k_kms[ii],
            np.sqrt(cov_emu) / pk,
            ls="--",
            lw=3,
        )
        ax[ii].plot(
            self.data[key].k_kms[ii],
            np.sqrt(cov_tot) / pk,
            ls="-.",
            lw=3,
        )
        ax[ii].text(
            0.05,
            0.95,
            "z=" + str(self.data[key].z[ii]),
            ha="left",
            va="top",
            transform=ax[ii].transAxes,
            fontsize=ftsize,
        )
        ax[ii].tick_params(axis="both", which="major", labelsize=ftsize)
    if len(ax) > nz:
        for ii in range(nz, len(ax)):
            ax[ii].axis("off")

    labs = ["stat", "syst", "emu", "total"]
    lss = ["-", ":", "--", "-."]
    for ii in range(4):
        ax[-1].plot(
            [0, 0],
            [0, 0],
            label=r"$\sigma_x = \sigma_\mathrm{" + labs[ii] + "}$",
            ls=lss[ii],
            lw=3,
        )
    ax[-1].legend(fontsize=ftsize, loc="upper left")
    fig.supxlabel(
        r"$k_\parallel\,[\mathrm{km}^{-1}\mathrm{s}]$", fontsize=ftsize + 2
    )
    fig.supylabel(r"$\sigma_x/P_\mathrm{1D}$", fontsize=ftsize + 2)
    ax[0].set_ylim(0.0, 0.06)
    ax[3].set_ylim(0.0, 0.06)
    plt.tight_layout()

    if fname is not None:
        plt.savefig(fname + ".pdf")
        plt.savefig(fname + ".png")
    else:
        plt.show()

    if store_data:
        return out_data


def plot_correlation_matrix(self, save_directory=None):
    """Plot the full P1D covariance normalized to a correlation matrix.

    Parameters
    ----------
    self : object
        Likelihood object exposing ``full_cov_Pk_kms`` in ``(km / s)**2``.
    save_directory : str or path-like, optional
        Directory for ``correlation.pdf`` and ``correlation.png``.

    Notes
    -----
    Zero covariance entries are assigned zero correlation.  The legacy local
    positive-definiteness helper is retained but not currently invoked.
    """
    def correlation_from_covariance(covariance):
        """Normalize a covariance matrix to correlation coefficients.

        Parameters
        ----------
        covariance : ndarray
            Square covariance matrix.

        Returns
        -------
        ndarray
            Correlation matrix of the same shape, with exact covariance zeros
            assigned zero correlation.
        """
        v = np.sqrt(np.diag(covariance))
        outer_v = np.outer(v, v)
        correlation = covariance / outer_v
        correlation[covariance == 0] = 0
        return correlation

    def is_pos_def(x):
        """Return whether all eigenvalues of a square matrix are positive.

        Parameters
        ----------
        x : ndarray
            Square matrix to test.

        Returns
        -------
        bool
            True only when every eigenvalue is strictly positive.
        """
        return np.all(np.linalg.eigvals(x) > 0)

    plt.imshow(correlation_from_covariance(self.full_cov_Pk_kms))
    plt.colorbar()

    if save_directory is not None:
        name = os.path.join(save_directory, "correlation")
        plt.savefig(name + ".pdf")
        plt.savefig(name + ".png")
    else:
        plt.show()


def plot_hull_fid(self, like_params=None):
    """Plot pairwise emulator-hull projections at a likelihood point.

    Parameters
    ----------
    self : object
        Theory/likelihood object exposing emulator-call construction and a
        redshift-indexed hull.
    like_params : mapping, optional
        Physical likelihood parameters used to construct emulator calls.

    Notes
    -----
    The resulting emulator parameter array has shape ``(nz, n_hull_params)``.
    Plotting is delegated to the hull object and this helper returns None.
    """
    emu_call, M_of_z = self.theory.get_emulator_calls(
        self.data.z, like_params=like_params
    )
    p1 = np.zeros(
        (
            self.theory.hull.nz,
            len(self.theory.hull.params),
        )
    )
    for jj, key in enumerate(self.theory.hull.params):
        p1[:, jj] = emu_call[key]

    self.theory.hull.plot_hulls(p1)
