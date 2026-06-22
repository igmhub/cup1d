import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import colormaps


def plot_p1d(
    zs,
    k_kms,
    Pk_kms,
    cov_Pk_kms,
    use_dimensionless=True,
    xlog=False,
    ylog=True,
    fname=None,
    cov_ext=None,
    ftsize=18,
    store_data=False,
):
    """Plot P1D mesurement. If use_dimensionless, plot k*P(k)/pi."""

    rcParams["mathtext.fontset"] = "stix"
    rcParams["font.family"] = "STIXGeneral"

    if store_data:
        out_data = {}

    fig, ax = plt.subplots(figsize=(8, 6))

    N = len(zs)
    for ii in range(N):
        if cov_ext is None:
            err_Pk_kms = np.sqrt(np.diagonal(cov_Pk_kms[ii]))
        else:
            err_Pk_kms = np.sqrt(np.diagonal(cov_ext[ii]))
        if use_dimensionless:
            fact = k_kms[ii] / np.pi
        else:
            fact = 1.0

        if store_data:
            out_data["x" + str(ii)] = k_kms[ii]
            out_data["y" + str(ii)] = fact * Pk_kms[ii]
            out_data["err" + str(ii)] = fact * err_Pk_kms

        ax.errorbar(
            k_kms[ii],
            fact * Pk_kms[ii],
            yerr=fact * err_Pk_kms,
            label=r"$z = {}$".format(np.round(zs[ii], 3)),
            color=colormaps["tab20"].colors[ii],
        )

    ax.legend(ncol=4, fontsize=ftsize - 4)
    if ylog:
        plt.yscale("log", nonpositive="clip")
    if xlog:
        plt.xscale("log")
    plt.xlabel(r"$k_\parallel\,[\mathrm{km}^{-1} \mathrm{s}]$", fontsize=ftsize)
    if use_dimensionless:
        plt.ylabel(r"$\mathrm{\pi}^{-1}k_\parallel\,P(k)$", fontsize=ftsize)
    else:
        plt.ylabel(r"$P(k) [km/s]$", fontsize=ftsize)

    ax.tick_params(axis="both", which="major", labelsize=ftsize)
    plt.tight_layout()

    if fname is not None:
        plt.savefig(fname + ".pdf")
        plt.savefig(fname + ".png")
    else:
        plt.show()

    if store_data:
        return out_data
