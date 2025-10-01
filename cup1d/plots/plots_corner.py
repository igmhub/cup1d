import numpy as np
from corner import corner
from emcee.autocorr import integrated_time
import matplotlib.pyplot as plt

from matplotlib import rcParams

rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "STIXGeneral"


def plots_chain(
    folder_in, labels, folder_out=None, nburn_extra=0, ftsize=20, truth=[0, 0]
):
    lnprob = np.load(folder_in + "lnprob.npy")
    plot_lnprob(lnprob, folder_out, nburn_extra, ftsize)
    lnprob = 0

    blobs = np.load(folder_in + "blobs.npy")
    corner_blobs(
        blobs,
        folder_out=folder_out,
        nburn_extra=nburn_extra,
        ftsize=ftsize,
        truth=truth,
        labels=labels,
    )

    chain = np.load(folder_in + "chain.npy")
    auto_time = integrated_time(chain[:, :, :2], c=5.0, quiet=True)
    print(auto_time, chain.shape)

    nelem = (chain.shape[0] - nburn_extra) * chain.shape[1]
    ndim = chain.shape[-1]
    dat = np.zeros((nelem, ndim))
    dat[:, 0] = blob["Delta2_star"][nburn_extra:, :].reshape(-1) - truth[0]
    dat[:, 1] = blob["n_star"][nburn_extra:, :].reshape(-1) - truth[1]
    dat[:, 2:] = chain[:, :, 2:].reshape(-1, ndim - 2)

    corner_chain(
        dat,
        folder_out=folder_out,
        nburn_extra=nburn_extra,
        ftsize=ftsize,
        truth=truth,
        labels=labels,
    )


def plot_lnprob(lnprob, folder_out=None, nburn_extra=0, ftsize=20):
    for ii in range(lnprob.shape[1]):
        plt.plot(lnprob[nburn_extra:, ii])

    plt.plot(np.mean(lnprob[nburn_extra:, :], axis=1), lw=3, label="mean")
    plt.plot(np.median(lnprob[nburn_extra:, :], axis=1), lw=3, label="median")

    if fname_out is None:
        plt.savefig(fname_out + "lnprob.pdf")
    else:
        plt.show()
    plt.close()

    return


def corner_blobs(
    blobs, folder_out=None, nburn_extra=0, ftsize=20, truth=[0, 0], labels=None
):
    nelem = int(blob["Delta2_star"][nburn_extra:].reshape(-1).shape[0])
    dat = np.zeros((nelem, 2))
    dat[:, 0] = blob["Delta2_star"][nburn_extra:, :].reshape(-1) - truth[0]
    dat[:, 1] = blob["n_star"][nburn_extra:, :].reshape(-1) - truth[1]
    fig = corner(
        dat,
        levels=[0.68, 0.95, 0.99],
        bins=50,
        # range=[1.] * 2,
        show_titles=True,
        # color="C0",
        title_fmt=".3f",
    )

    if truth != [0, 0]:
        fig.axes[0].axvline(color="k", ls=":")
        fig.axes[3].axvline(color="k", ls=":")
        fig.axes[2].axvline(color="k", ls=":")
        fig.axes[2].axhline(color="k", ls=":")

    if fname_out is None:
        plt.savefig(fname_out + "corner_compressed.pdf")
    else:
        plt.show()
    plt.close()

    return


def corner_chain(
    dat,
    folder_out=None,
    nburn_extra=0,
    ftsize=20,
    truth=[0, 0],
    labels=None,
):
    # fig corner
    fig = corner(
        dat,
        levels=[0.68, 0.95],
        bins=30,
        range=[0.98] * ndim,
        show_titles=True,
    )

    if fname_out is None:
        plt.savefig(fname_out + "corner_all.pdf")
    else:
        plt.show()
    plt.close()

    return


def corr_compressed(
    dat, folder_out=None, nburn_extra=0, ftsize=20, truth=[0, 0], labels=None
):
    # fig corr
    for ii in range(2, ndim):
        fig, ax = plt.subplots(2, sharex=True)
        ax[0].hist2d(dat[:, ii], dat[:, 0])
        ax[1].hist2d(dat[:, ii], dat[:, 1])

        ax[0].set_ylabel(r"$\Delta_\star$")
        ax[1].set_ylabel(r"$\Delta_\star$")
        ax[1].set_xlabel(labels[ii])
        plt.savefig(fname_out + "cosmo_" + str(ii) + ".pdf")
        plt.savefig(fname_out + "cosmo_" + str(ii) + ".png")

    return


def plot_corr(chain):
    mat = np.corrcoef(dat[:, :-11], rowvar=False)

    plt.imshow(mat, cmap="turbo")
    plt.colorbar()
