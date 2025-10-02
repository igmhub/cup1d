import numpy as np
from corner import corner
from emcee.autocorr import integrated_time
import matplotlib.pyplot as plt

from matplotlib.ticker import MaxNLocator

from matplotlib import rcParams

rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "STIXGeneral"

from cup1d.utils.various_dicts import param_dict


def prepare_data(folder_in, truth=[0, 0], nburn_extra=0):
    fdict = np.load(folder_in + "fitter_results.npy", allow_pickle=True).item()
    labels = fdict["like"]["free_param_names"]

    lnprob = np.load(folder_in + "lnprob.npy")[nburn_extra:, :]
    blobs = np.load(folder_in + "blobs.npy")
    chain = np.load(folder_in + "chain.npy")

    auto_time = integrated_time(chain[nburn_extra:, :, :2], c=5.0, quiet=True)
    print(auto_time, chain.shape)
    print(
        "length should be: ",
        np.mean(auto_time) * chain.shape[0],
        np.mean(auto_time) * chain.shape[0] * 20,
    )

    nelem = (chain.shape[0] - nburn_extra) * chain.shape[1]
    ndim = chain.shape[-1]
    dat = np.zeros((nelem, ndim))
    dat[:, 0] = blobs["Delta2_star"][nburn_extra:, :].reshape(-1) - truth[0]
    dat[:, 1] = blobs["n_star"][nburn_extra:, :].reshape(-1) - truth[1]
    dat[:, 2:] = chain[nburn_extra:, :, 2:].reshape(-1, ndim - 2)
    priors = np.zeros((ndim, 2))

    for ii in range(2, ndim):
        lab = labels[ii]
        vmax = fdict["like"]["free_params"][lab]["max_value"]
        vmin = fdict["like"]["free_params"][lab]["min_value"]
        dat[:, ii] = dat[:, ii] * (vmax - vmin) + vmin
        priors[ii, 0] = vmin
        priors[ii, 1] = vmax

    labels[0] = "Delta2_star"
    labels[1] = "n_star"

    return labels, lnprob, dat, priors


def plots_chain(
    folder_in, folder_out=None, nburn_extra=0, ftsize=20, truth=[0, 0]
):
    """
    Plot the chains
    """

    labels, lnprob, dat, priors = prepare_data(
        folder_in, truth, nburn_extra=nburn_extra
    )

    # plot_lnprob(lnprob, folder_out, ftsize)

    # corr_compressed(dat, labels, priors, folder_out=folder_out)

    # plot_corr(dat, folder_out=folder_out, ftsize=ftsize)

    # corner_blobs(dat, folder_out=folder_out, ftsize=ftsize, labels=labels)

    corner_chain(dat, folder_out=folder_out, ftsize=ftsize, labels=labels)

    return


def plot_lnprob(lnprob, folder_out=None, ftsize=20):
    for ii in range(lnprob.shape[1]):
        plt.plot(lnprob[:, ii])

    plt.plot(np.mean(lnprob, axis=1), lw=3, label="mean")
    plt.plot(np.median(lnprob, axis=1), lw=3, label="median")

    if folder_out is None:
        plt.savefig(folder_out + "lnprob.pdf")
    else:
        plt.show()
    plt.close()

    return


def corner_blobs(dat, folder_out=None, ftsize=20, labels=None):
    labs = []
    for ilab in range(2):
        labs.append(param_dict[labels[ilab]])

    fig = corner(
        dat[:, :2],
        levels=[0.68, 0.95],
        bins=50,
        range=[0.98] * 2,
        show_titles=True,
        # color="C0",
        title_fmt=".3f",
        label_kwargs={"fontsize": ftsize},
        title_kwargs={"fontsize": ftsize - 2},
        labels=labs,
    )

    fig.axes[0].axvline(color="k", ls=":")
    fig.axes[3].axvline(color="k", ls=":")
    fig.axes[2].axvline(color="k", ls=":")
    fig.axes[2].axhline(color="k", ls=":")

    for ax in fig.get_axes():
        ax.tick_params(labelsize=ftsize - 4)

    if folder_out is None:
        plt.savefig(folder_out + "corner_compressed.pdf")
    else:
        plt.show()
    plt.close()

    return


def corner_chain(dat, folder_out=None, ftsize=20, labels=None):
    # fig corner
    ndim = dat.shape[1]
    labs = []
    for ilab in range(ndim):
        labs.append(param_dict[labels[ilab]])
    fig = corner(
        dat,
        levels=[0.68, 0.95],
        bins=30,
        range=[0.98] * ndim,
        show_titles=True,
        # color="C0",
        title_fmt=".3f",
        label_kwargs={"fontsize": ftsize},
        title_kwargs={"fontsize": ftsize - 2},
        labels=labs,
    )

    for ax in fig.get_axes():
        ax.tick_params(labelsize=ftsize - 4)

    if folder_out is None:
        plt.savefig(folder_out + "corner_all.pdf")
    else:
        plt.show()
    plt.close()

    return


def corr_compressed(
    dat, labels, priors, folder_out=None, ftsize=20, truth=[0, 0], sigmas=2
):
    # labels = fdict["like"]["free_param_names"]
    frange = 0.1
    cmap = plt.get_cmap("Blues")
    groups = ["tau", "sigT_kms", "gamma", "Lya", ["SiIIa", "SiIIb"], "HCD"]
    egroups = ["cte", "cte", "cte", "exp", "exp", "exp"]
    # groups = [["SiIIa", "SiIIb"]]
    # egroups = ["exp"]
    for igroup, key in enumerate(groups):
        lab_use = {}
        for ii, lab in enumerate(labels):
            if len(key) == 2:
                if (key[0] in lab) | (key[1] in lab):
                    lab_use[lab] = ii
            else:
                if key in lab:
                    lab_use[lab] = ii

        if key in ["tau", "sigT_kms", "gamma"]:
            sharex = "all"
        else:
            sharex = "col"

        fig, ax = plt.subplots(
            2,
            len(lab_use),
            sharex="col",
            sharey="row",
            figsize=(len(lab_use) * 3, 8),
        )
        for ii, lab in enumerate(lab_use):
            pp = dat[:, lab_use[lab]]

            x, y, h, levels = get_contours(pp, dat[:, 0], sigmas=sigmas)
            if egroups[igroup] == "cte":
                xplot = x
            else:
                xplot = np.exp(x)
                if lab.startswith("s_"):
                    xplot = 1 / xplot
            cs1 = ax[0, ii].contour(xplot, y, h, levels=levels, colors="k")
            cs1 = ax[0, ii].contourf(xplot, y, h, levels=levels, cmap=cmap)

            x, y, h, levels = get_contours(pp, dat[:, 1], sigmas=sigmas)
            if egroups[igroup] == "cte":
                xplot = x
            else:
                xplot = np.exp(x)
                if lab.startswith("s_"):
                    xplot = 1 / xplot
            cs2 = ax[1, ii].contour(xplot, y, h, levels=levels, colors="k")
            cs2 = ax[1, ii].contourf(xplot, y, h, levels=levels, cmap=cmap)

            ##
            x_line = []
            y_line = []
            for collection in cs1.collections:
                for path in collection.get_paths():
                    v = path.vertices
                    x_line.extend(v[:, 0])
                    y_line.extend(v[:, 1])

            if ii == 0:
                yrange1 = [np.min(y_line), np.max(y_line)]
                change_alims = True
                change_blims = True
            else:
                if np.min(y_line) < yrange1[0]:
                    yrange1[0] = np.min(y_line)
                    change_alims = True
                else:
                    change_alims = False
                if np.max(y_line) > yrange1[1]:
                    yrange1[1] = np.max(y_line)
                    change_blims = True
                else:
                    change_blims = False
            if change_alims | change_blims:
                dy = yrange1[1] - yrange1[0]
                if change_alims:
                    yrange1[0] -= frange * dy
                if change_blims:
                    yrange1[1] += frange * dy
            ##

            ##
            x_line = []
            y_line = []
            for collection in cs2.collections:
                for path in collection.get_paths():
                    v = path.vertices
                    x_line.extend(v[:, 0])
                    y_line.extend(v[:, 1])

            if ii == 0:
                yrange2 = [np.min(y_line), np.max(y_line)]
                change_alims = True
                change_blims = True
            else:
                if np.min(y_line) < yrange2[0]:
                    yrange2[0] = np.min(y_line)
                    change_alims = True
                else:
                    change_alims = False
                if np.max(y_line) > yrange2[1]:
                    yrange2[1] = np.max(y_line)
                    change_blims = True
                else:
                    change_blims = False
            if change_alims | change_blims:
                dy = yrange2[1] - yrange2[0]
                if change_alims:
                    yrange2[0] -= frange * dy
                if change_blims:
                    yrange2[1] += frange * dy
            ##

            xrange = [np.min(x_line), np.max(x_line)]
            dy = xrange[1] - xrange[0]
            xrange[0] -= frange * dy
            xrange[1] += frange * dy

            # if sigmas == 2:
            #     range_par = np.percentile(dat[:, lab_use[lab]], [0.1, 99.9])
            # else:
            #     range_par = np.percentile(dat[:, lab_use[lab]], [3, 97])
            ax[1, ii].set_xlabel(param_dict[lab], fontsize=ftsize + 2)

            for i0 in range(2):
                ax[i0, ii].set_xlim(xrange)
                ax[i0, ii].xaxis.set_major_locator(
                    MaxNLocator(nbins=3, prune=None)
                )
                ax[i0, ii].tick_params(
                    axis="both", which="major", labelsize=ftsize
                )
                for j0 in range(2):
                    if egroups[igroup] == "cte":
                        xplot = priors[lab_use[lab], j0]
                    else:
                        xplot = np.exp(priors[lab_use[lab], j0])
                        if lab.startswith("s_"):
                            xplot = 1 / xplot
                    ax[i0, ii].axvline(xplot, ls=":", color="k")

        ax[0, 0].set_ylim(yrange1)
        ax[1, 0].set_ylim(yrange2)
        ax[0, 0].set_ylabel(r"$\Delta_\star$", fontsize=ftsize + 2)
        ax[1, 0].set_ylabel(r"$n_\star$", fontsize=ftsize + 2)

        plt.tight_layout()
        if folder_out is None:
            plt.show()
        else:
            plt.savefig(folder_out + "corr_compressed_" + str(igroup) + ".pdf")
            plt.savefig(folder_out + "corr_compressed_" + str(igroup) + ".png")

    return


def plot_corr(dat, ftsize=20, folder_out=None):
    mat = np.corrcoef(dat[:, :-11], rowvar=False)

    plt.imshow(mat, cmap="turbo")
    plt.colorbar()

    plt.tight_layout()
    if folder_out is None:
        plt.show()
    else:
        plt.savefig(folder_out + "corr_mat.pdf")
        plt.savefig(folder_out + "corr_mat.png")


def get_contours(x, y, sigmas=1, bins=30):
    """
    Return mesh (X,Y), histogram values H (shape matches X,Y), and contour
    thresholds that enclose 68% and optionally 95% of the samples.

    Usage:
        X, Y, H, levels = get_contours(x, y, sigmas=2, bins=50)
        plt.contour(X, Y, H, levels=levels)
    """
    # Use raw counts (simpler to work with probability mass)
    H, xedges, yedges = np.histogram2d(x, y, bins=bins, density=False)

    # bin centres for plotting (meshgrid expects shape (ny, nx))
    xcenters = 0.5 * (xedges[:-1] + xedges[1:])
    ycenters = 0.5 * (yedges[:-1] + yedges[1:])
    X, Y = np.meshgrid(xcenters, ycenters)

    # flatten and sort counts descending
    Hflat = H.flatten()
    idx = np.argsort(Hflat)[::-1]
    Hsorted = Hflat[idx]

    # cumulative probability (mass)
    cumsum = np.cumsum(Hsorted)
    cumsum = cumsum / cumsum[-1]

    probs = [0, 0.68] if sigmas == 1 else [0, 0.68, 0.95]

    levels = []
    for p in probs:
        k = np.searchsorted(cumsum, p)  # first index where cumsum >= p
        k = min(k, len(Hsorted) - 1)
        levels.append(Hsorted[k])

    levels = np.sort(levels)  # matplotlib requires ascending levels

    # Return H transposed so it matches X, Y shapes for plt.contour(X, Y, H, ...)
    return X, Y, H.T, levels
