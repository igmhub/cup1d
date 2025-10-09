import numpy as np
import os
from corner import corner
from emcee.autocorr import integrated_time
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter
from matplotlib.ticker import MaxNLocator
from scipy.stats import gaussian_kde

from matplotlib import rcParams
import matplotlib
from scipy.stats import chi2 as chi2_scipy

# from mpl_toolkits.axes_grid1.inset_locator import inset_axes


rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "STIXGeneral"

from cup1d.utils.various_dicts import param_dict


def prepare_data(folder_in, truth=[0, 0], nburn_extra=0):
    fdict = np.load(
        os.path.join(folder_in, "fitter_results.npy"), allow_pickle=True
    ).item()
    labels = fdict["like"]["free_param_names"]

    lnprob = np.load(os.path.join(folder_in, "lnprob.npy"))[nburn_extra:, :]
    blobs = np.load(os.path.join(folder_in, "blobs.npy"))
    chain = np.load(os.path.join(folder_in, "chain.npy"))

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
    dat_Asns = chain[nburn_extra:, :, :2].reshape(-1, 2)

    for ii in range(2):
        lab = labels[ii]
        vmax = fdict["like"]["free_params"][lab]["max_value"]
        vmin = fdict["like"]["free_params"][lab]["min_value"]
        dat_Asns[:, ii] = dat_Asns[:, ii] * (vmax - vmin) + vmin

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

    return labels, lnprob, dat, priors, dat_Asns


def plots_chain(
    folder_in, folder_out=None, nburn_extra=0, ftsize=20, truth=[0, 0]
):
    """
    Plot the chains
    """

    if folder_out is None:
        folder_out = folder_in

    labels, lnprob, dat, priors, dat_Asns = prepare_data(
        folder_in, truth, nburn_extra=nburn_extra
    )

    # plot_corr(dat, labels, folder_out=folder_out, ftsize=ftsize)
    # if 1 > 0:
    #     return

    try:
        plot_lnprob(lnprob, folder_out, ftsize)
    except:
        print("Could not plot lnprob")

    try:
        corr_compressed(dat, labels, priors, folder_out=folder_out)
    except:
        print("Could not plot corr_compressed")

    try:
        plot_corr(dat, labels, folder_out=folder_out, ftsize=ftsize)
    except:
        print("Could not plot corr")

    try:
        corner_blobs(dat, folder_out=folder_out, ftsize=ftsize, labels=labels)
    except:
        print("Could not plot corner_blobs")

    try:
        save_contours(dat[:, 0], dat[:, 1], folder_out=folder_out)
    except:
        print("Could not save contours")

    try:
        save_contours(
            dat_Asns[:, 0], dat_Asns[:, 1], folder_out=folder_out, flag="_Asns"
        )
    except:
        print("Could not save contours")

    try:
        get_summary(folder_out)
    except:
        print("Could not get summary")

    # corner_chain(dat, folder_out=folder_out, ftsize=ftsize, labels=labels)


def get_summary(folder_out):
    dict_out = {}

    data = np.load(
        os.path.join(folder_out, "fitter_results.npy"), allow_pickle=True
    ).item()
    dict_out["ndata"] = data["data"]["full_Pk_kms"].shape[0]
    dict_out["npar"] = data["fitter"]["mle_cube"].shape[0]
    dict_out["ndeg"] = dict_out["ndata"] - dict_out["npar"]
    dict_out["chi2"] = -2 * np.max(
        [
            data["fitter"]["lnprob_mle"],
            np.load(os.path.join(folder_out, "lnprob.npy")).max(),
        ]
    )
    dict_out["prob_chi2"] = chi2_scipy.sf(dict_out["chi2"], dict_out["ndeg"])

    data = np.load(
        os.path.join(folder_out, "line_sigmas.npy"), allow_pickle=True
    ).item()
    dict_out["delta2_star_2dcen"] = np.median(data[0.68][0][0])
    dict_out["nstar_2dcen"] = np.median(data[0.68][0][1])

    data = np.load(os.path.join(folder_out, "blobs.npy"))
    dict_out["delta2_star_16_50_84"] = np.percentile(
        data["Delta2_star"], [16, 50, 84]
    )
    dict_out["n_star_16_50_84"] = np.percentile(data["n_star"], [16, 50, 84])

    dict_out["delta2_star_err"] = 0.5 * (
        dict_out["delta2_star_16_50_84"][2]
        - dict_out["delta2_star_16_50_84"][0]
    )
    dict_out["n_star_err"] = 0.5 * (
        dict_out["n_star_16_50_84"][2] - dict_out["n_star_16_50_84"][0]
    )

    print(dict_out)

    np.save(os.path.join(folder_out, "summary.npy"), dict_out)


def save_contours(x, y, folder_out=None, bins=50, flag=""):
    """Extract contours from 2D histogram"""
    H, xedges, yedges = np.histogram2d(x, y, bins=bins, density=True)

    # Compute cumulative distribution in descending order
    H_flat = H.flatten()
    idx = np.argsort(H_flat)[::-1]
    H_sorted = H_flat[idx]
    cdf = np.cumsum(H_sorted) / np.sum(H_sorted)

    # Density thresholds for 1σ and 2σ
    levels = [H_sorted[np.searchsorted(cdf, p)] for p in [0.68, 0.95]]
    # Reverse for contour plotting
    levels_plot = np.sort(levels)  # increasing order for plt.contour

    # Map original sigma to threshold used for plotting
    sigma_to_level = {0.68: levels[0], 0.95: levels[1]}

    # Create meshgrid
    X = 0.5 * (xedges[:-1] + xedges[1:])
    Y = 0.5 * (yedges[:-1] + yedges[1:])
    X, Y = np.meshgrid(X, Y)

    # Compute contours
    cs = plt.contour(X, Y, H.T, levels=levels_plot)

    # Extract vertices for each level using allsegs
    contours_dict = {}
    for sigma, segs in zip([0.68, 0.95], cs.allsegs):
        level_contours = []
        # cs.allsegs is in the same order as levels_plot (increasing)
        # so match by density
        threshold = sigma_to_level[sigma]
        # find the matching index in levels_plot
        idx = np.where(levels_plot == threshold)[0][0]
        for seg in cs.allsegs[idx]:
            x_line, y_line = seg[:, 0], seg[:, 1]
            level_contours.append((x_line, y_line))
        contours_dict[sigma] = level_contours

    np.save(
        os.path.join(folder_out, "line_sigmas" + flag + ".npy"), contours_dict
    )
    plt.close()

    return


def plot_lnprob(lnprob, folder_out=None, ftsize=20):
    print("plotting lnprob")
    for ii in range(lnprob.shape[1]):
        plt.plot(lnprob[:, ii])

    plt.plot(np.mean(lnprob, axis=1), lw=3, label="mean")
    plt.plot(np.median(lnprob, axis=1), lw=3, label="median")

    if folder_out is not None:
        plt.savefig(os.path.join(folder_out, "lnprob.pdf"))
        plt.savefig(os.path.join(folder_out, "lnprob.png"))
    else:
        plt.show()
    plt.close()

    return


def corner_blobs(dat, folder_out=None, ftsize=20, labels=None):
    print("plotting corner_blobs")
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

    if folder_out is not None:
        plt.savefig(os.path.join(folder_out, "corner_compressed.pdf"))
        plt.savefig(os.path.join(folder_out, "corner_compressed.png"))
    else:
        plt.show()

    plt.close()

    return


def corner_chain(dat, folder_out=None, ftsize=20, labels=None, divs=2):
    print("plotting corner_chain")
    # fig corner
    ndim = dat.shape[1] - 11
    labs = []
    for ilab in range(ndim):
        labs.append(param_dict[labels[ilab]])

    inds = np.array_split(np.arange(ndim), divs)

    for ii in range(divs):
        if ii != 0:
            ind = np.concatenate([np.arange(2), inds[ii]])
        else:
            ind = inds[ii]
        fig = corner(
            dat[:, ind],
            levels=[0.68, 0.95],
            bins=30,
            range=[0.98] * len(ind),
            show_titles=True,
            title_fmt=".3f",
            label_kwargs={"fontsize": ftsize},
            title_kwargs={"fontsize": ftsize - 2},
            labels=np.array(labs)[ind],
        )

        for ax in fig.get_axes():
            ax.tick_params(labelsize=ftsize - 4)

        if folder_out is not None:
            plt.savefig(
                os.path.join(folder_out, "corner_all" + str(ii) + ".pdf")
            )
            plt.savefig(
                os.path.join(folder_out, "corner_all" + str(ii) + ".png")
            )
        else:
            plt.show()
        plt.close()

    return


def corr_compressed(
    dat,
    labels,
    priors,
    folder_out=None,
    ftsize=20,
    truth=[0, 0],
    sigmas=2,
    threshold=1e-4,
):
    print("plotting corr_compressed")
    # labels = fdict["like"]["free_param_names"]
    frange = 0.1
    cmap = plt.get_cmap("Blues")
    groups = [
        "tau",
        "sigT_kms",
        "gamma",
        "Lya",
        ["SiIIa", "SiIIb"],
        "HCD",
        "mix",
    ]
    # groups = [
    #     "mix",
    # ]
    # groups = [["HCD_damp1", "HCD_damp2"]]
    egroups = ["exp", "cte", "cte", "exp", "exp", "exp", "exp"]
    # groups = [["SiIIa", "SiIIb"]]
    # egroups = ["exp"]
    for igroup, key in enumerate(groups):
        lab_use = {}

        if key != "mix":
            for ii, lab in enumerate(labels):
                if len(key) == 2:
                    if (key[0] in lab) | (key[1] in lab):
                        lab_use[lab] = ii
                else:
                    if key in lab:
                        lab_use[lab] = ii
        else:
            lab_use = {
                "tau_eff_3": 5,
                "HCD_damp1_0": 30,
            }

        if key in ["tau", "sigT_kms", "gamma"]:
            sharex = "all"
        else:
            sharex = "col"

        if key != "mix":
            xsize = len(lab_use) * 3
        else:
            xsize = 8

        fig, ax = plt.subplots(
            2, len(lab_use), sharex="col", sharey="row", figsize=(xsize, 6)
        )
        for ii, lab in enumerate(lab_use):
            pp = dat[:, lab_use[lab]].copy()

            if egroups[igroup] == "cte":
                pass
            else:
                pp = np.exp(pp)
                if lab.startswith("s_"):
                    pp = 1 / pp

            x, y, h, levels = get_contours(
                pp, dat[:, 0], sigmas=sigmas, threshold=threshold
            )
            cs1 = ax[0, ii].contour(x, y, h, levels=levels, colors="k")
            cs1 = ax[0, ii].contourf(x, y, h, levels=levels, cmap=cmap)

            x, y, h, levels = get_contours(
                pp, dat[:, 1], sigmas=sigmas, threshold=threshold
            )
            cs2 = ax[1, ii].contour(x, y, h, levels=levels, colors="k")
            cs2 = ax[1, ii].contourf(x, y, h, levels=levels, cmap=cmap)

            for j1 in range(2):
                rr = np.corrcoef(pp, dat[:, j1])[0, 1]
                if rr < 0:
                    xppos = 0.95
                    ha = "right"
                else:
                    xppos = 0.05
                    ha = "left"
                ax[j1, ii].text(
                    xppos,
                    0.95,
                    r"$r=$" + str(np.round(rr, 2)),
                    transform=ax[j1, ii].transAxes,
                    ha=ha,
                    va="top",
                    fontsize=ftsize + 1,
                )

            ##
            x_line = []
            y_line = []
            for path in cs1.get_paths():
                v = path.vertices
                x_line.extend(v[:, 0])
                y_line.extend(v[:, 1])

            # print(lab, np.min(y_line), np.max(y_line))
            # print(np.min(x_line), np.max(x_line))

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
            for path in cs2.get_paths():
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
            if pp.min() < threshold:
                xrange[0] = 0

            # if sigmas == 2:
            #     range_par = np.percentile(dat[:, lab_use[lab]], [0.1, 99.9])
            # else:
            #     range_par = np.percentile(dat[:, lab_use[lab]], [3, 97])
            ax[1, ii].set_xlabel(param_dict[lab], fontsize=ftsize + 2)

            # ax[1, ii].axhline(-2.2, ls=":", color="k")

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
        ax[0, 0].set_ylabel(r"$\Delta^2_\star$", fontsize=ftsize + 2)
        ax[1, 0].set_ylabel(r"$n_\star$", fontsize=ftsize + 2)

        plt.tight_layout()
        if folder_out is None:
            plt.show()
        else:
            plt.savefig(
                os.path.join(
                    folder_out, "corr_compressed_" + str(igroup) + ".pdf"
                ),
                bbox_inches="tight",
            )
            plt.savefig(
                os.path.join(
                    folder_out, "corr_compressed_" + str(igroup) + ".png"
                ),
                bbox_inches="tight",
            )
        plt.close()

    return


def plot_corr(dat, labs, ftsize=20, folder_out=None, threshold=0.35):
    print("Plotting correlation matrix")

    groups = ["tau", "sigT_kms", "gamma", "Lya", ["SiIIa", "SiIIb"], "HCD"]

    for iiter in range(3):
        if iiter == 0:
            egroups = ["cte", "cte", "cte", "exp", "exp", "exp"]
            pdata = dat[:, :-11].copy()
        elif iiter == 1:
            egroups = ["cte", "cte", "cte", "cte", "cte", "cte"]
            pdata = dat[:, :-11].copy()
        elif iiter == 2:
            egroups = ["cte", "cte", "cte", "cte", "cte", "cte"]
            pdata = dat[:, :].copy()

        labels = []
        for ii in range(pdata.shape[1]):
            lab = labs[ii]
            labels.append(param_dict[lab])

            if ii < 2:
                continue

            for jj, key in enumerate(groups):
                inside = False
                if len(key) == 2:
                    if (key[0] in lab) | (key[1] in lab):
                        inside = True
                else:
                    if key in lab:
                        inside = True

                if inside:
                    # print("Group", key, "found in", lab, egroups[jj])
                    if egroups[jj] == "cte":
                        pass
                    else:
                        pdata[ii] = np.exp(pdata[ii])
                        if lab.startswith("s_"):
                            pdata[ii] = 1 / pdata[ii]
                    break

        mat = np.corrcoef(pdata, rowvar=False)

        # Mask upper triangle
        mask = np.triu(np.ones_like(mat, dtype=bool))

        fig, ax = plt.subplots(figsize=(12, 12))

        # Apply mask: set upper triangle to NaN
        mat_masked = np.ma.masked_where(mask, mat)

        # Heatmap
        im = ax.imshow(mat_masked, cmap="bwr", vmin=-1, vmax=1)

        # Colorbar
        # get the axes position in figure coordinates
        bbox = ax.get_position()

        # define the colorbar inset position (top-right inside the axes)
        left = bbox.x0 + bbox.width * 0.58  # horizontal offset from left
        bottom = bbox.y0 + bbox.height * 0.97  # vertical offset from bottom
        width = bbox.width * 0.45  # width of colorbar
        height = bbox.height * 0.05  # height of colorbar

        # add a small inset axes for the colorbar
        cax = fig.add_axes([left, bottom, width, height])

        # create the colorbar inside this inset axes
        cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
        cbar.set_label("Correlation", fontsize=ftsize)
        cbar.ax.tick_params(labelsize=ftsize - 6)

        # optional: ensure proper aspect and visibility
        cax.set_aspect("auto")

        # cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        # cbar.set_label("Correlation", fontsize=ftsize)
        # cbar.ax.tick_params(labelsize=ftsize - 6)

        # Axis labels
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=ftsize - 6)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=ftsize - 6)

        # Annotate only the lower triangle
        for i in range(mat.shape[0]):
            for j in range(i + 1):  # only lower triangle
                if np.abs(mat[i, j]) > threshold:
                    ax.text(
                        j,
                        i,
                        f"{mat[i,j]:.2f}",
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=ftsize - 12,
                    )

        plt.tight_layout()
        if folder_out is None:
            plt.show()
        else:
            plt.savefig(
                os.path.join(folder_out, "corr_mat" + str(iiter) + ".pdf"),
                bbox_inches="tight",
            )
            plt.savefig(
                os.path.join(folder_out, "corr_mat" + str(iiter) + ".png"),
                bbox_inches="tight",
            )
        plt.close()


def get_contours(x, y, sigmas=1, bins=40, threshold=1e-4):
    """
    Return mesh (X,Y), histogram values H (shape matches X,Y), and contour
    thresholds that enclose 68% and optionally 95% of the samples.

    Usage:
        X, Y, H, levels = get_contours(x, y, sigmas=2, bins=50)
        plt.contour(X, Y, H, levels=levels)
    """
    if np.min(x) < threshold:
        x = np.concatenate([x, -x])
        y = np.concatenate([y, y])

    # Use raw counts (simpler to work with probability mass)
    H, xedges, yedges = np.histogram2d(x, y, bins=bins, density=False)
    # if smooth:
    #     H = gaussian_filter(H, smooth)

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
