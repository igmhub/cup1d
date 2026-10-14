import numpy as np

from cup1d.configuration.args import Args
from cup1d.inference.analysis import Analysis


def format_asym_error(arr):
    """Format 16th, 50th, and 84th percentiles as a LaTex asymmetric error.

    Parameters
    ----------
    arr : array_like, shape (3,)
        Lower percentile, median, and upper percentile in that order.

    Returns
    -------
    str
        Math-mode LaTex string of the form ``$median^{+upper}_{-lower}$``.
    """
    lower = arr[1] - arr[0]
    upper = arr[2] - arr[1]

    # Determine number of decimal places based on the smaller error (2 significant figures)
    err = min(lower, upper)
    if err == 0:
        ndec = 2
    else:
        exp = int(np.floor(np.log10(abs(err))))
        ndec = max(-exp + 1, 0)

    m_str = f"{arr[1]:.{ndec}f}"
    upper_str = f"{upper:.{ndec}f}"
    lower_str = f"{lower:.{ndec}f}"

    return f"${m_str}^{{+{upper_str}}}_{{-{lower_str}}}$"


def plot_table_igm(
    base,
    save_fig=None,
    data_label="DESIY1_QMLE3",
    name_variation=None,
    chain="1",
    store_data=False,
):
    """Create an IGM summary figure and print its LaTex table rows.

    Parameters
    ----------
    base : str or path-like
        Parent directory containing the configured fitter products.
    save_fig : str or path-like, optional
        Directory passed to the IGM plotting routine.
    data_label : str, default: "DESIY1_QMLE3"
        cup1d P1D data-set label.
    name_variation : str, optional
        Analysis variation subdirectory; ``"nyx"`` selects the Nyx emulator.
    chain : str, default: "1"
        Chain subdirectory identifier.
    store_data : bool, default: False
        Return plotted numerical data when true.

    Returns
    -------
    dict, optional
        Data returned by the IGM plot when ``store_data`` is true.
    """
    emulator_label = "lace_mpg"
    if name_variation == "nyx":
        emulator_label = "lace_nyx"
        name_variation = None
        tit = None
        lab_fid = "lyssa-central"
        variation_label = "lace-lyssa"
        plot_more_igm = False
    else:
        tit = None
        lab_fid = "mpg-central"
        variation_label = "Baseline"
        plot_more_igm = True
    print(variation_label)

    args = Args(data_label=data_label, emulator_label=emulator_label)
    args.set_baseline(
        fit_type="global_opt",
        fix_cosmo=False,
        name_variation=name_variation,
    )
    pip = Analysis(args, out_folder=args.out_folder)
    if name_variation is None:
        name_variation = "global_opt"

    folder = (
        data_label
        + "/"
        + name_variation
        + "/"
        + emulator_label
        + "/chain_"
        + chain
        + "/"
    )
    print("Read data from: " + base + folder)

    data = np.load(
        base + folder + "fitter_results.npy", allow_pickle=True
    ).item()
    p0 = data["fitter"]["mle_cube"]
    free_params = pip.fitter.point_from_chain_row(p0)

    chain = np.load(base + folder + "chain.npy")

    out = pip.fitter.like.plot_igm(
        free_params=free_params,
        plot_fid=True,
        plot_type="tau_sigT",
        cloud=False,
        ftsize=20,
        chain_uformat=chain,
        save_directory=save_fig,
        title=tit,
        plot_more_igm=plot_more_igm,
        lab_fid=lab_fid,
        variation_label=variation_label,
        store_data=store_data,
    )

    z, mF, T0, gamma = out["tab_out"]
    # np.save(
    #     "data/tutorials/data/more_igm_data.npy",
    #     {"z": z, "mF": mF, "T0": T0, "gamma": gamma},
    # )
    # print(z)
    # print(mF)

    print(r"\begin{tabular}{cccc}")
    print(r"$z$ & $\bar{F}$ & $T_0[K]/10^4$ & $\gamma$ \\")
    print(r"\hline")

    for i in range(z.shape[0]):
        mF_str = format_asym_error(mF[:, i])
        T0_str = format_asym_error(T0[:, i])
        gamma_str = format_asym_error(gamma[:, i])
        print(f"{z[i]:.2f} & {mF_str} & {T0_str} & {gamma_str} \\\\")
        # print(r"\hline")

    print(r"\end{tabular}")

    if store_data:
        return out["out_data"]
