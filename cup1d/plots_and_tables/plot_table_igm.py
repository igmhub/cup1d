import numpy as np
import matplotlib.pyplot as plt

from cup1d.likelihood.input_pipeline import Args
from cup1d.likelihood.pipeline import Pipeline


def format_asym_error(arr):
    """Format median + errors from percentiles (arr[0]=16th, arr[1]=50th, arr[2]=84th)."""
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
):
    emulator_label = "CH24_mpgcen_gpr"
    if name_variation == "nyx":
        emulator_label = "CH24_nyxcen_gpr"

    args = Args(data_label=data_label, emulator_label=emulator_label)
    args.set_baseline(
        fit_type="global_opt",
        fix_cosmo=False,
        P1D_type=data_label,
        name_variation=name_variation,
    )
    pip = Pipeline(args, out_folder=args.out_folder)
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
    free_params = pip.fitter.like.parameters_from_sampling_point(p0)

    chain = np.load(base + folder + "chain.npy")

    tab_out = pip.fitter.like.plot_igm(
        free_params=free_params,
        plot_fid=True,
        plot_type="tau_sigT",
        cloud=False,
        ftsize=20,
        chain_uformat=chain,
        save_directory=save_fig,
    )

    z, mF, T0, gamma = tab_out

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

    return
