"""IGM summary table (computed by :mod:`cup1d.postprocessing.fit.igm`)."""

import numpy as np


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


def format_igm_table(table):
    """LaTeX table of the IGM summary (Table of the DR1 paper next to Fig. 19).

    Parameters
    ----------
    table : mapping
        Output of :func:`cup1d.postprocessing.fit.igm.igm_summary_table` with
        ``mF``, ``T0`` (in 10^4 K) and ``gamma`` 16/50/84 per cent quantiles.

    Returns
    -------
    str
        ``tabular`` environment, one row per redshift.
    """
    lines = [r"\begin{tabular}{cccc}", r"$z$ & $\bar{F}$ & $T_0[K]/10^4$ & $\gamma$ \\", r"\hline"]
    for i, z in enumerate(table["z"]):
        cells = [format_asym_error(table[name][:, i]) for name in ("mF", "T0", "gamma")]
        lines.append(f"{z:.2f} & " + " & ".join(cells) + r" \\")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)
