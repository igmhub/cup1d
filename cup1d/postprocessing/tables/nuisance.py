"""Nuisance-parameter table of a chain (Table C2 of the DR1 paper)."""

import numpy as np

from cup1d.postprocessing.labels import parameter_label


def format_value_with_error(m, ep, em):
    """Format a value and asymmetric errors as a LaTex math expression.

    Parameters
    ----------
    m : float
        Central value.
    ep, em : float
        Positive and negative error magnitudes.

    Returns
    -------
    str
        Formatted expression, using scientific notation for small errors.
    """
    if ep == 0 or em == 0 or np.isnan(ep) or np.isnan(em):
        return f"${m:.2f}^{{+{ep:.2f}}}_{{-{em:.2f}}}$"

    # Use scientific notation if either error is < 1e-3
    if min(ep, em) < 1e-3:
        exp = int(np.floor(np.log10(abs(m)))) if m != 0 else 0
        scale = 10**exp
        m_s, ep_s, em_s = m / scale, ep / scale, em / scale
        return f"${m_s:.2f}^{{+{ep_s:.2f}}}_{{-{em_s:.2f}}}\\times10^{{{exp}}}$"

    # Otherwise, use two significant digits for the errors
    exp_err = int(np.floor(np.log10(min(ep, em))))
    ndec = max(-exp_err + 1, 0)
    m_str = f"{m:.{ndec}f}"
    ep_str = f"{ep:.{ndec}f}"
    em_str = f"{em:.{ndec}f}"
    return f"${m_str}^{{+{ep_str}}}_{{-{em_str}}}$"


def table_nuisance(folder_variation):
    """Paired LaTeX nuisance-parameter summary rows of a chain folder.

    Parameters
    ----------
    folder_variation : str or path-like
        Chain folder read by
        :meth:`cup1d.postprocessing.posterior.PosteriorSamples.from_cup1d_folder`.

    Returns
    -------
    str
        Two blocks of LaTeX rows: value with errors, then value over error.
    """
    from cup1d.postprocessing.posterior import PosteriorSamples

    samples = PosteriorSamples.from_cup1d_folder(folder_variation)
    # Delta2_star, n_star, then the sampled parameters (As and ns replaced by
    # the star parameters, as in the paper table)
    labels = list(samples.names[:2]) + list(samples.sampled[2:])
    dat = np.column_stack([samples.column(name) for name in labels])

    for ii in range(2, dat.shape[-1]):
        if (
            ("sigT_kms" in labels[ii])
            | ("gamma" in labels[ii])
            | ("R_coeff_" in labels[ii])
        ):
            pass
        else:
            dat[:, ii] = np.exp(dat[:, ii])
            if labels[ii].startswith("s_"):
                dat[:, ii] = 1 / dat[:, ii]

    # Compute percentiles
    p16, p50, p84 = np.percentile(dat, [16, 50, 84], axis=0)

    # Asymmetric errors
    err_minus = p50 - p16
    err_plus = p84 - p50
    err_mid = 0.5 * (err_plus + err_minus)

    rows = []
    imin = 18
    n = len(labels) - imin
    half = (n + 1) // 2  # split index
    for ii in range(half):
        i = ii + imin
        left_label = parameter_label(labels[i])
        left_value = format_value_with_error(p50[i], err_plus[i], err_minus[i])
        if i > 1:
            if i + half < n + imin:
                right_label = parameter_label(labels[i + half])
                right_value = format_value_with_error(
                    p50[i + half], err_plus[i + half], err_minus[i + half]
                )
                rows.append(f"{left_label:<10} & {left_value:<20} & {right_label:<10} & {right_value} \\\\")
            else:
                # Odd number of parameters
                rows.append(f"{left_label:<10} & {left_value:<20} & & \\\\")

    rows += ["", ""]
    for ii in range(half):
        i = ii + imin
        left_label = parameter_label(labels[i])
        left_value = p50[i] / err_mid[i]
        if i > 1:
            if i + half < n + imin:
                right_label = parameter_label(labels[i + half])
                right_value = p50[i + half] / err_mid[i + half]
                rows.append(f"{left_label:<10} & {left_value:<20} & {right_label:<10} & {right_value} \\\\")
            else:
                # Odd number of parameters
                rows.append(f"{left_label:<10} & {left_value:<20} & & \\\\")

    return "\n".join(rows)
