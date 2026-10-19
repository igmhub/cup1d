"""Comparison plots for cosmological sampler and fit results."""

import numpy as np


def plot_cosmo_sampler_and_fit(
    samples,
    mle,
    covariance,
    names=("Delta2_star", "n_star"),
    labels=(r"$\Delta^2_\star$", r"$n_\star$"),
    sampler_label="Sampler",
    fit_label="MLE Gaussian",
    levels=(0.68, 0.95),
):
    """Plot two-parameter sampler contours and a local MLE Gaussian.

    When the MLE errors are zero (e.g. a fixed-cosmology fit, where no free
    parameter moves the compressed cosmology), only the MLE centre is drawn:
    a marker in the joint panel and vertical lines in the marginals.

    Parameters
    ----------
    samples : ndarray
        Posterior samples with shape ``(n_samples, 2)`` in display coordinates.
    mle : mapping
        Maximum-likelihood values keyed by the two entries in ``names``.
    covariance : ndarray
        MLE covariance with shape ``(2, 2)`` in the same coordinates as
        ``samples`` and ``mle``.
    names : tuple of str, default: ("Delta2_star", "n_star")
        Keys used to read MLE coordinates.
    labels : tuple of str
        Axis labels for the two displayed coordinates.
    sampler_label, fit_label : str
        Legend labels for posterior samples and MLE result.
    levels : tuple of float, default: (0.68, 0.95)
        Credible-region levels passed to :mod:`corner`.

    Returns
    -------
    matplotlib.figure.Figure
        Corner plot augmented with one-sigma MLE ellipse and Gaussian marginals.

    Raises
    ------
    ValueError
        If inputs do not describe two coordinates or covariance is not finite
        with non-negative eigenvalues and diagonal variances.

    Notes
    -----
    Callers must provide sampler and MLE values in the same, already unblinded
    display coordinates.  An additive blinding offset leaves covariance intact.
    """

    import corner
    from matplotlib.lines import Line2D
    from matplotlib.patches import Ellipse, Patch

    samples = np.asarray(samples)
    covariance = np.asarray(covariance)
    if samples.ndim != 2 or samples.shape[1] != 2:
        raise ValueError("samples must have shape (n_samples, 2)")
    if covariance.shape != (2, 2):
        raise ValueError("covariance must have shape (2, 2)")

    mean = np.asarray([mle[name] for name in names])
    variances = np.diag(covariance)
    if np.any(variances < 0) or not np.all(np.isfinite(covariance)):
        raise ValueError("MLE covariance must be finite with non-negative variances")
    errors = np.sqrt(variances)
    # zero errors: only the MLE centre is meaningful, no ellipse or Gaussian
    has_errors = bool(np.all(errors > 0))

    figure = corner.corner(
        samples,
        labels=labels,
        levels=levels,
        plot_datapoints=False,
        fill_contours=True,
        color="C0",
        hist_kwargs={"density": True},
    )
    axes = np.asarray(figure.axes).reshape(2, 2)
    joint_axis = axes[1, 0]
    legend_axis = axes[0, 1]

    if has_errors:
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        if np.any(eigenvalues < -1e-12 * np.max(np.abs(eigenvalues))):
            raise ValueError("MLE covariance must be positive semidefinite")
        eigenvalues = np.clip(eigenvalues, 0, None)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        joint_axis.add_patch(
            Ellipse(
                mean,
                width=2 * np.sqrt(eigenvalues[0]),
                height=2 * np.sqrt(eigenvalues[1]),
                angle=angle,
                facecolor="C3",
                edgecolor="C3",
                alpha=0.3,
            )
        )
    joint_axis.plot(*mean, "XC3")

    for index, axis in enumerate((axes[0, 0], axes[1, 1])):
        if has_errors:
            grid = np.linspace(*axis.get_xlim(), 300)
            density = np.exp(-0.5 * ((grid - mean[index]) / errors[index]) ** 2)
            density /= np.sqrt(2 * np.pi) * errors[index]
            axis.plot(grid, density, color="C3")
        else:
            axis.axvline(mean[index], color="C3")

    handles = [
        Line2D([], [], color="C0", label=sampler_label),
        Line2D([], [], marker="X", color="C3", linestyle="", label=fit_label),
    ]
    if has_errors:
        handles.append(
            Patch(facecolor="C3", edgecolor="C3", alpha=0.3, label="MLE 1$\\sigma$")
        )
    legend_axis.axis("off")
    legend_axis.legend(handles=handles, loc="center")

    return figure
