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
    """Plot sampler contours and the corresponding local MLE Gaussian.

    Parameters are supplied in their desired display coordinates.  In
    particular, callers should unblind both sampler values and MLE values
    before calling this function; an additive blinding offset leaves the
    covariance unchanged.
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
    errors = np.sqrt(np.diag(covariance))
    if np.any(errors <= 0) or not np.all(np.isfinite(errors)):
        raise ValueError("MLE covariance must have finite positive variances")

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

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if np.any(eigenvalues < 0):
        raise ValueError("MLE covariance must be positive semidefinite")
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
        grid = np.linspace(*axis.get_xlim(), 300)
        density = np.exp(-0.5 * ((grid - mean[index]) / errors[index]) ** 2)
        density /= np.sqrt(2 * np.pi) * errors[index]
        axis.plot(grid, density, color="C3")

    legend_axis.axis("off")
    legend_axis.legend(
        handles=[
            Line2D([], [], color="C0", label=sampler_label),
            Line2D([], [], marker="X", color="C3", linestyle="", label=fit_label),
            Patch(facecolor="C3", edgecolor="C3", alpha=0.3, label="MLE 1$\\sigma$"),
        ],
        loc="center",
    )

    return figure
