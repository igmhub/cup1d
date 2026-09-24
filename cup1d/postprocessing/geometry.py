"""Geometry plotting implementations; scientific objects retain compatibility wrappers."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.spatial import ConvexHull


def plot_hull(self, points, test_points=None):
    # Visualization: Project onto all 2D pairs of dimensions
    n_dimensions = points.shape[1]
    fig, axes = plt.subplots(
        n_dimensions,
        n_dimensions,
        figsize=(12, 12),
        constrained_layout=True,
    )

    for i in range(n_dimensions):
        for j in range(n_dimensions):
            if j > i:
                axes[i, j].set_visible(False)
                continue

            # Plot the points projected onto dimensions (i, j)
            if i == j:
                axes[i, j].hist(points[:, i])
            else:
                axes[i, j].scatter(points[:, j], points[:, i], s=10)

                # uncomment for test points
                # for icol in range(2):
                #     col = "C" + str(icol + 2)
                #     _ = np.argwhere(results == icol)[:, 0]
                # axes[i, j].scatter(test_points[_, j], test_points[_, i], s=20, color=col)

                # Project points onto dimensions (i, j)
                projected_points = self.hull.points[:, [j, i]]
                # Extract the hull vertices and sort them for the contour
                projected_hull_points = projected_points[self.hull.vertices]
                hull_2d = ConvexHull(projected_hull_points)
                for simplex in hull_2d.simplices:
                    axes[i, j].plot(
                        projected_hull_points[simplex, 0],
                        projected_hull_points[simplex, 1],
                        "k-",
                    )

    for j in range(n_dimensions):
        axes[-1, j].set_xlabel(self.params[j])
        axes[j, 0].set_ylabel(self.params[j])


def plot_hulls(self, points, test_points=None):
    # Visualization: Project onto all 2D pairs of dimensions
    n_dimensions = points.shape[1]
    fig, axes = plt.subplots(
        n_dimensions,
        n_dimensions,
        figsize=(12, 12),
        constrained_layout=True,
    )

    kk = 0
    for i in range(n_dimensions):
        for j in range(n_dimensions):
            if j > i:
                axes[i, j].set_visible(False)
                continue

            # Plot the points projected onto dimensions (i, j)
            if i == j:
                axes[i, j].hist(points[:, i])
            else:
                axes[i, j].scatter(points[:, j], points[:, i], s=10)

                for simplex in self.hulls[kk].simplices:
                    axes[i, j].plot(
                        self.hulls[kk].points[simplex, 1],
                        self.hulls[kk].points[simplex, 0],
                        "k-",
                    )
                kk += 1

    for j in range(n_dimensions):
        axes[-1, j].set_xlabel(self.params[j])
        axes[j, 0].set_ylabel(self.params[j])


def plot_ellipse(
    sigma1=0.2,
    sigma2=0.5,
    rho=0.6,
    mean=[1.0, 2.0],
    ax=None,
    color="C1",
    label="ellipse",
):
    # Covariance matrix
    cov = np.array(
        [
            [sigma1**2, rho * sigma1 * sigma2],
            [rho * sigma1 * sigma2, sigma2**2],
        ]
    )

    # Eigen-decomposition for ellipse axes
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Sort eigenvalues
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    # 68% chi-square value for 2 dof
    chi2_val = 2.30

    # Width and height of ellipse (2*sqrt because diameter)
    width, height = 2 * np.sqrt(eigvals * chi2_val)

    # Angle of ellipse (in degrees)
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    # print("angle")
    # angle = np.degrees(0.14250882064032286) * 2

    # Plot
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    ellipse = Ellipse(
        xy=mean,
        width=width,
        height=height,
        angle=angle,
        edgecolor=color,
        facecolor="none",
        lw=2,
        label=label,
    )
    ax.add_patch(ellipse)

    ax.scatter(*mean, c=color, marker="x")
