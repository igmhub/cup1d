"""Convex-hull helpers for emulator training domains."""

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull

from cup1d.utils.utils import get_path_repo


def in_hull(hull, p):
    """Return whether points ``p`` satisfy all stored hull half-spaces."""
    return np.all(hull.eq @ p.T + hull.eq2[:, : p.shape[0]] <= hull.tol, 0)


class Hull:
    """Compute and query emulator-domain convex hulls."""

    def __init__(
        self,
        zs=None,
        data_hull=None,
        suite="mpg",
        save=False,
        extra_factor=1.0,
        mpg_version="Cabayol23",
        nyx_version="Jul2024",
        recompute=False,
        tol=1e-12,
        multi_dim=False,
    ):
        """Build, load, or save convex hulls for emulator training data.

        Parameters
        ----------
        data_hull : numpy.ndarray
            Training points used to construct the hull.
        extra_factor : float, optional, default=1.05
            Scaling factor applied around the data mean before hull creation.
        """

        self.nz = len(zs)
        self.zs = zs
        self.tol = tol
        if suite == "mpg":
            self.params = [
                "Delta2_p",
                "n_p",
                "mF",
                "sigT_Mpc",
                "gamma",
                "kF_Mpc",
            ]
        elif suite == "nyx":
            self.params = [
                "Delta2_p",
                "n_p",
                "alpha_p",
                "mF",
                "sigT_Mpc",
                "gamma",
                "kF_Mpc",
            ]

        if multi_dim is True:
            self.hull = None
            if recompute is False:
                if suite == "mpg":
                    self.hull = self.load_hull(suite, mpg_version=mpg_version)
                elif suite == "nyx":
                    self.hull = self.load_hull(suite, nyx_version=nyx_version)

            if self.hull is None:
                self.hull = self.set_hull(data_hull, extra_factor=extra_factor)
                if save:
                    self.save_hull(
                        suite, mpg_version=mpg_version, nyx_version=nyx_version
                    )
            self.set_in_hull(zs)
        else:
            self.hulls = self.set_hulls(data_hull, extra_factor=extra_factor)

    def set_hulls(self, points, extra_factor=1.0):
        """Build all pairwise two-dimensional hulls."""
        int_factor = extra_factor - 0.01

        hulls = []
        for jj0 in range(points.shape[1]):
            for jj1 in range(points.shape[1]):
                if jj1 >= jj0:
                    continue

                data_hull = points[:, [jj0, jj1]]

                mean = data_hull.mean(axis=0)
                int_data = int_factor * (data_hull - mean) + mean
                ext_data = extra_factor * (data_hull - mean) + mean
                hull = ConvexHull(int_data)
                hull.eq = hull.equations[:, :-1]
                hull.eq2 = np.repeat(
                    hull.equations[:, -1][None, :], data_hull.shape[0], axis=0
                ).T
                hull.tol = self.tol

                mask = ~in_hull(hull, ext_data)
                data_for_hull = ext_data[mask]

                hull_2d = ConvexHull(data_for_hull)
                hull_2d.eq = hull_2d.equations[:, :-1]
                hull_2d.eq2 = np.repeat(
                    hull_2d.equations[:, -1][None, :], self.nz, axis=0
                ).T
                hull_2d.tol = self.tol
                hull_2d.dim0 = jj0
                hull_2d.dim1 = jj1
                hulls.append(hull_2d)

        return hulls

    def in_hulls(self, p):
        """Return whether all rows in ``p`` lie within every pairwise hull."""
        for jj in range(len(self.hulls)):
            res = in_hull(
                self.hulls[jj], p[:, [self.hulls[jj].dim0, self.hulls[jj].dim1]]
            )
            if not res.all():
                return False

        return True

    def set_hull(self, data_hull, extra_factor=1.050):
        """Build one multi-dimensional convex hull."""
        int_factor = extra_factor - 1e-3
        mean = data_hull.mean(axis=0)
        int_data = int_factor * (data_hull - mean) + mean
        ext_data = extra_factor * (data_hull - mean) + mean
        hull = ConvexHull(int_data)

        data_for_hull = []
        for ii in range(ext_data.shape[0]):
            if not self._in_hull(hull, ext_data[ii]):
                data_for_hull.append(ext_data[ii])
        data_for_hull = np.vstack(data_for_hull)

        return ConvexHull(data_for_hull)

    def _in_hull(self, hull, point):
        """Return whether one point is inside a SciPy convex hull."""
        return np.all(
            np.dot(hull.equations[:, :-1], point) + hull.equations[:, -1] <= 0
        )

    def save_hull(self, suite, mpg_version="Cabayol23", nyx_version="Jul2024"):
        """Save the current multi-dimensional hull to disk."""
        if suite == "nyx":
            folder = os.environ["NYX_PATH"]
            fname = os.path.join(folder, "hull_Nyx23_" + nyx_version + ".npy")
        elif suite == "mpg":
            folder = os.path.join(get_path_repo("cup1d"), "data", "hull")
            fname = os.path.join(folder, "hull_" + mpg_version + ".npy")

        np.save(fname, vars(self.hull))

    def load_hull(self, suite, mpg_version="Cabayol23", nyx_version="Jul2024"):
        """Load a saved multi-dimensional hull, if available."""
        if suite == "nyx":
            folder = os.environ["NYX_PATH"]
            fname = os.path.join(folder, "hull_Nyx23_" + nyx_version + ".npy")
        elif suite == "mpg":
            folder = os.path.join(get_path_repo("cup1d"), "data", "hull")
            fname = os.path.join(folder, "hull_" + mpg_version + ".npy")

        if not os.path.exists(fname):
            return None

        vars_hull = np.load(fname, allow_pickle=True).item()

        # create a tiny hull to fill it with that stored in disk
        hull = ConvexHull(vars_hull["_points"][:50])
        for key in vars_hull.keys():
            setattr(hull, key, vars_hull[key])

        return hull

    def plot_hull(self, points, test_points=None):
        """Plot pairwise projections of hull training points."""
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
