import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

from cup1d.utils.utils import get_path_repo


class Hull(object):
    """
    A class for computing and working with the convex hull of a dataset, with optional scaling.

    This class computes the convex hull of a given dataset, optionally scaling the data before
    calculating the hull. The data is first centered by subtracting the mean of the dataset, then scaled
    by a specified factor (`extra_factor`). The convex hull is then computed on the transformed data.
    The class also provides a method to check if a point is inside the computed convex hull.

    Attributes:
    -----------
    hull : scipy.spatial.ConvexHull
        A `ConvexHull` object that contains the vertices, simplices, and other information about the convex hull
        of the scaled dataset.

    Methods:
    --------
    in_hull(point):
        Checks if a given point lies inside the computed convex hull.

    """

    def __init__(
        self,
        data_hull=None,
        suite="mpg",
        save=True,
        extra_factor=1.05,
        mpg_version="Cabayol23",
        nyx_version="Jul2024",
        recompute=False,
        tol=1e-12,
    ):
        """
        Initializes the Hull object by computing the convex hull of a given dataset with an optional scaling factor.

        This method centers the provided dataset by subtracting its mean and then scales it by a specified factor
        (`extra_factor`). The convex hull of the scaled dataset is computed and stored as a `ConvexHull` object.
        The convex hull is stored as an attribute of the class, allowing for further operations such as checking
        if a point is inside the hull.

        Parameters:
        -----------
        data_hull : numpy.ndarray
            A 2D array of shape (n_samples, n_features) representing the dataset for which the convex hull is to be computed.
            Each row corresponds to a data point, and each column represents a feature (dimension).

        extra_factor : float, optional, default=1.05
            A scaling factor applied to the centered dataset before computing the convex hull.
            A value greater than 1.0 expands the dataset, while a value less than 1.0 contracts it.
            The default value is 1.05, slightly expanding the data.

        Returns:
        --------
        None
            This is the constructor of the `Hull` class, so it does not return any value. The resulting `ConvexHull` object
            is stored as an attribute `self.hull`.

        Notes:
        -----
        - The dataset is centered by subtracting the mean of the data along each feature (dimension).
        - The convex hull is computed using the scaled dataset, and the resulting `ConvexHull` object contains
          the vertices, simplices, and other details about the convex hull.
        """

        self.tol = tol

        self.hull = None
        if recompute == False:
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

    def set_hull(self, data_hull, extra_factor=1.050):
        int_factor = extra_factor - 1e-3
        mean = data_hull.mean(axis=0)
        int_data = int_factor * (data_hull - mean) + mean
        ext_data = extra_factor * (data_hull - mean) + mean
        hull = ConvexHull(int_data)

        data_for_hull = []
        for ii in range(ext_data.shape[0]):
            if self._in_hull(hull, ext_data[ii]) == False:
                data_for_hull.append(ext_data[ii])
        data_for_hull = np.vstack(data_for_hull)

        return ConvexHull(data_for_hull)

    def set_in_hull(self, p):
        self.eq = self.hull.equations[:, :-1]
        self.eq2 = np.repeat(
            self.hull.equations[:, -1][None, :], len(p), axis=0
        ).T

    def in_hull(self, p):
        p = np.atleast_2d(p)
        return np.all(self.eq @ p.T + self.eq2 <= self.tol, 0)

    def _in_hull(self, hull, point):
        """
        Check if a point is inside the convex hull.

        Parameters:
        -----------
        point : array-like
            The point to check, expected to be of shape (n_features,) where n_features is the number of features
            (dimensions) of the dataset.

        Returns:
        --------
        bool
            True if the point is inside the convex hull, False otherwise.

        Notes:
        -----
        This method uses the plane equations of the convex hull (derived from its faces) to determine if the point
        lies within the convex hull. The convex hull is considered to enclose all points whose projections
        onto the faces of the hull satisfy the inequality defined by the hull's equations.
        """
        return np.all(
            np.dot(hull.equations[:, :-1], point) + hull.equations[:, -1] <= 0
        )

    def save_hull(self, suite, mpg_version="Cabayol23", nyx_version="Jul2024"):
        if suite == "nyx":
            folder = os.environ["NYX_PATH"]
            fname = os.path.join(folder, "hull_Nyx23_" + nyx_version + ".npy")
        elif suite == "mpg":
            folder = os.path.join(get_path_repo("cup1d"), "data", "hull")
            fname = os.path.join(folder, "hull_" + mpg_version + ".npy")

        np.save(fname, vars(self.hull))

    def load_hull(self, suite, mpg_version="Cabayol23", nyx_version="Jul2024"):
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

    def plot_hull(self, data_params, test_points=None):
        # Visualization: Project onto all 2D pairs of dimensions
        points = self.hull.points
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
                    projected_points = points[:, [j, i]]
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
            axes[-1, j].set_xlabel(data_params[j])
            axes[j, 0].set_ylabel(data_params[j])