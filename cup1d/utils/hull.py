import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull


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

    def __init__(self, data_params, data_hull, extra_factor=1.05):
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
        self.data_params = data_params
        self.data_hull = data_hull
        mean = self.data_hull.mean(axis=0)
        self.hull = ConvexHull(extra_factor * (self.data_hull - mean) + mean)

    def in_hull(self, point):
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
        p0 = np.zeros(len(self.data_params))
        for ii, param in enumerate(self.data_params):
            p0[ii] = point[param]

        # Use the plane equations from the convex hull
        equations = (
            self.hull.equations
        )  # Ax + By + Cz + D = 0 (normal vector + offset)

        return np.all(np.dot(equations[:, :-1], p0) + equations[:, -1] <= 0)

    def plot_hull(self, test_points=None):
        # Visualization: Project onto all 2D pairs of dimensions
        points = self.data_hull
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
            axes[-1, j].set_xlabel(self.data_params[j])
            axes[j, 0].set_ylabel(self.data_params[j])
