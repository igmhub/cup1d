import numpy as np
import os
from scipy.spatial import ConvexHull
from lace.configuration import get_nyx_path

from cup1d.utils.utils import get_path_repo


def in_hull(hull, p):
    """Return membership flags for points in a two-dimensional convex hull.

    Parameters
    ----------
    hull : scipy.spatial.ConvexHull
        Hull augmented with the equation arrays used by cup1d.
    p : numpy.ndarray
        Points with shape ``(npoint, 2)``.

    Returns
    -------
    numpy.ndarray
        Boolean membership flags.
    """
    return np.all(hull.eq @ p.T + hull.eq2[:, : p.shape[0]] <= hull.tol, 0)


class Hull(object):
    """Represent expanded emulator-domain convex hulls.

    By default the class builds every two-dimensional projection used for
    inexpensive domain checks. ``multi_dim=True`` instead uses a persisted or
    newly calculated full-dimensional hull.
    """

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
        """Build projected or full-dimensional emulator-domain hulls.

        Parameters
        ----------
        zs : array_like
            Emulator redshift grid; its length is used by projected hulls.
        data_hull : ndarray, shape (n_points, n_parameters)
            Training-set emulator inputs.
        suite : {"mpg", "nyx"}, default: "mpg"
            Simulation suite selecting the expected parameter names.
        save : bool, default: False
            Save a newly computed full-dimensional hull.
        extra_factor : float, default: 1.0
            Expansion factor applied about the training-point mean.
        mpg_version, nyx_version : str
            Identifiers used for full-hull cache filenames.
        recompute : bool, default: False
            Ignore a cached full-dimensional hull.
        tol : float, default: 1e-12
            Numerical tolerance for projected-hull membership tests.
        multi_dim : bool, default: False
            Use one full-dimensional hull rather than all pairwise projections.
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

        if multi_dim == True:
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
            self.set_in_hull(zs)
        else:
            self.hulls = self.set_hulls(data_hull, extra_factor=extra_factor)

    def set_hulls(self, points, extra_factor=1.0):
        """Build expanded pairwise projected hulls for training parameters.

        Parameters
        ----------
        points : ndarray, shape (n_points, n_parameters)
            Training-set inputs.
        extra_factor : float, default: 1.0
            Expansion factor about each projected-data mean.

        Returns
        -------
        list of scipy.spatial.ConvexHull
            Hulls annotated with their parameter-column indices.
        """
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

                mask = in_hull(hull, ext_data) == False
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
        """Return whether all points lie in every pairwise projected hull.

        Parameters
        ----------
        p : ndarray, shape (n_points, n_parameters)
            Points to test.

        Returns
        -------
        bool
            True only when all point-projection combinations are admitted.
        """
        for jj in range(len(self.hulls)):
            res = in_hull(
                self.hulls[jj], p[:, [self.hulls[jj].dim0, self.hulls[jj].dim1]]
            )
            if res.all() == False:
                return False

        return True

    def set_hull(self, data_hull, extra_factor=1.050):
        """Build an expanded full-dimensional hull from training points.

        Parameters
        ----------
        data_hull : ndarray, shape (n_points, n_parameters)
            Training-set inputs.
        extra_factor : float, default: 1.050
            Expansion factor about the training-set mean.

        Returns
        -------
        scipy.spatial.ConvexHull
            Convex hull enclosing the expanded exterior points.
        """
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

    def _in_hull(self, hull, point):
        """Test one point against a full-dimensional hull's face equations.

        Parameters
        ----------
        hull : scipy.spatial.ConvexHull
            Full-dimensional hull to test.
        point : array_like, shape (n_parameters,)
            Point to test.

        Returns
        -------
        bool
            Whether the point satisfies every hull half-space.
        """
        return np.all(
            np.dot(hull.equations[:, :-1], point) + hull.equations[:, -1] <= 0
        )

    def save_hull(self, suite, mpg_version="Cabayol23", nyx_version="Jul2024"):
        """Save the full-dimensional hull for a simulation suite.

        Parameters
        ----------
        suite : {"mpg", "nyx"}
            Simulation suite selecting the output location.
        mpg_version, nyx_version : str
            Version strings included in the suite-specific filename.
        """
        if suite == "nyx":
            folder = get_nyx_path()
            fname = os.path.join(folder, "hull_Nyx23_" + nyx_version + ".npy")
        elif suite == "mpg":
            folder = os.path.join(get_path_repo("cup1d"), "data", "hull")
            fname = os.path.join(folder, "hull_" + mpg_version + ".npy")

        np.save(fname, vars(self.hull))

    def load_hull(self, suite, mpg_version="Cabayol23", nyx_version="Jul2024"):
        """Load a previously saved full-dimensional suite hull.

        Parameters
        ----------
        suite : {"mpg", "nyx"}
            Simulation suite selecting the cache location.
        mpg_version, nyx_version : str
            Version strings included in the suite-specific filename.

        Returns
        -------
        scipy.spatial.ConvexHull or None
            Reconstructed hull, or ``None`` when no cache exists.
        """
        if suite == "nyx":
            folder = get_nyx_path()
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
        """Plot a single hull through the shared geometry helper.

        Parameters
        ----------
        points : ndarray
            Training points used for the visualized projections.
        test_points : ndarray, optional
            Additional points highlighted for membership inspection.
        """
        from cup1d.postprocessing.geometry import plot_hull as _plot

        return _plot(self, points, test_points)

    def plot_hulls(self, points, test_points=None):
        """Plot pairwise hull projections through the shared geometry helper.

        Parameters
        ----------
        points : ndarray
            Training points used for the visualized projections.
        test_points : ndarray, optional
            Additional points highlighted for membership inspection.
        """
        from cup1d.postprocessing.geometry import plot_hulls as _plot

        return _plot(self, points, test_points)
