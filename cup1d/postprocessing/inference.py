"""Inference plotting implementations; scientific objects retain compatibility wrappers."""

import numpy as np
import matplotlib.pyplot as plt

from cup1d.likelihood import parameter as parameter_space


def plot_best_fit(self, plot_every_iz=1, residuals=True):
    """Plot the minimizer's best-fit P1D prediction against data.

    Parameters
    ----------
    self : object
        Inference object exposing ``minimizer`` and likelihood ``like``.
    plot_every_iz : int, default: 1
        Plot every nth redshift bin.
    residuals : bool, default: True
        Include residual panels in the likelihood P1D plot.

    Notes
    -----
    The minimizer must already have valid sampling-cube values.  This legacy
    helper delegates plotting and returns None.
    """

    # get best-fit values from minimizer (should check that it was run)
    best_fit_values = np.array(self.minimizer.values)
    if self.verbose:
        print("best-fit values =", best_fit_values)

    # plt.title("iminuit best fit")
    parameters = parameter_space.values_from_cube(self.like.free_params, best_fit_values)
    self.like.plot_p1d(
        values=parameters,
        plot_every_iz=plot_every_iz,
        residuals=residuals,
    )

    return


def plot_ellipses(self, pname_x, pname_y, nsig=2, cube_values=False):
    """Plot Gaussian confidence ellipses from the fitted covariance.

    Parameters
    ----------
    self : object
        Inference object exposing fitted parameter values, errors, covariance,
        and sampling-cube conversion helpers.
    pname_x, pname_y : str
        Names of the two free parameters to display.
    nsig : int, default: 2
        Number of nested integer-sigma ellipses.
    cube_values : bool, default: False
        Plot unit-cube coordinates rather than physical parameters.

    Notes
    -----
    When physical coordinates are plotted, ``As`` is multiplied by ``1e9`` for
    numerical display.  The helper creates axes in place and returns None.
    """

    from matplotlib.patches import Ellipse
    from numpy import linalg as LA

    # figure out true values of parameters
    if self.like.truth:
        if self.verbose:
            print("compute true values for", pname_x, pname_y)
        if pname_x in self.like.truth:
            true_x = self.like.truth[pname_x]
            if pname_x == "As":
                true_x *= 1e9
        else:
            true_x = 0.5 if cube_values else 0.0
        if pname_y in self.like.truth:
            true_y = self.like.truth[pname_y]
            if pname_y == "As":
                true_y *= 1e9
        else:
            true_y = 0.5 if cube_values else 0.0

    # figure out order of parameters in free parameters list
    ix = self.index_by_name(pname_x)
    iy = self.index_by_name(pname_y)

    # find out best-fit values, errors and covariance for parameters
    val_x = self.minimizer.values[ix]
    val_y = self.minimizer.values[iy]
    sig_x = self.minimizer.errors[ix]
    sig_y = self.minimizer.errors[iy]
    r = self.minimizer.covariance[ix, iy] / sig_x / sig_y

    # rescale from cube values (unless asked not to)
    if not cube_values:
        val_x = self.value_from_cube(pname_x, val_x)
        sig_x = self.error_from_cube(pname_x, sig_x)
        val_y = self.value_from_cube(pname_y, val_y)
        sig_y = self.error_from_cube(pname_y, sig_y)
        # multiply As by 10^9 for now, otherwise ellipse crashes
        if pname_x == "As":
            val_x *= 1e9
            sig_x *= 1e9
            pname_x += " x 1e9"
        if pname_y == "As":
            val_y *= 1e9
            sig_y *= 1e9
            pname_y += " x 1e9"

    # shape of ellipse from eigenvalue decomposition of covariance
    w, v = LA.eig(
        np.array(
            [
                [sig_x**2, sig_x * sig_y * r],
                [sig_x * sig_y * r, sig_y**2],
            ]
        )
    )

    # semi-major and semi-minor axis of ellipse
    a = np.sqrt(w[0])
    b = np.sqrt(w[1])

    # figure out inclination angle of ellipse
    alpha = np.arccos(v[0, 0])
    if v[1, 0] < 0:
        alpha = -alpha
    # compute angle in degrees (expected by matplotlib)
    alpha_deg = alpha * 180 / np.pi

    # make plot
    fig = plt.subplot(111)
    for isig in range(1, nsig + 1):
        ell = Ellipse(
            (val_x, val_y), 2 * isig * a, 2 * isig * b, angle=alpha_deg
        )
        ell.set_alpha(0.6 / isig)
        fig.add_artist(ell)
    plt.xlabel(pname_x)
    plt.ylabel(pname_y)
    plt.xlim(val_x - (nsig + 1) * sig_x, val_x + (nsig + 1) * sig_x)
    plt.ylim(val_y - (nsig + 1) * sig_y, val_y + (nsig + 1) * sig_y)
    if self.like.truth:
        plt.axhline(y=true_y, ls=":", color="gray")
        plt.axvline(x=true_x, ls=":", color="gray")
