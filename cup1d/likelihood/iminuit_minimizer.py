import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit

# our own modules
from cup1d.likelihood import likelihood


class IminuitMinimizer(object):
    """Wrapper around an iminuit minimizer for Lyman alpha likelihood"""

    def __init__(self, like, ini_values=None, error=0.02, verbose=False):
        """Setup minimizer from likelihood."""

        self.verbose = verbose
        self.like = like

        # set initial values (for now, center of the unit cube)
        if ini_values is None:
            ini_values = 0.5 * np.ones(len(self.like.free_params))

        # setup iminuit object (errordef=0.5 if using log-likelihood)
        self.minimizer = Minuit(like.minus_log_prob, ini_values)
        # self.minimizer = Minuit(like.get_chi2, ini_values)
        self.minimizer.errordef = 0.5
        # error only used to set initial parameter step
        self.minimizer.errors = error

    def minimize(self, compute_hesse=True):
        """Run migrad optimizer, and optionally compute Hessian matrix"""

        if self.verbose:
            print("will run migrad")
            self.minimizer.print_level = 0
        self.minimizer.migrad()

        if compute_hesse:
            if self.verbose:
                print("will compute Hessian matrix")
            self.minimizer.hesse()

        return

    def plot_best_fit(self, plot_every_iz=1, residuals=True):
        """Plot best-fit P1D vs data.
        - plot_every_iz (int): skip some redshift bins."""

        # get best-fit values from minimizer (should check that it was run)
        best_fit_values = np.array(self.minimizer.values)
        if self.verbose:
            print("best-fit values =", best_fit_values)

        # plt.title("iminuit best fit")
        self.like.plot_p1d(
            values=best_fit_values,
            plot_every_iz=plot_every_iz,
            residuals=residuals,
        )

        return

    def parameter_by_name(self, pname):
        """Find parameter in list of likelihood free parameters"""

        return [p for p in self.like.free_params if p.name == pname][0]

    def index_by_name(self, pname):
        """Find parameter index in list of likelihood free parameters"""

        return [
            i for i, p in enumerate(self.like.free_params) if p.name == pname
        ][0]

    def best_fit_value(self, pname, return_hesse=False):
        """Return best-fit value for pname parameter (assuming it was run).
        - return_hess: set to true to return also Gaussian error"""

        # get best-fit values from minimizer (in unit cube)
        cube_values = np.array(self.minimizer.values)
        if self.verbose:
            print("cube values =", cube_values)

        # get index for this parameter, and normalize value
        ipar = self.index_by_name(pname)
        par = self.like.free_params[ipar]
        par_value = par.value_from_cube(cube_values[ipar])

        # check if you were asked for errors as well
        if return_hesse:
            cube_errors = self.minimizer.errors
            par_error = cube_errors[ipar] * (par.max_value - par.min_value)
            return par_value, par_error
        else:
            return par_value

    def plot_ellipses(self, pname_x, pname_y, nsig=2, cube_values=False):
        """Plot Gaussian contours for parameters (pname_x,pname_y)
        - nsig: number of sigma contours to plot
        - cube_values: if True, will use unit cube values."""

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
            par_x = self.like.free_params[ix]
            val_x = par_x.value_from_cube(val_x)
            sig_x = sig_x * (par_x.max_value - par_x.min_value)
            par_y = self.like.free_params[iy]
            val_y = par_y.value_from_cube(val_y)
            sig_y = sig_y * (par_y.max_value - par_y.min_value)
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
