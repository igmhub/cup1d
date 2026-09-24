import numpy as np
from iminuit import Minuit

# our own modules
from cup1d.likelihood import likelihood
from cup1d.likelihood import parameter as parameter_space


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
        self.minimizer = Minuit(self.minus_log_prob, ini_values)
        # self.minimizer = Minuit(like.get_chi2, ini_values)
        self.minimizer.errordef = 0.5
        # error only used to set initial parameter step
        self.minimizer.errors = error

    def minus_log_prob(self, values):
        """Negative posterior for an iminuit point in unit-cube coordinates."""

        parameters = parameter_space.values_from_cube(self.like.free_params, values)
        return -self.like.log_prob(parameters)

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
        """Delegate to :func:`cup1d.postprocessing.inference.plot_best_fit`."""
        from cup1d.postprocessing.inference import plot_best_fit as _plot

        return _plot(self, plot_every_iz, residuals)

    def parameter_by_name(self, pname):
        """Find parameter in list of likelihood free parameters"""

        return self.like.free_params[pname]

    def index_by_name(self, pname):
        """Find parameter index in list of likelihood free parameters"""

        return self.like.free_param_names.index(pname)

    def best_fit_value(self, pname, return_hesse=False):
        """Return best-fit value for pname parameter (assuming it was run).
        - return_hess: set to true to return also Gaussian error"""

        # get best-fit values from minimizer (in unit cube)
        cube_values = np.array(self.minimizer.values)
        if self.verbose:
            print("cube values =", cube_values)

        # get index for this parameter, and normalize value
        ipar = self.index_by_name(pname)
        par_value = parameter_space.value_from_cube(self.like.free_params, pname, cube_values[ipar])

        # check if you were asked for errors as well
        if return_hesse:
            cube_errors = self.minimizer.errors
            par_error = parameter_space.error_from_cube(self.like.free_params, pname, cube_errors[ipar])
            return par_value, par_error
        else:
            return par_value

    def plot_ellipses(self, pname_x, pname_y, nsig=2, cube_values=False):
        """Delegate to :func:`cup1d.postprocessing.inference.plot_ellipses`."""
        from cup1d.postprocessing.inference import plot_ellipses as _plot

        return _plot(self, pname_x, pname_y, nsig, cube_values)
