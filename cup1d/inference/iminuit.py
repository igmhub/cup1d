import numpy as np
from iminuit import Minuit

# our own modules
from cup1d.likelihood import parameter as parameter_space


class IminuitMinimizer(object):
    """Adapt a cup1d likelihood to iminuit's bounded unit-cube interface.

    The adapter translates unit-cube coordinates to physical likelihood
    parameters, evaluates chi-squared, and exposes Minuit-compatible named
    objective arguments. It retains the likelihood and fitting configuration
    required to initialize and run iminuit optimizations.
    """

    def __init__(self, like, ini_values=None, error=0.02, verbose=False):
        """Initialize iminuit with a cup1d likelihood and unit-cube values.

        Parameters
        ----------
        like : cup1d.likelihood.likelihood.Likelihood
            Likelihood whose free parameters define the minimizer dimensions.
        ini_values : array-like, optional
            Unit-cube starting coordinates.  Defaults to 0.5 for each free
            parameter.
        error : float or array-like, default=0.02
            Initial iminuit parameter-step estimate; not a final uncertainty.
        verbose : bool, default=False
            Print optimization-stage messages.
        """

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
        """Evaluate negative log posterior at unit-cube coordinates.

        Parameters
        ----------
        values : array-like
            Unit-cube coordinates in free-parameter order.

        Returns
        -------
        float
            Negative likelihood log posterior.
        """

        parameters = parameter_space.values_from_cube(self.like.free_params, values)
        return -self.like.log_prob(parameters)

    def minimize(self, compute_hesse=True):
        """Run iminuit Migrad and optionally calculate Hesse uncertainties.

        Parameters
        ----------
        compute_hesse : bool, default=True
            Run ``minimizer.hesse()`` after Migrad.

        Returns
        -------
        None
        """

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
        """Plot the iminuit best-fit P1D model and data.

        Parameters
        ----------
        plot_every_iz : int, default=1
            Plot every nth redshift bin.
        residuals : bool, default=True
            Include residual panels.

        Returns
        -------
        object
            Result returned by :func:`cup1d.postprocessing.inference.plot_best_fit`.
        """
        from cup1d.postprocessing.inference import plot_best_fit as _plot

        return _plot(self, plot_every_iz, residuals)

    def parameter_by_name(self, pname):
        """Return the free-parameter definition for a name.

        Parameters
        ----------
        pname : str
            Free-parameter name.

        Returns
        -------
        dict
            Parameter definition stored by the likelihood.
        """

        return self.like.free_params[pname]

    def index_by_name(self, pname):
        """Return the unit-cube coordinate index for a parameter name.

        Parameters
        ----------
        pname : str
            Free-parameter name.

        Returns
        -------
        int
            Index in ``like.free_param_names``.
        """

        return self.like.free_param_names.index(pname)

    def best_fit_value(self, pname, return_hesse=False):
        """Return a physical best-fit value and optional Hesse uncertainty.

        Parameters
        ----------
        pname : str
            Free-parameter name.
        return_hesse : bool, default=False
            Also convert and return iminuit's Hesse error.

        Returns
        -------
        float or tuple of float
            Physical best-fit value, optionally followed by its physical Hesse
            uncertainty.
        """

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
        """Plot iminuit covariance ellipses for two parameters.

        Parameters
        ----------
        pname_x, pname_y : str
            Parameter names used as horizontal and vertical axes.
        nsig : float, default=2
            Ellipse radius in Gaussian standard deviations.
        cube_values : bool, default=False
            Plot unit-cube coordinates instead of physical values.

        Returns
        -------
        object
            Result returned by :func:`cup1d.postprocessing.inference.plot_ellipses`.
        """
        from cup1d.postprocessing.inference import plot_ellipses as _plot

        return _plot(self, pname_x, pname_y, nsig, cube_values)
