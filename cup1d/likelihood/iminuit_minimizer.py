"""Wrapper around an iminuit minimizer for Lyman alpha likelihood."""

from __future__ import annotations

from typing import Any

import numpy as np
from iminuit import Minuit

# our own modules


class IminuitMinimizer:
    """Wrapper around an iminuit minimizer for Lyman alpha likelihood.

    Parameters
    ----------
    like : Any
        Likelihood object to be minimized.
    ini_values : np.ndarray, optional
        Initial parameter values in the unit cube. If None, the center of the
        unit cube is used.
    error : float, optional
        Initial step size for the parameters. Default is 0.02.
    verbose : bool, optional
        Whether to print verbose output. Default is False.

    Attributes
    ----------
    verbose : bool
        Verbose flag.
    like : Any
        Likelihood object.
    minimizer : Minuit
        Iminuit minimizer object.
    """

    def __init__(
        self,
        like: Any,
        ini_values: np.ndarray | None = None,
        error: float = 0.02,
        verbose: bool = False,
    ):
        """Initialize the iminuit minimizer."""

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

    def minimize(self, compute_hesse: bool = True) -> None:
        """Run migrad optimizer, and optionally compute Hessian matrix.

        Parameters
        ----------
        compute_hesse : bool, optional
            Whether to compute the Hessian matrix. Default is True.
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

    def plot_best_fit(self, plot_every_iz: int = 1, residuals: bool = True) -> None:
        """Plot best-fit P1D vs data.

        Parameters
        ----------
        plot_every_iz : int, optional
            Skip some redshift bins. Default is 1.
        residuals : bool, optional
            Whether to plot residuals. Default is True.
        """

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

    def parameter_by_name(self, pname: str) -> Any:
        """Find parameter in list of likelihood free parameters.

        Parameters
        ----------
        pname : str
            Parameter name.

        Returns
        -------
        Any
            Likelihood parameter object.
        """

        return [p for p in self.like.free_params if p.name == pname][0]

    def index_by_name(self, pname: str) -> int:
        """Find parameter index in list of likelihood free parameters.

        Parameters
        ----------
        pname : str
            Parameter name.

        Returns
        -------
        int
            Index of the parameter.
        """

        return [
            i for i, p in enumerate(self.like.free_params) if p.name == pname
        ][0]

    def best_fit_value(self, pname: str, return_hesse: bool = False) -> Any:
        """Return best-fit value for pname parameter (assuming it was run).

        Parameters
        ----------
        pname : str
            Parameter name.
        return_hesse : bool, optional
            Whether to return also the Gaussian error. Default is False.

        Returns
        -------
        float or tuple[float, float]
            Best-fit value, or (value, error) if return_hesse is True.
        """

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

    def plot_ellipses(
        self,
        pname_x: str,
        pname_y: str,
        nsig: int = 2,
        cube_values: bool = False,
    ) -> None:
        """Plot Gaussian contours for parameters (pname_x, pname_y).

        Parameters
        ----------
        pname_x : str
            Name of the parameter on the x-axis.
        pname_y : str
            Name of the parameter on the y-axis.
        nsig : int, optional
            Number of sigma contours to plot. Default is 2.
        cube_values : bool, optional
            If True, will use unit cube values. Default is False.
        """

