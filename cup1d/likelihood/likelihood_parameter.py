"""Likelihood parameter representation and cube transforms."""

from __future__ import annotations


class LikelihoodParameter:
    """One scalar likelihood parameter with bounds and optional Gaussian prior.

    Parameters
    ----------
    name : str
        Parameter name.
    min_value : float
        Minimum value of the parameter.
    max_value : float
        Maximum value of the parameter.
    value : float, optional
        Current value of the parameter.
    Gauss_priors_width : float, optional
        Width of the Gaussian prior. If None, a uniform prior is used.
    fixed : bool, optional
        Whether the parameter is fixed. Default is False.

    Attributes
    ----------
    name : str
        Parameter name.
    min_value : float
        Minimum value.
    max_value : float
        Maximum value.
    value : float or None
        Current value.
    Gauss_priors_width : float or None
        Gaussian prior width.
    fixed : bool
        Fixed flag.
    """

    def __init__(
        self,
        name: str,
        min_value: float,
        max_value: float,
        value: float | None = None,
        Gauss_priors_width: float | None = None,
        fixed: bool = False,
    ):
        """Initialize the likelihood parameter."""
        self.name = name
        self.min_value = min_value
        self.max_value = max_value
        self.value = value
        self.Gauss_priors_width = Gauss_priors_width
        self.fixed = fixed

    def value_in_cube(self) -> float:
        """Normalize parameter value to [0, 1].

        Returns
        -------
        float
            Normalized value.
        """
        assert self.value is not None, "value not set in parameter " + self.name
        return (self.value - self.min_value) / (self.max_value - self.min_value)

    def get_value_in_cube(self, value: float) -> float:
        """Normalize parameter value to [0, 1].

        Parameters
        ----------
        value : float
            Physical parameter value.

        Returns
        -------
        float
            Normalized value.
        """
        return (value - self.min_value) / (self.max_value - self.min_value)

    def set_from_cube(self, x: float) -> None:
        """Set parameter value from value in cube [0, 1].

        Parameters
        ----------
        x : float
            Normalized value in the unit cube.
        """
        value = self.value_from_cube(x)
        self.value = value

    def set_without_cube(self, value: float) -> None:
        """Set the physical parameter value directly.

        Parameters
        ----------
        value : float
            Physical parameter value.
        """
        # Check to make sure parameter is within min/max
        assert self.min_value < value < self.max_value, (
            f"Parameter name: {self.name}"
        )
        self.value = value

    def info_str(self, all_info: bool = False) -> str:
        """Return a string with parameter name and value, for debugging.

        Parameters
        ----------
        all_info : bool, optional
            Whether to include min and max values. Default is False.

        Returns
        -------
        str
            Information string.
        """

        info = self.name + " = " + str(self.value)
        if all_info:
            info += " , " + str(self.min_value) + " , " + str(self.max_value)

        return info

    def value_from_cube(self, x: float) -> float:
        """Map a unit-cube value to the physical parameter range.

        Parameters
        ----------
        x : float
            Normalized value in the unit cube.

        Returns
        -------
        float
            Physical parameter value.
        """

        return self.min_value + x * (self.max_value - self.min_value)

    def err_from_cube(self, err: float) -> float:
        """Map a unit-cube error to the physical parameter range.

        Parameters
        ----------
        err : float
            Error in the unit cube.

        Returns
        -------
        float
            Error in the physical parameter range.
        """

        return err * (self.max_value - self.min_value)

    def get_new_parameter(self, value_in_cube: float) -> LikelihoodParameter:
        """Return copy of parameter, with updated value from cube.

        Parameters
        ----------
        value_in_cube : float
            Normalized value in the unit cube.

        Returns
        -------
        LikelihoodParameter
            A new LikelihoodParameter instance.
        """

        par = LikelihoodParameter(
            name=self.name,
            min_value=self.min_value,
            max_value=self.max_value,
            Gauss_priors_width=self.Gauss_priors_width,
        )
        par.set_from_cube(value_in_cube)

        return par
