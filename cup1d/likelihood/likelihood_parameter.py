"""Likelihood parameter representation and cube transforms."""

import numpy as np


class LikelihoodParameter(object):
    """One scalar likelihood parameter with bounds and optional Gaussian prior."""

    def __init__(
        self,
        name,
        min_value,
        max_value,
        value=None,
        Gauss_priors_width=None,
        fixed=False,
    ):
        """Create a bounded likelihood parameter."""
        self.name = name
        self.min_value = min_value
        self.max_value = max_value
        self.value = value
        self.Gauss_priors_width = Gauss_priors_width
        self.fixed = fixed
        return

    def value_in_cube(self):
        """Normalize parameter value to [0,1]."""
        assert self.value is not None, "value not set in parameter " + self.name
        return (self.value - self.min_value) / (self.max_value - self.min_value)

    def get_value_in_cube(self, value):
        """Normalize parameter value to [0,1]."""
        return (value - self.min_value) / (self.max_value - self.min_value)

    def set_from_cube(self, x):
        """Set parameter value from value in cube [0,1]."""
        value = self.value_from_cube(x)
        self.value = value
        return

    def set_without_cube(self, value):
        """Set the physical parameter value directly."""
        # Check to make sure parameter is within min/max
        assert self.min_value < value < self.max_value, (
            "Parameter name: %s" % self.name
        )
        self.value = value
        return

    def info_str(self, all_info=False):
        """Return a string with parameter name and value, for debugging"""

        info = self.name + " = " + str(self.value)
        if all_info:
            info += " , " + str(self.min_value) + " , " + str(self.max_value)

        return info

    def value_from_cube(self, x):
        """Map a unit-cube value to the physical parameter range."""

        return self.min_value + x * (self.max_value - self.min_value)

    def err_from_cube(self, err):
        """Map a unit-cube error to the physical parameter range."""

        return err * (self.max_value - self.min_value)

    def get_new_parameter(self, value_in_cube):
        """Return copy of parameter, with updated value from cube"""

        par = LikelihoodParameter(
            name=self.name,
            min_value=self.min_value,
            max_value=self.max_value,
            Gauss_priors_width=self.Gauss_priors_width,
        )
        par.set_from_cube(value_in_cube)

        return par
