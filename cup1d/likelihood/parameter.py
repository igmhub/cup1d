"""Plain-dictionary likelihood parameter definitions."""

from collections.abc import Mapping

import numpy as np


def make_parameter(
    name,
    min_value,
    max_value,
    value=None,
    Gauss_priors_width=None,
    fixed=False,
    hessian_transform=None,
):
    """Create the canonical plain-dictionary parameter definition.

    Parameters
    ----------
    name : str
        Parameter identifier.
    min_value, max_value : float
        Physical uniform-prior bounds.
    value : float, optional
        Current physical value.
    Gauss_priors_width : float, optional
        Physical Gaussian-prior width.
    fixed : bool, default=False
        Mark the coordinate fixed for minimizers that support masking.
    hessian_transform : str, optional
        Coordinate transformation hint for Hessian estimation.

    Returns
    -------
    dict
        Canonical parameter property mapping.
    """

    return {
        "name": name,
        "value": value,
        "min_value": min_value,
        "max_value": max_value,
        "Gauss_priors_width": Gauss_priors_width,
        "fixed": fixed,
        "hessian_transform": hessian_transform,
    }


# Compatibility for callers that imported the former constructor. This is a
# function alias, not a parameter class: calls now return plain dictionaries.
LikelihoodParameter = make_parameter


def info_str(parameter, all_info=False):
    """Format a compact parameter-definition description.

    Parameters
    ----------
    parameter : mapping
        Canonical parameter dictionary.
    all_info : bool, default=False
        Include physical lower and upper bounds.

    Returns
    -------
    str
        Name/value string, optionally with bounds.
    """

    info = f"{parameter['name']} = {parameter['value']}"
    if all_info:
        info += f" , {parameter['min_value']} , {parameter['max_value']}"
    return info


def value_in_cube(parameters, name, value=None):
    """Convert one physical parameter value to its unit-cube coordinate.

    Parameters
    ----------
    parameters : mapping
        Parameter definitions keyed by name.
    name : str
        Parameter name.
    value : float, optional
        Physical value. Defaults to the definition's ``'value'``.

    Returns
    -------
    float
        Unit-cube coordinate; values outside bounds are not clipped.

    Raises
    ------
    ValueError
        If neither ``value`` nor the definition contains a current value.
    """

    parameter = parameters[name]
    value = parameter["value"] if value is None else value
    if value is None:
        raise ValueError(f"value not set for parameter {name}")
    width = parameter["max_value"] - parameter["min_value"]
    return (value - parameter["min_value"]) / width


def value_from_cube(parameters, name, value):
    """Convert one unit-cube coordinate to a physical parameter value.

    Parameters
    ----------
    parameters : mapping
        Parameter definitions keyed by name.
    name : str
        Parameter name.
    value : float or ndarray
        Unit-cube coordinate or coordinates.

    Returns
    -------
    float or ndarray
        Physical value with the input shape.
    """

    parameter = parameters[name]
    width = parameter["max_value"] - parameter["min_value"]
    return parameter["min_value"] + value * width


def error_from_cube(parameters, name, error):
    """Convert unit-cube uncertainty to physical parameter units.

    Parameters
    ----------
    parameters : mapping
        Parameter definitions keyed by name.
    name : str
        Parameter name.
    error : float or ndarray
        Unit-cube uncertainty.

    Returns
    -------
    float or ndarray
        Physical uncertainty with the input shape.
    """

    parameter = parameters[name]
    return error * (parameter["max_value"] - parameter["min_value"])



def values_from_point(parameters, point):
    """Unwrap a public named point into scalar physical values.

    Parameters
    ----------
    parameters : mapping
        Free-parameter definitions determining required names and order.
    point : mapping or None
        Mapping from names to scalar values or full parameter dictionaries.

    Returns
    -------
    dict or None
        Scalar physical values for all free parameters, or ``None`` unchanged.
    """

    if point is None:
        return None
    values = {}
    for name in parameters:
        value = point[name]
        values[name] = value["value"] if isinstance(value, Mapping) else value
    return values


def point_from_values(parameters, values):
    """Copy definitions and replace current values from a scalar mapping.

    Parameters
    ----------
    parameters : mapping
        Base parameter definitions.
    values : mapping
        Physical values keyed by parameter name.

    Returns
    -------
    dict
        Deep-copied parameter definitions with matching ``'value'`` entries
        updated; unknown values are ignored.
    """

    import copy

    point = copy.deepcopy(parameters)
    for name, value in values.items():
        if name in point:
            point[name]["value"] = value
    return point

def values_to_cube(parameters, values=None):
    """Return an ordered unit-cube array from physical parameter values.

    Parameters
    ----------
    parameters : mapping
        Ordered parameter definitions.
    values : mapping, optional
        Direct physical values or full parameter dictionaries.

    Returns
    -------
    ndarray
        Unit-cube coordinates in ``parameters`` iteration order.

    Notes
    -----
    ``values`` may map names directly to physical values or to complete
    parameter dictionaries such as ``Likelihood.free_params``. The latter is
    the user-facing form and supplies each value under its ``"value"`` key.
    """

    if values is None:
        values = {name: parameter["value"] for name, parameter in parameters.items()}
    cube_values = []
    for name in parameters:
        value = values[name]
        if isinstance(value, Mapping):
            value = value["value"]
        cube_values.append(value_in_cube(parameters, name, value))
    return np.asarray(cube_values)


def values_from_cube(parameters, values):
    """Convert one unit-cube coordinate vector to physical values.

    Parameters
    ----------
    parameters : mapping
        Ordered parameter definitions.
    values : array-like
        One coordinate per parameter.

    Returns
    -------
    dict
        Physical values keyed by parameter name.

    Raises
    ------
    ValueError
        If coordinate count differs from parameter count.
    """

    if len(values) != len(parameters):
        raise ValueError("sampling-point size mismatch")
    return {
        name: value_from_cube(parameters, name, values[index])
        for index, name in enumerate(parameters)
    }


def values_from_cube_batch(parameters, values):
    """Convert unit-cube points to a columnar physical-parameter mapping.

    Parameters
    ----------
    parameters : mapping
        Ordered parameter definitions.
    values : array-like
        Unit-cube array with shape ``(n_batch, n_parameters)``.

    Returns
    -------
    dict
        Physical arrays of shape ``(n_batch,)`` keyed by parameter name.

    Raises
    ------
    ValueError
        If the array is not two-dimensional with the required trailing size.

    Notes
    -----
    ``values`` has shape ``(n_batch, n_parameters)``. The result maps every
    parameter to a one-dimensional ``(n_batch,)`` array, avoiding one Python
    dictionary per walker in batch-aware callers.
    """

    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(parameters):
        raise ValueError(
            "expected unit-cube points with shape "
            f"(n_batch, {len(parameters)}); got {values.shape}"
        )
    return {
        name: value_from_cube(parameters, name, values[:, index])
        for index, name in enumerate(parameters)
    }
