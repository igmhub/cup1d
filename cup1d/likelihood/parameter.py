"""Plain-dictionary likelihood parameter definitions."""

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
    """Return the canonical dictionary describing one model parameter."""

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
    """Return a compact description of a parameter dictionary."""

    info = f"{parameter['name']} = {parameter['value']}"
    if all_info:
        info += f" , {parameter['min_value']} , {parameter['max_value']}"
    return info


def value_in_cube(parameters, name, value=None):
    """Normalize one physical value using a parameter-property mapping."""

    parameter = parameters[name]
    value = parameter["value"] if value is None else value
    if value is None:
        raise ValueError(f"value not set for parameter {name}")
    width = parameter["max_value"] - parameter["min_value"]
    return (value - parameter["min_value"]) / width


def value_from_cube(parameters, name, value):
    """Convert one unit-cube coordinate to physical units."""

    parameter = parameters[name]
    width = parameter["max_value"] - parameter["min_value"]
    return parameter["min_value"] + value * width


def error_from_cube(parameters, name, error):
    """Convert one unit-cube uncertainty to physical units."""

    parameter = parameters[name]
    return error * (parameter["max_value"] - parameter["min_value"])


def values_to_cube(parameters, values=None):
    """Return an ordered unit-cube array from physical parameter values."""

    if values is None:
        values = {name: parameter["value"] for name, parameter in parameters.items()}
    return np.asarray(
        [value_in_cube(parameters, name, values[name]) for name in parameters]
    )


def values_from_cube(parameters, values):
    """Return an ordered mapping of names to physical values."""

    if len(values) != len(parameters):
        raise ValueError("sampling-point size mismatch")
    return {
        name: value_from_cube(parameters, name, values[index])
        for index, name in enumerate(parameters)
    }
