"""Tabulated mean-transmitted-flux measurements used for comparison plots."""

import numpy as np


def get_mean_flux_measurements():
    """Return the Gaikwad (2021) and Turner (2024) mean-flux measurements.

    Returns
    -------
    tuple[dict, dict]
        Dictionaries for Gaikwad et al. (2021) and Turner et al. (2024), each
        with ``z``, ``mF``, and ``mF_err`` arrays. The Gaikwad dictionary also
        contains ``T0`` and ``T0_err`` in K, plus ``gamma`` and ``gamma_err``.
        New dictionaries are created for every call so callers may add derived
        quantities safely.
    """
    gaikwad2021 = {
        "z": np.array([2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8]),
        "mF": np.array(
            [0.8690, 0.8261, 0.7919, 0.7665, 0.7398, 0.7105, 0.6731, 0.5927, 0.5320, 0.4695]
        ),
        "mF_err": np.array(
            [0.0214, 0.0206, 0.0210, 0.0216, 0.0212, 0.0213, 0.0223, 0.0247, 0.0280, 0.0278]
        ),
        "T0": np.array(
            [9500, 11000, 12750, 13500, 14750, 14750, 12750, 11250, 10250, 9250]
        ),
        "T0_err": np.array([1393, 1028, 1132, 1390, 1341, 1322, 1493, 1125, 1070, 876]),
        "gamma": np.array(
            [1.500, 1.425, 1.325, 1.275, 1.250, 1.225, 1.275, 1.350, 1.400, 1.525]
        ),
        "gamma_err": np.array(
            [0.096, 0.133, 0.122, 0.122, 0.109, 0.120, 0.129, 0.108, 0.101, 0.140]
        ),
    }
    turner2024 = {
        "z": np.arange(2.05, 4.16, 0.10),
        "mF": np.exp(
            -np.array(
                [
                    0.147, 0.158, 0.179, 0.200, 0.226, 0.235, 0.268, 0.292,
                    0.316, 0.342, 0.373, 0.410, 0.455, 0.498, 0.527, 0.579,
                    0.638, 0.694, 0.770, 0.830, 0.854, 0.928,
                ]
            )
        ),
        "mF_err": np.array(
            [
                0.012, 0.012, 0.015, 0.016, 0.016, 0.018, 0.019, 0.020,
                0.021, 0.022, 0.023, 0.023, 0.022, 0.025, 0.030, 0.032,
                0.031, 0.032, 0.033, 0.034, 0.036, 0.039,
            ]
        ),
    }
    return gaikwad2021, turner2024
