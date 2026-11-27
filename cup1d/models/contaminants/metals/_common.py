"""Line geometry shared by the silicon metal models."""

import numpy as np

C_KMS = 299792.458


def vel_diff(lambda1, lambda2):
    """Convert two rest wavelengths into their velocity separation.

    Wavelength inputs share any length unit; the returned absolute separation
    is in km/s using the logarithmic Doppler convention.
    """
    return np.abs(np.log(lambda2 / lambda1)) * C_KMS


def rstrength(lambda1, lambda2, f1, f2):
    """Return the optically-thin wavelength--oscillator-strength ratio."""
    return (lambda1 * f1) / (lambda2 * f2)
