"""Example script for using the IGM module.

This script demonstrates how to use the Thermal and MeanFlux classes
to model IGM properties.
"""

import numpy as np
from cup1d.igm import Thermal, MeanFlux


def example_thermal():
    """Example of using the Thermal class."""
    # Fiducial IGM values
    fid_igm = {
        "sigT_kms_z": np.array([2.0, 2.5, 3.0, 3.5, 4.0]),
        "sigT_kms": np.array([25.0, 28.0, 30.0, 32.0, 35.0]),
        "gamma_z": np.array([2.0, 2.5, 3.0, 3.5, 4.0]),
        "gamma": np.array([1.5, 1.6, 1.7, 1.8, 1.9]),
    }

    fid_vals = {
        "sigT_kms": np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        "gamma": np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
    }

    # Create thermal model
    thermal = Thermal(fid_igm=fid_igm, fid_vals=fid_vals)

    # Get thermal properties at z=2.5
    z = 2.5
    sigT = thermal.get_sigT_kms(z)
    T0 = thermal.get_T0(z)
    gamma = thermal.get_gamma(z)

    print(f"z = {z}")
    print(f"  sigT_kms = {sigT:.2f} km/s")
    print(f"  T0 = {T0:.2f} K")
    print(f"  gamma = {gamma:.2f}")


def example_mean_flux():
    """Example of using the MeanFlux class."""
    # Fiducial IGM values
    fid_igm = {
        "tau_eff_z": np.array([2.0, 2.5, 3.0, 3.5, 4.0]),
        "tau_eff": np.array([0.5, 0.4, 0.3, 0.25, 0.2]),
    }

    fid_vals = {
        "tau_eff": np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
    }

    # Create mean flux model
    mean_flux = MeanFlux(fid_igm=fid_igm, fid_vals=fid_vals)

    # Get mean flux at z=2.5
    z = 2.5
    tau = mean_flux.get_tau_eff(z)
    flux = mean_flux.get_mean_flux(z)

    print(f"z = {z}")
    print(f"  tau_eff = {tau:.4f}")
    print(f"  mean flux = {flux:.4f}")


if __name__ == "__main__":
    print("=" * 50)
    print("Thermal Model Example")
    print("=" * 50)
    example_thermal()

    print("\n" + "=" * 50)
    print("Mean Flux Model Example")
    print("=" * 50)
    example_mean_flux()
