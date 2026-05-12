"""Example script for loading P1D data.

This script demonstrates how to load and use P1D measurements
from various data sources.
"""

# Example data loading (requires actual data files)
# Uncomment to run with real data

# from cup1d.p1ds import DataDESIY1, DataChabanier2019, DataNyx


def example_data_descriptions():
    """Show available data sources and their descriptions."""

    data_sources = {
        "DataDESIY1": {
            "description": "DESI Year 1 P1D measurements",
            "reference": "DESI Collaboration (2024)",
            "redshifts": [2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0],
        },
        "DataChabanier2019": {
            "description": "Chabanier et al. (2019) P1D measurements",
            "reference": "Chabanier et al. (2019)",
            "redshifts": [2.0, 2.4, 2.8, 3.2, 3.6, 4.0],
        },
        "DataIrsic2017": {
            "description": "XQ-100 P1D measurements",
            "reference": "Irsic et al. (2017)",
            "redshifts": [2.0, 2.4, 2.8, 3.2, 3.6],
        },
        "DataNyx": {
            "description": "Nyx simulation P1D",
            "reference": "Armitage et al. (2018)",
            "redshifts": [2.0, 2.5, 3.0, 3.5, 4.0],
        },
    }

    print("=" * 60)
    print("Available P1D Data Sources")
    print("=" * 60)

    for name, info in data_sources.items():
        print(f"\n{name}")
        print(f"  Description: {info['description']}")
        print(f"  Reference: {info['reference']}")
        print(f"  Redshifts: {info['redshifts']}")


def example_data_structure():
    """Show the expected structure of P1D data."""

    print("\n" + "=" * 60)
    print("P1D Data Structure")
    print("=" * 60)

    structure = """
    BaseDataP1D attributes:
    -----------------------
    z : list
        Redshift values for each bin
    k_kms : list of arrays
        Wavenumber values in km/s for each redshift
    k_kms_min : list of arrays
        Minimum k values (bin edges)
    k_kms_max : list of arrays
        Maximum k values (bin edges)
    Pk_kms : list of arrays
        Power spectrum values
    cov_Pk_kms : list of arrays
        Covariance matrices
    covstat_Pk_kms : list of arrays
        Statistical covariance only
    Pksmooth_kms : list of arrays
        Smooth power spectrum (if available)
    full_Pk_kms : array
        Combined power spectrum (if available)
    full_cov_Pk_kms : array
        Combined covariance (if available)
    """
    print(structure)


def example_usage():
    """Example of how to use P1D data (pseudo-code)."""

    print("\n" + "=" * 60)
    print("Usage Example (pseudo-code)")
    print("=" * 60)

    code = """
    # Import data loader
    from cup1d.p1ds import DataDESIY1
    
    # Load data
    data = DataDESIY1()
    
    # Access data
    for iz, z in enumerate(data.z):
        print(f"z = {z}")
        print(f"  k range: {data.k_kms[iz].min():.4f} - {data.k_kms[iz].max():.4f} km/s")
        print(f"  P1D values: {len(data.Pk_kms[iz])} points")
        print(f"  covariance shape: {data.cov_Pk_kms[iz].shape}")
    
    # Use in likelihood
    from cup1d.likelihood import Likelihood
    like = Likelihood(data=data, theory=theory)
    """
    print(code)


if __name__ == "__main__":
    example_data_descriptions()
    example_data_structure()
    example_usage()
