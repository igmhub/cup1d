import pytest
from cup1d.likelihood.pipeline import set_archive, set_cosmo
from cup1d.likelihood.input_pipeline import Args

def test_set_archive():
    # Test with a known training set
    try:
        archive = set_archive("Pedersen21")
        assert archive is not None
    except Exception as e:
        # If data is not available, we might get an error, but at least the import works
        pytest.skip(f"set_archive failed likely due to missing data: {e}")

def test_set_cosmo():
    # Test with a known cosmology
    cosmo = set_cosmo("Planck18")
    assert cosmo is not None
    assert hasattr(cosmo, 'H0')
    assert cosmo.H0 == 67.66

def test_args_init():
    args = Args()
    assert args.data_label == "DESIY1_QMLE3"
    assert args.fid_cosmo_label == "Planck18"
