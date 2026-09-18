"""Regression test for the baseline DR1 tutorial analysis."""

import numpy as np

from cup1d import Analysis, Args


EXPECTED_CHI_SQUARED = np.float64(655.1284792810345)


def test_dr1_baseline_chi_squared(tmp_path):
    """The initial DR1 likelihood must reproduce the reference chi-squared."""

    arguments = Args.from_baseline(verbose=False)
    analysis = Analysis(arguments, out_folder=str(tmp_path))
    initial_point = analysis.like.sampling_point_from_parameters().copy()
    chi_squared = analysis.like.get_chi2(initial_point)

    assert isinstance(chi_squared, np.float64)
    assert chi_squared == EXPECTED_CHI_SQUARED
