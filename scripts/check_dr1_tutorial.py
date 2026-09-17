"""Run the first three executable cells of the DR1 tutorial."""

import os

import numpy as np

from cup1d import Analysis, Args
from cup1d.utils.utils import get_path_repo


EXPECTED_CHI2 = np.float64(655.1284792810345)


def main():
    config_file = os.path.join(
        get_path_repo("cup1d"),
        "configs",
        "cm2026",
        "cm2026_base.yaml",
    )
    args = Args.from_yaml(config_file, verbose=False)
    analysis = Analysis(args)
    p0 = analysis.like.sampling_point_from_parameters().copy()
    analysis.like.parameters_from_sampling_point(p0)
    chi2 = analysis.like.get_chi2(p0)

    print(repr(chi2))
    if chi2 != EXPECTED_CHI2:
        raise AssertionError(f"Expected {EXPECTED_CHI2!r}, obtained {chi2!r}")


if __name__ == "__main__":
    main()
