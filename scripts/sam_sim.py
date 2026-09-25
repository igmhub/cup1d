"""Run a Cup1D sampler from a YAML configuration."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

from cup1d import Analysis, Args
from cup1d.utils.utils import get_path_repo


def _initial_point(analysis):
    """Return the configured initial point in sampler coordinates."""

    point = analysis.fitter.sampling_point_from_parameters().copy()
    indices = {
        name: index for index, name in enumerate(analysis.like.free_params)
    }
    for name, value in analysis.args.initial_sampling_values.items():
        if name not in indices:
            raise ValueError(
                f"Initial sampling value provided for unknown parameter {name}"
            )
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"Initial sampling value for {name} must lie in [0, 1]"
            )
        point[indices[name]] = value
    return point


def run(config_path, refine=True):
    """Minimize, sample, and optionally refine a configured mock analysis."""

    config_path = Path(config_path).expanduser().resolve()
    args = Args.from_yaml(config_path, synthetic=True, verbose=False)
    output = Path(args.out_folder).expanduser()
    if not output.is_absolute():
        output = Path(get_path_repo("cup1d")) / output
    args.out_folder = str(output.resolve())
    analysis = Analysis(args)

    initial_point = _initial_point(analysis)
    analysis.run_minimizer(initial_point, restart=True)
    analysis.run_sampler()

    if refine:
        # run_sampler sets mle_cube to the best point in the chain. Because a
        # chain is present, this minimization updates sampler_results.npy and
        # leaves the independent minimizer_results.npy untouched.
        analysis.run_minimizer(analysis.fitter.mle_cube, restart=True)

    return analysis


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        type=Path,
        help="synthetic-analysis YAML configuration",
    )
    parser.add_argument(
        "--no-refine",
        action="store_true",
        help="do not minimize from the best point in the completed chain",
    )
    return parser.parse_args()


def main():
    options = parse_args()
    run(options.config, refine=not options.no_refine)


if __name__ == "__main__":
    main()
