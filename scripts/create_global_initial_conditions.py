#!/usr/bin/env python
"""Generate global-fit initial conditions from a YAML configuration."""

import argparse

from cup1d.inference import generate_global_initial_conditions


def parse_arguments():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="YAML file with fit_type: global_opt")
    parser.add_argument("--output", help="Optional destination .npy path.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing initial-condition file.",
    )
    return parser.parse_args()


def main():
    """Run global IC generation."""

    arguments = parse_arguments()
    generate_global_initial_conditions(
        arguments.config,
        output_path=arguments.output,
        overwrite=arguments.overwrite,
    )


if __name__ == "__main__":
    main()
