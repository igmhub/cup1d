#!/usr/bin/env python
"""Generate initial conditions from an at-a-time YAML configuration."""

import argparse

from cup1d.inference import generate_at_a_time_initial_conditions


def parse_arguments():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="YAML file with fit_type: at_a_time_global")
    parser.add_argument("--output", help="Optional destination .npy path.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing initial-condition file.",
    )
    return parser.parse_args()


def main():
    """Run the initial-condition workflow."""

    arguments = parse_arguments()
    generate_at_a_time_initial_conditions(
        arguments.config,
        output_path=arguments.output,
        overwrite=arguments.overwrite,
    )


if __name__ == "__main__":
    main()
