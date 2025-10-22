"""Entrypoint for running UG-ANTS applications.

conda env create -p <path/to/install/developer/environment> -f environment.yml
conda activate <path/to/install/developer/environment>
pip install -e .

ugants --help
"""

import argparse

from ugants import __version__


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(title="subcommands")
    run_help = "Run a UG-ANTS application"
    run_subparser = subparsers.add_parser("run", help=run_help, description=run_help)
    run_subparser.add_argument("recipe", help="Path to a recipe file to run")

    validate_help = "Validate a UG-ANTS recipe"
    validate_subparser = subparsers.add_parser(
        "validate", help=validate_help, description=validate_help
    )
    validate_subparser.add_argument("recipe", help="Path to a recipe file to run")

    args = parser.parse_args()
    print(args)
