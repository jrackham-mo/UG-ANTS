"""Entrypoint for running UG-ANTS applications.

conda env create -p <path/to/install/developer/environment> -f environment.yml
conda activate <path/to/install/developer/environment>
pip install -e .

ugants --help
"""

import argparse
import configparser
import importlib
import pathlib

import yaml

from ugants import __version__


def resolve_path(filepath: str):  # noqa: D103
    path = pathlib.Path(filepath)
    path = path.resolve(strict=True)
    return path


def load_recipe(filepath: pathlib.Path):  # noqa: D103
    print(f"Reading recipe from {filepath}")
    match filepath.suffix:
        case ".yml" | ".yaml":
            return _load_yaml(filepath)
        case ".conf" | ".ini":
            return _load_conf(filepath)
        case _:
            raise ValueError(f"Unsupported file type: {filepath.suffix}")


def _load_yaml(filepath):
    with open(filepath) as f:
        recipe = yaml.safe_load(f)
    return recipe


def _load_conf(filepath):
    config = configparser.ConfigParser()
    config.read(filepath)
    recipe = {section: dict(config[section]) for section in config.sections()}
    return recipe


def validate(recipe):  # noqa: D103
    print("Validating recipe:\n", recipe)


def run(recipe):  # noqa: D103
    print(recipe)
    app = importlib.import_module(recipe["ugants"]["app"])

    loaded_sources = {}
    for source_name, source_loader in app.SOURCES.items():
        recipe_value = recipe["ugants.sources"][source_name]
        constraint = recipe["ugants.constraints"].get(source_name, None)
        loaded_source = source_loader.load(recipe_value, constraint)
        loaded_sources[source_name] = loaded_source

    settings = {}
    for setting_name, setting_parser in app.SETTINGS.items():
        recipe_value = recipe["ugants.settings"][setting_name]
        parsed_setting = setting_parser.parse(recipe_value)
        settings[setting_name] = parsed_setting

    kwargs = loaded_sources | settings

    print("Parsed recipe:\n", kwargs)
    outputs = app.main(**kwargs)

    for output_name, output_handler in app.OUTPUTS.items():
        output = outputs[output_name]
        destination = recipe["ugants.outputs"][output_name]
        output_handler.save(output, destination)


def main():  # noqa: D103
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(title="subcommands")

    run_help = "Run a UG-ANTS application"
    run_subparser = subparsers.add_parser("run", help=run_help, description=run_help)
    run_subparser.add_argument(
        "recipe", help="Path to a recipe file to run", type=resolve_path
    )
    run_subparser.set_defaults(func=run)

    validate_help = "Validate a UG-ANTS recipe"
    validate_subparser = subparsers.add_parser(
        "validate", help=validate_help, description=validate_help
    )
    validate_subparser.add_argument(
        "recipe", help="Path to a recipe file to validate", type=resolve_path
    )
    validate_subparser.set_defaults(func=validate)

    args = parser.parse_args()
    recipe = load_recipe(args.recipe)

    args.func(recipe)
