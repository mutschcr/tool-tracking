from pathlib import Path
import configparser
import click

from .constants import CONFIG_INI
from .__version__ import __version__


@click.group()
@click.version_option(__version__)
def cli():
    pass


@cli.command()
@click.argument("path", type=click.Path(exists=True), required=True)
@click.argument("name", type=str, required=False)
@click.option("--default", is_flag=True)
def register(path, name, default):

    path = Path(path).absolute()

    # Load config file
    config = configparser.ConfigParser()
    config.read(CONFIG_INI)
    print("> Update data-tools registry ...")

    name = name if name is not None else path.name
    sources: dict = config["Sources"]
    sources[name] = str(path)

    print(f"> Add entry '{name} = {str(path)}' to data-tools registry")

    if default:
        config["Sources"]["default"] = name
        print(f"> Set '{name}' as default data source")

    # Save changes to config file
    with open(CONFIG_INI, 'w') as configfile:
        config.write(configfile)


@cli.command()
def show():
    with open(CONFIG_INI, "r") as configfile:
        file = configfile.read()

    print("*** DATA-TOOLS CONFIG ***")
    print(f"> config from {CONFIG_INI.absolute()}:\n")
    print(file)
    print("*** ***************** ***")


@cli.command()
def version():
    print(f"data-tools {__version__}")
