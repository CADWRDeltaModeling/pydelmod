# -*- coding: utf-8 -*-
"""Console script for pydelmod."""
import sys
import click


@click.command()
def main(args=None):
    """Console script for pydelmod."""
    click.echo("pydelmod main called")
    click.echo("Does nothing ... yet")
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
