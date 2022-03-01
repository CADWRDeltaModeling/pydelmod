# -*- coding: utf-8 -*-
"""Console script for pydelmod."""
from email.policy import default
from pydelmod.dsm2ui import DSM2FlowlineMap, build_output_plotter
import sys
import click
import panel as pn
pn.extension()


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.group(context_settings=CONTEXT_SETTINGS)
def main():
    pass


@click.command()
@click.argument("flowline_shapefile", type=click.Path(dir_okay=False, exists=True, readable=True))
@click.argument("hydro_echo_file", type=click.Path(dir_okay=False, exists=True, readable=True))
@click.option("-c","--colored-by", type=click.Choice(['MANNING', 'DISPERSION', 'LENGTH', 'ALL'],case_sensitive=False), default='MANNING')
def map_channels_colored(flowline_shapefile, hydro_echo_file, colored_by):
    mapui = DSM2FlowlineMap(flowline_shapefile, hydro_echo_file)
    if colored_by == 'ALL':
        return pn.panel(pn.Column(*[mapui.show_map_colored_by_column(c.upper()) for c in ['MANNING','DISPERSION', 'LENGTH']])).show()
    else:
        return pn.panel(mapui.show_map_colored_by_column(colored_by.upper())).show()

@click.command()
@click.argument("channel_shapefile", type=click.Path(dir_okay=False, exists=True, readable=True))
@click.argument("hydro_echo_file", type=click.Path(dir_okay=False, exists=True, readable=True))
@click.option("-v","--variable", type=click.Choice(['flow', 'stage'],case_sensitive=False), default='flow')
def output_map_plotter(channel_shapefile, hydro_echo_file, variable):
    plotter = build_output_plotter(channel_shapefile, hydro_echo_file, variable)
    pn.serve(plotter.get_panel(),kwargs={'websocket-max-message-size':100*1024*1024})

main.add_command(map_channels_colored)
main.add_command(output_map_plotter)
if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
