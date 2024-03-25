# -*- coding: utf-8 -*-
"""Console script for pydelmod."""
from pydelmod import dsm2ui
from pydelmod.dsm2ui import DSM2FlowlineMap, build_output_plotter
from pydelmod import postpro_dsm2, checklist_dsm2
from pydelmod import dsm2_chan_mann_disp
from pydelmod import create_ann_inputs
from pydelmod import datastore2dss
import sys
import click
import panel as pn

pn.extension()


CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.group(context_settings=CONTEXT_SETTINGS)
def main():
    pass


@click.command()
@click.argument(
    "flowline_shapefile", type=click.Path(dir_okay=False, exists=True, readable=True)
)
@click.argument(
    "hydro_echo_file", type=click.Path(dir_okay=False, exists=True, readable=True)
)
@click.option(
    "-c",
    "--colored-by",
    type=click.Choice(["MANNING", "DISPERSION", "LENGTH", "ALL"], case_sensitive=False),
    default="MANNING",
)
@click.option(
    "--base-file", "-b", type=click.Path(dir_okay=False, exists=True, readable=True)
)
def map_channels_colored(flowline_shapefile, hydro_echo_file, colored_by, base_file):
    mapui = DSM2FlowlineMap(flowline_shapefile, hydro_echo_file, base_file)
    if colored_by == "ALL":
        return pn.panel(
            pn.Column(
                *[
                    mapui.show_map_colored_by_column(c.upper())
                    for c in ["MANNING", "DISPERSION", "LENGTH"]
                ]
            )
        ).show()
    else:
        return pn.panel(mapui.show_map_colored_by_column(colored_by.upper())).show()


@click.command()
@click.argument(
    "channel_shapefile", type=click.Path(dir_okay=False, exists=True, readable=True)
)
@click.argument(
    "hydro_echo_file", type=click.Path(dir_okay=False, exists=True, readable=True)
)
@click.option(
    "-v",
    "--variable",
    type=click.Choice(["flow", "stage"], case_sensitive=False),
    default="flow",
)
def output_map_plotter(channel_shapefile, hydro_echo_file, variable):
    plotter = build_output_plotter(channel_shapefile, hydro_echo_file, variable)
    pn.serve(
        plotter.get_panel(), kwargs={"websocket-max-message-size": 100 * 1024 * 1024}
    )


@click.command()
@click.argument(
    "node_shapefile", type=click.Path(dir_okay=False, exists=True, readable=True)
)
@click.argument(
    "hydro_echo_file", type=click.Path(dir_okay=False, exists=True, readable=True)
)
def node_map_flow_splits(node_shapefile, hydro_echo_file):
    netmap = dsm2ui.DSM2GraphNetworkMap(node_shapefile, hydro_echo_file)
    pn.serve(
        netmap.get_panel(), kwargs={"websocket-max-message-size": 100 * 1024 * 1024}
    )


@click.command()
@click.argument(
    "process_name",
    type=click.Choice(
        [
            "observed",
            "model",
            "plots",
            "heatmaps",
            "validation_bar_charts",
            "copy_plot_files",
        ],
        case_sensitive=False,
    ),
    default="",
)
@click.argument("json_config_file")
@click.option("--dask/--no-dask", default=False)
def exec_postpro_dsm2(process_name, json_config_file, dask):
    print(process_name, dask, json_config_file)
    postpro_dsm2.run_process(process_name, json_config_file, dask)


@click.command()
@click.argument(
    "chan_to_group_filename",
    type=click.Path(dir_okay=False, exists=True, readable=True),
)
@click.argument(
    "chan_group_mann_disp_filename",
    type=click.Path(dir_okay=False, exists=True, readable=True),
)
@click.argument(
    "dsm2_channels_input_filename",
    type=click.Path(dir_okay=False, exists=True, readable=True),
)
@click.argument(
    "dsm2_channels_output_filename",
    type=click.Path(dir_okay=False, exists=False, readable=False),
)
def exec_dsm2_chan_mann_disp(
    chan_to_group_filename,
    chan_group_mann_disp_filename,
    dsm2_channels_input_filename,
    dsm2_channels_output_filename,
):
    dsm2_chan_mann_disp.prepro(
        chan_to_group_filename,
        chan_group_mann_disp_filename,
        dsm2_channels_input_filename,
        dsm2_channels_output_filename,
    )


@click.command()
@click.argument(
    "process_name",
    type=click.Choice(["resample", "extract", "plot"], case_sensitive=False),
    default="",
)
@click.argument("json_config_file")
def exec_checklist_dsm2(process_name, json_config_file):
    print(process_name, json_config_file)
    checklist_dsm2.run_checklist(process_name, json_config_file)


@click.command()
def exec_create_ann_inputs():
    print("create ann inputs")
    create_ann_inputs.create_ann_inputs()


@click.command(name="todss")
@click.argument(
    "datastore_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
)
@click.argument(
    "dssfile", type=click.Path(dir_okay=False, exists=False, readable=False)
)
@click.argument(
    "param",
    type=click.Choice(
        [
            "elev",
            "predictions",
            "flow",
            "temp",
            "do",
            "ec",
            "ssc",
            "turbidity",
            "ph",
            "velocity",
            "cla",
        ],
        case_sensitive=False,
    ),
)
@click.option(
    "--repo-level",
    type=click.Choice(["screened"], case_sensitive=False),
    default="screened",
)
def todss(datastore_dir, dssfile, param, repo_level="screened"):
    datastore2dss.read_from_datastore_write_to_dss(
        datastore_dir, dssfile, param, repo_level
    )


@click.command(name="tostationfile")
@click.argument(
    "datastore_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
)
@click.argument(
    "stationfile", type=click.Path(dir_okay=False, exists=False, readable=False)
)
@click.argument(
    "param",
    type=click.Choice(
        [
            "elev",
            "predictions",
            "flow",
            "temp",
            "do",
            "ec",
            "ssc",
            "turbidity",
            "ph",
            "velocity",
            "cla",
        ],
        case_sensitive=False,
    ),
)
def tostationfile(datastore_dir, stationfile, param):
    datastore2dss.write_station_lat_lng(datastore_dir, stationfile, param)


@click.group(context_settings=CONTEXT_SETTINGS)
def datastore():
    """
    CLI for interacting with the datastore.
    pydelmod datastore -h
    for more info
    """
    pass


datastore.add_command(todss)
datastore.add_command(tostationfile)


@click.command(name="stations_output_file")
@click.argument(
    "stations_file",
    type=click.Path(dir_okay=False, exists=True, readable=True),
)
@click.argument(
    "centerlines_file",
    type=click.Path(dir_okay=False, exists=True, readable=True),
)
@click.argument(
    "output_file", type=click.Path(dir_okay=False, exists=False, readable=False)
)
@click.option(
    "--distance-tolerance",
    type=click.INT,
    default=100,
    help="Maximum distance from a line that a station can be to be considered on that line",
)
def stations_output_file(
    stations_file, centerlines_file, output_file, distance_tolerance=100
):
    """
    Create DSM2 channels output compatible file for given stations info (station_id, lat lon)
    and centerlines geojson file (DSM2 channels centerlines) and writing out output_file

    stations_file :  The stations file should be a csv file with columns 'station_id', 'lat', 'lon'
        You can generate this file from a shapefile using the `pydelmod datastsore tostationfile` command

    centerlines_file : Path to the centerlines geojson file for dsm2 channel centerlines

    output_file : Path to the output file, the format will be a pandas dataframe with columns 'NAME', 'CHAN_NO', 'DISTANCE' and space separated

    distance_tolerance : default 100
    """
    from pydelmod import dsm2gis

    dsm2gis.create_stations_output_file(
        stations_file=stations_file,
        centerlines_file=centerlines_file,
        output_file=output_file,
        distance_tolerance=distance_tolerance,
    )


@click.group(context_settings=CONTEXT_SETTINGS)
def dsm2():
    """
    CLI for DSM2 related commands
    pydelmod dsm2 -h
    for more info
    """
    pass


dsm2.add_command(stations_output_file)

main.add_command(map_channels_colored)
main.add_command(node_map_flow_splits)
main.add_command(output_map_plotter)
main.add_command(exec_postpro_dsm2)
main.add_command(exec_dsm2_chan_mann_disp)
main.add_command(exec_checklist_dsm2)
main.add_command(exec_create_ann_inputs)
main.add_command(datastore)
main.add_command(dsm2)

if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
