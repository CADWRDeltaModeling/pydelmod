import pandas as pd
import numpy as np
import pydelmod.utilities as pdmu
import pydelmod.nbplot as pdmn
import yaml
import pathlib


def load_config(config_file):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    # get location of config file
    config_file_path = pathlib.Path(config_file)
    # get parent directory of config file
    config_dir = config_file_path.parent

    # change output_locations, fpath to absolute path relative to config_dir
    config["output_locations"] = str(config_dir / config["output_locations"])
    # change scenarios, fpath to absolute path relative to config_dir
    for scenario in config["scenarios"]:
        scenario["fpath"] = str(config_dir / scenario["fpath"])
    # change calsim_file to absolute path relative to config_dir
    config["calsim_file"] = str(config_dir / config["calsim_file"])
    return config


def read_output_locations(fpath):
    df_stations = pd.read_csv(fpath, comment="#")
    df_stations["ID"] = [x.upper() for x in df_stations["ID"]]
    return df_stations


def read_scenarios(scenarios):
    return [
        {"name": scenario["name"], "fpath": scenario["fpath"]} for scenario in scenarios
    ]


def main(config_file):
    config = load_config(config_file)

    df_stations = read_output_locations(config["output_locations"])
    stations_to_read = df_stations["ID"].values

    scenarios = read_scenarios(config["scenarios"])
    calsim_file = config["calsim_file"]
    df_wyt = pdmu.read_calsim3_wateryear_types(calsim_file)

    period = config["period"]
    df_flow = pdmu.prep_df(
        scenarios,
        stations_to_read,
        [config["variable"]],
        [config["interval"]],
        df_wyt,
        period,
    )

    # Dynamically call plotting functions based on config
    plots = []
    for plot_config in config["plots"]:
        plot_type = plot_config["type"]
        options = plot_config["options"]
        plot_func = getattr(pdmn, plot_type)
        plot = plot_func(df_flow, df_stations, options)
        plots.append((options["title"], plot))

    # Display the plots
    import panel as pn

    pn.Tabs(*plots).show(title=f"DSM2 Analysis: {config_file}")
