import pandas as pd
import numpy as np
import pydelmod.utilities as pdmu
import pydelmod.nbplot as pdmn
import yaml
import pathlib
import logging

# Display the plots
import panel as pn

pn.extension("plotly", "tabulator")

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


def milli_to_micro(df):
    df["value"] = df["value"] * 1000.0
    return df


def rsl_ec_to_cl(df):
    return df.apply(
        lambda x: max(x["value"] * 0.15 - 12.0, x["value"] * 0.285 - 50), axis=1
    )


def read_regulations(fpath, df_wyt):
    df_reg = pd.read_csv(fpath)
    df_reg = pdmu.read_regulations(fpath, df_wyt)
    return df_reg


def get_common_stations(df_stations, df_reg):
    reg_loc = df_reg.location.unique()
    return df_stations[df_stations["ID"].isin(reg_loc)]


def generate_regulations(df_reg, df, freq="D"):
    df_reg_ag = pdmu.generate_regulation_timeseries(df_reg, df, freq)
    return df_reg_ag


def read_scenarios(scenarios):
    return [
        {"name": scenario["name"], "fpath": scenario["fpath"]} for scenario in scenarios
    ]


def main(config_file):
    mainPanel = pn.Tabs()

    def serve_main_panel():
        return mainPanel.show(title=f"DSM2 Analysis: {config_file}")

    import threading

    thread = threading.Thread(target=serve_main_panel, daemon=True)
    thread.start()

    config = load_config(config_file)

    df_stations = read_output_locations(config["output_locations"])
    stations_to_read = df_stations["ID"].values

    scenarios = read_scenarios(config["scenarios"])
    calsim_file = config["calsim_file"]
    df_wyt = pdmu.read_calsim3_wateryear_types(calsim_file)

    period = config["period"]
    delta_standards = {}
    if "delta_standards" in config:
        for standard in config["delta_standards"]:
            delta_standards[standard["name"]] = {
                "name": standard["name"],
                "fpath": standard["fpath"],
                "variable": standard["variable"],
            }
    #
    # Dynamically call plotting functions based on config
    plots = []
    for plot_config in config["plots"]:
        logger.info(f"Plotting: {plot_config}")
        plot_type = plot_config["type"]
        options = plot_config["options"]
        try:
            plot_func = getattr(pdmn, plot_type)
            if "regulation" in plot_type:
                if "delta_standard" in options:
                    standard = delta_standards[options["delta_standard"]]
                    df_standards = read_regulations(
                        standard["fpath"],
                        df_wyt,
                    )
                    if "EC" in standard["variable"]:
                        df_standards = milli_to_micro(df_standards)
                    df_standard_stations = get_common_stations(
                        df_stations, df_standards
                    )
                    df = pdmu.prep_df(
                        scenarios,
                        list(df_standard_stations["ID"].unique()),
                        [config["variable"]],
                        [config["interval"]],
                        df_wyt,
                        period,
                    )
                    df_regulations = generate_regulations(df_standards, df, freq="D")
                    df_regulations["variable"] = standard["variable"]
                    df_regulations["scenario_name"] = standard["name"]
                    if "MI" in options["delta_standard"]:
                        df_cl = rsl_ec_to_cl(df.copy())
                        if options["delta_standard"] == "D1641 MI 150":
                            df_cl = pdmu.calculate_days_standard_met(df_cl, 150)
                        args = (df_cl, df_regulations, df_standard_stations, options)
                    else:
                        args = (df, df_regulations, df_standard_stations, options)
            else:
                df = pdmu.prep_df(
                    scenarios,
                    list(df_stations["ID"].unique()),
                    [config["variable"]],
                    [config["interval"]],
                    df_wyt,
                    period,
                )
                args = (df, df_stations, options)
            plot = plot_func(*args)
            mainPanel.append((options["title"], plot))
        except Exception as e:
            import traceback

            # create str representation of error with stack trace
            message = f"{e}\n{traceback.format_exc()}"
            mainPanel.append((options["title"], pn.pane.Str(message)))
            continue
    thread.join()  # wait for the thread to finish
