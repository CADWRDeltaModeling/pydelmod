import pathlib
import yaml
import pandas as pd
import geopandas as gpd
import hvplot.pandas
import panel as pn

pn.extension()
import holoviews as hv
from holoviews import opts

from schimpy import batch_metrics
from .dataui import DataUI, DataUIManager
from . import datastore, schismstudy
from vtools.functions.unit_conversions import (
    cfs_to_cms,
    ft_to_m,
    ec_psu_25c,
    fahrenheit_to_celsius,
)
from vtools.functions.filter import cosine_lanczos
from . import calibplot

# from .calibplot import tsplot, scatterplot, calculate_metrics, regression_line_plots

VAR_to_PARAM = {
    "flow": "flow",
    "elev": "elev",
    "salt": "ec",
    "temp": "temp",
    "ssc": "ssc",
}
variable_units = {
    "flow": "cms",
    "elev": "m",
    "salt": "PSU",
    "temp": "deg C",
    "ssc": "mg/L",
}


def interpret_file_relative_to(base_dir, fpath):
    full_path = base_dir / fpath
    if full_path.exists():
        return str(full_path)
    else:
        return str(fpath)


def replace_with_paths_relative_to(base_dir, params):
    params = params.copy()
    params["output_dir"] = [
        interpret_file_relative_to(base_dir, file) for file in params["outputs_dir"]
    ]
    params["stations_csv"] = interpret_file_relative_to(
        base_dir, params["stations_csv"]
    )
    params["obs_search_path"] = [
        interpret_file_relative_to(base_dir, file) for file in params["obs_search_path"]
    ]
    params["station_input"] = [
        interpret_file_relative_to(base_dir, file) for file in params["station_input"]
    ]
    params["obs_links_csv"] = interpret_file_relative_to(
        base_dir, params["obs_links_csv"]
    )
    params["flow_station_input"] = [
        interpret_file_relative_to(base_dir, file)
        for file in params["flow_station_input"]
    ]
    return params


def to_timewindow_string(window):
    return ":".join([pd.to_datetime(x).strftime("%Y-%m-%d") for x in window])


class SchismCalibPlotUIManager(DataUIManager):

    def __init__(self, config_file, base_dir=None, **kwargs):
        """
        config_file: str
            yaml file containing configuration

        base_dir: str
            base directory for config file, if None is assumed to be same as config file directory
        """
        base_dir = kwargs.pop("base_dir", None)
        super().__init__(**kwargs)
        self.config_file = config_file
        with open(self.config_file, "r") as file:
            config = yaml.safe_load(file)
        # substitue the base_dir in config paths
        if base_dir is None:
            base_dir = pathlib.Path(self.config_file).parent
        config = replace_with_paths_relative_to(base_dir, config)
        self.config = config
        # load studies and datastore
        self.reftime = pd.Timestamp(self.config["time_basis"])
        self.window_inst = (
            pd.Timestamp(self.config["start_inst"]),
            pd.Timestamp(self.config["end_inst"]),
        )
        self.window_avg = (
            pd.Timestamp(self.config["start_avg"]),
            pd.Timestamp(self.config["end_avg"]),
        )
        self.labels = self.config["labels"]
        self.studies = {}
        for schism_dir, flux_xsect_file, station_inf_file, label in zip(
            self.config["output_dir"],
            self.config["flow_station_input"],
            self.config["station_input"],
            self.labels[1:],
        ):
            study = schismstudy.SchismStudy(
                base_dir=schism_dir,
                flux_xsect_file=flux_xsect_file,
                flux_out="flux.out",
                station_in_file=station_inf_file,
                reftime=self.reftime,
            )
            self.studies[label] = study
        self.datastore = datastore.StationDatastore(
            repo_dir=self.config["obs_search_path"][0],
            inventory_file=self.config["obs_links_csv"],
        )
        self.dcat = self.datastore.get_catalog()
        self.dcat["id"] = (
            self.dcat["station_id"].astype(str) + "_" + self.dcat["subloc"].astype(str)
        )

    def get_widgets(self):
        return pn.Column(pn.pane.Markdown("UI Controls Placeholder"))

    # data related methods
    def get_data_catalog(self):
        scat = pd.concat([s.get_catalog() for s in self.studies.values()], axis=0)
        scat = scat[["id", "name", "variable", "unit", "geometry"]]
        scat.drop_duplicates(subset=["id", "variable"], inplace=True)
        scat = scat.reset_index(drop=True)
        scat = scat.astype(
            {
                "id": "str",
                "name": "str",
                "variable": "str",
                "unit": "str",
            },
            errors="raise",
        )
        scat = scat.dropna()
        scat = scat[
            scat["variable"].isin(["flow", "elev", "salt", "temp", "ssc"])
        ].reset_index(drop=True)
        return scat

    def get_table_column_width_map(self):
        """only columns to be displayed in the table should be included in the map"""
        column_width_map = {
            "id": "10%",
            "name": "65%",
            "variable": "15%",
            "unit": "10%",
        }
        return column_width_map

    def get_table_filters(self):
        table_filters = {
            "id": {"type": "input", "func": "like", "placeholder": "Enter match"},
            "name": {"type": "input", "func": "like", "placeholder": "Enter match"},
            "variable": {"type": "input", "func": "like", "placeholder": "Enter match"},
            "unit": {
                "type": "input",
                "func": "like",
                "placeholder": "Enter match",
            },
        }
        return table_filters

    def get_datastore_param_name(self, variable):
        return VAR_to_PARAM[variable]

    def convert_to_SI(self, ts, unit):
        """converts the time series to SI units"""
        if ts is None:
            raise ValueError("Cannot convert None")
        if unit in ["ft", "feet"]:
            ts = ft_to_m(ts)
        elif unit in ["cfs", "ft^3/s"]:
            ts = cfs_to_cms(ts)
        elif unit in ["ec", "microS/cm", "uS/cm"]:
            ts = ec_psu_25c(ts)
        elif unit in ("deg F", "degF", "deg_f"):
            ts = fahrenheit_to_celsius(ts)
        return ts

    def get_data(self, id, variable):
        dfs = []
        for study_name, study in self.studies.items():
            scat = study.get_catalog()
            rs = scat[scat.eval(f'(id=="{id}") & (variable=="{variable}")')]
            if not rs.empty:
                df = study.get_data(rs.iloc[0])
                df.columns = [study_name]
                dfs.append(df)
        dparam = self.get_datastore_param_name(variable)
        rd = self.dcat[self.dcat.eval(f'(id=="{id}") & (param=="{dparam}")')].iloc[0]
        if not rd.empty:
            dfobs = self.convert_to_SI(self.datastore.get_data(rd), rd["unit"])
        return dfobs, dfs

    def plot_metrics(self, row):
        station_id = row["id"]
        variable = row["variable"]

        dfobs, dfsimlist = self.get_data(station_id, variable)
        window_inst = to_timewindow_string(self.window_inst)
        inst_plot = calibplot.tsplot(
            [dfobs] + dfsimlist, self.labels, window_inst, True
        )
        # tidal averaging
        dfobsf = cosine_lanczos(dfobs, "40H")
        dfsimlistf = [cosine_lanczos(df, "40H") for df in dfsimlist]
        window_avg = to_timewindow_string(self.window_avg)
        # Now do the time slicing
        dfobsf = dfobsf.loc[slice(*window_avg.split(":")), :]
        dfsimlistf = [df.loc[slice(*window_avg.split(":")), :] for df in dfsimlistf]
        dfobs = dfobs.loc[slice(*window_inst.split(":")), :]
        dfsimlist = [df.loc[slice(*window_inst.split(":")), :] for df in dfsimlist]
        # Now do the plotting
        plotf = calibplot.tsplot([dfobsf] + dfsimlistf, self.labels, window_avg, True)
        dfmetrics = calibplot.calculate_metrics([dfobsf] + dfsimlistf, self.labels)
        splot = calibplot.scatterplot([dfobsf] + dfsimlistf, self.labels)
        # layout template
        grid = pn.GridSpec(sizing_mode="stretch_both", min_height=600)
        title = f"{variable}@{row['name']}"
        grid[0, 2:4] = pn.pane.HTML(f"<h2>{title}</h2>")
        grid[1:4, :] = inst_plot.opts(shared_axes=False, legend_position="right")
        grid[4:7, 0:5] = pn.Row(
            plotf.opts(show_legend=False), splot.opts(shared_axes=False)
        )
        grid[7:9, :] = pn.pane.DataFrame(dfmetrics)
        # save
        grid.save(f"{station_id}_{variable}_plot.html")

        return grid

    def create_panel(self, df):
        plots = []
        for _, row in df.iterrows():
            station_id = row["id"]
            varname = row["variable"]
            plot = self.plot_metrics(row)
            plots.append(
                (
                    varname + "@" + station_id,
                    pn.Row(plot),
                )
            )
        return pn.Tabs(*plots, dynamic=True, closable=True)

    # methods below if geolocation data is available
    def get_tooltips(self):
        return [
            ("id", "@id"),
            ("name", "@name"),
            ("variable", "@variable"),
            ("unit", "@unit"),
        ]

    def get_map_color_category(self):
        return "variable"


import click


@click.command()
@click.argument("config_file", type=click.Path(exists=True, readable=True))
@click.option("--base_dir", required=False, help="Base directory for config file")
def schism_calib_plot_ui(config_file, base_dir=None, **kwargs):
    """
    config_file: str
        yaml file containing configuration

    base_dir: str
        base directory for config file, if None is assumed to be same as config file directory
    """
    import cartopy.crs as ccrs

    manager = SchismCalibPlotUIManager(config_file, base_dir=base_dir, **kwargs)
    DataUI(manager, crs=ccrs.UTM(10)).create_view().show()
