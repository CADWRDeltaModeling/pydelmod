import pathlib
import yaml
import pandas as pd
import geopandas as gpd
import hvplot.pandas
import panel as pn

pn.extension()
import holoviews as hv
from holoviews import opts

import pyhecdss as dss
from pydsm import postpro
from pydelmod import postpro_dsm2

from .dataui import DataUI, DataUIManager


# substitue the base_dir in location_files_dict, observed_files_dict, study_files_dict
def substitute_base_dir(base_dir, dict):
    for key in dict:
        dict[key] = str((pathlib.Path(base_dir) / dict[key]).resolve())
    return dict


def load_location_file(location_file):
    df = postpro.load_location_file(location_file)
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude), crs="EPSG:4326"
    )
    return gdf


class CalibPlotUIManager(DataUIManager):

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
        # substitue the base_dir in location_files_dict, observed_files_dict, study_files_dict
        if base_dir is None:
            base_dir = pathlib.Path(self.config_file).parent
        config["location_files_dict"] = substitute_base_dir(
            base_dir, config["location_files_dict"]
        )
        config["observed_files_dict"] = substitute_base_dir(
            base_dir, config["observed_files_dict"]
        )
        config["study_files_dict"] = substitute_base_dir(
            base_dir, config["study_files_dict"]
        )
        self.config = config

    def get_studies(self, varname):
        studies = list(self.config["study_files_dict"].keys())
        obs_study = postpro.Study(
            "Observed", self.config["observed_files_dict"][varname]
        )
        model_studies = [
            postpro.Study(name, self.config["study_files_dict"][name])
            for name in self.config["study_files_dict"]
        ]
        studies = [obs_study] + model_studies
        return studies

    def build_location(self, row):
        return postpro.Location(
            row["Name"],
            row["BPart"],
            row["Description"],
            row["time_window_exclusion_list"],
            row["threshold_value"],
        )

    def get_locations(self, df):
        locations = [self.build_location(r) for i, r in df.iterrows()]
        return locations

    def get_widgets(self):
        return pn.Column(pn.pane.Markdown("UI Controls Placeholder"))

    # data related methods
    def get_data_catalog(self):
        gdfs = []
        for key, value in self.config["location_files_dict"].items():
            gdf = postpro.load_location_file(value)
            gdf.Latitude = pd.to_numeric(gdf.Latitude, errors="coerce")
            gdf.Longitude = pd.to_numeric(gdf.Longitude, errors="coerce")
            gdf.threshold_value = pd.to_numeric(gdf.threshold_value, errors="coerce")
            gdf = gpd.GeoDataFrame(
                gdf,
                geometry=gpd.points_from_xy(gdf.Longitude, gdf.Latitude),
                crs="EPSG:4326",
            )
            gdf["vartype"] = str(key)
            gdfs.append(gdf)
        gdf = pd.concat(gdfs, axis=0)
        gdf = gdf.reset_index(drop=True)
        gdf = gdf.astype(
            {
                "Name": "str",
                "BPart": "str",
                "Description": "str",
                "subtract": "str",
                "time_window_exclusion_list": "str",
                "vartype": "str",
            },
            errors="raise",
        )
        gdf = gdf.dropna(subset=["Latitude", "Longitude"])
        return gdf

    def get_table_column_width_map(self):
        """only columns to be displayed in the table should be included in the map"""
        column_width_map = {
            "Name": "20%",
            "BPart": "10%",
            "vartype": "5%",
            "Description": "30%",
            "subtract": "5%",
            "time_window_exclusion_list": "10%",
            "threshold_value": "5%",
        }
        return column_width_map

    def get_table_filters(self):
        table_filters = {
            "Name": {"type": "input", "func": "like", "placeholder": "Enter match"},
            "BPart": {"type": "input", "func": "like", "placeholder": "Enter match"},
            "vartype": {"type": "input", "func": "like", "placeholder": "Enter match"},
            "Description": {
                "type": "input",
                "func": "like",
                "placeholder": "Enter match",
            },
        }
        return table_filters

    def create_panel(self, df):
        plots = []
        for _, row in df.iterrows():
            varname = row["vartype"]
            vartype = postpro.VarType(varname, self.config["vartype_dict"][varname])
            studies = self.get_studies(varname)
            location = self.build_location(row)
            calib_plot_template_dict, metrics_df = postpro_dsm2.build_plot(
                self.config, studies, location, vartype
            )
            plots.append(
                (
                    location.name + "@" + varname,
                    pn.Row(calib_plot_template_dict["with"]),
                )
            )
        return pn.Tabs(*plots, dynamic=True, closable=True)

    # methods below if geolocation data is available
    def get_tooltips(self):
        return [
            ("Name", "@Name"),
            ("BPart", "@BPart"),
            ("Description", "@Description"),
            ("vartype", "@vartype"),
        ]

    def get_map_color_category(self):
        return "vartype"


import click


@click.command()
@click.argument("config_file", type=click.Path(exists=True, readable=True))
@click.option("--base_dir", required=False, help="Base directory for config file")
def calib_plot_ui(config_file, base_dir=None, **kwargs):
    """
    config_file: str
        yaml file containing configuration

    base_dir: str
        base directory for config file, if None is assumed to be same as config file directory
    """
    manager = CalibPlotUIManager(config_file, base_dir=base_dir, **kwargs)
    DataUI(manager).create_view().show()
