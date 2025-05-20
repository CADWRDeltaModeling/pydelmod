import pandas as pd
import cartopy.crs as ccrs
import holoviews as hv

hv.extension("bokeh")
from .dvue.dataui import DataUI
from .dvue.tsdataui import TimeSeriesDataUIManager
from pydelmod import schismstudy, datastore
import pathlib
import param
import panel as pn


class SchismOutputUIDataManager(TimeSeriesDataUIManager):

    convert_units = param.Boolean(default=True, doc="Convert units to SI")

    def __init__(self, *studies, datastore=None, time_range=None, **kwargs):
        """
        geolocations is a geodataframe with id, and geometry columns
        This is merged with the data catalog to get the station locations.
        """
        self.studies = studies
        self.study_dir_map = {str(s.output_dir): s for s in self.studies}
        self.datastore = datastore
        self.catalog = self._merge_catalogs(self.studies, self.datastore)
        self.catalog["filename"] = self.catalog["filename"].astype(str)
        self.catalog.reset_index(drop=True, inplace=True)
        self.time_range = time_range
        reftimes = [s.reftime for s in studies]
        stime = min(reftimes)
        etime = max(reftimes)
        if self.time_range is None:
            self.time_range = (
                pd.Timestamp(stime),
                pd.Timestamp(etime + pd.Timedelta(days=250)),
            )
        super().__init__(filename_column="filename", **kwargs)
        self.color_cycle_column = "id"
        self.dashed_line_cycle_column = "source"
        self.marker_cycle_column = "variable"


    def get_widgets(self):
        control_widgets = super().get_widgets()
        control_widgets.append(pn.Param(self.param.convert_units))
        return control_widgets

    def _merge_catalogs(self, studies, datastore):
        """
        Merge the schism study and the datastore catalogs
        """
        dfs = [s.get_catalog() for s in studies]
        df = pd.concat(dfs)
        if datastore is not None:
            dfobs = self._convert_to_study_format(datastore.get_catalog())
            dfobs["source"] = "datastore"
            dfcat = pd.concat([df, dfobs])
        return dfcat

    def _convert_to_study_format(self, df):
        df = df.copy()
        df["subloc"] = df["subloc"].apply(lambda v: "default" if len(v) == 0 else v)
        df["id"] = df["station_id"].astype(str) + "_" + df["subloc"]
        df = df.rename(columns={"param": "variable"})
        gdf = schismstudy.convert_station_to_gdf(df)
        return gdf[["id", "name", "variable", "unit", "filename", "geometry"]]

    def get_data_catalog(self):
        return self.catalog

    def get_time_range(self, dfcat):
        return self.time_range

    def build_station_name(self, r):
        name = r["id"] + ":" + r["variable"]
        if "source" not in r:
            return f"{name}"
        else:
            return f'{r["source"]}:{name}'

    def _get_table_column_width_map(self):
        """only columns to be displayed in the table should be included in the map"""
        column_width_map = {
            "id": "10%",
            "name": "15%",
            "variable": "15%",
            "unit": "15%",
            "source": "10%",
        }
        return column_width_map

    def get_table_filters(self):
        table_filters = {
            "id": {"type": "input", "func": "like", "placeholder": "Enter match"},
            "name": {"type": "input", "func": "like", "placeholder": "Enter match"},
            "variable": {"type": "input", "func": "like", "placeholder": "Enter match"},
            "unit": {"type": "input", "func": "like", "placeholder": "Enter match"},
            "source": {"type": "input", "func": "like", "placeholder": "Enter match"},
        }
        return table_filters

    def _append_value(self, new_value, value):
        if new_value not in value:
            value += f'{", " if value else ""}{new_value}'
        return value

    def append_to_title_map(self, title_map, unit, r):
        if unit in title_map:
            value = title_map[unit]
        else:
            value = ["", "", ""]
        value[0] = self._append_value(r["source"], value[0])
        value[2] = self._append_value(r["id"], value[0])
        value[1] = self._append_value(r["variable"], value[1])
        title_map[unit] = value

    def create_title(self, v):
        title = f"{v[1]} @ {v[2]} ({v[0]})"
        return title

    def create_curve(self, df, r, unit, file_index=None):
        crvlabel = f'{r["source"]}::{r["id"]}/{r["variable"]}'
        ylabel = f'{r["variable"]} ({unit})'
        title = f'{r["variable"]} @ {r["id"]}'
        crv = hv.Curve(df.iloc[:, [0]], label=crvlabel).redim(value=crvlabel)
        return crv.opts(
            xlabel="Time",
            ylabel=ylabel,
            title=title,
            responsive=True,
            active_tools=["wheel_zoom"],
            tools=["hover"],
        )

    def is_irregular(self, r):
        return False

    def get_data_for_time_range(self, r, time_range):
        unit = r["unit"]
        if r["source"] == "datastore":
            df = self.datastore.get_data(r)
            if self.convert_units:
                df, unit = schismstudy.convert_to_SI(df, r["unit"])
        else:
            base_dir = str(pathlib.Path(r["filename"]).parent)
            study = self.study_dir_map[base_dir]
            df = study.get_data(r)
        ptype = "INST-VAL"
        df = df[slice(df.first_valid_index(), df.last_valid_index())]
        return df, unit, ptype

    # methods below if geolocation data is available
    def get_tooltips(self):
        return [
            ("id", "@id"),
            ("name", "@name"),
            ("variable", "@variable"),
            ("unit", "@unit"),
            ("source", "@source"),
        ]

    def get_map_color_category(self):
        return "variable"

    def get_map_color_columns(self):
        """return the columns that can be used to color the map"""
        return ["variable", "source", "unit"]

    def get_map_marker_columns(self):
        """return the columns that can be used to color the map"""
        return ["variable", "source", "unit"]


import click
import yaml


@click.command()
@click.option("--schism_dir", default=".", help="Path to the schism study directory")
@click.option(
    "--repo_dir", default="screened", help="Path to the screened data directory"
)
@click.option(
    "--inventory_file",
    default="inventory_datasets.csv",
    help="Path to the inventory file",
)
@click.option(
    "--flux_xsect_file",
    default="flow_station_xsects.yaml",
    help="Path to the flux cross section file",
)
@click.option(
    "--station_in_file", default="station.in", help="Path to the station.in file"
)
@click.option("--flux_out", default="flux.out", help="Path to the flux.out file")
@click.option("--reftime", default=None, help="Reference time")
@click.option("--yaml_file", default=None, help="Path to the yaml file")
def show_schism_output_ui(
    schism_dir=".",
    flux_xsect_file="flow_station_xsects.yaml",
    station_in_file="station.in",
    flux_out="flux.out",
    reftime=None,
    repo_dir="screened",
    inventory_file="inventory_datasets.csv",
    yaml_file=None,
):
    """
    Shows Data UI for SCHISM output files.

    This function creates a Data UI for SCHISM output files, allowing users to visualize and analyze the data.
    It can handle multiple studies and datasets, and provides options for customizing the display.
    The function can be run from the command line or imported as a module.

    If a YAML file is provided, it will be used to create multiple studies.
    Otherwise, a single study will be created using the provided parameters.

    Example YAML file::

    .. code-block:: yaml
        \b
        schism_studies:
            - label: Study1
            base_dir: "study1_directory"
            flux_xsect_file: "study1_flow_station_xsects.yaml"
            station_in_file: "study1_station.in"
            output_dir: "outputs"
            param_nml_file: "param.nml"
            flux_out: "study1_flux.out"
            reftime: "2020-01-01"
            - label: Study2
            base_dir: "study2_directory"
        datastore:
            repo_dir: /repo/continuous/screened
            inventory_file: "inventory_datasets.csv"
    """
    if yaml_file:
        # Load the YAML file and create multiple studies
        with open(yaml_file, "r") as file:
            yaml_data = yaml.safe_load(file)

        studies = []
        for study_config in yaml_data.get("schism_studies", []):
            studies.append(
                schismstudy.SchismStudy(
                    study_name=study_config["label"],
                    base_dir=study_config["base_dir"],
                    output_dir=study_config.get("output_dir", "outputs"),
                    param_nml_file=study_config.get("param_nml_file", "param.nml"),
                    flux_xsect_file=study_config.get(
                        "flux_xsect_file", "flow_station_xsects.yaml"
                    ),
                    station_in_file=study_config.get("station_in_file", "station.in"),
                    flux_out=study_config.get("flux_out", "flux.out"),
                    reftime=reftime,
                    **study_config.get("additional_parameters", {}),
                )
            )

        datastore_config = yaml_data.get("datastore", {})
        repo_dir = datastore_config.get("repo_dir", repo_dir)
        inventory_file = datastore_config.get("inventory_file", inventory_file)
    else:
        # Create a single study if no YAML file is provided
        studies = [
            schismstudy.SchismStudy(
                schism_dir,
                flux_xsect_file=flux_xsect_file,
                station_in_file=station_in_file,
                flux_out=flux_out,
                reftime=reftime,
            )
        ]

    # study.reftime to study.endtime is the range of a single study
    # Initialize the union range
    union_start = min(study.reftime for study in studies)
    union_end = max(study.endtime for study in studies)

    # Create the union range as a single variable
    time_range = (union_start, union_end)

    # Create the datastore
    ds = datastore.StationDatastore(repo_dir=repo_dir, inventory_file=inventory_file)

    # Create the UI
    ui = DataUI(
        SchismOutputUIDataManager(
            *studies,
            datastore=ds,
            time_range=time_range,
        ),
        crs=ccrs.UTM(10),
    )
    ui.create_view(title="Schism Output UI").show()
