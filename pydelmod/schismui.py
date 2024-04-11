import pandas as pd
import cartopy.crs as ccrs
import holoviews as hv

hv.extension("bokeh")
from .dataui import DataUI
from .tsdataui import TimeSeriesDataUIManager
from pydelmod import schismstudy, datastore
import pathlib


class SchismOutputUIDataManager(TimeSeriesDataUIManager):

    def __init__(self, *studies, datastore=None, time_range=None, **kwargs):
        """
        geolocations is a geodataframe with id, and geometry columns
        This is merged with the data catalog to get the station locations.
        """
        self.studies = studies
        self.study_dir_map = {str(s.base_dir): s for s in self.studies}
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

    def _merge_catalogs(self, studies, datastore):
        """
        Merge the schism study and the datastore catalogs
        """
        dfs = [s.get_catalog() for s in studies]
        df = pd.concat(dfs)
        df["source"] = "schism"
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
        if "FILE_NUM" not in r:
            return f"{name}"
        else:
            return f'{r["FILE_NUM"]}:{name}'

    # data related methods
    def _get_station_ids(self, df):
        return list((df.apply(self.build_station_name, axis=1).astype(str).unique()))

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

    def _append_to_title_map(self, title_map, unit, r):
        if unit in title_map:
            value = title_map[unit]
        else:
            value = ["", "", ""]
        value[0] = self._append_value(r["source"], value[0])
        value[2] = self._append_value(r["id"], value[0])
        value[1] = self._append_value(r["variable"], value[1])
        title_map[unit] = value

    def _create_title(self, v):
        title = f"{v[1]} @ {v[2]} ({v[0]})"
        return title

    def _create_crv(self, df, r, unit, file_index=None):
        file_index_label = f"{file_index}:" if file_index is not None else ""
        crvlabel = f'{file_index_label}{r["id"]}/{r["variable"]}'
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

    def _get_data_for_time_range(self, r, time_range):
        unit = r["unit"]
        if r["source"] == "schism":
            base_dir = str(pathlib.Path(r["filename"]).parent)
            study = self.study_dir_map[base_dir]
            df = study.get_data(r)
        elif r["source"] == "datastore":
            df = self.datastore.get_data(r)
        else:
            df = pd.DataFrame(columns=["value"], dtype=float)  # empty dataframe
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


import click


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
@click.option("--reftime", default="2020-01-01", help="Reference time")
def show_schism_output_ui(
    schism_dir=".",
    flux_xsect_file="flow_station_xsects.yaml",
    station_in_file="station.in",
    flux_out="flux.out",
    reftime="2020-01-01",
    repo_dir="screened",
    inventory_file="inventory_datasets.csv",
):
    study = schismstudy.SchismStudy(
        schism_dir,
        flux_xsect_file=flux_xsect_file,
        station_in_file=station_in_file,
        flux_out=flux_out,
        reftime=reftime,
    )
    ds = datastore.StationDatastore(repo_dir=repo_dir, inventory_file=inventory_file)
    time_range = (pd.Timestamp(reftime), pd.Timestamp(reftime) + pd.Timedelta(days=250))
    ui = DataUI(
        SchismOutputUIDataManager(
            study,
            datastore=ds,
            time_range=time_range,
        ),
        crs=ccrs.UTM(10),
    )
    ui.create_view().show()
