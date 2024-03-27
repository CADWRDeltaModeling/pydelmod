# %%
# organize imports by category
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")
#
import pandas as pd
import geopandas as gpd

# viz and ui
import holoviews as hv
from holoviews import opts

hv.extension("bokeh")
import cartopy
import geoviews as gv

gv.extension("bokeh")
import param
import panel as pn

#
import pyhecdss as dss

from .dataui import DataUI, DataUIManager
from .dataui import full_stack


class DSSDataManager(param.Parameterized):
    def __init__(self, *dssfiles, **kwargs):
        self.time_range = kwargs.pop("time_range", None)
        self.geo_locations = kwargs.pop("geo_locations", None)
        self.geo_id_column = kwargs.pop("geo_id_column", "station_id")
        self.station_id_column = kwargs.pop(
            "station_id_column", "B"
        )  # The column in the data catalog that contains the station id
        super().__init__(**kwargs)
        self.dssfiles = dssfiles
        dfcats = []
        dssfh = {}
        dsscats = {}
        for dssfile in dssfiles:
            dssfh[dssfile] = dss.DSSFile(dssfile)
            dfcat = dssfh[dssfile].read_catalog()
            dsscats[dssfile] = self._build_map_pathname_to_catalog(dfcat)
            dfcat = dfcat.drop(columns=["T"])
            dfcat["filename"] = dssfile
            dfcats.append(dfcat)
        self.dssfh = dssfh
        self.dsscats = dsscats
        self.dfcat = pd.concat(dfcats)
        self.dfcat = self.dfcat.drop_duplicates().reset_index(drop=True)
        # add in the geo locations
        if self.geo_locations is not None:
            # DSS names are always in upper case
            self.geo_locations[self.geo_id_column] = (
                self.geo_locations[self.geo_id_column].astype(str).str.upper()
            )
            self.dfcat = pd.merge(
                self.geo_locations,
                self.dfcat,
                left_on=self.geo_id_column,
                right_on=self.station_id_column,
            )
        self.dssfiles = dssfiles
        self.dfcatpath = self._build_map_pathname_to_catalog(self.dfcat)

    def __del__(self):
        if hasattr(self, "dssfiles"):
            for dssfile in self.dssfiles:
                self.dssfh[dssfile].close()

    def build_pathname(self, r):
        return f'/{r["A"]}/{r["B"]}/{r["C"]}//{r["E"]}/{r["F"]}/'

    def _build_map_pathname_to_catalog(self, dfcat):
        dfcatpath = dfcat.copy()
        dfcatpath["pathname"] = dfcatpath.apply(self.build_pathname, axis=1)
        return dfcatpath

    def _slice_df(self, df, time_range):
        sdf = df.loc[slice(*time_range), :]
        if sdf.empty:
            return pd.DataFrame(
                columns=["value"],
                index=pd.date_range(*time_range, freq="D"),
                dtype=float,
            )
        else:
            return sdf

    def get_data_catalog(self):
        return self.dfcat

    def get_time_range(self, dfcat):
        """
        Calculate time range from the data catalog
        """
        if self.time_range is None:  # guess from catalog of DSS files
            dftw = dfcat.D.str.split("-", expand=True)
            dftw.columns = ["Tmin", "Tmax"]
            dftw["Tmin"] = pd.to_datetime(dftw["Tmin"])
            dftw["Tmax"] = pd.to_datetime(dftw["Tmax"])
            tmin = dftw["Tmin"].min()
            tmax = dftw["Tmax"].max()
            self.time_range = (tmin, tmax)
        return self.time_range

    def get_station_ids(self, df):
        return list((df.apply(self.build_pathname, axis=1).astype(str).unique()))

    def get_data_for_time_range(self, dssfile, r, irreg, time_range):
        try:
            dssfh = self.dssfh[dssfile]
            dfcatp = self.dsscats[dssfile]
            dfcatp = dfcatp[dfcatp["pathname"] == self.build_pathname(r)]
            pathname = dssfh.get_pathnames(dfcatp)[0]
            if irreg:
                df, unit, ptype = dssfh.read_its(
                    pathname,
                    time_range[0].strftime("%Y-%m-%d"),
                    time_range[1].strftime("%Y-%m-%d"),
                )
            else:
                df, unit, ptype = dssfh.read_rts(
                    pathname,
                    time_range[0].strftime("%Y-%m-%d"),
                    time_range[1].strftime("%Y-%m-%d"),
                )
        except Exception as e:
            print(full_stack())
            if pn.state.notifications:
                pn.state.notifications.error(
                    f"Error while fetching data for {dssfile}/{pathname}: {e}"
                )
            df = pd.DataFrame(columns=["value"], dtype=float)
            unit = "X"
            ptype = "INST-VAL"
        df = df[slice(df.first_valid_index(), df.last_valid_index())]
        return df, unit, ptype


class DSSDataUIManager(DataUIManager):
    def __init__(self, *dssfiles, **kwargs):
        """
        geolocations is a geodataframe with station_id, and geometry columns
        This is merged with the data catalog to get the station locations.
        """
        self.data_manager = DSSDataManager(*dssfiles, **kwargs)

    # data related methods
    def get_data_catalog(self):
        return self.data_manager.get_data_catalog()

    def get_station_ids(self, df):
        return self.data_manager.get_station_ids(df)

    def get_time_range(self, dfcat):
        return self.data_manager.get_time_range(dfcat)

    def get_table_column_width_map(self):
        """only columns to be displayed in the table should be included in the map"""
        column_width_map = {
            "A": "15%",
            "B": "15%",
            "C": "15%",
            "E": "10%",
            "F": "15%",
            "D": "20%",
            "filename": "10%",
        }
        return column_width_map

    def get_table_filters(self):
        table_filters = {
            "A": {"type": "input", "func": "like", "placeholder": "Enter match"},
            "B": {"type": "input", "func": "like", "placeholder": "Enter match"},
            "C": {"type": "input", "func": "like", "placeholder": "Enter match"},
            "E": {"type": "input", "func": "like", "placeholder": "Enter match"},
            "F": {"type": "input", "func": "like", "placeholder": "Enter match"},
        }
        return table_filters

    def _append_to_title_map(self, title_map, unit, r):
        value = title_map[unit]
        if r["C"] not in value[0]:
            value[0] += f',{r["C"]}'
        if r["B"] not in value[1]:
            value[1] += f',{r["B"]}'
        if r["A"] not in value[2]:
            value[2] += f',{r["A"]}'
        if r["F"] not in value[3]:
            value[3] += f',{r["F"]}'
        title_map[unit] = value

    def _create_title(self, v):
        title = f"{v[1]} @ {v[2]} ({v[3]}::{v[0]})"
        return title

    def _create_crv(self, df, crvlabel, ylabel, title, irreg=False):
        if irreg:
            crv = hv.Scatter(df.iloc[:, [0]], label=crvlabel).redim(value=crvlabel)
        else:
            crv = hv.Curve(df.iloc[:, [0]], label=crvlabel).redim(value=crvlabel)
        return crv.opts(
            xlabel="Time",
            ylabel=ylabel,
            title=title,
            responsive=True,
            active_tools=["wheel_zoom"],
            tools=["hover"],
        )

    def create_layout(self, df, time_range):
        layout_map = {}
        title_map = {}
        range_map = {}
        station_map = {}  # list of stations for each unit
        for _, r in df.iterrows():
            irreg = r["E"].startswith("IR-")
            data, unit, _ = self.data_manager.get_data_for_time_range(
                r["filename"], r, irreg, time_range
            )
            crv = self._create_crv(
                data,
                f'{r["B"]}/{r["C"]}',
                f'{r["C"]} ({unit})',
                f'{r["C"]} @ {r["B"]} ({r["A"]}/{r["F"]})',
                irreg=irreg,
            )
            if unit not in layout_map:
                layout_map[unit] = []
                title_map[unit] = [
                    r["C"],
                    r["B"],
                    r["A"],
                    r["F"],
                ]
                range_map[unit] = None
                station_map[unit] = []
            layout_map[unit].append(crv)
            station_map[unit].append(self.data_manager.build_pathname(r))
            self._append_to_title_map(title_map, unit, r)
        title_map = {k: self._create_title(v) for k, v in title_map.items()}
        return layout_map, station_map, range_map, title_map

    # methods below if geolocation data is available
    def get_tooltips(self):
        return [
            ("station_id", "@station_id"),
            ("A", "@A"),
            ("B", "@B"),
            ("C", "@C"),
            ("E", "@E"),
            ("F", "@F"),
        ]

    def get_map_color_category(self):
        return "C"


import glob
import click


@click.command()
@click.argument("dssfiles", nargs=-1)
@click.option(
    "--location-file",
    default=None,
    help="Location file as geojson containing station locations as lat and lon columns",
)
@click.option(
    "--location-id-column",
    default="station_id",
    help="Station ID column in location file",
)
@click.option(
    "--station-id-column",
    default="B",
    help="Station ID column in data catalog, e.g. B part for DSS file pathname",
)
def show_dss_ui(
    dssfiles, location_file=None, location_id_column="station_id", station_id_column="B"
):
    """
    Show DSS UI for the given DSS files

    dssfiles : list of DSS files
    location_file : Location file as geojson containing station locations as lat and lon columns
    location_id_column : Station ID column in location file
    station_id_column : Station ID column in data catalog, e.g. B part for DSS file pathname
    """
    geodf = None
    # TODO: Add support for other location file formats and move to a utility module
    if location_file is not None:
        if location_file.endswith(".shp") or location_file.endswith(".geojson"):
            geodf = gpd.read_file(location_file)
        elif location_file.endswith(".csv"):
            df = pd.read_csv(location_file)
            if all(column in df.columns for column in ["lat", "lon"]):
                geodf = gpd.GeoDataFrame(
                    df, geometry=gpd.points_from_xy(df.lon, df.lat, crs="EPSG:4326")
                )
            elif all(
                column in df.columns for column in ["utm_easting", "utm_northing"]
            ) or all(column in df.columns for column in ["utm_x", "utm_y"]):
                geodf = gpd.GeoDataFrame(
                    df,
                    geometry=gpd.points_from_xy(df.utm_easting, df.utm_northing),
                    crs="EPSG:26910",
                )
            else:
                raise ValueError(
                    "Location file should be a geojson file or should have lat and lon or utm_easting and utm_northing columns"
                )
        if not (location_id_column in geodf.columns):
            raise ValueError(
                f"Station ID column {location_id_column} not found in location file"
            )

    dssuimgr = DSSDataUIManager(
        *dssfiles,
        geo_locations=geodf,
        geo_id_column=location_id_column,
        station_id_column=station_id_column,
    )
    ui = DataUI(dssuimgr)
    ui.create_view().show()
