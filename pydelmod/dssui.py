# %%
# organize imports by category
from datetime import datetime, timedelta
from functools import lru_cache
import warnings

warnings.filterwarnings("ignore")
#
import pandas as pd
import geopandas as gpd
import holoviews as hv
import cartopy.crs as ccrs

hv.extension("bokeh")
# viz and ui
import param
import panel as pn

pn.extension()
#
import pyhecdss as dss
from vtools.functions.filter import cosine_lanczos

from .dvue.dataui import DataUI, full_stack
from .dvue.tsdataui import TimeSeriesDataUIManager


class DSSDataUIManager(TimeSeriesDataUIManager):

    def __init__(self, *dssfiles, **kwargs):
        """
        geolocations is a geodataframe with station_id, and geometry columns
        This is merged with the data catalog to get the station locations.
        """
        self.time_range = kwargs.pop("time_range", None)
        self.geo_locations = kwargs.pop("geo_locations", None)
        self.geo_id_column = kwargs.pop("geo_id_column", "station_id")
        self.station_id_column = kwargs.pop(
            "station_id_column", "B"
        )  # The column in the data catalog that contains the station id
        if len(dssfiles) == 0:
            raise ValueError("At least one DSS file is required")
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
        super().__init__(**kwargs)

    def __del__(self):
        if hasattr(self, "dssfiles"):
            for dssfile in self.dssfiles:
                self.dssfh[dssfile].close()

    def build_pathname(self, r):
        return f'/{r["A"]}/{r["B"]}/{r["C"]}//{r["E"]}/{r["F"]}/'

    def build_station_name(self, r):
        pathname = self.build_pathname(r)
        if "FILE_NUM" not in r:
            return f"{pathname}"
        else:
            return f'{r["FILE_NUM"]}:{pathname}'

    def _build_map_pathname_to_catalog(self, dfcat):
        dfcatpath = dfcat.copy()
        dfcatpath["pathname"] = dfcatpath.apply(self.build_pathname, axis=1)
        return dfcatpath

    # data related methods
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

    def _get_table_column_width_map(self):
        """only columns to be displayed in the table should be included in the map"""
        column_width_map = {
            "A": "15%",
            "B": "15%",
            "C": "15%",
            "E": "10%",
            "F": "15%",
            "D": "20%",
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

    def _append_value(self, new_value, value):
        if new_value not in value:
            value += f'{", " if value else ""}{new_value}'
        return value

    def append_to_title_map(self, title_map, unit, r):
        if unit in title_map:
            value = title_map[unit]
        else:
            value = ["", "", "", ""]
        value[0] = self._append_value(r["C"], value[0])
        value[1] = self._append_value(r["B"], value[1])
        value[2] = self._append_value(r["A"], value[2])
        value[3] = self._append_value(r["F"], value[3])
        title_map[unit] = value

    def create_title(self, v):
        title = f"{v[1]} @ {v[2]} ({v[3]}::{v[0]})"
        return title

    def is_irregular(self, r):
        return r["E"].startswith("IR-")

    def create_curve(self, df, r, unit, file_index=None):
        file_index_label = f"{file_index}:" if file_index is not None else ""
        crvlabel = f'{file_index_label}{r["B"]}/{r["C"]}'
        ylabel = f'{r["C"]} ({unit})'
        title = f'{r["C"]} @ {r["B"]} ({r["A"]}/{r["F"]})'
        crv = hv.Curve(df.iloc[:, [0]], label=crvlabel).redim(value=crvlabel)
        return crv.opts(
            xlabel="Time",
            ylabel=ylabel,
            title=title,
            responsive=True,
            active_tools=["wheel_zoom"],
            tools=["hover"],
        )

    def get_data_for_time_range(self, r, time_range):
        irreg = self.is_irregular(r)
        dssfile = r[self.filename_column]
        dssfh = self.dssfh[dssfile]
        dfcatp = self.dsscats[dssfile]
        dfcatp = dfcatp[dfcatp["pathname"] == self.build_pathname(r)]
        pathname = dssfh.get_pathnames(dfcatp)[0]
        if irreg:
            df, unit, ptype = lru_cache(maxsize=32)(dssfh.read_its)(
                pathname,
                time_range[0].strftime("%Y-%m-%d"),
                time_range[1].strftime("%Y-%m-%d"),
            )
        else:
            df, unit, ptype = lru_cache(maxsize=32)(dssfh.read_rts)(
                pathname,
                time_range[0].strftime("%Y-%m-%d"),
                time_range[1].strftime("%Y-%m-%d"),
            )
        df = df[slice(df.first_valid_index(), df.last_valid_index())]
        return df, unit, ptype

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

    def get_map_color_columns(self):
        """return the columns that can be used to color the map"""
        return ["C", "A", "F"]

    def get_map_marker_columns(self):
        """return the columns that can be used to color the map"""
        return ["C", "A", "F"]


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
    crs_cartopy = None
    # TODO: Add support for other location file formats and move to a utility module
    if location_file is not None:
        if location_file.endswith(".shp") or location_file.endswith(".geojson"):
            geodf = gpd.read_file(location_file)
            # Extract EPSG code
            epsg_code = geodf.crs.to_epsg()
            # Create Cartopy CRS from EPSG
            crs_cartopy = ccrs.epsg(epsg_code)
        elif location_file.endswith(".csv"):
            df = pd.read_csv(location_file)
            if all(column in df.columns for column in ["lat", "lon"]):
                geodf = gpd.GeoDataFrame(
                    df, geometry=gpd.points_from_xy(df.lon, df.lat, crs="EPSG:4326")
                )
                crs_cartopy = ccrs.PlateCarree()
            elif all(
                column in df.columns for column in ["utm_easting", "utm_northing"]
            ) or all(column in df.columns for column in ["utm_x", "utm_y"]):
                geodf = gpd.GeoDataFrame(
                    df,
                    geometry=gpd.points_from_xy(df.utm_easting, df.utm_northing),
                    crs="EPSG:26910",
                )
                crs_cartopy = ccrs.UTM(10)
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
        filename_column="filename",
    )
    ui = DataUI(dssuimgr, crs=crs_cartopy)
    ui.create_view().show()
