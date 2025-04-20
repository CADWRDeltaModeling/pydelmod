from functools import lru_cache
import os
import pathlib
import schimpy
import pandas as pd
import schimpy.station as station
from schimpy import param as schimpyparam
import schimpy.schism_yaml as schism_yaml
import schimpy.batch_metrics as schism_metrics
import cartopy.crs as ccrs
import yaml
from shapely.geometry import LineString
import geopandas as gpd
import param
import diskcache
import datetime

# use logging
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def read_station_in(station_in_file):
    return station.read_station_in(station_in_file)


def convert_station_to_gdf(stations):
    return gpd.GeoDataFrame(
        stations,
        geometry=gpd.points_from_xy(stations.x, stations.y),
        crs="EPSG:32610",
    )


def load_flux_dataframe(flux_names):
    # Load the YAML file
    with open(str(flux_names), "r") as file:
        yaml_data = yaml.safe_load(file)["linestrings"]

    # Process the data to create LineString objects
    for item in yaml_data:
        # Convert the list of coordinates into a LineString
        item["line_string"] = LineString(item["coordinates"])
        # Remove the original coordinates to avoid redundancy
        del item["coordinates"]

    return pd.DataFrame(yaml_data)


def convert_flux_to_gdf(flux_df):
    flux_df = flux_df.rename(columns={"line_string": "geometry"})
    flux_gdf = gpd.GeoDataFrame(flux_df, geometry="geometry")
    flux_gdf.crs = "EPSG:32610"  # (UTM Zone 10 N)
    return flux_gdf


def convert_flux_to_points_gdf(flux_df):
    flux_pts_df = convert_flux_to_gdf(flux_df)
    flux_pts_df.geometry = flux_pts_df.geometry.centroid
    return flux_pts_df


STATION_VARS = [
    "elev",
    "air pressure",
    "wind_x",
    "wind_y",
    "temp",
    "salt",
    "u",
    "v",
    "w",
    "ssc",
]

from vtools.functions.unit_conversions import (
    cfs_to_cms,
    ft_to_m,
    ec_psu_25c,
    fahrenheit_to_celsius,
)


def convert_to_SI(ts, unit):
    """converts the time series to SI units"""
    if ts is None:
        raise ValueError("Cannot convert None")
    if unit in ["ft", "feet"]:
        ts = ft_to_m(ts)
        unit = "meters"
    elif unit in ["cfs", "ft^3/s"]:
        ts = cfs_to_cms(ts)
        unit = "m^3/s"
    elif unit in ["ec", "microS/cm", "uS/cm"]:
        ts = ec_psu_25c(ts)
        unit = "PSU"
    elif unit in ("deg F", "degF", "deg_f"):
        ts = fahrenheit_to_celsius(ts)
        unit = "deg_c"
    return ts, unit


class SchismStudy(param.Parameterized):

    def __init__(
        self,
        base_dir=".",
        output_dir="outputs",
        param_nml_file="param.nml",
        flux_xsect_file="flow_station_xsects.yaml",
        station_in_file="station.in",
        flux_out="flux.out",
        reftime=None,
        **kwargs,
    ):
        self.base_dir = pathlib.Path(base_dir)
        self.param_nml_file = self.interpret_file_relative_to(
            self.base_dir, pathlib.Path(param_nml_file)
        )
        self.output_dir = self.interpret_file_relative_to(
            self.base_dir, pathlib.Path(output_dir)
        )
        try:
            self.cache = diskcache.Cache(self.base_dir / ".cache-schismstudy")
        except:
            logger.warning("Could not create cache. Using temporary cache.")
            self.cache = diskcache.Cache()
        if not reftime:
            nml = schimpyparam.read_params(self.param_nml_file)
            self.reftime = nml.run_start
            self.endtime = nml.run_start + datetime.timedelta(days=nml["rnday"])
        else:
            self.reftime = pd.Timestamp(reftime)
        self.flux_xsect_file = self.interpret_file_relative_to(
            self.base_dir, pathlib.Path(flux_xsect_file)
        )
        self.station_in_file = self.interpret_file_relative_to(
            self.base_dir, pathlib.Path(station_in_file)
        )
        self.flux_out = self.interpret_file_relative_to(
            self.output_dir, pathlib.Path(flux_out)
        )
        super().__init__(**kwargs)
        stations = read_station_in(self.station_in_file)
        self.stations_in = stations
        stations = stations.reset_index()
        # only add subloc if subloc value is not == "default"
        stations["station_id"] = stations["id"] + stations["subloc"].apply(
            lambda x: ("_" + x) if x.lower() != "default" else ""
        )
        self.stations_gdf = convert_station_to_gdf(stations)
        if self.flux_xsect_file.suffix in [".yaml", ".yml"]:
            flux_df = load_flux_dataframe(self.flux_xsect_file)
            flux_df["station_id"] = flux_df["name"].str.lower()  # + "_default"
            self.flux_gdf = convert_flux_to_gdf(flux_df)
            self.flux_pts_gdf = convert_flux_to_points_gdf(flux_df)
            self.flux_names = flux_df["name"].tolist()
        else:
            print(
                "flux xsect file is not a yaml file, so only names loaded with no geometry.\n"
                + "Filling in with matching stations.in geometry where station_id matches.\n"
            )
            names = station.station_names_from_file(str(self.flux_xsect_file))
            flux_df = pd.DataFrame(names, columns=["flux_name"])
            flux_df["station_id"] = flux_df["flux_name"].str.lower()  # + "_default"
            self.flux_pts_gdf = flux_df.merge(
                self.stations_gdf, on="station_id", how="left"
            )
            self.flux_pts_gdf = self.flux_pts_gdf.rename(
                columns={"name": "station_name"}
            )
            self.flux_names = names

    def interpret_file_relative_to(self, base_dir, fpath):
        full_path = base_dir / fpath
        if not full_path.exists():
            logger.warning(f"File {full_path} does not exist. Using {fpath} instead.")
            full_path = fpath
        return full_path

    def get_unit_for_variable(self, var):
        if var in ["elev"]:
            return "meters"
        elif var in ["flow"]:
            return "m^3/s"
        elif var in ["temp"]:
            return "deg_c"
        elif var in ["wind_x", "wind_y", "u", "v", "w"]:
            return "m/s"
        elif var in ["salt"]:
            return "PSU"
        elif var in ["ssc"]:
            return "mg/L"
        else:
            return "unknown"

    def get_catalog(self):
        catalog_key = str(self.base_dir / "catalog")
        if catalog_key in self.cache:
            return self.cache[catalog_key]
        else:
            var_stations = []
            for var in STATION_VARS:
                s = self.stations_gdf.copy()
                s["variable"] = var
                s["unit"] = self.get_unit_for_variable(var)
                s["filename"] = self.interpret_file_relative_to(
                    self.output_dir, station.staout_name(var)
                )
                s.drop(columns=["id", "subloc", "x", "y", "z"], inplace=True)
                s.rename(columns={"station_id": "id"}, inplace=True)
                var_stations.append(s)
            # TODO: if flux_pts_gdf is empty, then use the flux_names to create geometry
            if self.flux_pts_gdf is not None:
                flux_stations = self.flux_pts_gdf.copy()
                flux_stations = flux_stations[
                    ["station_id", "station_name", "geometry"]
                ]
                flux_stations.rename(
                    columns={"station_id": "id", "station_name": "name"}, inplace=True
                )
                flux_stations["variable"] = "flow"
                flux_stations["unit"] = "m^3/s"
                flux_stations["filename"] = str(self.flux_out)
            df = pd.concat(var_stations + [flux_stations])
            self.cache[catalog_key] = df
            return df

    def cache_vars(self, vars):
        # get data for these vars
        for var in vars:
            if var == "flow":
                flux = self.get_flux()
            else:
                staout = self.get_staout(var)

    def clear_cache(self):
        self.cache.clear()

    def get_data(self, row):
        """get data for a row of the catalog"""
        var = row["variable"]
        id = row["id"]
        filename = row["filename"]
        if var == "flow":
            flux = self.get_flux()
            return flux[[id]]
        else:
            staout = self.get_staout(var)
            if not ("_" in id):
                id = id + "_default"
            return staout[[id]]

    @lru_cache(maxsize=32)
    def get_flux(self):
        if self.flux_out in self.cache:
            logger.info(f"Using cached flux from disk: {self.base_dir}")
            return self.cache[self.flux_out]
        else:
            logger.info(f"Reading flux: {self.base_dir}")
            flux = station.read_flux_out(self.flux_out, self.flux_names, self.reftime)
            # flux.columns = [col + "_default" for col in flux.columns]
            flux.index.name = "Time"
            self.cache[self.flux_out] = flux
            return flux

    @lru_cache(maxsize=32)
    def get_staout(self, variable, fpath=None):
        if fpath is None:
            fpath = self.interpret_file_relative_to(
                self.output_dir, station.staout_name(variable)
            )
        if fpath in self.cache:
            logger.info(f"Using cached staout from disk: {fpath}")
            return self.cache[fpath]
        else:
            logger.info(f"Reading staout: {fpath}")
            staout = station.read_staout(
                fpath,
                self.stations_in,
                self.reftime,
            )
            staout.index.name = "Time"
            self.cache[fpath] = staout
            return staout

    def clear_cache(self):
        self.cache.clear()

    def get_flux_for(self, station_id):
        flux = self.get_flux()
        return flux[station_id]

    def get_staout_for(self, variable, station_id):
        staout = self.get_staout(variable)
        return staout[station_id]
