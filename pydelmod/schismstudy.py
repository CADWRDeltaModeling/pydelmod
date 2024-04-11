import os
import pathlib
import schimpy
import pandas as pd
import schimpy.station as station
import schimpy.schism_yaml as schism_yaml
import cartopy.crs as ccrs
import yaml
from shapely.geometry import LineString
import geopandas as gpd
import param
import diskcache


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


class SchismStudy(param.Parameterized):

    def __init__(
        self,
        base_dir=".",
        flux_xsect_file="flow_station_xsects.yaml",
        station_in_file="station.in",
        flux_out="flux.out",
        reftime="2020-01-01",
        **kwargs
    ):
        self.base_dir = pathlib.Path(base_dir)
        self.cache = diskcache.Cache(self.base_dir / ".schismstudy-cache")
        self.reftime = pd.Timestamp(reftime)
        self.flux_xsect_file = self.interpret_file_relative_to(
            self.base_dir, pathlib.Path(flux_xsect_file)
        )
        self.station_in_file = self.interpret_file_relative_to(
            self.base_dir, pathlib.Path(station_in_file)
        )
        self.flux_out = self.interpret_file_relative_to(
            self.base_dir, pathlib.Path(flux_out)
        )
        self.reftime = pd.Timestamp(kwargs.pop("reftime", "2020-01-01"))
        super().__init__(**kwargs)
        stations = read_station_in(self.station_in_file)
        self.stations_in = stations
        stations = stations.reset_index()
        stations["station_id"] = stations["id"] + "_" + stations["subloc"]
        self.stations_gdf = convert_station_to_gdf(stations)
        flux_df = load_flux_dataframe(self.flux_xsect_file)
        flux_df["station_id"] = flux_df["name"].str.lower() + "_default"
        self.flux_gdf = convert_flux_to_gdf(flux_df)
        self.flux_pts_gdf = convert_flux_to_points_gdf(flux_df)
        self.flux_names = flux_df["name"].tolist()

    def interpret_file_relative_to(self, base_dir, fpath):
        full_path = base_dir / fpath
        if full_path.exists():
            return full_path
        else:
            return fpath

    def get_unit_for_variable(self, var):
        if var in ["elev", "air pressure", "temp"]:
            return "m"
        elif var in ["wind_x", "wind_y", "u", "v", "w"]:
            return "m/s"
        elif var in ["salt"]:
            return "ppt"
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
                s["filename"] = str(self.base_dir / station.staout_name(var))
                s.drop(columns=["id", "subloc", "x", "y", "z"], inplace=True)
                s.rename(columns={"station_id": "id"}, inplace=True)
                var_stations.append(s)
            flux_stations = self.flux_pts_gdf.copy()
            flux_stations = flux_stations.drop(
                columns=["name", "new_name", "comment", "agency_id"],
            )
            flux_stations.rename(
                columns={"station_id": "id", "station_name": "name"}, inplace=True
            )
            flux_stations["variable"] = "flow"
            flux_stations["unit"] = "m^3/s"
            flux_stations["filename"] = str(self.flux_out)
            df = pd.concat(var_stations + [flux_stations])
            self.cache[catalog_key] = df
            return df

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
            return staout[[id]]

    def get_flux(self):
        if self.flux_out in self.cache:
            return self.cache[self.flux_out]
        else:
            flux = station.read_flux_out(self.flux_out, self.flux_names, self.reftime)
            flux.columns = [col + "_default" for col in flux.columns]
            flux.index.name = "Time"
            self.cache[self.flux_out] = flux
            return flux

    def get_staout(self, variable, fpath=None):
        if fpath is None:
            fpath = str(self.base_dir / station.staout_name(variable))
        if fpath in self.cache:
            return self.cache[fpath]
        else:
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
