from functools import lru_cache
import param
import os
import diskcache
from dms_datastore.read_ts import read_ts
from pathlib import Path
import pandas as pd
import geopandas as gpd


def convert_station_to_gdf(stations):
    return gpd.GeoDataFrame(
        stations,
        geometry=gpd.points_from_xy(stations.x, stations.y),
        crs="EPSG:32610",
    )


# this should be a util function
def find_lastest_fname(pattern, dir="."):
    d = Path(dir)
    fname, mtime = None, 0
    for f in d.glob(pattern):
        fmtime = f.stat().st_mtime
        if fmtime > mtime:
            mtime = fmtime
            fname = f.absolute()
    return fname, mtime


class StationDatastore(param.Parameterized):
    caching = param.Boolean(default=True, doc="Use caching")

    def __init__(self, **kwargs):
        self.repo_dir = kwargs.pop("repo_dir", "screened")
        self.inventory_file = kwargs.pop("inventory_file", "inventory_datasets.csv")
        self.cache_dir = kwargs.pop("cache_dir", ".cache-ds")
        super().__init__(**kwargs)
        self.cache = diskcache.Cache(self.cache_dir, size_limit=1e11)
        self.caching_read_ts = lru_cache(maxsize=32)(self.cache.memoize()(read_ts))
        # read inventory file for each repo level
        print("Using inventory file: ", self.inventory_file)
        self.df_dataset_inventory = pd.read_csv(self.inventory_file, comment="#")
        # replace nan with empty string for column subloc
        self.df_dataset_inventory["subloc"] = self.df_dataset_inventory[
            "subloc"
        ].fillna("")
        self.unique_params = self.df_dataset_inventory["param"].unique()
        common_cols = [
            "subloc",
            "name",
            "unit",
            "param",
            "min_year",
            "max_year",
            "filename",
            "agency",
            "agency_id_dbase",
            "lat",
            "lon",
            "x",
            "y",
        ]
        try:  # inventory.csv file ?
            group_cols = ["station_id"] + common_cols
            self.df_station_inventory = (
                self.df_dataset_inventory.groupby(group_cols)
                .count()
                .reset_index()[group_cols]
            )
        except KeyError:  # obs_links.csv ?
            group_cols = ["id"] + common_cols
            self.df_station_inventory = (
                self.df_dataset_inventory.groupby(group_cols)
                .count()
                .reset_index()[group_cols]
            )
            self.df_station_inventory.rename(columns={"id": "station_id"}, inplace=True)
        # calculate min (min year) and max of max_year
        self.min_year = self.df_station_inventory["min_year"].min()
        self.max_year = self.df_station_inventory["max_year"].max()

    def get_catalog(self):
        return self.df_station_inventory

    def get_station_inventory_gdf(self):
        df = self.get_station_inventory().copy()
        df["subloc"] = df["subloc"].apply(lambda v: "default" if len(v) == 0 else v)
        df["station_id"] = df["station_id"].astype(str) + "_" + df["subloc"]
        df.rename(columns={"param": "variable"})
        return convert_station_to_gdf(df)

    def last_part_path(self, dir):
        return os.path.basename(os.path.normpath(dir))

    def get_data(self, row):
        repo_dir = self.repo_dir
        filename = row["filename"]
        if self.caching:
            return self.caching_read_ts(os.path.join(repo_dir, filename))
        else:
            return read_ts(os.path.join(repo_dir, filename))

    def clear_cache(self):
        if self.caching:
            self.cache.clear()
            print("Cache cleared")
