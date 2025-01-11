# %%
import schimpy
import pandas as pd
import schimpy.station as station
import schimpy.schism_yaml as schism_yaml
from dms_datastore.read_ts import *
import hvplot.pandas
import cartopy.crs as ccrs
from dms_datastore import read_ts
import yaml
from shapely.geometry import LineString
import geopandas as gpd
import os
import pathlib

base_dir = pathlib.Path("D:/temp/schism_output_sample")
reftime = pd.Timestamp("2020-01-01")
flux_file = base_dir / "preprocessed/flow_station_xsects.yaml"
repo_dir = pathlib.Path("Y:/repo_staging/continuous/screened")


# %%
from pydelmod import schismstudy

study = schismstudy.SchismStudy(base_dir=str(base_dir), flux_xsect_file=str(flux_file))

# %%
stations_pts = study.stations_gdf.hvplot(
    tiles="CartoLight", crs=ccrs.UTM(10), geo=True, hover_cols=["id", "subloc", "name"]
)
flux_lines = study.flux_gdf.hvplot(
    tiles="CartoLight",
    crs=ccrs.UTM(10),
    geo=True,
    line_width=8,
    hover_cols=["name", "station_id", "station_name"],
)
flux_pts = study.flux_pts_gdf.hvplot(
    tiles="CartoLight",
    crs=ccrs.UTM(10),
    geo=True,
    hover_cols=["name", "station_id", "station_name"],
)
flux_lines + flux_pts
# %%
stations_pts * flux_pts
# %%
flux = study.load_flux()
salt = study.load_staout("salt")
# %%
stations = station.read_station_dbase(str(base_dir / "station_dbase.csv"))
# %%
station_obs_links = station.read_obs_links(str(base_dir / "obs_links_20230315.csv"))
station_obs_links = station_obs_links.reset_index()
station_obs_links.set_index("id", inplace=True)
# %%


station_name = "mrz"
station_links = station_obs_links.loc[station_name]
fpath = str(repo_dir / station_links.iloc[0]["filename"])
mrz_ts = read_ts.read_flagged(fpath)
#
# station.read_station_in(fpath)
# station.read_station_out(fpath, stationinfo, var, start)
# %%
merged_stations = stations.merge(
    station_obs_links[["subloc", "variable", "filename"]],
    left_index=True,
    right_index=True,
    how="left",
)
# %%
import diskcache

# %%
cache = diskcache.Cache(".schismui_cache")
cache[str(base_dir / "flux.out")] = flux
