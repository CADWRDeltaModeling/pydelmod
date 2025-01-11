# %%
import pydelmod
from pydelmod import dssui
from pydelmod import dataui
import pandas as pd
import geopandas as gpd
import cartopy.crs as ccrs
from pydelmod.dssui import DSSDataUIManager, DataUI

# %%
location_file = "data/elev_stations.csv"
location_id_column = "station_id"
station_id_column = "station_id"


# %%
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
    return ui


# %%
