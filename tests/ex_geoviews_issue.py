# %%
import pandas as pd
import geopandas as gpd
import geoviews as gv
import hvplot.pandas
import numpy as np

gv.extension("bokeh")
# %%
# create a GeoDataFrame with a few points
df = pd.DataFrame(
    {
        "City": ["Buenos Aires", "Brasilia", "Santiago", "Bogota", "Caracas", "Quito"],
        "Country": ["Argentina", "Brazil", "Chile", "Colombia", "Venezuela", "Ecuador"],
        "Latitude": [-34.58, -15.78, -33.45, 4.60, 10.48, np.nan],
        "Longitude": [-58.66, -47.91, -70.66, -74.08, -66.86, np.nan],
    }
)
gdf = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude, crs="EPSG:4326")
)
gdf.hvplot.points(geo=True, tiles="CartoLight")
# %%
# create a GeoDataFrame with a few points
df = pd.DataFrame(
    {
        "City": ["Buenos Aires", "Brasilia", "Santiago", "Bogota", "Caracas", "Quito"],
        "Country": ["Argentina", "Brazil", "Chile", "Colombia", "Venezuela", "Ecuador"],
        "Latitude": [-34.58, -15.78, -33.45, 4.60, 10.48, ""],
        "Longitude": [-58.66, -47.91, -70.66, -74.08, -66.86, ""],
    }
)
gdf = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude, crs="EPSG:4326")
)
gdf.hvplot.points(geo=True, tiles="CartoLight")
# %%
gdf["Latitude"] = pd.to_numeric(gdf["Latitude"], errors="coerce")
gdf["Longitude"] = pd.to_numeric(gdf["Longitude"], errors="coerce")
# %%
gdf.hvplot.points(geo=True, tiles="CartoLight")
# %%
