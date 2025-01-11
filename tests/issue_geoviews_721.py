# %%
import geopandas as gpd
import pandas as pd
import holoviews as hv
import geoviews as gv

hv.extension("bokeh")
import cartopy.crs as ccrs

# %%
# create a dataframe with some x,y coordinates and a couple of columns
df = pd.DataFrame(
    {
        "x": [1, 2, 3, 4, 5],
        "y": [1, 2, 3, 4, 5],
        "value1": [10, 20, 30, 40, 50],
        "value2": [5, "4", 3, 2, 1],
    }
)
# emulate UTM zone 10n coordinates
df["x"] = df["x"].astype(float) + 625000
df["y"] = df["y"].astype(float) + 4180000
# %%
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y))

# %%
gv.Points(gdf)
# %%
df.min()
# %%
df.min(numeric_only=True)
