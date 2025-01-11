# %%
import geopandas as gpd
import geoviews as gv

gv.extension("bokeh")
import cartopy.crs as ccrs

# %%
gdf = gpd.read_file("some_channels.geojson")
# %%
gv.Path(gdf).opts(width=400, height=500)
# %%
