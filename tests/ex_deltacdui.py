#%%
import geoviews as gv
import geopandas as gpd
from bokeh.plotting import show
import holoviews as hv
import cartopy.crs as ccrs
# Initialize renderers
gv.extension('bokeh')

# Load the GeoJSON file
file_path = '../examples/dsm2gis/detaw168subareas.geojson'
gdf = gpd.read_file(file_path)

# Create a GeoViews polygon feature
geom = gv.Polygons(gdf, crs=ccrs.epsg(26910)).opts(responsive=True,
    tools=['hover'],)

# Display the GeoJSON data with a proper background
tiles = gv.tile_sources.OSM()
map_view = tiles * geom.opts(line_width=1, fill_alpha=0.5, tools=['hover'])

# Show the result
import panel as pn
pn.extension()
pn.panel(pn.Row(map_view, sizing_mode="stretch_both")).servable(title="GeoJSON Map with Background Tiles").show()
# %%
detaw_output_file = "d:/delta/deltacd_inputs/historical/outputs/detawoutput_dsm2.nc"
import xarray as xr
# %%
ds = xr.open_dataset(detaw_output_file)
# %%
ds
# %%
from pydelmod.deltacduimgr import DeltaCDUIManager
detaw_output_file = "d:/delta/deltacd_inputs/historical/outputs/detawoutput_dsm2.nc"
file_path = '../pydelmod/dsm2gis/detaw168subareas.geojson'
#%%
#dcd_ui = DeltaCDUIManager(detaw_output_file, geojson_file_path=file_path)
dcd_ui = DeltaCDUIManager(detaw_output_file)
dfcat=dcd_ui.get_data_catalog()
# %%
from pydelmod.dvue import dataui
import cartopy.crs as ccrs
dui=dataui.DataUI(dcd_ui, station_id_column="area_id", crs=ccrs.epsg(26910))
dui.create_view().servable(title="DeltaCD UI Manager").show()

# %%
dcd_ui.get_data_for_time_range(dfcat.iloc[0],dcd_ui.get_time_range(dfcat))
# %%
