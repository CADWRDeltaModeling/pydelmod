# %%
import pandas as pd
import geopandas as gpd
import hvplot.pandas
import pydsm
from pydsm.hydroh5 import HydroH5
from pydsm.qualh5 import QualH5
import geoviews as gv
from geoviews import opts, dim, streams
import cartopy.crs as ccrs

# %%
channels = gpd.read_file(
    "../examples/dsm2gis_cleaned/dsm2_channels_centerlines_8_2.geojson"
)

# channels = gpd.read_file("../examples/dsm2gis/dsm2_channels_centerlines_8_2.geojson")
# %%

# %%
# channels_buffered.geometry = channels.buffer(100)
# %%
hydro_tidefile = "D:/dev/pydsm/tests/data/historical_v82.h5"
qual_tidefile = "d:/dev/pydsm/tests/data/historical_v82_ec.h5"
# %%
from pydelmod import dsm2ui
from pydelmod import dataui

uim = dsm2ui.DSM2TidefileUIManager([hydro_tidefile, qual_tidefile], channels=channels)
ui = dataui.DataUI(uim, crs=ccrs.epsg("26910"), station_id_column="geoid")
# %%
ui.create_view().show()

# %%
