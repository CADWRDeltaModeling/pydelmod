# %%
import pandas as pd
from pydelmod import dsm2ui
from pydelmod.dvue import dataui

# %%
dir = ".."
dir = "D:/dev/pydelmod/"
channel_line_geojson_file = (
    f"{dir}/examples/dsm2gis/dsm2_channels_centerlines_8_2.geojson"
)
hydro_echo_file = (
    "D:/delta/dsm2v822/study_templates/historical/output/hydro_echo_hist_v822.inp"
)
# %%
# from pydelmod.dsm2study import *
# hydro_tables = load_echo_file(hydro_echo_file)
# time_range = get_runtime(hydro_tables)
# output_channels = hydro_tables["OUTPUT_CHANNEL"]
# output_dir = os.path.dirname(hydro_echo_file)
# output_channels["FILE"] = output_channels["FILE"].str.replace(
#     "./output", output_dir, regex=False
# )
# import cartopy
# from cartopy.crs import GOOGLE_MERCATOR
# dsm2_chan_lines = load_dsm2_channelline_shapefile(channel_line_geojson_file)
# dsm2_chan_lines.hvplot(geo=True, tiles='OSM', project=True, crs=GOOGLE_MERCATOR)
# import geoviews as gv
# import geopandas as gpd
# gdf = gpd.read_file(channel_line_geojson_file)


# %%
uimgr = dsm2ui.build_output_plotter(
    channel_line_geojson_file,
    hydro_echo_file,
    variable="FLOW",
)
ui = dataui.DataUI(uimgr)
ui.create_view().show()
