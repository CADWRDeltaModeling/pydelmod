# %%
import pandas as pd
import hvplot.pandas
import holoviews as hv
from holoviews import opts, dim, streams
from pydelmod.calibplotui import CalibPlotUIManager, substitute_base_dir
from pydelmod.dataui import DataUI
import geoviews as gv

# %%
# Load the calibration results
# config_file = "X:/DSM2/full_calibration_8_3/delta/dsm2_FC.2024.01/studies/run2_1_newgeom_salt_loading/postprocessing/postpro_config_hydro.yml"
config_file = "X:/DSM2/full_calibration_8_3/delta/dsm2_FC.2024.01/studies/run2_1_newgeom_salt_loading/postprocessing/postpro_config_hydro.yml"
ui = DataUI(CalibPlotUIManager(config_file))
# %%
df = ui.dfcat
row = df.iloc[0]
from pydsm import postpro
from pydelmod import postpro_dsm2

dui = ui.dataui_manager
varname = row["vartype"]
vartype = postpro.VarType(varname, dui.config["vartype_dict"][varname])
studies = dui.get_studies(varname)
location = dui.build_location(row)
layout, metrics = postpro_dsm2.build_plot(dui.config, studies, location, vartype)
overlay = layout["with"]
# %%
import panel as pn

pn.extension()

# %%
import cartopy.crs as ccrs

ui.dfcat.hvplot(geo=True, crs=ccrs.PlateCarree(), tiles="CartoLight")
# %%
ui.dfcat.describe()
# %%
from shapely.geometry import Point, Polygon

# %%
california = Polygon(
    [
        (-124.848974, 42.009518),
        (-114.131211, 42.009518),
        (-114.131211, 32.534156),
        (-124.848974, 32.534156),
    ]
)
# %%
ui.dfcat.loc[~(ui.dfcat.within(california))]

# %%
ui.dfcat.loc[(ui.dfcat.within(california))].hvplot(
    geo=True, crs=ccrs.PlateCarree(), tiles="CartoLight"
)
# %%
