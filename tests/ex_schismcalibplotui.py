# %%
from pydelmod import dataui
from pydelmod import schismcalibplotui
from cartopy import crs as ccrs

# config_file = "d:/dev/schimpy/tests/example_full_mss/input_compare.yaml"
config_file = "D:/temp/itp202411/postpro/batch_metrics_itp2024.yaml"
manager = schismcalibplotui.SchismCalibPlotUIManager(config_file)
dataui.DataUI(manager, crs=ccrs.UTM(10)).create_view().show()

# %%
# import cProfile
# cProfile.run('manager.get_data("odm_default","flow")', "schismcalibplotui.prof")
