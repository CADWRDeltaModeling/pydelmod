# %%
import schimpy
from schimpy import batch_metrics
import pydelmod
from pydelmod import datastore
from pydelmod import schism_plot_metrics, schismcalibplotui, schismstudy

# %%
from schimpy import batch_metrics
from pathlib import Path


def interpret_file_relative_to(base_dir, fpath):
    full_path = base_dir / fpath
    if full_path.exists():
        return str(full_path)
    else:
        return str(fpath)


def replace_with_paths_relative_to(base_dir, params):
    params = params.copy()
    params["output_dir"] = [
        interpret_file_relative_to(base_dir, file) for file in params["outputs_dir"]
    ]
    params["stations_csv"] = interpret_file_relative_to(
        base_dir, params["stations_csv"]
    )
    params["obs_search_path"] = [
        interpret_file_relative_to(base_dir, file) for file in params["obs_search_path"]
    ]
    params["station_input"] = [
        interpret_file_relative_to(base_dir, file) for file in params["station_input"]
    ]
    params["obs_links_csv"] = interpret_file_relative_to(
        base_dir, params["obs_links_csv"]
    )
    params["flow_station_input"] = [
        interpret_file_relative_to(base_dir, file)
        for file in params["flow_station_input"]
    ]
    return params


# %%

calib_config_file = r"D:\dev\schimpy\tests\example_full_mss\input_compare.yaml"
base_dir = Path(calib_config_file).parent
params = batch_metrics.get_params(calib_config_file)
selected_stations = params["selected_stations"].split(",")
params = replace_with_paths_relative_to(base_dir, params)
# %%
for station in selected_stations[0:1]:
    sparam = params.copy()
    sparam["selected_stations"] = station
    bm = batch_metrics.BatchMetrics(sparam)
    # bm.plot()

# %%
studyid = 0
schism_dir = params["output_dir"][studyid]
flux_xsect_file = params["flow_station_input"][studyid]
station_in_file = params["station_input"][studyid]
reftime = params["time_basis"]
flux_out = "flux.out"
study = schismstudy.SchismStudy(
    schism_dir,
    flux_xsect_file=flux_xsect_file,
    station_in_file=station_in_file,
    flux_out=flux_out,
    reftime=reftime,
)
# %%
repo_dir = params["obs_search_path"][0]
inventory_file = params["obs_links_csv"]
ds = datastore.StationDatastore(repo_dir=repo_dir, inventory_file=inventory_file)
import pandas as pd

time_range = (pd.Timestamp(reftime), pd.Timestamp(reftime) + pd.Timedelta(days=250))

# %%
