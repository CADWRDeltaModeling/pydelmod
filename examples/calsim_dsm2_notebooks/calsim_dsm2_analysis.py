# %%
# Import modules
import pandas as pd
import numpy as np
import pydelmod.utilities as pdmu
import pydelmod.nbplot as pdmn

# Read output locations
fpath_output_locations = "info/DSM2_bnd_loc.csv"
df_stations = pd.read_csv(fpath_output_locations, comment="#")
df_stations["ID"] = [x.upper() for x in df_stations["ID"]]
station_ids = df_stations["ID"].values
stations_to_read = df_stations["ID"].values

# Read in scenarios
dir_scenarios = "scenarios/"
scenarios = [
    {"name": "1EX_2020", "fpath": dir_scenarios + "1ex_2020c.dss"},
    {"name": "9B_2020", "fpath": dir_scenarios + "9b_v2_2020.dss"},
]

wyt_c3f2020 = dir_scenarios + "calsim.dss"
df_wyt2020 = pdmu.read_calsim3_wateryear_types(wyt_c3f2020)

# period93 = ['1922-10-1','2015-9-30']
period93 = ["1921-1-1", "2021-9-30"]

df_flow = pdmu.prep_df(
    scenarios, stations_to_read, ["FLOW"], ["1MON"], df_wyt2020, period93
)
df_flow

# %%
options = {
    "yaxis_name": "Monthly Mean Flow (cfs)",
    "title": "Flow Monthly Mean Timelines",
}
plot = pdmn.plot_step_w_variable_station_filters(df_flow, df_stations, options)
plot.show()
# %%
options = {
    "yaxis_name": "Monthly Mean Flow (cfs)",
    "title": "Flow Monthly Barcharts of Monthly Mean",
}
pdmn.plot_bar_monthly_w_controls(df_flow, df_stations, options)

# %%
options = {
    "yaxis_name": "Monthly Mean Flow (cfs)",
    "title": "Flow Monthly Mean Exceedances",
}
pdmn.plot_exceedance_w_variable_station_filters(df_flow, df_stations, options)

# %%
options = {
    "xaxis_name": "Monthly Mean Flow (cfs)",
    "title": "Flow Monthly Mean Box-Whiskers",
}
pdmn.plot_box_w_variable_station_filters(df_flow, df_stations, options)

# %% [markdown]
# ### Flow Boundary (Calsim) Monthly Mean Diff

# %%
# df_flow1 = df_flow[df_flow['scenario_name']=='NAA_2020']
# df_flow2 = df_flow[df_flow['scenario_name']=='PA7K5_2020']
# df_flow_dff = df_flow1.copy()
# df_flow_dff['scenario_name'] = 'PA7K5-NAA_2020'
# df_flow_dff['value'] = df_flow2['value'] - df_flow1['value']

# %%
# options = {'yaxis_name': 'Monthly Mean Flow Difference (cfs)', 'title': 'Flow Monthly Mean Difference Timelines'}
# pdmn.plot_step_w_variable_station_filters(df_flow_dff, df_stations, options)

# %%
# options = {'yaxis_name': 'Monthly Mean Flow Difference (cfs)', 'title': 'Flow Monthly Barcharts of Monthly Mean Difference'}
# pdmn.plot_bar_monthly_w_controls(df_flow_dff, df_stations, options)

# %%
# options = {'yaxis_name': 'Monthly Mean Flow Difference (cfs)', 'title': 'Flow Monthly Mean Difference Exceedances'}
# pdmn.plot_exceedance_w_variable_station_filters(df_flow_dff, df_stations,options)

# %%
# options = {'xaxis_name': 'Monthly Mean EC Difference (mmhos/cm)', 'title': 'Flow Monthly Mean Difference Box-Whiskers'}
# pdmn.plot_box_w_variable_station_filters(df_flow_dff, df_stations, options)

# %% [markdown]
# # Structure Operation

# %%
# ['OP']
# ['IR-YEAR']

# %% [markdown]
# # EC Vernalis Daily Mean

# %%
df_stations_ver = pd.DataFrame(
    [["SJR Vernalis", "VERNWQFINAL"]], columns=["Location", "ID"]
)
# df_stations_ver

# %%
# df_ec_ver = prep_df_bnd(np.array(['VERNWQFINAL'],dtype=object),['SALINITY-EC'],['1DAY'])
df_ec_ver = pdmu.prep_df(
    scenarios,
    np.array(["VERNWQFINAL"], dtype=object),
    ["SALINITY-EC"],
    ["1DAY"],
    df_wyt2020,
    period93,
)
df_ec_ver

# %%
options = {"yaxis_name": "Daily Mean EC (mmhos/cm)", "title": "EC Daily Mean Timelines"}
pdmn.plot_step_w_variable_station_filters(df_ec_ver, df_stations_ver, options)

# %%
options = {
    "yaxis_name": "Daily Mean EC (mmhos/cm)",
    "title": "EC Monthly Barcharts of Daily Mean",
}
pdmn.plot_bar_monthly_w_controls(df_ec_ver, df_stations_ver, options)

# %%
options = {
    "yaxis_name": "Daily Mean EC (mmhos/cm)",
    "title": "EC Daily Mean Exceedances",
}
pdmn.plot_exceedance_w_variable_station_filters(df_ec_ver, df_stations_ver, options)

# %%
options = {
    "yaxis_name": "Daily Mean EC (mmhos/cm)",
    "title": "EC Daily Mean Box-Whiskers",
}
pdmn.plot_box_w_variable_station_filters(df_ec_ver, df_stations_ver, options)

# %% [markdown]
# # EC Boundary (Martinez) Monthly Mean

# %%
df_stations_mtz = pd.DataFrame([["Martinez", "RSAC054"]], columns=["Location", "ID"])

# %%
df_ec_mtz = pdmu.prep_df(
    scenarios,
    np.array(["RSAC054"], dtype=object),
    ["EC-MEAN"],
    ["1DAY"],
    df_wyt2020,
    period93,
)

# %%
options = {
    "yaxis_name": "Monthly Mean EC (mmhos/cm)",
    "title": "EC Monthly Mean Timelines",
}
pdmn.plot_step_w_variable_station_filters(df_ec_mtz, df_stations_mtz, options)

# %%
options = {"yaxis_name": "Mean EC (mmhos/cm)", "title": "EC Means by Months"}
pdmn.plot_bar_monthly_w_controls(df_ec_mtz, df_stations_mtz, options)

# %%


# %%


# %%
