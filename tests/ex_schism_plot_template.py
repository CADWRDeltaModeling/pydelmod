# %%
import pandas as pd
import hvplot.pandas
import holoviews as hv
from holoviews import opts, dim
import panel as pn

pn.extension()

from vtools.functions.filter import cosine_lanczos


def shift_cycle(vals, shift=1):
    return hv.Cycle(vals[shift:] + vals[:shift])


# %%
# Load the station data
dfobs = pd.read_csv("odm_default_flow_obs.csv", index_col=0, parse_dates=True)
dfobs.index.freq = pd.infer_freq(dfobs.index)
import glob

simfiles = glob.glob("odm_default_flow_sim_*.csv")
dfsimlist = []
for i, simfile in enumerate(simfiles):
    dfsim = pd.read_csv(simfile, index_col=0, parse_dates=True)
    dfsim.index.freq = pd.infer_freq(dfsim.index)
    dfsimlist.append(dfsim)
# %%
inst_time_window = slice("2022-01-22", "2022-04-13")
avg_time_window = slice("2022-01-27", "2022-07-13")


# %%
def schism_plot_template(dfobs, dfsimlist, inst_time_window, avg_time_window):
    ex_avg_time_window = slice(
        pd.Timestamp(avg_time_window.start) - pd.Timedelta("2D"),
        pd.Timestamp(avg_time_window.stop) + pd.Timedelta("2D"),
    )
    df = pd.concat([dfobs] + dfsimlist, axis=1)
    df.columns = ["Observed", "Simulation 1", "Simulation 2"]
    df = df.loc[ex_avg_time_window]
    df.index.freq = pd.infer_freq(df.index)
    dff = cosine_lanczos(df.loc[ex_avg_time_window], "40H")
    default_cycle = hv.Cycle(hv.Cycle.default_cycles["default_colors"])
    plot_inst = df.loc[inst_time_window].hvplot(
        responsive=True, color=default_cycle, grid=True
    )
    plot_scatter = df.hvplot.scatter(
        x="Observed",
        color=shift_cycle(default_cycle.values),
        grid=True,
    ).opts(
        opts.Scatter(
            data_aspect=1,
        )
    )
    plot_avg = dff.loc[avg_time_window].hvplot(
        responsive=True,
        color=default_cycle,
        grid=True,
        xlim=(avg_time_window.start, avg_time_window.stop),
    )
    gs = pn.layout.GridSpec()
    gs[0, 4:6] = pn.pane.Markdown("## Flow Calibration")
    gs[1:6, 0:11] = plot_inst.opts(shared_axes=False)
    gs[6:10, 0:6] = plot_avg.opts(shared_axes=False, show_legend=False)
    gs[6:10, 6:8] = plot_scatter.opts(shared_axes=False)
    return gs


# %%
pn.Tabs(
    (
        "Flow Calibration",
        schism_plot_template(dfobs, dfsimlist, inst_time_window, avg_time_window),
    )
).show()
# %%
