import pydelmod
import pandas as pd
import hvplot.pandas
import glob
from pydelmod import calibplot
import warnings

warnings.filterwarnings("ignore")
import yaml
import vtools.functions.filter as tidal_filter
import holoviews as hv
from holoviews import opts
import panel as pn

pn.extension()


def to_timewindow_string(str):
    wx = str.replace("(", "").replace(")", "").split(",")
    return ":".join([pd.to_datetime(x).strftime("%Y-%m-%d") for x in wx])


def scatter_plot(
    dflist,
    dfmetrics,
    names,
    xaxislabel="Observed",
    yaxislabel="Model",
    legend_position="top_left",
    color_cycle=hv.Cycle("Category10"),
):
    scatterplot = calibplot.scatterplot(dflist, names)
    scatterplot.opts(opts.Scatter(color=calibplot.shift_cycle(color_cycle))).opts(
        ylabel=yaxislabel, legend_position=legend_position
    ).opts(show_grid=True, frame_height=250, frame_width=250, data_aspect=1).opts(
        toolbar="right"
    )
    reglineplots = calibplot.regression_line_plots(dfmetrics, False)
    scatter_plot = (
        reglineplots.opts(opts.Slope(color=calibplot.shift_cycle(color_cycle)))
        * scatterplot
    )
    scatter_plot = scatter_plot.opts(
        xlabel=xaxislabel, ylabel=yaxislabel, legend_position=legend_position
    ).opts(
        show_grid=True,
        frame_height=250,
        frame_width=250,
        data_aspect=1,
        show_legend=False,
    )
    return scatter_plot


def load_dfs(station_var):
    dfobs = pd.read_csv(f"{station_var}_obs.csv", index_col=0, parse_dates=True)
    dfobs = dfobs.resample(dfobs.index.inferred_freq).mean()
    obs_data_freq = dfobs.index.freqstr
    dfsimlist = [
        pd.read_csv(f, index_col=0, parse_dates=True)
        for f in glob.glob(f"{station_var}_sim_*.csv")
    ]
    dfsimlist = [df.resample(obs_data_freq).mean() for df in dfsimlist]
    return dfobs, dfsimlist


def plot_metrics(station_yaml_file):
    configmap = yaml.safe_load(open(f"{station_yaml_file}"))
    station_var = station_yaml_file.split("_plot.yaml")[0]
    dfobs, dfsimlist = load_dfs(station_var)
    window_inst = to_timewindow_string(configmap["window_inst"])
    inst_plot = calibplot.tsplot(
        [dfobs] + dfsimlist, configmap["labels"], window_inst, True
    )
    # tidal averaging
    dfobsf = tidal_filter.cosine_lanczos(dfobs, "40H")
    dfsimlistf = [tidal_filter.cosine_lanczos(df, "40H") for df in dfsimlist]
    window_avg = to_timewindow_string(configmap["window_avg"])
    # Now do the time slicing
    dfobsf = dfobsf.loc[slice(*window_avg.split(":")), :]
    dfsimlistf = [df.loc[slice(*window_avg.split(":")), :] for df in dfsimlistf]
    dfobs = dfobs.loc[slice(*window_inst.split(":")), :]
    dfsimlist = [df.loc[slice(*window_inst.split(":")), :] for df in dfsimlist]
    # Now do the plotting
    plotf = calibplot.tsplot(
        [dfobsf] + dfsimlistf, configmap["labels"], window_avg, True
    )
    dfmetrics = calibplot.calculate_metrics([dfobsf] + dfsimlistf, configmap["labels"])
    splot = scatter_plot([dfobsf] + dfsimlistf, dfmetrics, configmap["labels"])
    # layout template
    grid = pn.GridSpec(sizing_mode="stretch_both", min_height=600)
    grid[0, 2:4] = pn.pane.HTML(f'<h2>{configmap["title"]}</h2>')
    grid[1:4, :] = inst_plot.opts(shared_axes=False, legend_position="right")
    grid[4:7, 0:5] = pn.Row(
        plotf.opts(show_legend=False), splot.opts(shared_axes=False)
    )
    grid[7:9, :] = pn.pane.DataFrame(dfmetrics)
    # save
    grid.save(f"{station_var}_plot.html")

    return grid
