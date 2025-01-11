# %%
import pandas as pd
import hvplot.pandas
import holoviews as hv
from holoviews import opts
import panel as pn
from vtools.functions.filter import cosine_lanczos

# %%
file = "data/schism_data_anh_upper_salt.csv"
df = pd.read_csv(file, index_col=0, parse_dates=True)
df.columns = ["Observed"] + df.columns[1:].tolist()
df
# %%


def shift_cycle(vals, shift=1):
    return hv.Cycle(vals[shift:] + vals[:shift])


def scatter_plot_with_slopes(df, y_axis_label, default_cycle=None):
    scatters = []
    slopes = []
    for col in df.columns[1:]:
        scatter = hv.Scatter(df, "Observed", col).opts(ylabel=y_axis_label)
        slope = hv.Slope.from_scatter(scatter)
        scatter = scatter.relabel(
            f"Slope: {slope.slope:.2f}*obs + {slope.y_intercept:.2f}"
        )
        scatters.append(scatter)
        slopes.append(slope)
    overlay = (
        hv.Overlay(scatters + slopes)
        .opts(opts.Slope(color=default_cycle))
        .opts(
            opts.Scatter(
                show_legend=True,
                color=default_cycle,
                legend_position="top",
                legend_cols=1,
            )
        )
    )

    return overlay


def plot_template(
    df, avg_time_window, inst_time_window, y_axis_label, title, resample_freq="1H"
):
    ex_avg_time_window = slice(
        pd.Timestamp(avg_time_window[0]) - pd.Timedelta("2D"),
        pd.Timestamp(avg_time_window[1]) + pd.Timedelta("2D"),
    )
    df = df.loc[ex_avg_time_window]
    df = df.resample(resample_freq).mean()
    dff = cosine_lanczos(df.loc[ex_avg_time_window], "40H")
    df = df.loc[slice(*inst_time_window), :]
    dff = dff.loc[slice(*avg_time_window), :]
    #
    default_cycle = hv.Cycle(hv.Cycle.default_cycles["default_colors"])
    plot_inst = df.hvplot(
        color=default_cycle,
        grid=True,
        responsive=True,
    ).opts(ylabel=y_axis_label, legend_position="top_left")
    plot_scatter = scatter_plot_with_slopes(
        df, y_axis_label, shift_cycle(default_cycle.values)
    )
    plot_avg = dff.hvplot(
        ylabel=f"Tidal Averaged {y_axis_label}",
        color=default_cycle,
        grid=True,
        responsive=True,
    )
    gs = pn.layout.GridSpec()  # 12x12 grid
    gs[0, 4:6] = pn.pane.Markdown(f"## {title}")
    gs[1:7, 0:11] = plot_inst.opts(shared_axes=False)
    gs[7:11, 0:11] = pn.Row(
        plot_avg.opts(shared_axes=False, show_legend=False),
        pn.Column(
            pn.pane.HoloViews(
                plot_scatter.opts(shared_axes=False, toolbar=None),
                sizing_mode="scale_height",
            ),
        ),
    )
    return gs


# %%
avg_time_window = ["2024-06-15", "2024-09-20"]
inst_time_window = ["2024-06-15", "2024-07-10"]
y_axis_label = "Salinity"
title = "Salinity @ Antioch Upper"

# %%
gs = plot_template(df, avg_time_window, inst_time_window, y_axis_label, title)
gs
# %%
gs.show()
#
# %%
import pandas as pd
import holoviews as hv

hv.extension("bokeh")

# Create a sample DataFrame
data = {
    "observed": [1, 2, 3, 4, 5],
    "model1": [1.1, 1.9, 3.2, 3.8, 5.1],
    "model2": [0.8, 2.2, 3.1, 4.0, 5.3],
}
df = pd.DataFrame(data)

# Create scatter plots for observed vs model1 and model2
scatter1 = hv.Scatter(df, "observed", "model1", label="Model1 Data")
scatter2 = hv.Scatter(df, "observed", "model2", label="Model2 Data")

# Calculate slopes and manually add labels
slope1 = hv.Slope.from_scatter(scatter1)
slope2 = hv.Slope.from_scatter(scatter2)

# Add slope equations as labels
scatter1 = scatter1.relabel(
    f"Slope1: {slope1.slope:.2f}*obs + {slope1.y_intercept:.2f}"
)
scatter2 = scatter2.relabel(
    f"Slope2: {slope2.slope:.2f}*obs + {slope2.y_intercept:.2f}"
)
# Combine everything into an overlay
overlay = (scatter1 * slope1 * scatter2 * slope2).opts(
    legend_position="top",
    legend_cols=1,
    shared_axes=False,
)

overlay

# %%
