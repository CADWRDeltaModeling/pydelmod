# %%
import numpy as np
import pandas as pd
import holoviews as hv
from holoviews import opts

hv.extension("bokeh")

# %%
# Generate sample data
np.random.seed(42)
dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
series1 = np.random.normal(loc=0, scale=1, size=100).cumsum()
series2 = np.random.normal(loc=0.5, scale=1.2, size=100).cumsum()

# Create DataFrame
df = pd.DataFrame({"date": dates, "series1": series1, "series2": series2})

# Create time series plots
ts1 = hv.Curve(df, "date", "series1", label="Series 1")
ts2 = hv.Curve(df, "date", "series2", label="Series 2")
time_series = (
    (ts1 * ts2)
    .opts(
        opts.Curve(
            tools=["box_select", "lasso_select", "hover"],
            width=700,
            height=300,
            line_width=2,
        ),
        opts.Overlay(title="Time Series Plot", xlabel="Date", ylabel="Value"),
    )
    .opts(legend_position="top_left")
)

# Create scatter plot
scatter = hv.Scatter((df["series1"], df["series2"]), "Series 1", "Series 2").opts(
    opts.Scatter(
        tools=["box_select", "lasso_select", "hover"],
        width=400,
        height=400,
        size=8,
        color="navy",
        selection_color="red",
        nonselection_color="lightgray",
        title="Series 1 vs Series 2",
    )
)
# %%
# Link the selections
selection_linker = hv.link_selections.instance()
linked_plots = selection_linker((time_series + scatter).cols(1))

# Apply global options
linked_plots.opts(
    opts.Curve(tools=["box_select", "lasso_select", "hover"]),
    opts.Scatter(tools=["box_select", "lasso_select", "hover"]),
)

# Show the plots
linked_plots

# %%
