# %%
import pandas as pd
import numpy as np
import hvplot.pandas

hvplot.extension("matplotlib")
import holoviews as hv
from holoviews import opts

# %%
# create a dataframe with randome values and datetime index
df = pd.DataFrame(
    data=np.random.randn(100, 4),
    index=pd.date_range("1/1/2000", periods=100),
    columns=["var1", "var2", "var3", "var4"],
)


# %%
# create a exceedance plot using hvplot
# An exceedance plot is the probability of a value being exceeded in a dataset.
# The x-axis is the value, and the y-axis is the probability of that value being exceeded.
# The exceedance plot is useful for understanding the distribution of the data.
# The exceedance plot is the complement of the cumulative distribution function (CDF).
def cdf(df):
    df = df.apply(lambda x: x.sort_values().reset_index(drop=True), axis=0)
    df.index = np.linspace(0, 100, len(df))
    return df


# %%
def customize_legend(fig, legend_position="upper center", bbox_to_anchor=(0.5, 1.15)):
    axes = fig.get_axes()[0]
    # put the legend outside the plot at the top centered
    axes.legend(
        df.columns,
        loc=legend_position,
        bbox_to_anchor=bbox_to_anchor,
        ncols=len(df.columns),
    )
    return fig


def rectangle_around_plot(fig, edgecolor="black", facecolor="none"):
    import matplotlib.patches as patches

    # fig.subplots_adjust(left=0.1, right=0.9, wspace=0.4, hspace=0.4)
    rect = patches.Rectangle(
        (0, 0.3),
        1,
        0.4,
        linewidth=1,
        edgecolor=edgecolor,
        facecolor=facecolor,
        transform=fig.transFigure,
        figure=fig,
    )

    # Add the patch to the figure
    fig.add_artist(rect)
    return fig


def exceedance_plot(
    df,
    ylabel,
    xlabel,
    line_styles=[
        (0, (5, 1, 2, 1)),
        "-",
        ":",
        "--",
        "-.",
        ":",
    ],
    legend_position="upper center",
):
    edf = cdf(df)
    line_plot = edf.hvplot.line(
        linestyle=line_styles,
        ylabel=ylabel,
        xlabel=xlabel,
        grid=True,
        legend="top",
    )
    line_plot.opts(fig_inches=5, show_frame=True).opts(opts.Layout(tight=True)).opts(
        backend_opts={
            "legend.frame_on": False,
        }
    )
    fig = hvplot.render(line_plot, backend="matplotlib")
    fig = customize_legend(fig, legend_position=legend_position)
    fig = rectangle_around_plot(fig)
    return fig


def save_figure(fig, filename):
    fig.savefig(filename, dpi=300, bbox_inches="tight")


# %%
# %matplotlib inline # uncomment this line if you want to see it in jupyter notebook
fig = exceedance_plot(df, "Value (units)", "Probability (%)")
# Save the figure to a PNG file
fig.savefig("exceedance_plot.png", dpi=300, bbox_inches="tight")
# %%
