import pandas as pd
import numpy as np
import hvplot.pandas
import holoviews as hv
from holoviews import opts, dim


def cdf(df):
    """Return the cumulative distribution function of a dataframe."""
    df = df.apply(lambda x: x.sort_values().reset_index(drop=True), axis=0)
    df.index = np.linspace(0, 100, len(df))
    return df


# %%
def customize_legend(
    fig, legend_labels, legend_position="upper center", bbox_to_anchor=(0.5, 1.15)
):
    axes = fig.get_axes()[0]
    # put the legend outside the plot at the top centered
    axes.legend(
        legend_labels,
        loc=legend_position,
        bbox_to_anchor=bbox_to_anchor,
        ncols=len(legend_labels),
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
    fig = customize_legend(fig, df.columns, legend_position=legend_position)
    fig = rectangle_around_plot(fig)
    return fig


def save_figure(fig, filename):
    fig.savefig(filename, dpi=300, bbox_inches="tight")
