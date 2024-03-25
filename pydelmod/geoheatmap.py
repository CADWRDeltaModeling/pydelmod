# %%
import numpy as np
import pandas as pd
import geopandas as gpd

# viz stuff
import holoviews as hv
from holoviews import opts, dim
import geoviews as gv
import panel as pn
import hvplot.pandas

pn.extension()


def create_metrics_map(
    dfmetrics,
    title,
    metrics,
    alpha=1,
    scale=1,
    metric="NMSE",
    cmap=None,
):
    return dfmetrics.hvplot.points(
        "utm_easting",
        "utm_northing",
        geo=True,
        tiles="CartoLight",
        crs="EPSG:26910",
        project=True,
        responsive=True,
        c=metric,
        cmap=cmap,
        title=title,
        alpha=alpha,
        hover_cols=[
            "DSM2 ID",
            "Station Name",
        ]
        + metrics,
    ).opts(opts.Points(size=scale * np.sqrt(np.abs(dim(metric)))))


import click


@click.command()
@click.argument("summary_file", type=click.Path(exists=True, readable=True))
@click.argument("station_location_file", type=click.Path(exists=True, readable=True))
@click.option("--metric", default="NMSE", help="name of metric column")
def show_metrics_geo_heatmap(summary_file, station_location_file, metric="NMSE"):
    """
    Create a map of the metrics for the stations in the summary file
    The summary file has the following columns:

    DSM2 Run, Location, R Squared, Mean Error, NMean Error, NMSE, NRMSE, RMSE, NSE, PBIAS, RSR, Mnly Mean Err, Mnly RMSE, Mnly NMean Err, Mnly NRMSE

    The station location file has the following columns:

    DSM2 ID, station_id, station_name, lat, lon, utm_easting, utm_northing

    The Location column is split on the "bpart=" and the first part is taken as the station_id. These are then merged with the station location file on the DSM2_ID column.

    The DSM2 Run column is used to create a map for each run with the NMSE metric. The difference between the first run and the other runs is also calculated and displayed in a difference map

    The map displays circles for each station with the size and color of the circle representing the NMSE metric. The circles are semi-transparent to allow for overlapping circles to be visible.

    Parameters
    ----------
    summary_file : str
        Summary file with the metrics for the stations
    station_location_file : str
        Station location file with the lat, lon, utm_easting, utm_northing
    metric : str, optional
        name of metric column, by default "NMSE"

    Returns
    -------
    Panel
        A panel object with the maps (side by side and difference maps)
    """
    dfs = pd.read_csv(summary_file)
    dfl = pd.read_csv(station_location_file)
    station_id = (
        dfs.Location.str.split("bpart=", expand=True)
        .loc[:, 1]
        .str.split(",", expand=True)[0]
    )
    station_id = station_id.str.replace("'", "")
    dfs["station_id"] = station_id
    df = dfs.merge(dfl, right_on="DSM2 ID", left_on="station_id", how="inner")
    metrics_columns = [
        "R Squared",
        "Mean Error",
        "NMean Error",
        "NMSE",
        "NRMSE",
        "RMSE",
        "NSE",
        "PBIAS",
        "RSR",
        "Mnly Mean Err",
        "Mnly RMSE",
        "Mnly NMean Err",
        "Mnly NRMSE",
    ]

    dsm2_runs = list(df["DSM2 Run"].unique())
    maps = [
        create_metrics_map(
            df.loc[df["DSM2 Run"] == dsm2_run]
            .reset_index()
            .drop_duplicates(subset=["DSM2 ID"], keep="first"),
            dsm2_run,
            metrics_columns,
            metric="NMSE",
        ).opts(title=f"{dsm2_run}: {metric}")
        for dsm2_run in dsm2_runs
    ]
    side_by_side_map = hv.Layout(maps).cols(len(dsm2_runs)).opts(shared_axes=False)
    diffmaps = []
    dfrun0 = (
        df.loc[df["DSM2 Run"] == dsm2_runs[0]]
        .reset_index()
        .drop_duplicates(subset=["DSM2 ID"], keep="first")
    )
    dfrun0 = dfrun0.set_index("DSM2 ID")
    for run in range(1, len(dsm2_runs)):
        dfrun1 = (
            df.loc[df["DSM2 Run"] == dsm2_runs[run]]
            .reset_index()
            .drop_duplicates(subset=["DSM2 ID"], keep="first")
        )
        dfrun1 = dfrun1.set_index("DSM2 ID")
        dfdm = dfrun0.loc[:, metrics_columns] - dfrun1.loc[:, metrics_columns]
        dfdiff = dfrun0.copy()
        dfdiff[metrics_columns] = dfdm[metrics_columns]
        dfdiff = dfdiff.reset_index()
        diffmap = create_metrics_map(
            dfdiff,
            f"Difference {dsm2_runs[0]} - {dsm2_runs[run]}: {metric} [Negative values are better]",
            metrics_columns,
            metric=metric,
            scale=2,
            cmap="bkr",
        )
        diffmaps.append(diffmap)
    diffmap = hv.Layout(diffmaps).cols(len(dsm2_runs) - 1).opts(shared_axes=False)
    pn.Column(side_by_side_map, diffmap).show()
