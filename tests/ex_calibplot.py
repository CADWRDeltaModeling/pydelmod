# %%
import pandas as pd
import numpy as np
from pydelmod import calibplot
import hvplot.pandas
import holoviews as hv

hv.extension("bokeh")
import logging

# %%
# create a time series dataframe
n = 100
df1 = pd.DataFrame(
    np.random.randn(n, 1),
    columns=["series1"],
    index=pd.date_range("1/1/2018", periods=n),
)
df2 = pd.DataFrame(
    np.random.randn(n, 1),
    columns=["series2"],
    index=pd.date_range("1/1/2018", periods=n),
)
dflist = [df1, df2]
names = ["series1", "series2"]


# %%
def tsplot(dflist, names, timewindow=None, zoom_inst_plot=False):
    """Time series overlay plots

    Handles missing DataFrame, just put None in the list

    Args:
        dflist (List): Time-indexed DataFrame list
        names (List): Names list (same size as dflist)
        timewindow (str, optional): time window for plot. Must be in format: 'YYYY-MM-DD:YYYY-MM-DD'
        zoom_inst_plot (bool): if true, display only data in timewindow for plot
    Returns:
        Overlay: Overlay of Curve
    """
    start_dt = None
    end_dt = None
    if dflist[0] is not None:
        start_dt = dflist[0].index.min()
        end_dt = dflist[0].index.max()
    if zoom_inst_plot and (timewindow is not None):
        try:
            parts = timewindow.split(":")
            start_dt = parts[0]
            end_dt = parts[1]
        except:
            errmsg = "error in calibplot.tsplot"
            print(errmsg)
            logging.error(errmsg)
    # This doesn't work. Need to find a way to get this working.
    # plt = [df[start_dt:end_dt].hvplot(label=name, x_range=(timewindow)) if df is not None else hv.Curve(None, label=name)

    from bokeh.models import DatetimeTickFormatter

    plt = [
        (
            df[start_dt:end_dt].hvplot(
                label=name,
                xformatter=DatetimeTickFormatter(
                    years="%b-%Y", months="%b-%y", days="%d-%b-%y"
                ),
            )
            if df is not None
            else hv.Curve(None, label=name)
        )
        for df, name in zip(dflist, names)
    ]
    plt = [(c.redim(**{c.vdims[0].name: c.label}) if c.name != "" else c) for c in plt]
    return hv.Overlay(plt)


# %%
tsplot(dflist, names)
# %%
calibplot.tsplot(dflist, names)
# %%
