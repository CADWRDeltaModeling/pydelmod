from .dataui import DataUIManager
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

import pandas as pd

# viz and ui
import holoviews as hv
from holoviews import opts

hv.extension("bokeh")
import param
import panel as pn

pn.extension("tabulator", notifications=True, design="native")
#


def get_colors(stations, dfc):
    """
    Create a dictionary with station names and colors
    """
    return hv.Cycle(list(dfc.loc[stations].values.flatten()))


# from stackoverflow.com https://stackoverflow.com/questions/6086976/how-to-get-a-complete-exception-stack-trace-in-python
def full_stack():
    import traceback, sys

    exc = sys.exc_info()[0]
    stack = traceback.extract_stack()[:-1]  # last one would be full_stack()
    if exc is not None:  # i.e. an exception is present
        del stack[-1]  # remove call of full_stack, the printed exception
        # will contain the caught exception caller instead
    trc = "Traceback (most recent call last):\n"
    stackstr = trc + "".join(traceback.format_list(stack))
    if exc is not None:
        stackstr += "  " + traceback.format_exc().lstrip(trc)
    return stackstr


def get_color_dataframe(stations, color_cycle=hv.Cycle()):
    """
    Create a dataframe with station names and colors
    """
    cc = color_cycle.values
    # extend cc to the size of stations
    while len(cc) < len(stations):
        cc = cc + cc
    dfc = pd.DataFrame({"stations": stations, "color": cc[: len(stations)]})
    dfc.set_index("stations", inplace=True)
    return dfc


def get_colors(stations, dfc):
    """
    Create a dictionary with station names and colors
    """
    return hv.Cycle(list(dfc.loc[stations].values.flatten()))


class TimeSeriesDataUIManager(DataUIManager):
    time_range = param.CalendarDateRange(
        default=(datetime.now() - timedelta(days=10), datetime.now()),
        doc="Time window for data. Default is last 10 days",
    )
    show_legend = param.Boolean(default=True, doc="Show legend")
    legend_position = param.Selector(
        objects=["top_right", "top_left", "bottom_right", "bottom_left"],
        default="top_right",
        doc="Legend position",
    )

    def __init__(self, **params):
        super().__init__(**params)
        self.time_range = self.get_time_range(self.get_data_catalog())

    def get_widgets(self):
        control_widgets = pn.Row(
            pn.Column(
                pn.pane.HTML("Change time range of data to display:"),
                pn.Param(
                    self.param.time_range,
                    widgets={
                        "time_range": {
                            "widget_type": pn.widgets.DatetimeRangeInput,
                            "format": "%Y-%m-%d %H:%M",
                        }
                    },
                ),
                pn.WidgetBox(
                    self.param.show_legend,
                    self.param.legend_position,
                ),
            ),
        )
        return control_widgets

    # data related methods
    def get_data_catalog(self):
        raise NotImplementedError("Method get_data_catalog not implemented")

    def get_station_ids(self, df):
        raise NotImplementedError("Method get_station_ids not implemented")

    def get_time_range(self, dfcat):
        raise NotImplementedError("Method get_time_range not implemented")

    # display related support for tables
    def get_table_columns(self):
        return list(self.get_table_column_width_map().keys())

    def get_table_column_width_map(self):
        raise NotImplementedError("Method get_table_column_width_map not implemented")

    def get_table_filters(self):
        raise NotImplementedError("Method get_table_filters not implemented")

    def create_panel(self, df):
        time_range = self.time_range
        try:
            stationids = self.get_station_ids(df)
            color_df = get_color_dataframe(stationids, hv.Cycle())
            layout_map, station_map, range_map, title_map = self.create_layout(
                df, time_range
            )
            if len(layout_map) == 0:
                return hv.Div("<h3>Select rows from table and click on button</h3>")
            else:
                return (
                    hv.Layout(
                        [
                            hv.Overlay(layout_map[k])
                            .opts(
                                opts.Curve(color=get_colors(station_map[k], color_df))
                            )
                            .opts(
                                opts.Scatter(color=get_colors(station_map[k], color_df))
                            )
                            .opts(
                                show_legend=self.show_legend,
                                legend_position=self.legend_position,
                                ylim=(
                                    tuple(range_map[k])
                                    if range_map[k] is not None
                                    else (None, None)
                                ),
                                title=title_map[k],
                            )
                            for k in layout_map
                        ]
                    )
                    .cols(1)
                    .opts(axiswise=True, sizing_mode="stretch_both")
                )
        except Exception as e:
            stackmsg = full_stack()
            print(stackmsg)
            pn.state.notifications.error(f"Error while fetching data for {e}")
            return hv.Div(f"<h3> Exception while fetching data </h3> <pre>{e}</pre>")

    # methods below if geolocation data is available
    def get_tooltips(self):
        raise NotImplementedError("Method get_tooltips not implemented")

    def get_map_color_category(self):
        raise NotImplementedError("Method get_map_color_category not implemented")
