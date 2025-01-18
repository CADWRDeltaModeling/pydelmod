from .dataui import DataUIManager, full_stack
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
import colorcet as cc

pn.extension("tabulator", notifications=True, design="native")
#
LINE_DASH_MAP = ["solid", "dashed", "dotted", "dotdash", "dashdot"]
from vtools.functions.filter import cosine_lanczos


def get_colors(stations, dfc):
    """
    Create a dictionary with station names and colors
    """
    return hv.Cycle(list(dfc.loc[stations].values.flatten()))


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
        default=None,
        doc="Time window for data. If None, all data is displayed. Format: (start, end)",
    )
    show_legend = param.Boolean(default=True, doc="Show legend")
    legend_position = param.Selector(
        objects=["top_right", "top_left", "bottom_right", "bottom_left"],
        default="top_right",
        doc="Legend position",
    )
    fill_gap = param.Integer(
        default=0, doc="Fill gaps in data upto this limit, only when a positive integer"
    )
    do_tidal_filter = param.Boolean(default=False, doc="Apply tidal filter")
    irregular_curve_connection = param.Selector(
        objects=["steps-post", "steps-pre", "steps-mid", "linear"],
        default="steps-post",
        doc="Curve connection method for irregular data",
    )
    regular_curve_connection = param.Selector(
        objects=["linear", "steps-pre", "steps-post", "steps-mid"],
        default="steps-pre",
        doc="Curve connection method for regular period type data",
    )
    sensible_range_yaxis = param.Boolean(
        default=False,
        doc="Sensible range (in percentile) or auto range for y axis",
    )
    sensible_percentile_range = param.Range(
        default=(0.01, 0.99), bounds=(0, 1), step=0.01, doc="Percentile range"
    )
    file_number_column_name = param.String(default="FILE_NUM")

    def __init__(
        self, filename_column="FILE", file_number_column_name="FILE_NUM", **params
    ):
        self.color_cycle = hv.Cycle(cc.glasbey_dark)
        # modify catalog if filename_column is present to include file number if multiple files are present
        catalog = self.get_data_catalog()
        self.filename_column = filename_column
        self.file_number_column_name = file_number_column_name
        self.display_fileno = False
        if self.filename_column in catalog.columns:
            unique_files = catalog[self.filename_column].unique()
            if len(unique_files) > 1:
                catalog[self.file_number_column_name] = catalog[
                    self.filename_column
                ].apply(lambda x: unique_files.tolist().index(x))
                self.display_fileno = True
        self.time_range = self.get_time_range(self.get_data_catalog())
        super().__init__(**params)

    def get_widgets(self):
        control_widgets = pn.Column(
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
            self.param.fill_gap,
            self.param.do_tidal_filter,
            pn.Row(
                self.param.sensible_range_yaxis, self.param.sensible_percentile_range
            ),
            self.param.irregular_curve_connection,
            self.param.regular_curve_connection,
        )

        return control_widgets

    # data related methods
    def get_data_catalog(self):
        raise NotImplementedError("Method get_data_catalog not implemented")

    def _get_station_ids(self, df):
        raise NotImplementedError("Method get_station_ids not implemented")

    def get_time_range(self, dfcat):
        raise NotImplementedError("Method get_time_range not implemented")

    def get_data(self, df):
        for _, r in df.iterrows():
            data, _, _ = self._get_data_for_time_range(r, self.time_range)
            yield data

    # display related support for tables
    def get_table_columns(self):
        return list(self.get_table_column_width_map().keys())

    def get_table_width_sum(self, column_width_map):
        width = 0
        for k, v in column_width_map.items():
            width += float(v[:-1])  # drop % sign
        return width

    def adjust_column_width(self, column_width_map, max_width=100):
        width_sum = self.get_table_width_sum(column_width_map)
        if width_sum > max_width:
            for k, v in column_width_map.items():
                column_width_map[k] = f"{(float(v[:-1]) / width_sum) * max_width}%"
        return column_width_map

    def get_table_column_width_map(self):
        column_width_map = self._get_table_column_width_map()
        column_width_map[self.filename_column] = "10%"
        if self.display_fileno:
            column_width_map[self.file_number_column_name] = "5%"
        self.adjust_column_width(column_width_map)
        return column_width_map

        raise NotImplementedError("Method get_table_column_width_map not implemented")

    def get_table_filters(self):
        raise NotImplementedError("Method get_table_filters not implemented")

    def is_irregular(self, r):
        raise NotImplementedError("Method is_irregular not implemented")

    def create_layout(self, df, time_range):
        layout_map = {}
        title_map = {}
        range_map = {}
        station_map = {}  # list of stations for each unit
        if self.display_fileno:
            local_unique_files = df[self.filename_column].unique()
        for _, r in df.iterrows():
            try:
                data, unit, _ = self._get_data_for_time_range(r, time_range)
                if isinstance(data.index, pd.PeriodIndex):
                    data = data[
                        (data.index.start_time >= time_range[0])
                        & (data.index.end_time <= time_range[1])
                    ]
                else:  # Assume DatetimeIndex
                    data = data[
                        (data.index >= time_range[0]) & (data.index <= time_range[1])
                    ]
                if self.fill_gap > 0:
                    data = data.interpolate(limit=self.fill_gap)
                if self.do_tidal_filter and not self.is_irregular(r):
                    data = cosine_lanczos(data, "40h")
            except Exception as e:
                print(full_stack())
                if pn.state.notifications:
                    pn.state.notifications.error(
                        f"Error while fetching data for row: {r}: {e}"
                    )
                data = pd.DataFrame(columns=["value"], dtype=float)
                unit = "X"
            file_index = r[self.file_number_column_name] if self.display_fileno else ""
            crv = self._create_crv(
                data,
                r,
                unit,
                file_index=file_index,
            )
            if isinstance(data.index, pd.PeriodIndex):
                crv.opts(opts.Curve(interpolation=self.regular_curve_connection))
            if self.is_irregular(r):
                crv.opts(opts.Curve(interpolation=self.irregular_curve_connection))

            if unit not in layout_map:
                layout_map[unit] = []
                range_map[unit] = None
                station_map[unit] = []
            layout_map[unit].append(crv)
            station_map[unit].append(self.build_station_name(r))
            self._append_to_title_map(title_map, unit, r)
        title_map = {k: self._create_title(v) for k, v in title_map.items()}
        if self.sensible_range_yaxis:
            for unit in layout_map.keys():
                for crv in layout_map[unit]:
                    range_map[unit] = self._calculate_range(range_map[unit], crv.data)
        return layout_map, station_map, range_map, title_map

    def _calculate_range(self, current_range, df, factor=0.0):
        if df.empty:
            return current_range
        else:
            new_range = (
                df.iloc[:, 0].quantile(list(self.sensible_percentile_range)).values
            )
            scaleval = new_range[1] - new_range[0]
            new_range = [
                new_range[0] - scaleval * factor,
                new_range[1] + scaleval * factor,
            ]
        if current_range is not None:
            new_range = [
                min(current_range[0], new_range[0]),
                max(current_range[1], new_range[1]),
            ]
        return new_range

    def create_panel(self, df):
        time_range = self.time_range
        try:
            stationids = self._get_station_ids(df)
            color_df = get_color_dataframe(stationids, self.color_cycle)
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
                                min_height=400,
                            )
                            for k in layout_map
                        ]
                    )
                    .cols(1)
                    .opts(
                        shared_axes=False,
                        axiswise=True,
                        sizing_mode="stretch_both",
                    )
                )
        except Exception as e:
            stackmsg = full_stack()
            print(stackmsg)
            pn.state.notifications.error(f"Error while fetching data for {e}")
            return hv.Div(f"<h3> Exception while fetching data </h3> <pre>{e}</pre>")

    # methods below if geolocation data is available
    def get_tooltips(self):
        raise NotImplementedError("Method get_tooltips not implemented")

    def get_map_color_columns(self):
        """return the columns that can be used to color the map"""
        pass

    def get_name_to_color(self):
        """return a dictionary mapping column names to color names"""
        return hv.Cycle("Category10").values

    def get_map_marker_columns(self):
        """return the columns that can be used to color the map"""
        pass

    def get_name_to_marker(self):
        """return a dictionary mapping column names to marker names"""
        from bokeh.core.enums import MarkerType

        return list(MarkerType)
