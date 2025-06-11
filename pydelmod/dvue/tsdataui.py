from .utils import get_unique_short_names
from .dataui import DataUIManager, full_stack
from datetime import datetime, timedelta
import warnings
from functools import lru_cache
import os

warnings.filterwarnings("ignore")

import pandas as pd

# viz and ui
import holoviews as hv
from holoviews import opts

hv.extension("bokeh")
import param
import panel as pn
import colorcet as cc
from holoviews.plotting.util import process_cmap

pn.extension("tabulator", notifications=True, design="native")
#
LINE_DASH_MAP = ["solid", "dashed", "dotted", "dotdash", "dashdot"]
from vtools.functions.filter import cosine_lanczos


def unique_preserve_order(seq):
    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]


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


@lru_cache
def get_categorical_color_maps():
    cmaps = hv.plotting.util.list_cmaps(
        records=True, category="Categorical", reverse=False
    )
    cmaps = {c.name + "." + c.provider: c for c in cmaps}
    return cmaps


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
    color_cycle_name = param.Selector(
        objects=list(get_categorical_color_maps().keys()),
        default="glasbey_dark.colorcet",
        doc="Color cycle name",
    )
    plot_group_by_column = param.Selector(
        default=None,
        objects=[],
        doc="Column to group plots by. When None, curves are grouped by unit.",
    )
    shared_axes = param.Boolean(default=True, doc="Share axes across plots")
    marker_cycle_column = param.Selector(
        default=None, objects=[], doc="Column to use for marker cycle"
    )
    dashed_line_cycle_column = param.Selector(
        default=None, objects=[], doc="Column to use for dashed line cycle"
    )
    color_cycle_column = param.Selector(
        default=None, objects=[], doc="Column to use for color cycle"
    )

    def __init__(
        self, filename_column="FILE", file_number_column_name="FILE_NUM", **params
    ):
        # modify catalog if filename_column is present to include file number if multiple files are present
        catalog = self.get_data_catalog()
        self.change_color_cycle()
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
        table_columns = list(self.get_table_columns())
        # Add blank (None) option at the start
        columns_with_blank = [None] + table_columns
        self.param.marker_cycle_column.objects = columns_with_blank
        self.param.dashed_line_cycle_column.objects = columns_with_blank
        self.param.color_cycle_column.objects = columns_with_blank
        self.param.plot_group_by_column.objects = columns_with_blank

    def get_data_catalog(self):
        raise NotImplementedError("Method get_data_catalog not implemented")

    def get_time_range(self, dfcat):
        raise NotImplementedError("Method get_time_range not implemented")

    def get_table_filters(self):
        raise NotImplementedError("Method get_table_filters not implemented")

    def is_irregular(self, r):
        raise NotImplementedError("Method is_irregular not implemented")

    def get_data_for_time_range(self, r, time_range):
        raise NotImplementedError("Method get_data_for_time_range not implemented")

    def get_tooltips(self):
        raise NotImplementedError("Method get_tooltips not implemented")

    def create_curve(self, data, r, unit, file_index=""):
        raise NotImplementedError("Method create_curve not implemented")

    # methods below if geolocation data is available

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
        # from bokeh.core.enums import MarkerType
        # list(MarkerType) -> ['asterisk', 'circle', 'circle_cross', 'circle_dot', 'circle_x', 'circle_y', 'cross', 'dash', 'diamond', 'diamond_cross', 'diamond_dot', 'dot', 'hex', 'hex_dot', 'inverted_triangle', 'plus', 'square', 'square_cross', 'square_dot', 'square_pin', 'square_x', 'star', 'star_dot', 'triangle', 'triangle_dot', 'triangle_pin', 'x', 'y']
        return [
            "circle",
            "triangle",
            "square",
            "diamond",
            "cross",
            "x",
            "star",
            "plus",
            "dot",
            "hex",
            "inverted_triangle",
            "asterisk",
            "circle_cross",
            "square_cross",
            "diamond_cross",
            "circle_dot",
            "square_dot",
            "diamond_dot",
            "star_dot",
            "hex_dot",
            "triangle_dot",
            "circle_x",
            "square_x",
            "circle_y",
            "y",
            "dash",
            "square_pin",
            "triangle_pin",
        ]

    @param.depends("color_cycle_name", watch=True)
    def change_color_cycle(self):
        cmapinfo = get_categorical_color_maps()[self.color_cycle_name]
        color_list = unique_preserve_order(
            process_cmap(cmapinfo.name, provider=cmapinfo.provider)
        )
        self.color_cycle = hv.Cycle(color_list)

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
        )
        plot_widgets = pn.Column(
            pn.WidgetBox(
                self.param.show_legend,
                self.param.legend_position,
            ),
            pn.WidgetBox(
                self.param.irregular_curve_connection,
                self.param.regular_curve_connection,
            ),
            pn.WidgetBox(
                pn.pane.Markdown("**Group and Style Options:**"),
                self.param.plot_group_by_column,  # Option for grouping plots
                self.param.color_cycle_column,  # Group related options together
                self.param.dashed_line_cycle_column,
                self.param.marker_cycle_column,
            ),
            self.param.color_cycle_name,
            self.param.shared_axes,  # Add checkbox for shared_axes
        )
        transform_widgets = pn.Column(
            self.param.fill_gap,
            self.param.do_tidal_filter,
            pn.Row(
                self.param.sensible_range_yaxis, self.param.sensible_percentile_range
            ),
        )
        widget_tabs = pn.Tabs(
            ("Time", control_widgets),
            ("Plot", plot_widgets),
            ("Transform", transform_widgets),
        )
        return widget_tabs

    def get_data(self, df):
        # Start with 0 progress
        # Get the DataUI instance from the caller
        dataui = self._dataui if hasattr(self, "_dataui") else None
        if dataui:
            dataui.set_progress(0)

        # Calculate progress increment per row
        total_rows = len(df)
        if total_rows == 0:  # Avoid division by zero
            return

        progress_per_row = 50 / total_rows  # We'll use 0-50% range for the iteration

        # Process each row, updating progress as we go
        for i, (_, r) in enumerate(df.iterrows()):
            data, _, _ = self.get_data_for_time_range(r, self.time_range)

            # Update progress - scale from 0 to 50%
            if dataui:
                current_progress = int(progress_per_row * (i + 1))
                dataui.set_progress(current_progress)

            yield data

        # After completing all rows, ensure progress is at 50%
        if dataui:
            dataui.set_progress(50)

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

    def get_color_style_mapping(self, unique_values):
        """
        Map unique values to colors.
        """
        color_df = get_color_dataframe(unique_values, self.color_cycle)
        return {
            value: color_df.loc[value].values.flatten()[0] for value in unique_values
        }

    def get_line_style_mapping(self, unique_values):
        """
        Map unique values to line dash styles.
        """
        return {
            value: LINE_DASH_MAP[i % len(LINE_DASH_MAP)]
            for i, value in enumerate(unique_values)
        }

    def get_marker_style_mapping(self, unique_values):
        """
        Map unique values to marker styles.
        """
        from bokeh.core.enums import MarkerType

        marker_types = [None] + list(MarkerType)
        return {
            value: marker_types[i % len(marker_types)]
            for i, value in enumerate(unique_values)
        }

    def _process_curve_data(self, data, r, time_range):
        """Process time series data based on index type and apply transformations."""
        if isinstance(data.index, pd.PeriodIndex):
            data = data[
                (data.index.start_time >= time_range[0])
                & (data.index.end_time <= time_range[1])
            ]
        else:  # Assume DatetimeIndex
            data = data[(data.index >= time_range[0]) & (data.index <= time_range[1])]

        # Apply optional data transformations
        if self.fill_gap > 0:
            data = data.interpolate(limit=self.fill_gap)
        if self.do_tidal_filter and not self.is_irregular(r):
            data = cosine_lanczos(data, "40h")

        return data

    def _add_curve_to_layout(
        self,
        layout_map,
        station_map,
        title_map,
        range_map,
        curve,
        row,
        unit,
        station_name,
        group_key=None,
    ):
        """Add a curve to the layout maps using the specified group key.

        Args:
            layout_map: Dictionary mapping group keys to lists of (curve, row) tuples
            station_map: Dictionary mapping group keys to lists of station names
            title_map: Dictionary mapping group keys to title information
            range_map: Dictionary mapping group keys to y-axis ranges
            curve: The curve to add
            row: The data row
            unit: The unit of measure (used as default group key if group_key is None)
            station_name: The name of the station
            group_key: The key to group by (defaults to unit if None)
        """
        # Use unit as the default group key if group_key is None
        group_key = group_key if group_key is not None else unit

        if group_key not in layout_map:
            layout_map[group_key] = []
            range_map[group_key] = None
            station_map[group_key] = []

        layout_map[group_key].append((curve, row))
        station_map[group_key].append(station_name)
        self.append_to_title_map(title_map, group_key, row)

    def create_layout(self, df, time_range):
        """
        Create layout maps for visualizing time series data.

        Groups curves based on plot_group_by_column if specified, otherwise by unit.

        Returns:
            tuple: (layout_map, station_map, range_map, title_map)
                layout_map: Dictionary mapping group keys to lists of (curve, row) tuples
                station_map: Dictionary mapping group keys to lists of station names
                range_map: Dictionary mapping group keys to y-axis ranges
                title_map: Dictionary mapping group keys to title information
        """
        layout_map = {}
        title_map = {}
        range_map = {}
        station_map = {}  # list of stations for each unit

        # Prepare file index mapping if needed
        file_index_map = {}
        if self.display_fileno:
            local_unique_files = df[self.filename_column].unique()
            short_unique_files = get_unique_short_names(local_unique_files)
            file_index_map = dict(zip(local_unique_files, short_unique_files))

        # Setup progress tracking
        dataui = self._dataui if hasattr(self, "_dataui") else None
        if dataui:
            dataui.set_progress(50)

        # Calculate progress increment
        total_rows = len(df)
        progress_per_row = 40 / max(
            total_rows, 1
        )  # We'll use 50-90% range for the iteration

        # Process each row
        for i, (_, row) in enumerate(df.iterrows()):
            try:
                # Get and process data
                data, unit, _ = self.get_data_for_time_range(row, time_range)
                unit = unit.lower()  # lowercase the units
                data = self._process_curve_data(data, row, time_range)

                # Create curve
                file_index = (
                    file_index_map.get(row[self.filename_column], "")
                    if self.display_fileno
                    else ""
                )
                curve = self.create_curve(data, row, unit, file_index=file_index)

                # Add curve to layout
                station_name = self.build_station_name(row)

                # Determine group key based on plot_group_by_column
                group_key = None
                if self.plot_group_by_column:
                    # Use the value from the column specified by plot_group_by_column
                    # if that column exists in the row's data
                    if self.plot_group_by_column in row:
                        group_value = row[self.plot_group_by_column]
                        if group_value is not None and str(group_value).strip() != "":
                            group_key = str(group_value)

                self._add_curve_to_layout(
                    layout_map,
                    station_map,
                    title_map,
                    range_map,
                    curve,
                    row,
                    unit,
                    station_name,
                    group_key=group_key,
                )

            except Exception as e:
                print(full_stack())
                if pn.state.notifications:
                    pn.state.notifications.error(f"Error processing row: {row}: {e}")

            # Update progress
            if dataui and total_rows > 0:
                current_progress = 50 + int(progress_per_row * (i + 1))
                dataui.set_progress(current_progress)

        # Post-processing
        title_map = self._update_title_for_custom_grouping(title_map)

        # Calculate y-axis ranges if needed
        if self.sensible_range_yaxis:
            for unit, curves in layout_map.items():
                for curve, _ in curves:
                    range_map[unit] = self._calculate_range(range_map[unit], curve.data)

        # Finalize progress
        if dataui:
            dataui.set_progress(90)

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

    def _prepare_style_maps(self, df):
        """Prepare color, line style, and marker style mappings."""
        style_maps = {"color": None, "line": None, "marker": None}

        # Color map
        if self.color_cycle_column:
            color_values = df[self.color_cycle_column].unique()
            style_maps["color"] = self.get_color_style_mapping(color_values)

        # Line style map
        if self.dashed_line_cycle_column:
            line_style_values = df[self.dashed_line_cycle_column].unique()
            style_maps["line"] = self.get_line_style_mapping(line_style_values)

        # Marker map
        if self.marker_cycle_column:
            marker_values = df[self.marker_cycle_column].unique()
            style_maps["marker"] = self.get_marker_style_mapping(marker_values)

        return style_maps

    def _calculate_has_duplicates(self, curves_data):
        """Check if there are duplicate station names in the curves data."""
        # If no color cycle column is specified, return False
        if not self.color_cycle_column:
            return False

        try:
            station_names = []
            for i, (_, row) in enumerate(curves_data):
                if self.color_cycle_column in row:
                    station_names.append(row[self.color_cycle_column])
                else:
                    # If missing, use index to avoid duplicates
                    station_names.append(f"curve_{i}")
            return len(station_names) != len(set(station_names))
        except Exception as e:
            # Fallback to avoid breaking the app
            print(f"Error in _calculate_has_duplicates: {e}")
            return False

    def _get_style_combinations(self, stations, curves_data, style_maps):
        """
        Determine which color and line style combinations exist within a unit.

        Args:
            stations: List of station names
            curves_data: List of (curve, row) tuples
            style_maps: Dictionary of style mappings

        Returns:
            tuple: (combinations_dict, has_duplicates, has_style_duplicates)
        """
        has_duplicates = self._calculate_has_duplicates(curves_data)
        color_map, line_map = style_maps["color"], style_maps["line"]

        # First pass to collect color + line style combinations
        combinations = {}
        for i, (_, row) in enumerate(curves_data):
            color_val = (
                row[self.color_cycle_column]
                if color_map and self.color_cycle_column
                else None
            )
            line_style_val = (
                row[self.dashed_line_cycle_column]
                if line_map and self.dashed_line_cycle_column and has_duplicates
                else None
            )

            combo_key = (color_val, line_style_val)
            if combo_key not in combinations:
                combinations[combo_key] = []
            combinations[combo_key].append(i)

        # Check for duplicate combinations
        has_style_duplicates = any(
            len(indices) > 1 for indices in combinations.values()
        )

        return combinations, has_duplicates, has_style_duplicates

    def _apply_curve_styling(
        self,
        curve,
        row,
        has_duplicates,
        has_style_duplicates,
        style_maps,
        style_combinations,
    ):
        """
        Apply styling options to a curve based on context and available styles.

        The logic ensures markers are only used when there are multiple curves
        with the same color and line style combination in the layout.
        """
        color_map, line_map, marker_map = (
            style_maps["color"],
            style_maps["line"],
            style_maps["marker"],
        )

        # Base styling options
        curve_opts = {}

        # Apply color
        if color_map and self.color_cycle_column:
            curve_opts["color"] = color_map.get(row[self.color_cycle_column], "black")

        # Apply line style if needed
        if has_duplicates and line_map and self.dashed_line_cycle_column:
            curve_opts["line_dash"] = line_map.get(
                row[self.dashed_line_cycle_column], "solid"
            )

        # Apply basic styling
        styled_curve = curve.opts(opts.Curve(**curve_opts))

        # Add markers only when there are multiple curves with the same color and line style
        if marker_map and self.marker_cycle_column:
            # Get the combo key for this curve
            current_color = (
                row[self.color_cycle_column] if self.color_cycle_column else None
            )
            current_line_style = (
                row[self.dashed_line_cycle_column]
                if has_duplicates and self.dashed_line_cycle_column
                else None
            )
            combo_key = (current_color, current_line_style)

            # Only add markers if this specific style combination appears multiple times
            if (
                combo_key in style_combinations
                and len(style_combinations[combo_key]) > 1
            ):
                marker_style = marker_map.get(row[self.marker_cycle_column], None)
                if marker_style is not None:
                    scatter = hv.Scatter(curve.data, label=curve.label).opts(
                        opts.Scatter(
                            marker=marker_style,
                            size=5,
                            color=curve_opts.get("color", "black"),
                        )
                    )
                    styled_curve = styled_curve * scatter

        return styled_curve

    def create_panel(self, df):
        """
        Create visualization panel from the data.

        This method applies styling to curves based on selected attributes and
        arranges them in overlays and layouts.
        """
        time_range = self.time_range
        try:
            # Get station IDs and prepare color dataframe
            stationids = self.get_station_ids(df)
            color_df = get_color_dataframe(stationids, self.color_cycle)

            # Prepare style mappings (color, line style, marker)
            style_maps = self._prepare_style_maps(df)

            # Create data layout
            layout_map, station_map, range_map, title_map = self.create_layout(
                df, time_range
            )

            if len(layout_map) == 0:
                return hv.Div(self.get_no_selection_message()).opts(
                    sizing_mode="stretch_both"
                )

            # Build visualization layout
            overlays = []
            for group_key, curves_data in layout_map.items():
                stations = station_map[group_key]

                # Analyze style combinations
                style_combinations, has_duplicates, has_style_duplicates = (
                    self._get_style_combinations(stations, curves_data, style_maps)
                )

                # Apply styling to each curve
                styled_curves = []
                for i, (curve, row) in enumerate(curves_data):
                    styled_curve = self._apply_curve_styling(
                        curve,
                        row,
                        has_duplicates,
                        has_style_duplicates,
                        style_maps,
                        style_combinations,
                    )
                    styled_curves.append(styled_curve)

                # Create overlay for this group
                overlay = hv.Overlay(styled_curves).opts(
                    show_legend=self.show_legend,
                    legend_position=self.legend_position,
                    ylim=(
                        tuple(range_map[group_key])
                        if range_map[group_key] is not None
                        else (None, None)
                    ),
                    title=title_map[group_key],
                    min_height=400,
                )
                overlays.append(overlay)

            # Return final layout
            return (
                hv.Layout(overlays)
                .cols(1)
                .opts(
                    shared_axes=self.shared_axes,
                    axiswise=True,
                    sizing_mode="stretch_both",
                )
            )
        except Exception as e:
            stackmsg = full_stack()
            print(stackmsg)
            pn.state.notifications.error(f"Error while creating panel: {e}")
            return hv.Div(f"<h3> Exception while creating panel </h3> <pre>{e}</pre>")

    def _update_title_for_custom_grouping(self, title_map):
        """
        Update title map when custom grouping is used.

        This method adds grouping information to titles when a custom plot_group_by_column
        is used instead of the default unit-based grouping.

        Args:
            title_map: The title map to update

        Returns:
            Updated title map with grouping information
        """
        # Process each key-value pair to create titles
        processed_titles = {}
        for group_key, title_info in title_map.items():
            base_title = self.create_title(title_info)

            # When using custom grouping, add the group info and column name
            if self.plot_group_by_column:
                column_name = self.plot_group_by_column
                if str(group_key) != base_title:  # Avoid redundancy
                    title = f"{column_name}: {group_key} - {base_title}"
                else:
                    title = f"{column_name}: {group_key}"
            else:
                # No custom grouping, use base title
                title = base_title

            processed_titles[group_key] = title

        return processed_titles
