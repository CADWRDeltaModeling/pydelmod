# organize imports by category
import warnings

warnings.filterwarnings("ignore")
#
import pandas as pd
import geopandas as gpd
from io import StringIO
from functools import lru_cache

# viz and ui
import hvplot.pandas  # noqa
import holoviews as hv
from holoviews import opts, dim, streams

hv.extension("bokeh")
import cartopy.crs as ccrs
import geoviews as gv

gv.extension("bokeh")
import param
import panel as pn
from panel.io import location

pn.extension("tabulator", notifications=True, design="native")
#
from . import fullscreen
from .actions import (
    PlotAction,
    PermalinkAction,
    DownloadDataAction,
    DownloadDataCatalogAction,
)

from bokeh.models import HoverTool
from bokeh.core.enums import MarkerType

import logging

import urllib.parse
from .utils import full_stack

# setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DataUIManager(param.Parameterized):

    def get_widgets(self):
        return pn.pane.Markdown("No widgets available")

    # data related methods
    def get_data_catalog(self):
        """return a dataframe or geodataframe with the data catalog"""
        raise NotImplementedError("This method should be implemented by subclasses")

    # display related support for tables
    def get_table_columns(self):
        return list(self.get_table_column_width_map().keys())

    def get_table_column_width_map(self):
        raise NotImplementedError("This method should be implemented by subclasses")

    def get_table_filters(self):
        raise NotImplementedError("This method should be implemented by subclasses")

    def get_data(self, df):
        raise NotImplementedError("This method should be implemented by subclasses")

    def create_panel(self, df):
        raise NotImplementedError("This method should be implemented by subclasses")

    def get_station_ids(self, df):
        return list((df.apply(self.build_station_name, axis=1).astype(str).unique()))

    def build_station_name(self, r):
        raise NotImplementedError("This method should be implemented by subclasses")

    # FIXME: this should not be here
    def append_to_title_map(self, title_map, unit, r):
        raise NotImplementedError("This method should be implemented by subclasses")

    # FIXME: this should not be here
    def create_title(self, title_map, unit, r):
        raise NotImplementedError("This method should be implemented by subclasses")

    @lru_cache(maxsize=128)
    def get_no_selection_message(self):
        """return the message to be displayed when no selection is made"""
        import os

        resource_path = os.path.join(
            os.path.dirname(__file__), "dataui.noselection.html"
        )
        with open(resource_path, "r") as file:
            no_selection_message = file.read()
        return no_selection_message

    def get_tooltips(self):
        raise NotImplementedError("This method should be implemented by subclasses")

    def get_map_color_columns(self):
        """return the columns that can be used to color the map"""
        raise NotImplementedError("This method should be implemented by subclasses")

    def get_name_to_color(self):
        """return a dictionary mapping column names to color names"""
        raise NotImplementedError("This method should be implemented by subclasses")

    def get_map_marker_columns(self):
        """return the columns that can be used to color the map"""
        raise NotImplementedError("This method should be implemented by subclasses")

    def get_name_to_marker(self):
        """return a dictionary mapping column names to marker names"""
        raise NotImplementedError("This method should be implemented by subclasses")

    def get_data_actions(self):
        """Return a list of default data actions."""
        plot_action = PlotAction()
        download_action = DownloadDataAction()
        permalink_action = PermalinkAction()
        download_catalog = DownloadDataCatalogAction()

        plot_button = dict(
            name="Plot",
            button_type="primary",
            icon="chart-line",
            action_type="display",
            callback=plot_action.callback,
        )
        download_button = dict(
            name="Download",
            button_type="primary",
            icon="file-download",
            action_type="download",
            filename="data.csv",
            callback=download_action.callback,
        )
        permalink_button = dict(
            name="Permalink",
            button_type="primary",
            icon="link",
            action_type="link",
            callback=permalink_action.callback,
        )
        download_catalog_button = dict(
            name="Download Catalog",
            button_type="primary",
            icon="file-download",
            action_type="download",
            filename="catalog.csv",
            callback=download_catalog.callback,
        )

        return [plot_button, download_button, permalink_button, download_catalog_button]


notifications = pn.state.notifications


class DataUI(param.Parameterized):
    """
    Show table (and map) of data from a catalog. IIf the catalog manager returns a catalog that is a GeoDataFrame it will display a map of the data.

    Selection on table rows or map will select the corresponding rows in the other view (map or table). It supports 1-to-many mapping of stations to rows in the catalog.

    Actions on the selections are supported via the buttons on the table. These are configurable by the catalog manager.
    """

    view_type = param.Selector(
        objects=["combined", "table", "display"],
        default="combined",
        doc="Type of view to display: combined, table only, or display panel only",
    )

    map_color_category = param.Selector(
        objects=[],
        doc="Options for the map color category selection",
    )
    show_map_colors = param.Boolean(
        default=True, doc="Show map colors for selected category"
    )
    map_marker_category = param.Selector(
        objects=[],
        doc="Options for the map marker category selection",
    )
    show_map_markers = param.Boolean(
        default=False, doc="Show map markers for selected category"
    )
    map_default_span = param.Number(
        default=15000, doc="Default span for map zoom in meters"
    )
    map_non_selection_alpha = param.Number(default=0.2, doc="Non selection alpha")
    map_point_size = param.Number(default=10, doc="Point size for map")

    query = param.String(
        default="",
        doc='Query to filter stations. See <a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html">Pandas Query</a> for details. E.g. max_year <= 2023',
    )
    use_regex_filter = param.Boolean(
        default=False,
        doc="Use regex for table filtering instead of 'like' functionality",
    )

    def __init__(
        self, dataui_manager, crs=ccrs.PlateCarree(), station_id_column=None, **kwargs
    ):
        self._crs = crs
        self._station_id_column = station_id_column
        super().__init__(**kwargs)
        self._dataui_manager = dataui_manager
        self._dataui_manager._dataui = self  # insert a reference to self in the dataui_manager for progress bar updates for example
        self._dfcat = self._dataui_manager.get_data_catalog()
        self.param.map_color_category.objects = (
            self._dataui_manager.get_map_color_columns()
        )
        self.map_color_category = self.param.map_color_category.objects[0]
        self.param.map_marker_category.objects = (
            self._dataui_manager.get_map_marker_columns()
        )
        self.map_marker_category = self.param.map_marker_category.objects[0]
        self._dfmapcat = self._get_map_catalog()

        if isinstance(self._dfcat, gpd.GeoDataFrame):
            self._tmap = gv.tile_sources.CartoLight()
            self.build_map_of_features(self._dfmapcat, crs=self._crs)
            if hasattr(self, "_station_select"):
                self._station_select.source = self._map_features
            else:
                self._station_select = streams.Selection1D(source=self._map_features)
        else:
            warnings.warn(
                "No geolocation data found in catalog. Not displaying map of stations."
            )

    def _get_map_catalog(self):
        if (
            isinstance(self._station_id_column, str)
            and self._station_id_column in self._dfcat.columns
        ):
            dfx = self._dfcat.groupby(self._station_id_column).first().reset_index()
            if isinstance(dfx, gpd.GeoDataFrame):
                dfx = dfx.dropna(subset=["geometry"])
                dfx = dfx.set_crs(self._dfcat.crs)
            else:
                dfx = dfx.dropna(subset=["Latitude", "Longitude"])  # FIXME: ?
        else:
            dfx = self._dfcat
        return dfx

    def build_map_of_features(self, dfmap, crs):
        tooltips = self._dataui_manager.get_tooltips()
        # if station_id column is defined then consolidate the self._dfcat into a single row per station
        # this is useful when we have multiple rows per station
        hover = HoverTool(tooltips=tooltips)
        # check if the dfmap is a geodataframe
        try:
            if isinstance(dfmap, gpd.GeoDataFrame):
                geom_type = dfmap.geometry.iloc[0].geom_type
                if geom_type == "Point":
                    self._map_features = gv.Points(dfmap, crs=crs)
                elif geom_type == "LineString":
                    self._map_features = gv.Path(dfmap, crs=crs)
                elif geom_type == "Polygon":
                    self._map_features = gv.Polygons(dfmap, crs=crs)
                else:  # pragma: no cover
                    raise "Unknown geometry type " + geom_type
        except Exception as e:
            logger.error(f"Error building map of features: {e}")
            self._map_features = gv.Points(dfmap, crs=crs)
        if self.show_map_colors:
            self._map_features = self._map_features.opts(
                color=dim(self.map_color_category).categorize(
                    self._dataui_manager.get_name_to_color(), default="blue"
                )
            )
        else:
            self._map_features = self._map_features.opts(color="blue")
        if isinstance(self._map_features, gv.Points):
            self._map_features = self._map_features.opts(
                opts.Points(
                    tools=["tap", hover, "lasso_select", "box_select"],
                    nonselection_alpha=self.map_non_selection_alpha,  # nonselection_color='gray',
                    size=10,
                )
            )
        elif isinstance(self._map_features, gv.Path):
            self._map_features = self._map_features.opts(
                opts.Path(
                    tools=["tap", hover, "lasso_select", "box_select"],
                    nonselection_alpha=self.map_non_selection_alpha,  # nonselection_color='gray',
                    line_width=2,
                )
            )
        elif isinstance(self._map_features, gv.Polygons):
            self._map_features = self._map_features.opts(
                opts.Polygons(
                    tools=["tap", hover, "lasso_select", "box_select"],
                    nonselection_alpha=self.map_non_selection_alpha,  # nonselection_color='gray',
                )
            )
        else:
            raise "Unknown map feature type " + str(type(self._map_features))
        self._map_features = self._map_features.opts(
            active_tools=["wheel_zoom"], responsive=True
        )
        return self._map_features

    def update_map_features(
        self,
        show_color_by,
        color_by,
        show_marker_by,
        marker_by,
        query,
        filters,
        selection,
    ):
        """Update the map features based on the selection in the table or filters or query. Also updates if the color or marker by columns are changed"""
        query = query.strip()
        dfs = self._get_map_catalog()
        # select only those rows in dfs that have station_id_column in self.display_table.current_view
        if (
            self._station_id_column
            and self._station_id_column in self.display_table.current_view.columns
        ):
            current_view = dfs[
                dfs[self._station_id_column].isin(
                    self.display_table.current_view[self._station_id_column]
                )
            ]
            # if current_view is a geodataframe, keep only valid geometries
            if isinstance(current_view, gpd.GeoDataFrame):
                current_view = current_view.loc[current_view.is_valid]
            current_table_selected = self._dfcat.iloc[selection]
            current_selected = current_view[
                current_view[self._station_id_column].isin(
                    current_table_selected[self._station_id_column]
                )
            ]
        else:
            current_view = dfs.loc[self.display_table.current_view.index]
            if isinstance(current_view, gpd.GeoDataFrame):
                current_view = current_view.loc[current_view.is_valid]
            current_table_selected = self._dfcat.iloc[selection]
            current_selected = current_table_selected
        current_selection = current_view.index.get_indexer(
            current_selected.index
        ).tolist()
        try:
            if len(query) > 0:
                current_view = current_view.query(query)
        except Exception as e:
            str_stack = full_stack()
            logger.error(str_stack)
            notifications.error(
                f"Error while fetching data for {str_stack}", duration=0
            )
        self.map_color_category = color_by
        self.show_map_colors = show_color_by
        self._map_features = self.build_map_of_features(current_view, self._crs)
        if isinstance(self._map_features, gv.Points):
            if show_marker_by:
                self._map_features = self._map_features.opts(
                    marker=dim(marker_by).categorize(
                        self._dataui_manager.get_name_to_marker(), default="circle"
                    )
                )
            else:
                self._map_features = self._map_features.opts(marker="circle")
        with param.discard_events(self._station_select):
            self._map_features = self._map_features.opts(
                default_span=self.map_default_span,  # for max zoom this is the default span in meters
                selected=current_selection,
            )
        return self._map_features

    def select_data_catalog(self, index=[]):
        """Select the rows in the table that correspond to the selected features in the map"""
        if index is None or (len(index) == 1 and index[0] == -1):
            return
        idcol = self._station_id_column
        table = self.display_table
        if idcol and idcol in self._dfcat.columns:
            # get station ids from the _map_features being displayed
            stations_map_selected = (
                self._map_features.dframe().iloc[index][idcol].unique()
            )
            # get the stations selected in table already
            stations_table_selected = table.selected_dataframe[idcol].unique()
            # get stations in stations_map_selected that are not in stations_selected
            stations_to_be_selected = list(
                set(stations_map_selected) - set(stations_table_selected)
            )
            # get the indices of the stations that are not in the selected stations in the current view
            current_view_selected_indices = table.current_view[
                table.current_view[idcol].isin(stations_to_be_selected)
            ].index.to_list()
            # First get the indices of matching rows
            matching_indices = table.selected_dataframe[
                table.selected_dataframe[idcol].isin(stations_map_selected)
            ].index

            # Then convert to integer positions (iloc indices)
            keep_selected_from_map = list(
                map(int, self._dfcat.index.get_indexer(matching_indices))
            )
            i_selected_indices = list(
                map(int, self._dfcat.index.get_indexer(current_view_selected_indices))
            )
            selected_indices = i_selected_indices + list(keep_selected_from_map)
        else:
            dfs = self._map_features.dframe().iloc[index]
            merged_indices = (
                self._dfcat.reset_index().merge(dfs)["index"].to_list()
            )  # index matching
            selected_indices = self._dfcat.index.get_indexer(
                merged_indices
            ).tolist()  # positional indices on table
        # with param.discard_events(table.param.selection):
        table.param.update(selection=selected_indices)

    def create_data_actions(self, actions):
        action_buttons = []
        for action in actions:
            if action["action_type"] == "download":
                # Create a closure that captures the current action
                def create_download_callback(current_action):
                    def _download_callback():
                        sio = current_action["callback"](None, self)
                        # Hide progress when download is initiated
                        import asyncio

                        pn.state.curdoc.add_next_tick_callback(
                            lambda: asyncio.create_task(
                                self._hide_progress_after_delay()
                            )
                        )
                        if sio:
                            return sio
                        else:
                            return None

                    return _download_callback

                # Pass the current action to create a specific callback function for this action
                button = pn.widgets.FileDownload(
                    label=action["name"],
                    callback=create_download_callback(action),
                    filename=action["filename"],
                    button_type=action["button_type"],
                    icon=action["icon"],
                    embed=False,
                )
            else:
                button = pn.widgets.Button(
                    name=action["name"],
                    button_type=action["button_type"],
                    icon=action["icon"],
                )

                # For regular buttons, we can use a function factory to create a proper closure
                def create_click_handler(current_action):
                    def on_click(event):
                        current_action["callback"](event, self)

                    return on_click

                button.on_click(create_click_handler(action))

            action_buttons.append(button)
        return action_buttons

    async def _hide_progress_after_delay(self):
        """Hide the progress bar after a short delay to show completion"""
        import asyncio

        await asyncio.sleep(0.5)
        self.hide_progress()

    @param.depends("use_regex_filter", watch=True)
    def update_data_table_filters(self):
        """Update the table filters based on the use_regex_filter parameter."""
        if self.use_regex_filter:
            # Update filters to use regex
            for column in self.display_table.header_filters:
                # self.display_table.header_filters[column]["type"] = "regex"
                self.display_table.header_filters[column]["func"] = "regex"
        else:
            # Revert to 'like' functionality
            for column in self.display_table.header_filters:
                # self.display_table.header_filters[column]["type"] = "input"
                self.display_table.header_filters[column]["func"] = "like"
        self.display_table.header_filters = self.display_table.header_filters

    def create_data_table(self, dfs):
        column_width_map = self._dataui_manager.get_table_column_width_map()
        dfs = dfs[self._dataui_manager.get_table_columns()]
        self.display_table = pn.widgets.Tabulator(
            dfs,
            disabled=True,
            widths=column_width_map,
            show_index=False,
            sizing_mode="stretch_width",
            header_filters=self._dataui_manager.get_table_filters(),
            page_size=200,
            configuration={
                "headerFilterLiveFilterDelay": 600,
                "columnDefaults": {"tooltip": True},
            },
        )

        self._display_panel = pn.Row(sizing_mode="stretch_both")
        self._action_panel = pn.Row()
        actions = self._dataui_manager.get_data_actions()

        if actions:
            action_buttons = self.create_data_actions(actions)
            self._action_panel.extend(action_buttons)
        self._action_panel.append(pn.layout.HSpacer())
        self._display_panel.append(
            pn.pane.HTML(
                self._dataui_manager.get_no_selection_message(),
                sizing_mode="stretch_both",
            )
        )
        gspec = pn.GridStack(
            sizing_mode="stretch_both", allow_resize=True, allow_drag=False
        )
        gspec[0, 0:5] = self._action_panel
        gspec[1:5, 0:10] = fullscreen.FullScreen(
            pn.Row(self.display_table, scroll=True)
        )
        gspec[6:15, 0:10] = fullscreen.FullScreen(
            pn.Row(self._display_panel, scroll=True)
        )
        self._main_panel = gspec
        return gspec

    def setup_location_sync(self):
        if pn.state.location:
            pn.state.location.param.watch(self.update_view_from_location, "hash")
            self.update_view_from_location()

    def get_version(self):
        try:
            return self._dataui_manager.get_version()
        except Exception as e:
            return "Unknown version"

    def get_about_text(self):
        try:
            return self._dataui_manager.get_about_text()
        except Exception as e:
            version = self.get_version()

            # insert app version with date time of last commit and commit id
            version_string = f"Data UI: {version}"
            about_text = f"""
            ## App version:
            ### {version}

            ## An App to view data using Holoviews and Panel
            """
            return about_text

    def create_about_button(self, template):
        about_btn = pn.widgets.Button(
            name="About App", button_type="primary", icon="info-circle"
        )

        def about_callback(event):
            template.open_modal()

        about_btn.on_click(about_callback)
        return about_btn

    def _create_main_view(self):
        """Create the main view content based on the current view_type"""
        if self.view_type == "table":
            gspec = pn.GridStack(
                sizing_mode="stretch_both", allow_resize=False, allow_drag=False
            )
            gspec[0, 0:5] = self._action_panel
            gspec[1:15, 0:10] = fullscreen.FullScreen(
                pn.Row(self.display_table, scroll=True)
            )
            return gspec
        elif self.view_type == "display":
            gspec = pn.GridStack(
                sizing_mode="stretch_both", allow_resize=False, allow_drag=False
            )
            gspec[0, 0:5] = self._action_panel
            gspec[1:15, 0:10] = fullscreen.FullScreen(
                pn.Row(self._display_panel, scroll=True)
            )
            return gspec
        else:  # combined view
            return pn.Column(
                pn.Row(self._main_panel, sizing_mode="stretch_both", scroll=True),
                sizing_mode="stretch_both",
            )

    def set_progress(self, value):
        """
        Set the progress bar value.

        Parameters:
        -----------
        value : int
            Value between 0-100 for progress percentage, or -1 for indeterminate progress
        """
        self.progress_bar.visible = True
        if value == -1:
            # Set to indeterminate mode
            self.progress_bar.indeterminate = True
        else:
            self.progress_bar.indeterminate = False
            self.progress_bar.value = max(
                0, min(100, value)
            )  # Ensure value is between 0-100

    def hide_progress(self):
        """Hide the progress bar."""
        self.progress_bar.visible = False
        self.progress_bar.value = 0
        self.progress_bar.indeterminate = False

    def show_map_in_display_panel(self, event):
        """Display the map in the display panel area as a closable tab"""
        try:
            # Set progress indicator while loading the map
            self._display_panel.loading = True
            self.set_progress(-1)  # Start indeterminate progress

            # Create a copy of the map for the display panel
            map_display = pn.Column(
                self._tmap * self._map_function,
                min_width=800,
                min_height=600,
                sizing_mode="stretch_both",
            )

            # Show 90% progress
            self.set_progress(90)

            # Check if there are already tabs in the display panel
            if len(self._display_panel.objects) > 0 and isinstance(
                self._display_panel.objects[0], pn.Tabs
            ):
                # Add to existing tabs
                tabs = self._display_panel.objects[0]

                # Initialize tab_count if it doesn't exist
                if not hasattr(self, "_tab_count"):
                    self._tab_count = 0

                self._tab_count += 1
                tabs.append((f"Interactive Map {self._tab_count}", map_display))
                tabs.active = len(tabs) - 1  # Activate the new tab
            else:
                # Create a new tabs panel
                self._tab_count = 1
                self._display_panel.objects = [
                    pn.Tabs(
                        (f"Interactive Map {self._tab_count}", map_display),
                        closable=True,
                        dynamic=True,
                    )
                ]

            # Complete the progress
            self.set_progress(100)
        except Exception as e:
            stack_str = full_stack()
            logger.error(stack_str)
            if pn.state.notifications is not None:
                pn.state.notifications.error(
                    "Error displaying map: " + str(stack_str), duration=0
                )
        finally:
            self._display_panel.loading = False
            # Hide progress after a short delay to show completion
            import asyncio

            pn.state.curdoc.add_next_tick_callback(
                lambda: asyncio.create_task(self._hide_progress_after_delay())
            )

    def create_view_navigation(self):
        """Create navigation buttons for switching between views"""
        nav_buttons = pn.Row(
            pn.widgets.Button(name="Combined", button_type="primary"),
            pn.widgets.Button(name="Table", button_type="primary"),
            pn.widgets.Button(name="Display", button_type="primary"),
        )

        def set_hash(event):
            button_name = event.obj.name.lower().split()[0]
            pn.state.location.hash = f"#{button_name}"

        for btn in nav_buttons:
            btn.on_click(set_hash)

        return nav_buttons

    def update_view_from_location(self, event=None):
        """Update the view based on the URL hash value"""
        hash_value = pn.state.location.hash.lstrip("#")

        # Map hash values to view types
        if hash_value == "table":
            self.view_type = "table"
        elif hash_value == "display":
            self.view_type = "display"
        else:
            self.view_type = "combined"

        # Update the template main content based on the selected view
        if hasattr(self, "_main_view"):
            self._main_view.objects = [self._create_main_view()]

    def create_view(self, title="Data User Interface"):
        main_panel = self.create_data_table(self._dfcat)
        control_widgets = self._dataui_manager.get_widgets()

        # Create progress bar
        self.progress_bar = pn.indicators.Progress(
            name="Progress",
            value=0,
            min_width=400,
            sizing_mode="stretch_width",
            margin=(10, 5, 10, 5),
            bar_color="primary",
            visible=False,
        )

        table_options = pn.WidgetBox(
            "Table Options",
            self.param.use_regex_filter,
        )
        if hasattr(self, "_map_features"):
            map_options = pn.WidgetBox(
                "Map Options",
                self.param.show_map_colors,
                self.param.map_color_category,
                self.param.show_map_markers,
                self.param.map_marker_category,
                self.param.map_default_span,
                self.param.map_non_selection_alpha,
                self.param.map_point_size,
                self.param.query,
            )
            self._map_function = hv.DynamicMap(
                pn.bind(
                    self.update_map_features,
                    show_color_by=self.param.show_map_colors,
                    color_by=self.param.map_color_category,
                    show_marker_by=self.param.show_map_markers,
                    marker_by=self.param.map_marker_category,
                    query=self.param.query,
                    filters=self.display_table.param.filters,
                    selection=self.display_table.param.selection,
                )
            )
            self._station_select.source = self._map_function
            self._station_select.param.watch_values(self.select_data_catalog, "index")
            map_tooltip = pn.widgets.TooltipIcon(
                value="""Map of geographical features. Click on a feature to see data available in the table. <br/>
                See <a href="https://docs.bokeh.org/en/latest/docs/user_guide/interaction/tools.html">Bokeh Tools</a> for toolbar operation"""
            )

            # Create a button to show map in display panel
            map_display_btn = pn.widgets.Button(
                name="Show Map in Display", button_type="primary", icon="map", width=150
            )
            map_display_btn.on_click(self.show_map_in_display_panel)

            map_view = pn.Column(
                pn.Row(map_display_btn, pn.layout.HSpacer(), map_tooltip),
                self._tmap * self._map_function,
                min_width=450,
            )

            sidebar_view = pn.Column(
                pn.Tabs(
                    ("Map", map_view),
                    ("Options", control_widgets),
                    ("Table Options", table_options),
                    ("Map Options", map_options),
                ),
                self.progress_bar,
                sizing_mode="stretch_both",
            )
        else:
            sidebar_view = pn.Column(
                pn.Tabs(("Options", control_widgets), ("Table Options", table_options)),
                self.progress_bar,
            )
        # Create view navigation buttons
        nav_buttons = pn.Row(self.create_view_navigation())

        # Create the initial main view based on URL hash
        self._main_view = pn.Column(
            self._create_main_view(), sizing_mode="stretch_both"
        )

        template = pn.template.VanillaTemplate(
            title=title,
            sidebar=[sidebar_view],
            sidebar_width=450,
            header_color="lightgray",
        )

        # Add navigation before the main content
        about_button = self.create_about_button(template)
        template.header.append(pn.Row(nav_buttons, pn.layout.HSpacer(), about_button))
        template.main.append(self._main_view)

        # Adding about button
        template.modal.append(self.get_about_text())
        # sidebar_view.append(self.create_about_button(template))
        self._template = template

        # finally sync location views
        self.setup_location_sync()

        return template
