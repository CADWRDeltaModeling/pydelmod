# organize imports by category
import warnings

warnings.filterwarnings("ignore")
#
import pandas as pd
import geopandas as gpd
from io import StringIO

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

pn.extension("tabulator", notifications=True, design="native")
#
from . import fullscreen

from bokeh.models import HoverTool
from bokeh.core.enums import MarkerType

import logging

# setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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


notifications = pn.state.notifications


class DataUI(param.Parameterized):
    """
    Show table of data from a catalog
    Furthermore select the data rows and click on button to display plots for selected rows
    """

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
    map_default_span = param.Number(default=15000, doc="Default span for map zoom")

    query = param.String(
        default="",
        doc='Query to filter stations. See <a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html">Pandas Query</a> for details. E.g. max_year <= 2023',
    )

    def __init__(
        self, dataui_manager, crs=ccrs.PlateCarree(), station_id_column=None, **kwargs
    ):
        self._crs = crs
        self._station_id_column = station_id_column
        super().__init__(**kwargs)
        self._dataui_manager = dataui_manager
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
                dfx = dfx.dropna(subset=["Latitude", "Longitude"])
            return dfx
        else:
            return self._dfcat

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
                    nonselection_alpha=0.2,  # nonselection_color='gray',
                    size=10,
                )
            )
        elif isinstance(self._map_features, gv.Path):
            self._map_features = self._map_features.opts(
                opts.Path(
                    tools=["tap", hover, "lasso_select", "box_select"],
                    nonselection_alpha=0.2,  # nonselection_color='gray',
                    line_width=2,
                )
            )
        elif isinstance(self._map_features, gv.Polygons):
            self._map_features = self._map_features.opts(
                opts.Polygons(
                    tools=["tap", hover, "lasso_select", "box_select"],
                    nonselection_alpha=0.2,  # nonselection_color='gray',
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
        query = query.strip()
        dfs = self._get_map_catalog()
        # select only those rows in dfs that have station_id_column in self.display_table.current_view
        if (
            self._station_id_column
            and self._station_id_column in self.display_table.current_view.columns
        ):
            current_selected = dfs[
                dfs[self._station_id_column].isin(
                    self._dfcat.iloc[selection][self._station_id_column]
                )
            ]
            current_view = dfs[
                dfs[self._station_id_column].isin(
                    self.display_table.current_view[self._station_id_column]
                )
            ]
            current_selection = current_view.index.intersection(current_selected.index)

            # Convert to list of integers
            current_selection = list(map(int, current_selection))
        else:
            current_view = self.display_table.current_view
            current_selected = self.display_table.selected_dataframe
            current_selection = selection
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
        # select rows from self._dfcat where station_id is in dfs station_ids
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

            keep_selected_from_map = table.selected_dataframe[
                table.selected_dataframe[idcol].isin(stations_map_selected)
            ].index

            i_selected_indices = list(
                map(int, self._dfcat.index.get_indexer(current_view_selected_indices))
            )
            selected_indices = i_selected_indices + list(keep_selected_from_map)
        else:
            dfs = self._dfcat.iloc[index]
            current_view = self.display_table.current_view
            selected_indices = current_view.reset_index().merge(dfs)["index"].to_list()
        # with param.discard_events(table.param.selection):
        table.param.update(selection=selected_indices)

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
            configuration={"headerFilterLiveFilterDelay": 00},
        )

        self._plot_button = pn.widgets.Button(
            name="Plot", button_type="primary", icon="chart-line"
        )
        self._plot_button.on_click(self.update_plots)
        self._download_button = self.create_save_button()
        self._plot_panel = pn.panel(
            hv.Div("<h3>Select rows from table and click on button</h3>"),
            sizing_mode="stretch_both",
        )
        self._plots_panel = pn.Row(self._plot_panel)
        gspec = pn.GridStack(
            sizing_mode="stretch_both", allow_resize=True, allow_drag=False
        )  # ,
        self._table_panel = pn.Row(
            self._plot_button,
            self._download_button,
            pn.layout.HSpacer(),
        )
        gspec[0, 0:5] = self._table_panel
        gspec[1:5, 0:10] = fullscreen.FullScreen(pn.Row(self.display_table))
        gspec[6:15, 0:10] = fullscreen.FullScreen(self._plots_panel)
        return gspec

    def update_plots(self, event):
        try:
            self._plots_panel.loading = True
            dfselected = self.display_table.value.iloc[self.display_table.selection]
            plot_panel = self._dataui_manager.create_panel(dfselected)
            if isinstance(self._plots_panel.objects[0], pn.Tabs):
                tabs = self._plots_panel.objects[0]
                self._tab_count += 1
                tabs.append((str(self._tab_count), plot_panel))
                tabs.active = len(tabs) - 1
            else:
                self._tab_count = 0
                self._plots_panel.objects = [
                    pn.Tabs((str(self._tab_count), plot_panel), closable=True)
                ]
        except Exception as e:
            stack_str = full_stack()
            logger.error(stack_str)
            notifications.error("Error updating plots: " + str(stack_str), duration=0)
        finally:
            self._plots_panel.loading = False

    def download_data(self):
        self._download_button.loading = True
        try:
            dfselected = self.display_table.value.iloc[self.display_table.selection]
            dfdata = pd.concat(
                [df for df in self._dataui_manager.get_data(dfselected)], axis=1
            )
            sio = StringIO()
            dfdata.to_csv(sio)
            sio.seek(0)
            return sio
        except Exception as e:
            notifications.error("Error downloading data: " + str(e), duration=0)
        finally:
            self._download_button.loading = False

    def create_save_button(self):
        # add a button to trigger the save function
        return pn.widgets.FileDownload(
            label="Save Data",
            callback=self.download_data,
            filename="data.csv",
            button_type="primary",
            icon="file-download",
            embed=False,
        )

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

    def create_view(self):
        main_panel = self.create_data_table(self._dfcat)
        control_widgets = self._dataui_manager.get_widgets()
        if hasattr(self, "_map_features"):
            map_options = pn.WidgetBox(
                "Map Options",
                self.param.show_map_colors,
                self.param.map_color_category,
                self.param.show_map_markers,
                self.param.map_marker_category,
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
            map_view = fullscreen.FullScreen(
                pn.Column(
                    self._tmap * self._map_function,
                    map_tooltip,
                    min_width=450,
                    min_height=550,
                )
            )
            sidebar_view = pn.Column(
                pn.Tabs(
                    ("Map", map_view),
                    ("Options", control_widgets),
                    ("Map Options", map_options),
                )
            )
        else:
            sidebar_view = pn.Column(pn.Tabs(("Options", control_widgets)))
        main_view = pn.Column(pn.Row(main_panel, sizing_mode="stretch_both"))

        template = pn.template.VanillaTemplate(
            title="Data User Interface",
            sidebar=[sidebar_view],
            sidebar_width=450,
            header_color="lightgray",
        )
        template.main.append(main_view)
        # Adding about button
        template.modal.append(self.get_about_text())
        sidebar_view.append(self.create_about_button(template))
        self._template = template
        return template
