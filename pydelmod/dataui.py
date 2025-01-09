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
        pass

    # display related support for tables
    def get_table_columns(self):
        return list(self.get_table_column_width_map().keys())

    def get_table_column_width_map(self):
        pass

    def get_table_filters(self):
        pass

    def get_data(self, df):
        pass

    def create_panel(self, df):
        pass

    def get_tooltips(self):
        pass

    def get_map_color_columns(self):
        """return the columns that can be used to color the map"""
        pass

    def get_name_to_color(self):
        """return a dictionary mapping column names to color names"""
        pass

    def get_map_marker_columns(self):
        """return the columns that can be used to color the map"""
        pass

    def get_name_to_marker(self):
        """return a dictionary mapping column names to marker names"""
        pass


notifications = pn.state.notifications


class DataUI(param.Parameterized):
    """
    Show table of data froma catalog
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
    query = param.String(
        default="",
        doc='Query to filter stations. See <a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html">Pandas Query</a> for details. E.g. max_year <= 2023',
    )

    def __init__(
        self, dataui_manager, crs=ccrs.PlateCarree(), station_id_column=None, **kwargs
    ):
        self.crs = crs
        self.station_id_column = station_id_column
        super().__init__(**kwargs)
        self.dataui_manager = dataui_manager
        self.dfcat = self.dataui_manager.get_data_catalog()
        self.param.map_color_category.objects = (
            self.dataui_manager.get_map_color_columns()
        )
        self.map_color_category = self.param.map_color_category.objects[0]
        self.param.map_marker_category.objects = (
            self.dataui_manager.get_map_marker_columns()
        )
        self.map_marker_category = self.param.map_marker_category.objects[0]
        self.dfmapcat = self._get_map_catalog()

        if isinstance(self.dfcat, gpd.GeoDataFrame):
            self.tmap = gv.tile_sources.CartoLight
            self.build_map_of_features(self.dfmapcat, crs=self.crs)
        else:
            warnings.warn(
                "No geolocation data found in catalog. Not displaying map of stations."
            )

    def _get_map_catalog(self):
        if self.station_id_column and self.station_id_column in self.dfcat.columns:
            dfx = self.dfcat.groupby(self.station_id_column).first().reset_index()
            if isinstance(dfx, gpd.GeoDataFrame):
                dfx = dfx.dropna(subset=["geometry"])
                dfx = dfx.set_crs(self.dfcat.crs)
            else:
                dfx = dfx.dropna(subset=["Latitude", "Longitude"])
            return dfx
        else:
            return self.dfcat

    def build_map_of_features(self, dfmap, crs):
        tooltips = self.dataui_manager.get_tooltips()
        # if station_id column is defined then consolidate the self.dfcat into a single row per station
        # this is useful when we have multiple rows per station
        hover = HoverTool(tooltips=tooltips)
        # check if the dfmap is a geodataframe
        if isinstance(dfmap, gpd.GeoDataFrame):
            geom_type = dfmap.geometry.iloc[0].geom_type
            if geom_type == "Point":
                self.map_features = gv.Points(dfmap, crs=crs)
            elif geom_type == "LineString":
                self.map_features = gv.Path(dfmap, crs=crs)
            elif geom_type == "Polygon":
                self.map_features = gv.Polygons(dfmap, crs=crs)
            else:  # pragma: no cover
                raise "Unknown geometry type " + geom_type
        if self.show_map_colors:
            self.map_features = self.map_features.opts(
                color=dim(self.map_color_category).categorize(
                    self.dataui_manager.get_name_to_color(), default="blue"
                )
            )
        else:
            self.map_features = self.map_features.opts(color="blue")
        if isinstance(self.map_features, gv.Points):
            self.map_features = self.map_features.opts(
                opts.Points(
                    tools=["tap", hover, "lasso_select", "box_select"],
                    nonselection_alpha=0.2,  # nonselection_color='gray',
                    size=10,
                )
            )
        elif isinstance(self.map_features, gv.Path):
            self.map_features = self.map_features.opts(
                opts.Path(
                    tools=["tap", hover, "lasso_select", "box_select"],
                    nonselection_alpha=0.2,  # nonselection_color='gray',
                    line_width=2,
                )
            )
        elif isinstance(self.map_features, gv.Polygons):
            self.map_features = self.map_features.opts(
                opts.Polygons(
                    tools=["tap", hover, "lasso_select", "box_select"],
                    nonselection_alpha=0.2,  # nonselection_color='gray',
                )
            )
        else:
            raise "Unknown map feature type " + str(type(self.map_features))
        self.map_features = self.map_features.opts(
            active_tools=["wheel_zoom"], responsive=True
        )
        if hasattr(self, "station_select"):
            self.station_select.source = self.map_features
        else:
            self.station_select = streams.Selection1D(source=self.map_features)
        return self.map_features

    def update_map_features(
        self, show_color_by, color_by, show_marker_by, marker_by, query
    ):
        query = query.strip()
        dfs = self._get_map_catalog()
        try:
            if len(query) > 0:
                dfs = dfs.query(query)
        except Exception as e:
            str_stack = full_stack()
            logger.error(str_stack)
            notifications.error(
                f"Error while fetching data for {str_stack}", duration=0
            )
        self.map_color_category = color_by
        self.show_map_colors = show_color_by
        self.build_map_of_features(dfs, self.crs)
        self.map_features.data = dfs
        if isinstance(self.map_features, gv.Points):
            if show_marker_by:
                self.map_features = self.map_features.opts(
                    marker=dim(marker_by).categorize(
                        self.dataui_manager.get_name_to_marker(), default="circle"
                    )
                )
            else:
                self.map_features = self.map_features.opts(marker="circle")
        return self.tmap * self.map_features

    def show_data_catalog(self, index=slice(None)):
        # called when map selects stations
        if index == []:
            index = slice(None)
        # select rows from self.dfcat where station_id is in dfs station_ids
        if self.station_id_column and self.station_id_column in self.dfcat.columns:
            dfs = (
                self.map_features.dframe()
                .iloc[index]
                .groupby(self.station_id_column)
                .first()
                .reset_index()
            )
            dfs = self.dfcat[
                self.dfcat[self.station_id_column].isin(dfs[self.station_id_column])
            ]
        else:
            dfs = self.dfcat.iloc[index]
        dfs = dfs[self.dataui_manager.get_table_columns()]
        # return a UI with controls to plot and show data
        return self.update_data_table(dfs)

    def update_data_table(self, dfs):
        if not hasattr(self, "display_table"):
            column_width_map = self.dataui_manager.get_table_column_width_map()
            self.display_table = pn.widgets.Tabulator(
                dfs,
                disabled=True,
                widths=column_width_map,
                show_index=False,
                sizing_mode="stretch_width",
                header_filters=self.dataui_manager.get_table_filters(),
            )

            self.plot_button = pn.widgets.Button(
                name="Plot", button_type="primary", icon="chart-line"
            )
            self.plot_button.on_click(self.update_plots)
            self.download_button = self.create_save_button()
            self.plot_panel = pn.panel(
                hv.Div("<h3>Select rows from table and click on button</h3>"),
                sizing_mode="stretch_both",
            )
            self.plots_panel = pn.Row(self.plot_panel)
            gspec = pn.GridStack(
                sizing_mode="stretch_both", allow_resize=True, allow_drag=False
            )  # ,
            gspec[0, 0:5] = pn.Row(
                self.plot_button,
                self.download_button,
                pn.layout.HSpacer(),
            )
            gspec[1:5, 0:10] = fullscreen.FullScreen(pn.Row(self.display_table))
            gspec[6:15, 0:10] = fullscreen.FullScreen(self.plots_panel)
            self.main_panel = pn.Row(gspec)

        else:
            self.display_table.value = dfs

        return self.main_panel

    def update_plots(self, event):
        try:
            self.plots_panel.loading = True
            # FIXME: needs a PR to panel to fix this. Assured that panel 1.5.x will fix all these issues
            # use self.display_table._index_mapping to map the selection (original indices) to the processed indices
            # for tables with no filters this works
            if True:
                dfselected = self.display_table.value.iloc[self.display_table.selection]
            else:
                dfselected = self.display_table._processed.loc[
                    [
                        self.display_table._index_mapping.get(i, None)
                        for i in self.display_table.selection
                    ]
                ]
            plot_panel = self.dataui_manager.create_panel(dfselected)
            if isinstance(self.plots_panel.objects[0], pn.Tabs):
                tabs = self.plots_panel.objects[0]
                self.tab_count += 1
                tabs.append((str(self.tab_count), plot_panel))
                tabs.active = len(tabs) - 1
            else:
                self.tab_count = 0
                self.plots_panel.objects = [
                    pn.Tabs((str(self.tab_count), plot_panel), closable=True)
                ]
        except Exception as e:
            stack_str = full_stack()
            logger.error(stack_str)
            notifications.error("Error updating plots: " + str(stack_str), duration=0)
        finally:
            self.plots_panel.loading = False

    def download_data(self):
        self.download_button.loading = True
        try:
            dfselected = self.display_table._processed.iloc[
                self.display_table.selection
            ]
            dfdata = self.dataui_manager.get_data(dfselected)
            sio = StringIO()
            dfdata.to_csv(sio)
            sio.seek(0)
            return sio
        except Exception as e:
            notifications.error("Error downloading data: " + str(e), duration=0)
        finally:
            self.download_button.loading = False

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

    def get_about_text(self):
        version = "0.1.0"

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
        control_widgets = self.dataui_manager.get_widgets()
        if hasattr(self, "map_features"):
            map_options = pn.WidgetBox(
                "Map Options",
                self.param.show_map_colors,
                self.param.map_color_category,
                self.param.show_map_markers,
                self.param.map_marker_category,
                self.param.query,
            )
            map_function = pn.bind(
                self.update_map_features,
                show_color_by=self.param.show_map_colors,
                color_by=self.param.map_color_category,
                show_marker_by=self.param.show_map_markers,
                marker_by=self.param.map_marker_category,
                query=self.param.query,
            )
            map_tooltip = pn.widgets.TooltipIcon(
                value="""Map of geographical features. Click on a feature to see data available in the table. <br/>
                See <a href="https://docs.bokeh.org/en/latest/docs/user_guide/interaction/tools.html">Bokeh Tools</a> for toolbar operation"""
            )
            map_view = fullscreen.FullScreen(
                pn.Column(
                    map_function,
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
        if hasattr(self, "station_select"):
            show_data_catalog_bound = pn.bind(
                self.show_data_catalog, index=self.station_select.param.index
            )
        else:
            show_data_catalog_bound = pn.bind(self.show_data_catalog)
        main_view = pn.Column(show_data_catalog_bound)

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
        self.template = template
        return template
