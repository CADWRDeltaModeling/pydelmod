# %%
# organize imports by category
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")
#
import pandas as pd
import geopandas as gpd

# viz and ui
import holoviews as hv
from holoviews import opts, dim, streams

hv.extension("bokeh")
from cartopy.crs import GOOGLE_MERCATOR
import geoviews as gv

gv.extension("bokeh")
import param
import panel as pn

pn.extension("tabulator", notifications=True, design="native")
#
from vtools.functions.filter import godin, cosine_lanczos

from . import fullscreen


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


from bokeh.models import HoverTool
from bokeh.core.enums import MarkerType


class DataUIManager(param.Parameterized):
    show_legend = param.Boolean(default=True, doc="Show legend")
    legend_position = param.Selector(
        objects=["top_right", "top_left", "bottom_right", "bottom_left"],
        default="top_right",
        doc="Legend position",
    )

    def get_widgets(self):
        return pn.WidgetBox(
            self.param.show_legend,
            self.param.legend_position,
        )

    # data related methods
    def get_data_catalog(self):
        pass

    def get_station_ids(self, df):
        pass

    def get_time_range(self, dfcat):
        pass

    # display related support for tables
    def get_table_columns(self):
        return list(self.get_table_column_width_map().keys())

    def get_table_column_width_map(self):
        pass

    def get_table_filters(self):
        pass

    # display related methods for plots
    def create_layout(self, df, time_range):
        pass

    def create_panel(self, df, time_range):
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
        pass

    def get_map_color_category(self):
        pass


class DataUI(param.Parameterized):
    """
    Show table of data froma catalog
    Furthermore select the data rows and click on button to display plots for selected rows
    """

    time_range = param.CalendarDateRange(
        default=(datetime.now() - timedelta(days=10), datetime.now()),
        doc="Time window for data. Default is last 10 days",
    )

    def __init__(self, dataui_manager, **kwargs):
        super().__init__(**kwargs)
        self.dataui_manager = dataui_manager
        self.dfcat = self.dataui_manager.get_data_catalog()
        self.time_range = self.dataui_manager.get_time_range(self.dfcat)
        if isinstance(self.dfcat, gpd.GeoDataFrame):
            self.tmap = gv.tile_sources.CartoLight
            tooltips = self.dataui_manager.get_tooltips()

            map_color_category = self.dataui_manager.get_map_color_category()
            hover = HoverTool(tooltips=tooltips)
            self.map_stations = gv.Points(self.dfcat, crs=GOOGLE_MERCATOR).opts(
                color=dim(map_color_category),
                cmap="Category10",
            )
            self.map_stations = self.map_stations.opts(
                opts.Points(
                    tools=["tap", hover, "lasso_select", "box_select"],
                    nonselection_alpha=0.3,  # nonselection_color='gray',
                    size=10,
                    responsive=True,
                    height=550,
                )
            ).opts(active_tools=["wheel_zoom"], responsive=True)
            self.station_select = streams.Selection1D(source=self.map_stations)
        else:
            warnings.warn(
                "No geolocation data found in catalog. Not displaying map of stations."
            )

    def show_data_catalog(self, index=slice(None)):
        if index == []:
            index = [0]
        dfs = self.dfcat.iloc[index]  # FIXME: later add filters
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
            self.plot_panel = pn.panel(
                hv.Div("<h3>Select rows from table and click on button</h3>"),
                sizing_mode="stretch_both",
            )
            gspec = pn.GridStack(
                sizing_mode="stretch_both", allow_resize=True, allow_drag=False
            )  # ,
            gspec[0, 0:5] = pn.Row(
                self.plot_button,
                pn.layout.HSpacer(),
            )
            gspec[1:5, 0:10] = fullscreen.FullScreen(pn.Row(self.display_table))
            gspec[6:15, 0:10] = fullscreen.FullScreen(pn.Row(self.plot_panel))
            self.plots_panel = pn.Row(
                gspec
            )  # fails with object of type 'GridSpec' has no len()

        else:
            self.display_table.value = dfs

        return self.plots_panel

    def update_plots(self, event):
        self.plot_panel.loading = True
        df = self.display_table.value.iloc[self.display_table.selection]
        self.plot_panel.object = self.dataui_manager.create_panel(df, self.time_range)
        self.plot_panel.loading = False

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
                self.dataui_manager.get_widgets(),
            ),
        )
        sidebar_view = pn.Column(control_widgets)
        if hasattr(self, "map_stations"):
            map_tooltip = pn.widgets.TooltipIcon(
                value="""Map of stations. Click on a station to see data available in the table. <br/>
                See <a href="https://docs.bokeh.org/en/latest/docs/user_guide/interaction/tools.html">Bokeh Tools</a> for toolbar operation"""
            )
            sidebar_view.append(
                pn.Column(
                    self.tmap * self.map_stations,
                    map_tooltip,
                    sizing_mode="stretch_both",
                )
            )
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
        control_widgets[0].append(self.create_about_button(template))
        return template


# %%
