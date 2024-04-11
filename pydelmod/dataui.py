# %%
# organize imports by category
import warnings

warnings.filterwarnings("ignore")
#
import pandas as pd
import geopandas as gpd
from io import StringIO

# viz and ui
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


class DataUIManager(param.Parameterized):

    def get_widgets(self):
        return pn.pane.Markdown("No widgets available")

    # data related methods
    def get_data_catalog(self):
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

    def get_map_color_category(self):
        pass


class DataUI(param.Parameterized):
    """
    Show table of data froma catalog
    Furthermore select the data rows and click on button to display plots for selected rows
    """

    def __init__(self, dataui_manager, **kwargs):
        crs = kwargs.pop("crs", ccrs.PlateCarree())
        super().__init__(**kwargs)
        self.dataui_manager = dataui_manager
        self.dfcat = self.dataui_manager.get_data_catalog()
        if isinstance(self.dfcat, gpd.GeoDataFrame):
            self.tmap = gv.tile_sources.CartoLight
            tooltips = self.dataui_manager.get_tooltips()

            map_color_category = self.dataui_manager.get_map_color_category()
            hover = HoverTool(tooltips=tooltips)
            self.map_stations = gv.Points(self.dfcat, crs=crs).opts(
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
            index = slice(None)
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
        self.plots_panel.loading = True
        # FIXME: needs a PR to panel to fix this
        dfselected = self.display_table._processed.iloc[self.display_table.selection]
        self.plots_panel.objects = [self.dataui_manager.create_panel(dfselected)]
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
        sidebar_view.append(self.create_about_button(template))
        return template
