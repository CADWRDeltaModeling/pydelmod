# -*- coding: utf-8 -*-
""" Common mixins for interactive controls
"""

import pandas as pd
import panel as pn
import panel.widgets as pnw
import plotly.io as pio

pn.extension("tabulator", "plotly")

MONTHS = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}
YEARTYPES = ["W", "AN", "BN", "D", "C"]


class FilterVariableMixin:
    """A mixin dropdown ipywidget to filter variable."""

    def __init__(self, *args, **kwargs):
        """Constructor"""
        self.variables = None
        self.colname_variable = None
        super().__init__(*args, **kwargs)

    def preprocess_data(self):
        """Process data.
        Collect available variables in the DataFrame, self.df.
        The column name for variables is from self.colname_variable
        """
        self.variables = list(self.df[self.colname_variable].unique())
        super().preprocess_data()

    def create_widgets(self):
        """Create a dropdown widget."""
        self.dd_variable = pnw.Select(
            options=list(self.variables),
            value=self.variables[0],
            name="Variable",
        )
        self.variable_selected = self.dd_variable.value
        self.dd_variable.param.watch(self.response_filter_variable, "value")
        super().create_widgets()

    def response_filter_variable(self, change):
        """A response function to the event from the dropdown"""
        self.variable_selected = self.dd_variable.value
        self.filter_data()
        self.update()

    def filter_data(self):
        if self.variable_selected is not None:
            self.mask = self.mask & (
                self.df[self.colname_variable] == self.variable_selected
            )
        super().filter_data()

    def update(self):
        """Update the figure and others"""
        super().update()


class FilterStationMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_widgets(self):
        locations = list(self.df_stations["Location"])
        self.dd_station = pnw.Select(
            options=locations,
            value=locations[0],
            name="Station",
        )
        self.station_selected = self.dd_station.value
        self.station_id_selected = self.df_stations[
            self.df_stations["Location"] == self.station_selected
        ]["ID"].values[0]
        self.dd_station.param.watch(self.response_filter_station, "value")
        super().create_widgets()

    def response_filter_station(self, change):
        self.station_selected = self.dd_station.value
        self.station_id_selected = self.df_stations[
            self.df_stations["Location"] == self.station_selected
        ]["ID"].values[0]
        self.filter_data()
        self.update()

    def filter_data(self):
        if self.colname_station_id is not None:
            self.mask = self.mask & (
                self.df[self.colname_station_id] == self.station_id_selected
            )
        super().filter_data()


class FilterWateryearTypeMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_widgets(self):
        self.tb_yeartypes = [pnw.Toggle(value=True, name=f"{t}") for t in YEARTYPES]
        self.box_yeartypes = pn.Row(
            *([pn.panel("Wateryear Type")] + self.tb_yeartypes),
        )
        for tb in self.tb_yeartypes:
            tb.param.watch(self.response_filter_yeartype, "value")
        self.yt_selected = [
            YEARTYPES[i] for i, tb in enumerate(self.tb_yeartypes) if tb.value
        ]
        super().create_widgets()

    def response_filter_yeartype(self, change):
        self.yt_selected = [
            YEARTYPES[i] for i, tb in enumerate(self.tb_yeartypes) if tb.value
        ]
        self.filter_data()
        self.update()

    def filter_data(self):
        mask = self.df["sac_yrtype"].isin(self.yt_selected)
        self.mask = self.mask & mask
        super().filter_data()


class FilterMonthMixin:
    """A mixin 12 toggle buttons to filter by months."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_widgets(self):
        self.tb_months = [pnw.Toggle(value=True, name="{}".format(t)) for t in MONTHS]
        self.box_months = pn.Row(*([pn.panel("Month")] + self.tb_months))
        self.mo_selected = [
            MONTHS[tb.name] for i, tb in enumerate(self.tb_months) if tb.value
        ]
        for tb in self.tb_months:
            tb.param.watch(self.response_filter_month, "value")
        super().create_widgets()

    def response_filter_month(self, change):
        self.mo_selected = [
            MONTHS[tb.name] for i, tb in enumerate(self.tb_months) if tb.value
        ]
        self.filter_data()
        self.update()

    def filter_data(self):
        self.mask = self.mask & self.df["month"].isin(self.mo_selected)
        super().filter_data()


class ShowDataMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # forwards all unused arguments

    def create_widgets(self):
        self.tb_showdata = pnw.Toggle(value=False, name="Show Data")
        self.table_panel = pn.Row()
        self.tb_showdata.param.watch(self.response_showdata, "value")
        super().create_widgets()

    def response_showdata(self, change):
        self.update()

    def update(self):
        if self.tb_showdata.value:
            self.table_panel.objects = [
                pnw.Tabulator(
                    self.df_to_plot,
                    layout="fit_data",
                    height=400,
                    width=800,
                )
            ]
            if not self.widgets.objects[-1] == self.table_panel:
                self.widgets.objects += [self.table_panel]
        else:
            if self.widgets.objects[-1] == self.table_panel:
                self.widgets.objects = self.widgets.objects[:-1]
        super().update()


class SaveDataMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_widgets(self):
        self.tb_savedata = pnw.Toggle(
            value=False,
            name="Save data",
        )
        self.tb_savedata.param.watch(self.response_savedata, "value")
        self.box_savedata = pn.Row(self.tb_savedata, self.lb_msg)
        super().create_widgets()

    def response_savedata(self, change):
        fpath_csv = "export.csv"
        self.lb_msg.object = "Saving the data into {}".format(fpath_csv)
        self.df_to_plot.to_csv(fpath_csv)
        self.lb_msg.object = "Saved the data into {}".format(fpath_csv)
        self.tb_savedata.value = False


class ExportPlotForStationsMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_widgets(self):
        self.tb_exportplots = pnw.Toggle(
            value=False,
            name="Export Plots",
        )

        # allow user to specify a plot prefix, which is a string that will be used for filenames
        self.t_exportplots_prefix = pnw.TextInput(
            value="plot",
            placeholder="Prefix for plots",
            name="Plot prefix",
            disabled=False,
        )
        self.lb_msg = pn.panel("")
        self.tb_exportplots.param.watch(self.response_exportplots, "value")
        self.box_exportplots = pn.Row(self.tb_exportplots, self.t_exportplots_prefix)
        super().create_widgets()

    def response_exportplots(self, change):
        self.lb_msg.value = f"Exporting plots..."
        self.export_plots()
        self.tb_exportplots.value = False

    def export_plots(self):
        table_plots = {"station": [], "filename": [], "station_long_name": []}
        plot_prefix = self.t_exportplots_prefix.value
        for i, row in self.df_stations.iterrows():
            station_name = row["Location"]
            self.dd_station.value = station_name
            station_id = row["ID"]
            self.update()
            fpath_plot = f"{plot_prefix}_{station_id}.png"
            # don't know why, but for some reason it is necessary to reapply the layout
            self.fig.layout = self.layout
            self.fig.layout.title = station_name
            pio.write_image(self.fig, fpath_plot, scale=3)
            table_plots["station"].append(station_id)
            table_plots["filename"].append(fpath_plot)
            table_plots["station_long_name"].append(station_name)
        fpath_map = f"{plot_prefix}_description.csv"
        df_plots = pd.DataFrame(data=table_plots)
        df_plots.to_csv(fpath_map)
        self.lb_msg.value = (
            f"Finished saving plots. See {fpath_map} for plot information."
        )
