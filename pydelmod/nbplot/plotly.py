# -*- coding: utf-8 -*-
""" Jupyter plotting routine with Plot.ly.
"""

import pandas as pd
import numpy as np

import plotly.graph_objs as go

# import holoviews as hv
# hv.extension("plotly")
import panel as pn

pn.extension("tabulator", "plotly")

from .ipw_mixins import *

__all__ = [
    "plot_step_w_variable_station_filters",
    "plot_bar_monthly_w_controls",
    "plot_box_w_variable_station_filters",
    "plot_exceedance_w_variable_station_filters",
    "plot_step_w_regulation",
    "plot_exceedance_w_regulation",
]

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
WY_MONTHS = {
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
}
YEARTYPES = ["W", "AN", "BN", "D", "C"]

# Default margin for plots. Can be overridden for individual plots by including differnt margin in options. Example:
# 'margin': dict(l=80, r=10, t=40, b=0)
DEFAULT_PLOT_MARGIN = dict(l=80, r=10, t=40, b=0)


def plot_step_w_variable_station_filters(df, df_stations=None, options=None):
    """ """
    p = PlotStepWithControls(df, df_stations, options)
    return p.plot()


def plot_bar_monthly_w_controls(df, df_stations=None, options=None):
    """ """
    p = PlotMonthlyBarWithControls(df, df_stations, options)
    return p.plot()


def plot_box_w_variable_station_filters(df, df_stations=None, options=None):
    """ """
    p = PlotBoxWithControls(df, df_stations, options)
    return p.plot()


def plot_exceedance_w_variable_station_filters(df, df_stations=None, options=None):
    """ """
    p = PlotExceedanceWithControls(df, df_stations, options)
    return p.plot()


def plot_step_w_regulation(df, df_reg, df_stations=None, options=None):
    """ """
    p = PlotStepWithRegulation(df, df_reg, df_stations, options)
    return p.plot()


def plot_exceedance_w_regulation(df, df_reg, df_stations=None, options=None):
    """ """
    p = PlotExceedanceWithRegulation(df, df_reg, df_stations, options)
    return p.plot()


def set_default_plot_margin(margin):
    PLOT_MARGIN = margin


class PlotNotebookBase:
    def __init__(self, *args, **kwargs):
        self.lb_msg = pn.panel("")

    def create_widgets(self):
        pass


class PlotStepBase(PlotNotebookBase):
    """
    Example Options dictionary for one time series per scenario
    options = {'xaxis_name': 'Time', 'yaxis_name': 'Stage (cfs)', 'title': 'Stage Plot'}

    Example Options dictionary for > one time series per scenario
    values_to_plot = ['value', 'ssw_highs', 'ssw_lows']
    options = {'yaxis': 'Stage (ft)', 'title': 'Stage Plot', 'plot_multiple_series_per_study': True,
            'colnames_y': values_to_plot, 'multi_series_line_modes': ['lines', 'markers', 'markers'], 'multi_series_line_or_marker_widths': [1, 7, 7]}
    """

    def __init__(
        self,
        df,
        df_stations,
        options,
        colname_x="time",
        colname_y="value",
        colname_variable="variable",
        colname_case="scenario_name",
        colname_station_id="station",
    ):
        super().__init__()
        self.df = df
        self.df_stations = df_stations
        self.options = {} if options is None else options
        # if plotting more than one series per study (for example, stage with high tides and low tides)
        self.multiple_series_per_study = False
        if (
            self.options.get("plot_multiple_series_per_study") is not None
            and self.options.get("plot_multiple_series_per_study")
            and self.options.get("colnames_y") is not None
        ):
            self.multiple_series_per_study = True
        self.colname_x = colname_x
        self.colname_y = colname_y
        self.colname_variable = colname_variable
        # an example of a case is a scenario name
        self.colname_case = colname_case
        self.colname_station_id = colname_station_id
        self.height = self.options["height"] if "height" in self.options else None
        self.margin = (
            self.options["margin"] if "margin" in self.options else DEFAULT_PLOT_MARGIN
        )
        self.layout = None
        self.preprocess_data()
        self.create_widgets()
        self.create_figure()

    def preprocess_data(self):
        self.cases = self.df[self.colname_case].unique()

    def create_widgets(self):
        # Does nothing
        pass

    def generate_title(self):
        return f"{self.station_selected}"

    def create_figure(self):
        self.filter_data()
        data = []
        title = self.generate_title()
        yaxis_name = self.options.get("yaxis_name", "EC (micromhos/cm)")
        # adjusted layout to include margins that are more appropriate for exporting.
        self.layout = go.Layout(
            template="seaborn",
            title=dict(text=title),
            yaxis=dict(title=yaxis_name),
            height=self.height,
            margin=self.margin,
        )
        for case in self.cases:
            mask = self.df_to_plot[self.colname_case] == case
            if self.multiple_series_per_study:
                colnames_y = self.options.get("colnames_y")
                colname_index = 0
                for cy in colnames_y:
                    line_mode = "lines"
                    line_or_marker_width = 1
                    if (
                        self.options.get("multi_series_line_or_marker_widths")
                        is not None
                        and self.options.get("multi_series_line_modes") is not None
                    ):
                        line_mode = self.options.get("multi_series_line_modes")[
                            colname_index
                        ]
                        line_or_marker_width = self.options.get(
                            "multi_series_line_or_marker_widths"
                        )[colname_index]
                    marker_dict = None
                    line_dict = None
                    if line_mode == "lines":
                        line_dict = dict(shape="hv", width=line_or_marker_width)
                    else:
                        marker_dict = dict(symbol="circle", size=line_or_marker_width)
                    data.append(
                        go.Scatter(
                            x=self.df_to_plot[mask]["time"],
                            y=self.df_to_plot[mask][cy],
                            name=case,
                            mode=line_mode,
                            marker=marker_dict,
                            line=line_dict,
                        )
                    )
                    colname_index += 1
            else:
                # otherwise, default to single series, with name determined by colname_y, which defaults to 'value'
                data.append(
                    go.Scatter(
                        x=self.df_to_plot[mask]["time"],
                        y=self.df_to_plot[mask][self.colname_y],
                        name=case,
                        line={"shape": "hv"},
                    )
                )
        self.fig = go.Figure(data=data, layout=self.layout)

    def make_all_true_mask(self):
        self.mask = pd.Series(True, self.df.index)

    def filter_data(self):
        self.df_to_plot = self.df[self.mask]

    def update(self):
        self.fig.layout.title.text = self.generate_title()
        # If user requested plotting multiple series per study, get column name from colnames_y rather than colname_y
        colnames_y = self.options.get("colnames_y")
        col_index = 0
        # loops through all the data sets in the graph, for all cases
        # A scenario is an example of a case
        for i, trace in enumerate(self.fig.data):
            case = trace["name"]
            mask = self.df_to_plot[self.colname_case] == case
            x = self.df_to_plot[mask]["time"]
            trace.x = x
            if self.multiple_series_per_study:
                cy = colnames_y[col_index]
                y = self.df_to_plot[mask][cy]
                if col_index == 2:
                    col_index = 0
                else:
                    col_index += 1
            else:
                y = self.df_to_plot[mask][self.colname_y]
            trace.y = y


class PlotMonthlyBarBase(PlotNotebookBase):
    def __init__(
        self,
        df,
        df_stations,
        options,
        colname_x="month",
        colname_y="value",
        colname_variable="variable",
        colname_case="scenario_name",
        colname_station_id="station",
    ):
        super().__init__()
        self.df = df
        self.df_stations = df_stations
        self.options = {} if options is None else options
        self.colname_x = colname_x
        self.colname_y = colname_y
        self.colname_variable = colname_variable
        self.colname_case = colname_case
        self.colname_station_id = colname_station_id
        self.height = self.options["height"] if "height" in self.options else None
        self.margin = (
            self.options["margin"] if "margin" in self.options else DEFAULT_PLOT_MARGIN
        )
        self.layout = None
        self.preprocess_data()
        self.create_widgets()
        self.create_figure()

    def preprocess_data(self):
        self.cases = self.df[self.colname_case].unique()
        # Some column names are hard-wired.

    def create_widgets(self):
        # Does nothing
        pass

    def generate_title(self):
        return f"{self.station_selected}"

    def create_figure(self):
        self.filter_data()
        data = []
        title = self.generate_title()
        yaxis_name = self.options.get("yaxis_name", "EC (micromhos/cm)")
        self.layout = go.Layout(
            template="seaborn",
            title=dict(text=title),
            yaxis=dict(title=yaxis_name),
            height=self.height,
            margin=self.margin,
        )
        for case in self.cases:
            mask = self.df_to_plot[self.colname_case] == case
            df = self.df_to_plot[mask].set_index("month").reindex(WY_MONTHS.values())
            data.append(
                go.Bar(x=list(WY_MONTHS.keys()), y=df[self.colname_y], name=case)
            )
        self.fig = go.Figure(data=data, layout=self.layout)

    def make_all_true_mask(self):
        self.mask = pd.Series(True, self.df.index)

    def filter_data(self):
        self.df_to_plot = (
            self.df[self.mask]
            .groupby([self.colname_case, "month"])["value"]
            .mean()
            .reset_index()
        )

    def update(self):
        self.fig.layout.title.text = self.generate_title()
        for i, trace in enumerate(self.fig.data):
            case = self.cases[i]
            mask = self.df_to_plot[self.colname_case] == case
            df = self.df_to_plot[mask].set_index("month").reindex(WY_MONTHS.values())
            trace.y = df[self.colname_y]


class PlotMonthlyBarWithControls(
    ExportPlotForStationsMixin,
    SaveDataMixin,
    ShowDataMixin,
    FilterWateryearTypeMixin,
    FilterStationMixin,
    FilterVariableMixin,
    PlotMonthlyBarBase,
):
    def __init__(self, *args, **kwargs):
        """ """
        super().__init__(*args, **kwargs)

    def filter_data(self):
        self.make_all_true_mask()
        super().filter_data()

    def plot(self):
        self.widgets = pn.Column(
            self.fig,
            pn.Row(self.dd_variable, self.dd_station),
            self.box_yeartypes,
            pn.Row(self.tb_showdata, self.tb_savedata),
            self.box_exportplots,
            self.lb_msg,
        )
        return self.widgets


class PlotBoxBase(PlotNotebookBase):
    def __init__(
        self,
        df,
        df_stations,
        options,
        colname_x="time",
        colname_y="value",
        colname_variable="variable",
        colname_case="scenario_name",
        colname_station_id="station",
    ):
        super().__init__()
        self.df = df
        self.df_stations = df_stations
        self.options = {} if options is None else options
        self.colname_x = colname_x
        self.colname_y = colname_y
        self.colname_variable = colname_variable
        self.colname_case = colname_case
        self.colname_station_id = colname_station_id
        self.height = self.options["height"] if "height" in self.options else None
        self.margin = (
            self.options["margin"] if "margin" in self.options else DEFAULT_PLOT_MARGIN
        )
        self.layout = None
        self.preprocess_data()
        self.create_widgets()
        self.create_figure()

    def preprocess_data(self):
        self.cases = self.df[self.colname_case].unique()

    def create_widgets(self):
        pass

    def generate_title(self):
        return f"{self.station_selected}"

    def create_figure(self):
        self.filter_data()
        data = []
        title = self.generate_title()
        xaxis_name = self.options.get("xaxis_name", "EC (micromhos/cm)")
        self.layout = go.Layout(
            template="seaborn",
            title=dict(text=title),
            xaxis=dict(title=xaxis_name),
            height=self.height,
            margin=self.margin,
        )
        for case in self.cases:
            mask = self.df_to_plot[self.colname_case] == case
            data.append(go.Box(name=case, x=self.df_to_plot[mask][self.colname_y]))
        self.fig = go.Figure(data=data, layout=self.layout)

    def make_all_true_mask(self):
        self.mask = pd.Series(True, self.df.index)

    def filter_data(self):
        self.df_to_plot = self.df[self.mask]

    def update(self):
        self.fig.layout.title.text = self.generate_title()
        for i, trace in enumerate(self.fig.data):
            case = self.cases[i]
            mask = self.df_to_plot[self.colname_case] == case
            y = self.df_to_plot[mask][self.colname_y]
            trace.x = y


class PlotExceedanceBase(PlotNotebookBase):
    def __init__(
        self,
        df,
        df_stations,
        options,
        colname_x="time",
        colname_y="value",
        colname_variable="variable",
        colname_case="scenario_name",
        colname_station_id="station",
    ):
        super().__init__()
        self.df = df
        self.df_stations = df_stations
        self.options = {} if options is None else options
        self.colname_x = colname_x
        self.colname_y = colname_y
        self.colname_variable = colname_variable
        self.colname_case = colname_case
        self.colname_station_id = colname_station_id
        self.height = self.options["height"] if "height" in self.options else None
        self.margin = (
            self.options["margin"] if "margin" in self.options else DEFAULT_PLOT_MARGIN
        )
        self.layout = None
        self.preprocess_data()
        self.create_widgets()
        self.create_figure()

    def preprocess_data(self):
        self.cases = self.df[self.colname_case].unique()

    def create_widgets(self):
        pass

    def generate_title(self):
        return f"{self.station_selected}"

    def create_figure(self):
        self.filter_data()
        data = []
        title = self.generate_title()
        xaxis_name = self.options.get("xaxis_name", "Probability of Exceedance (%)")
        yaxis_name = self.options.get("yaxis_name", "EC (micromhos/cm)")
        self.layout = go.Layout(
            template="seaborn",
            title=dict(text=title),
            yaxis=dict(title=yaxis_name),
            xaxis=dict(title=xaxis_name),
            height=self.height,
            margin=self.margin,
        )
        for case in self.cases:
            mask = self.df_to_plot[self.colname_case] == case
            yval = (
                self.df_to_plot[mask][self.colname_y]
                .sort_values(ascending=False)
                .values
            )
            n = len(yval)
            xval = np.arange(1, n + 1) / n * 100.0
            data.append(go.Scatter(x=xval, y=yval, name=case))
        self.fig = go.Figure(data=data, layout=self.layout)

    def make_all_true_mask(self):
        self.mask = pd.Series(True, self.df.index)

    def filter_data(self):
        self.df_to_plot = self.df[self.mask]

    def update(self):
        self.fig.layout.title.text = self.generate_title()
        for i, trace in enumerate(self.fig.data):
            case = self.cases[i]
            mask = self.df_to_plot[self.colname_case] == case
            yval = (
                self.df_to_plot[mask][self.colname_y]
                .sort_values(ascending=False)
                .values
            )
            n = len(yval)
            xval = np.arange(1, n + 1) / n * 100.0
            trace.x = xval
            trace.y = yval


class PlotStepWithControls(
    ExportPlotForStationsMixin,
    SaveDataMixin,
    ShowDataMixin,
    FilterStationMixin,
    FilterVariableMixin,
    PlotStepBase,
):
    def __init__(self, *args, **kwargs):
        """ """
        super().__init__(*args, **kwargs)

    def filter_data(self):
        self.make_all_true_mask()
        super().filter_data()

    def plot(self):
        self.widgets = pn.Column(
            self.fig,
            pn.Row(self.dd_variable, self.dd_station),
            pn.Row(self.tb_showdata, self.tb_savedata),
            self.box_exportplots,
            self.lb_msg,
        )
        return self.widgets


class PlotBoxWithControls(
    ExportPlotForStationsMixin,
    SaveDataMixin,
    ShowDataMixin,
    FilterMonthMixin,
    FilterWateryearTypeMixin,
    FilterStationMixin,
    FilterVariableMixin,
    PlotBoxBase,
):
    def __init__(self, *args, **kwargs):
        """ """
        super().__init__(*args, **kwargs)

    def filter_data(self):
        self.make_all_true_mask()
        super().filter_data()

    def plot(self):
        self.widgets = pn.Column(
            self.fig,
            pn.Row(self.dd_variable, self.dd_station),
            self.box_yeartypes,
            self.box_months,
            pn.Row(self.tb_showdata, self.tb_savedata),
            self.box_exportplots,
            self.lb_msg,
        )
        return self.widgets


class PlotExceedanceWithControls(
    ExportPlotForStationsMixin,
    SaveDataMixin,
    ShowDataMixin,
    FilterMonthMixin,
    FilterWateryearTypeMixin,
    FilterStationMixin,
    FilterVariableMixin,
    PlotExceedanceBase,
):

    def __init__(self, *args, **kwargs):
        """ """
        super().__init__(*args, **kwargs)

    def filter_data(self):
        self.make_all_true_mask()
        super().filter_data()

    def plot(self):
        self.widgets = pn.Column(
            self.fig,
            pn.Row(self.dd_variable, self.dd_station),
            self.box_yeartypes,
            self.box_months,
            pn.Row(self.tb_showdata, self.tb_savedata),
            self.box_exportplots,
            self.lb_msg,
        )
        return self.widgets


class PlotStepWithRegulationBase(PlotNotebookBase):
    def __init__(
        self,
        df,
        df_reg,
        df_stations,
        options,
        colname_x="time",
        colname_y="value",
        colname_variable="variable",
        colname_case="scenario_name",
        colname_station_id="station",
    ):
        super().__init__()
        self.df = df
        self.df_reg = df_reg
        self.df_stations = df_stations
        self.options = {} if options is None else options
        self.colname_x = colname_x
        self.colname_y = colname_y
        self.colname_variable = colname_variable
        self.colname_case = colname_case
        self.colname_station_id = colname_station_id
        self.height = self.options["height"] if "height" in self.options else None
        self.margin = (
            self.options["margin"] if "margin" in self.options else DEFAULT_PLOT_MARGIN
        )
        self.layout = None
        self.preprocess_data()
        self.create_widgets()
        self.create_figure()

    def preprocess_data(self):
        cases = self.df[self.colname_case].unique()
        dfs = []
        for case in cases:
            for variable in self.df[self.colname_variable].unique():
                for station_id in self.df_stations["ID"].unique():
                    mask = (
                        (self.df[self.colname_case] == case)
                        & (self.df[self.colname_variable] == variable)
                        & (self.df[self.colname_station_id] == station_id)
                    )
                    # processed data based on stations/criteria
                    if self.df_reg["scenario_name"].unique() in [
                        "D1641 AG WI",
                        "D1641 FWS SJR",
                    ]:
                        df = (
                            self.df[mask]
                            .set_index("time")
                            .rolling("14d")["value"]
                            .mean()
                            .reset_index()
                        )
                        df[self.colname_variable] = variable + "-14DAY"
                    elif self.df_reg["scenario_name"].unique() == "D1641 AG South":
                        df = (
                            self.df[mask]
                            .set_index("time")
                            .rolling("30d")["value"]
                            .mean()
                            .reset_index()
                        )
                        df[self.colname_variable] = variable + "-30DAY"
                    elif self.df_reg["scenario_name"].unique() == "D1641 AG Export":
                        df = (
                            self.df[mask]
                            .set_index("time")
                            .resample("1m")["value"]
                            .mean()
                            .reset_index()
                        )
                        df[self.colname_variable] = variable + "-MAVG"
                    elif self.df_reg["scenario_name"].unique() == "D1641 FWS Suisun":
                        # print("D1641_FWS")
                        # print(station_id)
                        df = (
                            self.df[mask]
                            .set_index("time")
                            .resample("1m")["value"]
                            .mean()
                            .reset_index()
                        )
                        df[self.colname_variable] = variable + "-MAVG"
                    elif self.df_reg["scenario_name"].unique() in [
                        "D1641 MI 250",
                        "D1641 MI 150",
                        "D1641 Monthly",
                    ]:
                        df = self.df[mask].set_index("time").reset_index()
                        df[self.colname_variable] = variable
                    else:
                        print("not stations in D1641_AG or FWS or MI:", station_id)

                    df.rename(columns={"index": "time"}, inplace=True)
                    df[self.colname_station_id] = station_id
                    df[self.colname_case] = case
                    dfs.append(df)

        self.df = pd.concat(dfs)
        self.df = pd.concat([self.df, self.df_reg], sort=False)
        self.cases_to_plot = self.df[self.colname_case].unique()
        self.regulation_name = self.df_reg[self.colname_case].unique()

    def generate_title(self):
        return f"{self.station_selected}"

    def create_figure(self):
        self.filter_data()
        data = []
        title = self.generate_title()
        yaxis_name = self.options.get("yaxis_name", "EC (micromhos/cm)")
        self.layout = go.Layout(
            template="seaborn",
            title=dict(text=title),
            yaxis=dict(title=yaxis_name, rangemode="tozero"),
            height=self.height,
            margin=self.margin,
        )
        for case in self.cases_to_plot:
            mask = self.df_to_plot[self.colname_case] == case
            # Hard-wired, assuming that the regulation comes with this name.
            if case == self.regulation_name:
                data.append(
                    go.Scatter(
                        x=self.df_to_plot[mask]["time"],
                        y=self.df_to_plot[mask][self.colname_y],
                        line={"shape": "hv"},
                        fill="tozeroy",
                        name=case,
                    )
                )
            else:
                data.append(
                    go.Scatter(
                        x=self.df_to_plot[mask]["time"],
                        y=self.df_to_plot[mask][self.colname_y],
                        line={"shape": "hv"},
                        name=case,
                    )
                )
        self.fig = go.Figure(data=data, layout=self.layout)

    def make_all_true_mask(self):
        self.mask = pd.Series(True, self.df.index)

    def filter_data(self):
        self.df_to_plot = self.df[self.mask]

    def update(self):
        self.fig.layout.title.text = self.generate_title()
        for i, trace in enumerate(self.fig.data):
            case = self.cases_to_plot[i]
            mask = self.df_to_plot[self.colname_case] == case
            x = self.df_to_plot[mask]["time"]
            trace.x = x
            y = self.df_to_plot[mask][self.colname_y]
            trace.y = y


class PlotStepWithRegulation(
    ExportPlotForStationsMixin,
    SaveDataMixin,
    ShowDataMixin,
    FilterStationMixin,
    PlotStepWithRegulationBase,
):
    def __init__(self, *args, **kwargs):
        """ """
        super().__init__(*args, **kwargs)

    def filter_data(self):
        self.make_all_true_mask()
        super().filter_data()

    def plot(self):
        self.widgets = pn.Column(
            self.fig,
            pn.Row(
                self.dd_station,
            ),
            pn.Row(
                self.tb_showdata,
                self.tb_savedata,
            ),
            self.box_exportplots,
            self.lb_msg,
        )
        return self.widgets


class PlotExceedanceWithRegulationBase(PlotNotebookBase):
    def __init__(
        self,
        df,
        df_reg,
        df_stations,
        options,
        colname_x="time",
        colname_y="value",
        colname_variable="variable",
        colname_case="scenario_name",
        colname_station_id="station",
    ):
        super().__init__()
        self.df = df
        self.df_reg = df_reg
        self.df_stations = df_stations
        self.options = {} if options is None else options
        self.colname_x = colname_x
        self.colname_y = colname_y
        self.colname_variable = colname_variable
        self.colname_case = colname_case
        self.colname_station_id = colname_station_id
        self.height = self.options["height"] if "height" in self.options else None
        self.margin = (
            self.options["margin"] if "margin" in self.options else DEFAULT_PLOT_MARGIN
        )
        self.layout = None
        self.preprocess_data()
        self.create_widgets()
        self.create_figure()

    def preprocess_data(self):
        cases = self.df[self.colname_case].unique()
        dfs = []
        for case in cases:
            for variable in self.df[self.colname_variable].unique():
                for station_id in self.df_stations["ID"].unique():
                    mask = (
                        (self.df[self.colname_case] == case)
                        & (self.df[self.colname_variable] == variable)
                        & (self.df[self.colname_station_id] == station_id)
                    )
                    mask_reg = self.df_reg[self.colname_station_id] == station_id
                    ds_reg = self.df_reg[mask_reg][
                        self.df_reg[mask_reg]["value"] > 0.0
                    ].set_index("time")["value"]

                    # diff processed data based on stations/criteria
                    if self.df_reg["scenario_name"].unique() in [
                        "D1641 AG WI",
                        "D1641 FWS SJR",
                    ]:
                        ds = (
                            self.df[mask]
                            .set_index("time")
                            .rolling("14d")["value"]
                            .mean()
                        )
                        ds_diff = ds.reindex(ds_reg.index) - ds_reg
                        df = ds_diff.to_frame().reset_index()
                        df[self.colname_variable] = variable + "-14DAY-DIFF"
                    elif self.df_reg["scenario_name"].unique() == "D1641 AG South":
                        ds = (
                            self.df[mask]
                            .set_index("time")
                            .rolling("30d")["value"]
                            .mean()
                        )
                        ds_diff = ds.reindex(ds_reg.index) - ds_reg
                        df = ds_diff.to_frame().reset_index()
                        df[self.colname_variable] = variable + "-30DAY-DIFF"
                    elif self.df_reg["scenario_name"].unique() == "D1641 AG Export":
                        ds = (
                            self.df[mask]
                            .set_index("time")
                            .resample("1m")["value"]
                            .mean()
                        )
                        ds = ds.reindex(
                            pd.date_range(
                                start=ds.index[0].replace(day=1),
                                end=ds.index[-1],
                                freq="D",
                            )
                        ).bfill()
                        ds_diff = ds.reindex(ds_reg.index) - ds_reg
                        df = ds_diff.to_frame().reset_index()
                        df[self.colname_variable] = variable + "-MAVG-DIFF"
                    elif self.df_reg["scenario_name"].unique() == "D1641 FWS Suisun":
                        ds = (
                            self.df[mask]
                            .set_index("time")
                            .resample("1m")["value"]
                            .mean()
                        )
                        ds = ds.reindex(
                            pd.date_range(
                                start=ds.index[0].replace(day=1),
                                end=ds.index[-1],
                                freq="D",
                            )
                        ).bfill()
                        ds_diff = ds.reindex(ds_reg.index) - ds_reg
                        df = ds_diff.to_frame().reset_index()
                        df[self.colname_variable] = variable + "-MAVG-DIFF"
                    elif self.df_reg["scenario_name"].unique() == "D1641 MI 250":
                        ds = self.df[mask].set_index("time")["value"]
                        ds = ds.reindex(
                            pd.date_range(
                                start=ds.index[0].replace(day=1),
                                end=ds.index[-1],
                                freq="D",
                            )
                        ).bfill()
                        ds_diff = ds.reindex(ds_reg.index) - ds_reg
                        df = ds_diff.to_frame().reset_index()
                        df[self.colname_variable] = variable + "-DIFF"
                    elif (
                        self.df_reg["scenario_name"].unique() == "D1641 MI 150",
                        "D1641 Monthly",
                    ):
                        ds = self.df[mask].set_index("time")["value"]
                        ds_diff = ds.reindex(ds_reg.index) - ds_reg
                        df = ds_diff.to_frame().reset_index()
                        df[self.colname_variable] = variable + "-DIFF"
                    else:
                        print("not stations in D1641_AG or FWS or MI:", station_id)

                    df.rename(columns={"index": "time"})
                    df[self.colname_station_id] = station_id
                    df[self.colname_case] = case
                    df = df.merge(
                        self.df[mask][["time", "sac_yrtype"]], on="time", how="left"
                    )
                    dfs.append(df)
        self.df = pd.concat(dfs)
        self.cases = self.df[self.colname_case].unique()

    def generate_title(self):
        return f"{self.station_selected}<br>(Scenario minus Standard)"

    def create_figure(self):
        self.filter_data()
        data = []
        title = self.generate_title()
        xaxis_name = self.options.get(
            "xaxis_name", "Probability of Meeting D-1641 Water Quality Objective (%)"
        )
        yaxis_name = self.options.get("yaxis_name", "Difference in EC (micromhos/cm)")
        self.layout = go.Layout(
            template="seaborn",
            title=dict(text=title),
            yaxis=dict(
                zeroline=True,
                zerolinecolor="#000000",
                title=yaxis_name,
                rangemode="tozero",
            ),
            xaxis=dict(title=xaxis_name),
            height=self.height,
            margin=self.margin,
        )
        if self.df_reg["scenario_name"].unique() == "D1641 MI 150":  # CCC 150 Chloride
            results = {
                "Scenario": [],
                "# of Years Standards are Applicable": [],
                "# of Years Exceeded": [],
                r"% of Years Exceeded": [],
            }
        elif (
            self.df_reg["scenario_name"].unique() == "D1641 Monthly"
        ):  # Calsim Monthly, EMM/JER/RSU
            results = {
                "Scenario": [],
                "# of Months Standards are Applicable": [],
                "# of Months Exceeded": [],
                r"% of Months Exceeded": [],
            }
        else:
            results = {
                "Scenario": [],
                "# of Days Standards are Applicable": [],
                "# of Days Exceeded": [],
                r"% of Days Exceeded": [],
            }
        for case in self.cases:
            mask = self.df_to_plot[self.colname_case] == case
            yval = self.df_to_plot[mask][self.colname_y].sort_values(ascending=True)
            n = yval.count()
            xval = np.arange(1, n + 1) / n * 100.0
            name = case
            data.append(go.Scatter(x=xval, y=yval.values, name=case))
            results["Scenario"].append(case)
            if (
                self.df_reg["scenario_name"].unique() == "D1641 MI 150"
            ):  # CCC 150 Chloride, count under standard
                results["# of Years Standards are Applicable"].append(n)
                n_exceeded = yval[yval < 0.0].count()
                results["# of Years Exceeded"].append(n_exceeded)
                results[r"% of Years Exceeded"].append(f"{n_exceeded / n * 100.:.2f}")
            elif (
                self.df_reg["scenario_name"].unique() == "D1641 Monthly"
            ):  # Calsim Monthly, EMM/JER/RSU
                results["# of Months Standards are Applicable"].append(n)
                n_exceeded = yval[yval > 0.0].count()
                results["# of Months Exceeded"].append(n_exceeded)
                results[r"% of Months Exceeded"].append(f"{n_exceeded / n * 100.:.2f}")
            else:  # most cases, count over standard
                results["# of Days Standards are Applicable"].append(n)
                n_exceeded = yval[yval > 0.0].count()
                results["# of Days Exceeded"].append(n_exceeded)
                # print(n_exceeded) #todel
                # print(n)#todel
                results[r"% of Days Exceeded"].append(f"{n_exceeded / n * 100.:.2f}")
        self.fig = go.Figure(data=data, layout=self.layout)
        self.df_results = pd.DataFrame(data=results)
        self.results = go.Figure(
            data=[
                go.Table(
                    header=dict(values=[[v] for v in self.df_results.columns]),
                    cells=dict(
                        values=[self.df_results[k] for k in self.df_results.columns],
                        height=30,
                    ),
                )
            ],
            layout=go.Layout(
                template="seaborn",
                height=(self.df_results.shape[0] + 2) * 20 + 100,
                margin=dict(t=30, b=10),
                font=dict(size=14),
            ),
        )

    def make_all_true_mask(self):
        self.mask = pd.Series(True, self.df.index)

    def filter_data(self):
        self.df_to_plot = self.df[self.mask]

    def update(self):
        self.fig.layout.title.text = self.generate_title()
        if (
            self.df_reg["scenario_name"].unique() == "D1641 MI 150"
        ):  # CCC 150 Chloride, count under standard
            results = {
                "Scenario": [],
                "# of Years Standards are Applicable": [],
                "# of Years Exceeded": [],
                r"% of Years Exceeded": [],
            }
        elif (
            self.df_reg["scenario_name"].unique() == "D1641 Monthly"
        ):  # Calsim Monthly, EMM/JER/RSU
            results = {
                "Scenario": [],
                "# of Months Standards are Applicable": [],
                "# of Months Exceeded": [],
                r"% of Months Exceeded": [],
            }
        else:
            results = {
                "Scenario": [],
                "# of Days Standards are Applicable": [],
                "# of Days Exceeded": [],
                r"% of Days Exceeded": [],
            }
        for i, trace in enumerate(self.fig.data):
            case = self.cases[i]
            mask = self.df_to_plot[self.colname_case] == case
            yval = self.df_to_plot[mask][self.colname_y].sort_values(ascending=True)
            n = yval.count()
            xval = np.arange(1, n + 1) / n * 100.0
            results["Scenario"].append(case)
            if (
                self.df_reg["scenario_name"].unique() == "D1641 MI 150"
            ):  # CCC 150 Chloride, count under standard
                results["# of Years Standards are Applicable"].append(n)
                n_exceeded = yval[yval < 0.0].count()
                results["# of Years Exceeded"].append(n_exceeded)
                results[r"% of Years Exceeded"].append(f"{n_exceeded / n * 100.:.2f}")
            elif (
                self.df_reg["scenario_name"].unique() == "D1641 Monthly"
            ):  # Calsim Monthly, EMM/JER/RSU
                results["# of Months Standards are Applicable"].append(n)
                n_exceeded = yval[yval > 0.0].count()
                results["# of Months Exceeded"].append(n_exceeded)
                results[r"% of Months Exceeded"].append(f"{n_exceeded / n * 100.:.2f}")
            else:  # most cases, count over standard
                results["# of Days Standards are Applicable"].append(n)
                n_exceeded = yval[yval > 0.0].count()
                results["# of Days Exceeded"].append(n_exceeded)
                results[r"% of Days Exceeded"].append(f"{n_exceeded / n * 100.:.2f}")
            trace.x = xval
            trace.y = yval
        self.df_results = pd.DataFrame(data=results)
        self.results.data[0].cells.values = [
            self.df_results[k] for k in self.df_results.columns
        ]


class PlotExceedanceWithRegulation(
    ExportPlotForStationsMixin,
    SaveDataMixin,
    ShowDataMixin,
    FilterWateryearTypeMixin,
    FilterStationMixin,
    PlotExceedanceWithRegulationBase,
):
    def __init__(self, *args, **kwargs):
        """ """
        super().__init__(*args, **kwargs)

    def filter_data(self):
        self.make_all_true_mask()
        super().filter_data()

    def plot(self):
        self.widgets = pn.Column(
            self.fig,
            self.results,
            pn.Row(
                self.dd_station,
            ),
            self.box_yeartypes,
            pn.Row(self.tb_showdata, self.tb_savedata),
            self.box_exportplots,
            self.lb_msg,
        )
        return self.widgets
