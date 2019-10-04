# -*- coding: utf-8 -*-
""" Jupyter plotting routine with Plot.ly.
"""

import pandas as pd
import numpy as np
import plotly.graph_objs as go
from .ipw_mixins import *

__all__ = ['plot_step_w_variable_station_filters',
           'plot_bar_monthly_w_controls',
           'plot_box_w_variable_station_filters',
           'plot_exceedance_w_variable_station_filters',
           'plot_step_w_regulation',
           'plot_exceedance_w_regulation']

MONTHS = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5,
          'JUN': 6, 'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12}
WY_MONTHS = {'OCT': 10, 'NOV': 11, 'DEC': 12, 'JAN': 1, 'FEB': 2,
             'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6, 'JUL': 7, 'AUG': 8, 'SEP': 9, }
YEARTYPES = ['W', 'AN', 'BN', 'D', 'C']


def plot_step_w_variable_station_filters(df, df_stations=None, options=None):
    """
    """
    p = PlotStepWithControls(df, df_stations, options)
    return p.plot()


def plot_bar_monthly_w_controls(df, df_stations=None, options=None):
    """
    """
    p = PlotMonthlyBarWithControls(df, df_stations, options)
    return p.plot()


def plot_box_w_variable_station_filters(df, df_stations=None, options=None):
    """
    """
    p = PlotBoxWithControls(df, df_stations, options)
    return p.plot()


def plot_exceedance_w_variable_station_filters(df, df_stations=None, options=None):
    """
    """
    p = PlotExceedanceWithControls(df, df_stations, options)
    return p.plot()


def plot_step_w_regulation(df, df_reg, df_stations=None, options=None):
    """
    """
    p = PlotStepWithRegulation(df, df_reg, df_stations, options)
    return p.plot()


def plot_exceedance_w_regulation(df, df_reg, df_stations=None, options=None):
    """
    """
    p = PlotExceedanceWithRegulation(df, df_reg, df_stations, options)
    return p.plot()


class PlotNotebookBase():
    def __init__(self, *args, **kwargs):
        self.lb_msg = ipw.Label(value='')

    def create_widgets(self):
        pass


class PlotStepBase(PlotNotebookBase):
    def __init__(self, df, df_stations, options,
                 colname_x='time', colname_y='value',
                 colname_variable='variable', colname_case='scenario_name',
                 colname_station_id='station'):
        super().__init__()
        self.df = df
        self.df_stations = df_stations
        self.options = {} if options is None else options
        self.colname_x = colname_x
        self.colname_y = colname_y
        self.colname_variable = colname_variable
        self.colname_case = colname_case
        self.colname_station_id = colname_station_id
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
        yaxis_name = self.options.get('yaxis_name',
                                      'EC (micromhos/cm)')
        layout = go.Layout(template='seaborn',
                           title=title,
                           yaxis=dict(title=yaxis_name),
                           )
        for case in self.cases:
            mask = (self.df_to_plot[self.colname_case] == case)
            data.append(go.Scatter(x=self.df_to_plot[mask]['time'],
                                   y=self.df_to_plot[mask][self.colname_y],
                                   name=case,
                                   line={'shape': 'hv'}))
        self.fig = go.FigureWidget(data=data, layout=layout)

    def make_all_true_mask(self):
        self.mask = pd.Series(True, self.df.index)

    def filter_data(self):
        self.df_to_plot = self.df[self.mask]

    def update(self):
        self.fig.layout.title.text = self.generate_title()
        for i, trace in enumerate(self.fig.data):
            case = self.cases[i]
            mask = self.df_to_plot[self.colname_case] == case
            x = self.df_to_plot[mask]['time']
            trace.x = x
            y = self.df_to_plot[mask][self.colname_y]
            trace.y = y


class PlotMonthlyBarBase(PlotNotebookBase):
    def __init__(self, df, df_stations, options,
                 colname_x='month', colname_y='value',
                 colname_variable='variable', colname_case='scenario_name',
                 colname_station_id='station'):
        super().__init__()
        self.df = df
        self.df_stations = df_stations
        self.options = {} if options is None else options
        self.colname_x = colname_x
        self.colname_y = colname_y
        self.colname_variable = colname_variable
        self.colname_case = colname_case
        self.colname_station_id = colname_station_id
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
        yaxis_name = self.options.get('yaxis_name',
                                      'EC (micromhos/cm)')
        layout = go.Layout(template='seaborn',
                           title=title,
                           yaxis=dict(title=yaxis_name),
                           )
        for case in self.cases:
            mask = (self.df_to_plot[self.colname_case] == case)
            df = self.df_to_plot[mask].set_index(
                'month').reindex(WY_MONTHS.values())
            data.append(go.Bar(x=list(WY_MONTHS.keys()),
                               y=df[self.colname_y],
                               name=case))
        self.fig = go.FigureWidget(data=data, layout=layout)

    def make_all_true_mask(self):
        self.mask = pd.Series(True, self.df.index)

    def filter_data(self):
        self.df_to_plot = self.df[self.mask].groupby([self.colname_case,
                                                      'month'])['value'].mean().reset_index()

    def update(self):
        self.fig.layout.title.text = self.generate_title()
        for i, trace in enumerate(self.fig.data):
            case = self.cases[i]
            mask = (self.df_to_plot[self.colname_case] == case)
            df = self.df_to_plot[mask].set_index(
                'month').reindex(WY_MONTHS.values())
            trace.y = df[self.colname_y]


class PlotMonthlyBarWithControls(ExportPlotForStationsMixin,
                                 SaveDataMixin, ShowDataMixin,
                                 FilterWateryearTypeMixin,
                                 FilterStationMixin,
                                 FilterVariableMixin,
                                 PlotMonthlyBarBase):
    def __init__(self, *args, **kwargs):
        """
        """
        super().__init__(*args, **kwargs)

    def filter_data(self):
        self.make_all_true_mask()
        super().filter_data()

    def plot(self):
        self.widgets = ipw.VBox((self.fig,
                                 ipw.HBox((self.dd_variable,
                                           self.dd_station)),
                                 self.box_yeartypes,
                                 ipw.HBox((self.tb_showdata,
                                           self.tb_savedata)),
                                 self.box_exportplots,
                                 self.lb_msg))
        return self.widgets


class PlotBoxBase(PlotNotebookBase):
    def __init__(self, df, df_stations, options,
                 colname_x='time', colname_y='value',
                 colname_variable='variable', colname_case='scenario_name',
                 colname_station_id='station'):
        super().__init__()
        self.df = df
        self.df_stations = df_stations
        self.options = {} if options is None else options
        self.colname_x = colname_x
        self.colname_y = colname_y
        self.colname_variable = colname_variable
        self.colname_case = colname_case
        self.colname_station_id = colname_station_id
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
        xaxis_name = self.options.get('xaxis_name',
                                      'EC (micromhos/cm)')
        layout = go.Layout(template='seaborn',
                           title=dict(text=title),
                           xaxis=dict(title=xaxis_name)
                           )
        for case in self.cases:
            mask = (self.df_to_plot[self.colname_case] == case)
            data.append(go.Box(name=case,
                               x=self.df_to_plot[mask][self.colname_y]))
        self.fig = go.FigureWidget(data=data, layout=layout)

    def make_all_true_mask(self):
        self.mask = pd.Series(True, self.df.index)

    def filter_data(self):
        self.df_to_plot = self.df[self.mask]

    def update(self):
        for i, trace in enumerate(self.fig.data):
            case = self.cases[i]
            mask = self.df_to_plot[self.colname_case] == case
            y = self.df_to_plot[mask][self.colname_y]
            trace.x = y


class PlotExceedanceBase(PlotNotebookBase):
    def __init__(self, df, df_stations, options,
                 colname_x='time', colname_y='value',
                 colname_variable='variable', colname_case='scenario_name',
                 colname_station_id='station'):
        super().__init__()
        self.df = df
        self.df_stations = df_stations
        self.options = {} if options is None else options
        self.colname_x = colname_x
        self.colname_y = colname_y
        self.colname_variable = colname_variable
        self.colname_case = colname_case
        self.colname_station_id = colname_station_id
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
        yaxis_name = self.options.get('yaxis_name',
                                      'EC (micromhos/cm)')
        layout = go.Layout(template='seaborn',
                           title=dict(text=title),
                           yaxis=dict(title=yaxis_name)
                           )
        for case in self.cases:
            mask = (self.df_to_plot[self.colname_case] == case)
            yval = self.df_to_plot[mask][self.colname_y].sort_values(
                ascending=False).values
            n = len(yval)
            xval = np.arange(1, n + 1) / n * 100.
            data.append(go.Scatter(x=xval,
                                   y=yval,
                                   name=case))
        self.fig = go.FigureWidget(data=data, layout=layout)

    def make_all_true_mask(self):
        self.mask = pd.Series(True, self.df.index)

    def filter_data(self):
        self.df_to_plot = self.df[self.mask]

    def update(self):
        self.fig.layout.title.text = self.generate_title()
        for i, trace in enumerate(self.fig.data):
            case = self.cases[i]
            mask = self.df_to_plot[self.colname_case] == case
            yval = self.df_to_plot[mask][self.colname_y].sort_values(
                ascending=False).values
            n = len(yval)
            xval = np.arange(1, n + 1) / n * 100.
            trace.x = xval
            trace.y = yval


class PlotStepWithControls(ExportPlotForStationsMixin,
                           SaveDataMixin, ShowDataMixin,
                           FilterStationMixin,
                           FilterVariableMixin, PlotStepBase):
    def __init__(self, *args, **kwargs):
        """
        """
        super().__init__(*args, **kwargs)

    def filter_data(self):
        self.make_all_true_mask()
        super().filter_data()

    def plot(self):
        self.widgets = ipw.VBox((self.fig,
                                 ipw.HBox((self.dd_variable,
                                           self.dd_station)),
                                 ipw.HBox((self.tb_showdata,
                                           self.tb_savedata)),
                                 self.box_exportplots,
                                 self.lb_msg))
        return self.widgets


class PlotBoxWithControls(ExportPlotForStationsMixin,
                          SaveDataMixin, ShowDataMixin,
                          FilterMonthMixin,
                          FilterWateryearTypeMixin,
                          FilterStationMixin,
                          FilterVariableMixin, PlotBoxBase):
    def __init__(self, *args, **kwargs):
        """
        """
        super().__init__(*args, **kwargs)

    def filter_data(self):
        self.make_all_true_mask()
        super().filter_data()

    def plot(self):
        self.widgets = ipw.VBox((self.fig,
                                 ipw.HBox((self.dd_variable,
                                           self.dd_station)),
                                 ipw.HBox((self.tb_showdata,
                                           self.tb_savedata)),
                                 self.box_yeartypes,
                                 self.box_months,
                                 self.box_exportplots,
                                 self.lb_msg))
        return self.widgets


class PlotExceedanceWithControls(ExportPlotForStationsMixin,
                                 SaveDataMixin, ShowDataMixin,
                                 FilterMonthMixin,
                                 FilterWateryearTypeMixin,
                                 FilterStationMixin,
                                 FilterVariableMixin, PlotExceedanceBase):

    def __init__(self, *args, **kwargs):
        """
        """
        super().__init__(*args, **kwargs)

    def filter_data(self):
        self.make_all_true_mask()
        super().filter_data()

    def plot(self):
        self.widgets = ipw.VBox((self.fig,
                                 ipw.HBox((self.dd_variable,
                                           self.dd_station)),
                                 ipw.HBox((self.tb_showdata,
                                           self.tb_savedata)),
                                 self.box_yeartypes,
                                 self.box_months,
                                 self.box_exportplots,
                                 self.lb_msg))
        return self.widgets


class PlotStepWithRegulationBase(PlotNotebookBase):
    def __init__(self, df, df_reg, df_stations, options,
                 colname_x='time', colname_y='value',
                 colname_variable='variable', colname_case='scenario_name',
                 colname_station_id='station'):
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
        self.preprocess_data()
        self.create_widgets()
        self.create_figure()

    def preprocess_data(self):
        cases = self.df[self.colname_case].unique()
        dfs = []
        for case in cases:
            for variable in self.df[self.colname_variable].unique():
                for station_id in self.df_stations['ID'].unique():
                    mask = ((self.df[self.colname_case] == case) & (
                        self.df[self.colname_variable] == variable) &
                        (self.df[self.colname_station_id] == station_id))
                    df = self.df[mask].set_index('time').rolling('14d')[
                        'value'].mean().reset_index()
                    df.rename(columns={'index': 'time'}, inplace=True)
                    df[self.colname_station_id] = station_id
                    df[self.colname_variable] = variable + '-14DAY'
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
        yaxis_name = self.options.get('yaxis_name',
                                      'EC (micromhos/cm)')
        layout = go.Layout(template='seaborn',
                           title=dict(text=title),
                           #    yaxis=dict(rangemode='tozero')
                           yaxis=dict(title=yaxis_name)
                           )
        for case in self.cases_to_plot:
            mask = (self.df_to_plot[self.colname_case] == case)
            # Hard-wired, assuming that the regulation comes with this name.
            if case == self.regulation_name:
                data.append(go.Scatter(x=self.df_to_plot[mask]['time'],
                                       y=self.df_to_plot[mask][self.colname_y],
                                       line={'shape': 'hv'},
                                       fill='tozeroy',
                                       name=case))
            else:
                data.append(go.Scatter(x=self.df_to_plot[mask]['time'],
                                       y=self.df_to_plot[mask][self.colname_y],
                                       line={'shape': 'hv'},
                                       name=case))
        self.fig = go.FigureWidget(data=data, layout=layout)

    def make_all_true_mask(self):
        self.mask = pd.Series(True, self.df.index)

    def filter_data(self):
        self.df_to_plot = self.df[self.mask]

    def update(self):
        self.fig.layout.title.text = self.generate_title()
        for i, trace in enumerate(self.fig.data):
            case = self.cases_to_plot[i]
            mask = self.df_to_plot[self.colname_case] == case
            x = self.df_to_plot[mask]['time']
            trace.x = x
            y = self.df_to_plot[mask][self.colname_y]
            trace.y = y


class PlotStepWithRegulation(ExportPlotForStationsMixin,
                             SaveDataMixin, ShowDataMixin,
                             FilterStationMixin,
                             PlotStepWithRegulationBase):
    def __init__(self, *args, **kwargs):
        """
        """
        super().__init__(*args, **kwargs)

    def filter_data(self):
        self.make_all_true_mask()
        super().filter_data()

    def plot(self):
        self.widgets = ipw.VBox((self.fig,
                                 ipw.HBox((self.dd_station,)),
                                 ipw.HBox((self.tb_showdata,
                                           self.tb_savedata,)),
                                 self.box_exportplots,
                                 self.lb_msg))
        return self.widgets


class PlotExceedanceWithRegulationBase(PlotNotebookBase):
    def __init__(self, df, df_reg, df_stations, options,
                 colname_x='time', colname_y='value',
                 colname_variable='variable', colname_case='scenario_name',
                 colname_station_id='station'):
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
        self.preprocess_data()
        self.create_widgets()
        self.create_figure()

    def preprocess_data(self):
        cases = self.df[self.colname_case].unique()
        dfs = []
        for case in cases:
            for variable in self.df[self.colname_variable].unique():
                for station_id in self.df_stations['ID'].unique():
                    mask = ((self.df[self.colname_case] == case) & (
                        self.df[self.colname_variable] == variable) &
                        (self.df[self.colname_station_id] == station_id))
                    ds_14d = self.df[mask].set_index('time').rolling('14d')[
                        'value'].mean()
                    mask_reg = (
                        self.df_reg[self.colname_station_id] == station_id)
                    ds_reg = self.df_reg[mask_reg][self.df_reg[mask_reg]
                                                   ['value'] > 0.].set_index('time')['value']
                    ds_diff = ds_14d.loc[ds_reg.index] - ds_reg
                    df = ds_diff.to_frame().reset_index()
                    df.rename(columns={'index': 'time'})
                    df[self.colname_station_id] = station_id
                    df[self.colname_variable] = variable + '-14DAY-DIFF'
                    df[self.colname_case] = case
                    dfs.append(df)
        self.df = pd.concat(dfs)
        self.cases = self.df[self.colname_case].unique()

    def generate_title(self):
        return f"{self.station_selected}<br>(Scenario minus Standard)"

    def create_figure(self):
        self.filter_data()
        data = []
        title = self.generate_title()
        xaxis_name = self.options.get('xaxis_name',
                                      'Probability of Compliance (%)')
        yaxis_name = self.options.get('yaxis_name',
                                      'Difference in EC (micromhos/cm)')
        layout = go.Layout(template='seaborn',
                           title=dict(text=title),
                           yaxis=dict(zeroline=True,
                                      zerolinecolor='#000000',
                                      title=yaxis_name,
                                      rangemode='tozero'),
                           xaxis=dict(title=xaxis_name)
                           )
        results = {'Scenario': [],
                   '# of Days Standards are Applicable': [],
                   '# of Days Violated': [],
                   r'% of Days Violated': []}
        for case in self.cases:
            mask = (self.df_to_plot[self.colname_case] == case)
            yval = self.df_to_plot[mask][self.colname_y].sort_values(
                ascending=True)
            n = yval.count()
            xval = np.arange(1, n + 1) / n * 100.
            name = case
            data.append(go.Scatter(x=xval,
                                   y=yval.values,
                                   name=case))
            results['Scenario'].append(case)
            results['# of Days Standards are Applicable'].append(n)
            n_violated = yval[yval > 0.].count()
            results['# of Days Violated'].append(n_violated)
            results[r'% of Days Violated'].append(
                f'{n_violated / n * 100.:.2f}')
        self.fig = go.FigureWidget(data=data, layout=layout)
        self.df_results = pd.DataFrame(data=results)
        self.results = go.FigureWidget(data=[go.Table(
            header=dict(values=[[v] for v in self.df_results.columns]),
            cells=dict(values=[self.df_results[k] for k in self.df_results.columns],
                       height=30))],
            layout=go.Layout(
                template='seaborn',
                height=(self.df_results.shape[0] + 2) * 20 + 100,
                margin=dict(t=30, b=10),
                font=dict(size=14)
        ))

    def make_all_true_mask(self):
        self.mask = pd.Series(True, self.df.index)

    def filter_data(self):
        self.df_to_plot = self.df[self.mask]

    def update(self):
        self.fig.layout.title.text = self.generate_title()
        results = {'Scenario': [],
                   '# of Days Standards are Applicable': [],
                   '# of Days Violated': [],
                   r'% of Days Violated': []}
        for i, trace in enumerate(self.fig.data):
            case = self.cases[i]
            mask = self.df_to_plot[self.colname_case] == case
            yval = self.df_to_plot[mask][self.colname_y].sort_values(
                ascending=True)
            n = yval.count()
            xval = np.arange(1, n + 1) / n * 100.
            results['Scenario'].append(case)
            results['# of Days Standards are Applicable'].append(n)
            n_violated = yval[yval > 0.].count()
            results['# of Days Violated'].append(n_violated)
            results[r'% of Days Violated'].append(
                f'{n_violated / n * 100.:.2f}')
            trace.x = xval
            trace.y = yval
        self.df_results = pd.DataFrame(data=results)
        self.results.data[0].cells.values = [self.df_results[k]
                                             for k in self.df_results.columns]


class PlotExceedanceWithRegulation(ExportPlotForStationsMixin,
                                   SaveDataMixin, ShowDataMixin,
                                   FilterStationMixin,
                                   PlotExceedanceWithRegulationBase):
    def __init__(self, *args, **kwargs):
        """
        """
        super().__init__(*args, **kwargs)

    def filter_data(self):
        self.make_all_true_mask()
        super().filter_data()

    def plot(self):
        self.widgets = ipw.VBox((self.fig,
                                 self.results,
                                 ipw.HBox((self.dd_station,)),
                                 ipw.HBox((self.tb_showdata,
                                           self.tb_savedata)),
                                 self.box_exportplots,
                                 self.lb_msg))
        return self.widgets
