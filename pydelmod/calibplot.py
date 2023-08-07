from operator import add
import panel as pn
from scipy import stats
import pandas as pd
# display stuff
import hvplot.pandas
import holoviews as hv
from holoviews import opts
# styling plots
from bokeh.themes.theme import Theme
from bokeh.themes import built_in_themes
#
from pydsm.functions import tsmath
from pydsm import postpro
import datetime
import sys
import logging
## - Generic Plotting Functions ##
import pyhecdss
import numpy as np
import copy
import re


def parse_time_window(timewindow):
    """
    Args:
        timewindow (str, optional): time window for plot. Must be in format: 'YYYY-MM-DD:YYYY-MM-DD'

    Returns:
        list:str containing starting and ending times
    """
    return_list = []
    try:
        parts = timewindow.split(":")
        for p in parts:
            date_parts = [int(i) for i in p.split('-')]
            return_list.append(date_parts)
    except:
        errmsg = 'error in calibplot.parse_time_window, while parsing timewindow. Timewindow must be in format yyyy-mm-dd:yyyy-mm-dd or ' + \
            'yyyy-mm-dd hhmm:yyyy-mm-dd hhmm. Ignoring timewindow'
        print(errmsg)
        logging.error(errmsg)
    return return_list


def tsplot(dflist, names, timewindow=None, zoom_inst_plot=False):
    """Time series overlay plots

    Handles missing DataFrame, just put None in the list

    Args:
        dflist (List): Time-indexed DataFrame list
        names (List): Names list (same size as dflist)
        timewindow (str, optional): time window for plot. Must be in format: 'YYYY-MM-DD:YYYY-MM-DD'
        zoom_inst_plot (bool): if true, display only data in timewindow for plot
    Returns:
        Overlay: Overlay of Curve
    """
    start_dt = None
    end_dt = None
    if dflist[0] is not None:
        start_dt = dflist[0].index.min()
        end_dt = dflist[0].index.max()
    if zoom_inst_plot and (timewindow is not None):
        try:
            parts = timewindow.split(':')
            start_dt = parts[0]
            end_dt = parts[1]
        except:
            errmsg = 'error in calibplot.tsplot'
            print(errmsg)
            logging.error(errmsg)
    # This doesn't work. Need to find a way to get this working.
    # plt = [df[start_dt:end_dt].hvplot(label=name, x_range=(timewindow)) if df is not None else hv.Curve(None, label=name)

    from bokeh.models import DatetimeTickFormatter




    plt = [df[start_dt:end_dt].hvplot(label=name, xformatter=DatetimeTickFormatter(years='%b-%Y',months='%b-%y', days='%d-%b-%y')) if df is not None else hv.Curve(None, label=name)
        for df, name in zip(dflist, names)]
    plt = [c.redim(**{c.vdims[0].name:c.label, c.kdims[0].name: 'Time'})
        if c.name != '' else c for c in plt]
    return hv.Overlay(plt)


def scatterplot(dflist, names, index_x=0):
    """Scatter plot overlay with index_x determining the x (independent variable)

    Args:
        dflist (List): DataFrame list
        names (List): Names list
        index_x (int, optional): Index for determining independent variable. Defaults to 0.

    Returns:
        Overlay: Overlay of Scatter
    """
    dfa = pd.concat(dflist, axis=1)
    dfa.columns = names
    dfa = dfa.resample('D').mean()
    return dfa.hvplot.scatter(x=dfa.columns[index_x], hover_cols='all')


def remove_data_for_time_windows_thresholds(df: pd.DataFrame, time_window_exclusion_list_str, invert_selection=False, upper_threshold=None):
    """removes data from dataframe that is within time windows in the time_window_exclusion_list
    Args:
        df (DataFrame): The DataFrame from which to remove data
        time_window_exclusion_list_str (str): A string consisting of one or more time windows separated by commas, each time window 
        using the format 'yyyy-mm-dd_yyyy-mm-dd' Data in each of the specified time windows will be excluded from the metrics calculations
        invert_selection (bool): If True, keep data in the time windows rather than removing it.
        threshold_value (float): If specified, and if invert_selection==True, then data will be retained if value is above threshold OR 
            datetime is outside all specified timewindows.
    Returns:
        DataFrame: DataFrame with data removed
    """
    # df = df.copy()
    cols = df.columns
    if upper_threshold is None:
        upper_threshold = 999999
    else:
        if(len(str(upper_threshold))>0):
            upper_threshold = float(upper_threshold)
        else:
            upper_threshold = 999999

    time_window_exclusion_list = None
    if time_window_exclusion_list_str is not None and len(time_window_exclusion_list_str.strip())>0:
        time_window_exclusion_list = time_window_exclusion_list_str.split(',')
    if (time_window_exclusion_list is not None and len(time_window_exclusion_list) > 0 and df is not None):
        tw_index = 0
        last_tw = None

        if invert_selection:
            # set all values NOT in any timewindow to nan.
            cols = df.columns
            df['outside_all_tw'] = True
            df['above_threshold'] = False
            df['keep_inverted'] = False
            for tw in time_window_exclusion_list:
                start_dt_str, end_dt_str = tw.split('_')
                df.loc[((df.index>=start_dt_str) & (df.index<end_dt_str)), 'outside_all_tw'] = False
            df.loc[(df[cols[0]]>=upper_threshold), 'above_threshold'] = True
            df.loc[((df['outside_all_tw']==False) | (df['above_threshold']==True)), 'keep_inverted'] = True
            df.loc[df['keep_inverted']==False, cols[0]] = np.nan
            df.drop(columns=['outside_all_tw', 'above_threshold', 'keep_inverted'], inplace=True)
            # df[(df.index>=pd.Timestamp(last_end_dt_str)) & (df.index<pd.Timestamp(start_dt_str)) & (df[cols[0]] < threshold_value)] = np.nan
            # conditions = [ (df.index >= pd.Timestamp(s)) & (df.index <= pd.Timestamp(e)) for s,e in array_of_tuples] # [(3,5), (19, 38)]
            # functools.reduce
            # c=conditions[0]
            # for c2 in conditions[1:]: 
            #  c = c | c2
            # df[c] = np.nan
            # date_range_list = []
            # start_dt_list = []
            # end_dt_list = []
            # for tw in time_window_exclusion_list:
            #     start_dt_str, end_dt_str = tw.split('_')
            #     # date_range_list.append(pd.date_range(start=pd.Timestamp(start_dt_str), end=pd.Timestamp(end_dt_str, freq='15T')))
            #     start_dt_list.append(start_dt_str)
            #     end_dt_list.append(end_dt_str)        
            # print('*****************************************************************************************')
            # print('lengths of start, end date lists='+str(len(start_dt_list))+','+str(len(end_dt_list)))
            # print('*****************************************************************************************')
            # # if the timestamp is outside every time window, AND is above the threshold
            # df[(all((df.index < start_dt) | (df.index >= end_dt)) for start_dt, end_dt in zip(start_dt_list, end_dt_list)) & df>=threshold_value] = np.nan
            # # df[all(df.index not in date_range for date_range in date_range_list) & (df[cols[0]] < threshold_value)] = np.nan
            # # df[test_function(df, start_dt_list, end_dt_list) & df>=threshold_value] = np.nan




        for tw in time_window_exclusion_list:
            if len(tw)>0:
                start_dt_str, end_dt_str = tw.split('_')
                if not invert_selection:
                    # remove data in the time windows
                    # This is the old way: not good for plotting, because it becomes an ITS
                    # df = df[(df.index < start_dt_str) | (df.index > end_dt_str)]
                    # df[start_dt_str:end_dt_str] = np.nan
                    df[((df.index>pd.Timestamp(start_dt_str)) & (df.index<=pd.Timestamp(end_dt_str))) | (df[cols[0]]>=upper_threshold)] = np.nan
                # else:
                #     # keep data in the timewindows, and remove all other data, except those that are above the threshold
                #     if tw_index == 0:
                #         df[(df.index<=pd.Timestamp(start_dt_str)) & (df[cols[0]]<threshold_value)] = np.nan
                #     else:
                #         # if in any time window
                #         last_start_dt_str, last_end_dt_str = last_tw.split('_')
                #         # df[last_end_dt_str:start_dt_str | df < threshold_value] = np.nan
                #         #     # if the timestamp is outside every time window, AND is above the threshold
                #         # df[(all((df.index < start_dt) | (df.index >= end_dt)) for start_dt, end_dt in zip(start_dt_list, end_dt_list)) & df>=threshold_value] = np.nan

                #         # df[(df.index>=pd.Timestamp(last_end_dt_str)) & (df.index<pd.Timestamp(start_dt_str)) & (df[cols[0]] < threshold_value)] = np.nan
                # last_tw = tw
            tw_index += 1
        # now remove the data after the end of the last timewindow
        # if invert_selection and last_tw is not None and len(last_tw)>0:
        #     last_start_dt_str, last_end_dt_str = last_tw.split('_')
        #     df[(df.index>=pd.Timestamp(last_end_dt_str)) & (df[cols[0]] < threshold_value)] = np.nan
    elif upper_threshold is not None:
        if not invert_selection:
            df[df>=upper_threshold] = np.nan
        else:
            df[df<upper_threshold] = np.nan
    return df


# def remove_data_for_time_windows_thresholds(df: pd.DataFrame, time_window_exclusion_list_str, invert_selection=False, upper_threshold=None, \
#     lower_threshold=None):
#     """removes data from dataframe that is within time windows in the time_window_exclusion_list
#         if data masking does not remove any data (which could happen if invert_selection=True and the data masking timewindow is outside the 
#         time window of the data set), then this will return a dataframe with only nans. Code that calls this method must be prepared to
#         deal with this situation.
#     Args:
#         df (DataFrame): The DataFrame from which to remove data
#         time_window_exclusion_list_str (str): A string consisting of one or more time windows separated by commas, each time window 
#         using the format 'yyyy-mm-dd_yyyy-mm-dd' Data in each of the specified time windows will be excluded from the metrics calculations
#         invert_selection (bool): If True, keep data in the time windows rather than removing it.
#         upper_threshold (float): If specified, and if invert_selection==True, then data will be retained if value is above threshold OR 
#             datetime is outside all specified timewindows.
#         lower_threshold (float): If specified, and if invert_selection==True, then data will be retained if value is below threshold OR 
#             datetime is outside all specified timewindows.
#     Returns:
#         DataFrame: DataFrame with data removed
#     """
#     # df = df.copy()
#     cols = df.columns
#     if upper_threshold is None:
#         upper_threshold = 999999
#     else:
#         if(len(str(upper_threshold))>0):
#             upper_threshold = float(upper_threshold)
#         else:
#             upper_threshold = 999999

#     if lower_threshold is None:
#         lower_threshold = -999999
#     else:
#         if(len(str(lower_threshold))>0):
#             lower_threshold = float(lower_threshold)
#         else:
#             lower_threshold = -999999
    
#     time_window_exclusion_list = None
#     if time_window_exclusion_list_str is not None and len(time_window_exclusion_list_str.strip())>0:
#         time_window_exclusion_list = time_window_exclusion_list_str.split(',')
#     if (time_window_exclusion_list is not None and len(time_window_exclusion_list) > 0 and df is not None):
#         tw_index = 0
#         last_tw = None

#         if invert_selection:
#             # set all values NOT in any timewindow to nan.
#             cols = df.columns
#             df['outside_all_tw'] = True
#             df['above_upper_threshold'] = False
#             df['below_lower_threshold'] = False
#             df['keep_inverted'] = False
#             for tw in time_window_exclusion_list:
#                 start_dt_str, end_dt_str = tw.split('_')
#                 df.loc[((df.index>=start_dt_str) & (df.index<end_dt_str)), 'outside_all_tw'] = False
#             df.loc[(df[cols[0]]>=upper_threshold), 'above_lower_threshold'] = True
#             df.loc[(df[cols[0]]<=lower_threshold), 'below_lower_threshold'] = True
#             df.loc[((df['outside_all_tw']==False) | (df['above_upper_threshold']==True) | (df['below_lower_threshold']==True)), 'keep_inverted'] = True
#             df.loc[df['keep_inverted']==False, cols[0]] = np.nan
#             df.drop(columns=['outside_all_tw', 'above_upper_threshold', 'below_lower_threshold', 'keep_inverted'], inplace=True)

#         for tw in time_window_exclusion_list:
#             if len(tw)>0:
#                 start_dt_str, end_dt_str = tw.split('_')
#                 if not invert_selection:
#                     # remove data in the time windows
#                     # This is the old way: not good for plotting, because it becomes an ITS
#                     # df = df[(df.index < start_dt_str) | (df.index > end_dt_str)]
#                     # df[start_dt_str:end_dt_str] = np.nan
#                     df[((df.index>pd.Timestamp(start_dt_str)) & (df.index<=pd.Timestamp(end_dt_str))) | \
#                         (df[cols[0]]>=upper_threshold) | (df[cols[0]]<=lower_threshold)] = np.nan
#             tw_index += 1
#     else:
#         if not invert_selection:
#             df[df>=upper_threshold] = np.nan
#             df[df<=lower_threshold] = np.nan
#         else:
#             df[df<upper_threshold] = np.nan
#             df[df>lower_threshold] = np.nan
#     return df


def calculate_metrics(dflist, names, index_x=0, location=None):
    """Calculate metrics between the index_x column and other columns

    Args:
        dflist (List): DataFrame list
        names (List): Names list
        index_x (int, optional): Index of base DataFrame. Defaults to 0.
        time_window_exclusion_list is not required here because the methods that call this method can remove data first.
        location (str): optional, for debugging. If specified, data frame before and after will be written to a file.
    Returns:
        DataFrame: DataFrame of metrics
    """
    dfa = pd.concat(dflist, axis=1)
    dfa = dfa.dropna()

    # dfa = remove_data_for_time_windows(dfa, time_window_exclusion_list, location, invert_selection=invert_timewindow_exclusion)
    dfa.dropna(inplace=True) # this is necessary for metrics
    # x_series contains observed data
    # y_series contains model output for each of the studies
    x_series = dfa.iloc[:, index_x]
    dfr = dfa.drop(columns=dfa.columns[index_x])
    names.remove(names[index_x])
    slopes, interceps, equations, r2s, pvals, stds = [], [], [], [], [], []
    mean_errors, nmean_errors, ses, nmses, mses, nmses, rmses, nrmses, percent_biases, nses, rsrs = [], [], [], [], [], [], [], [], [], [], []

    metrics_calculated = False
    if len(x_series) > 0:
        for col in dfr.columns:
            y_series = dfr.loc[:, col]

            if len(y_series) > 0:
                slope, intercep, rval, pval, std = stats.linregress(x_series, y_series)
                slopes.append(slope)
                interceps.append(intercep)
                sign = '-' if intercep <= 0 else '+'
                equation = 'y=%.2fx%s%.2f' % (slope, sign, abs(intercep))
                equations.append(equation)
                r2s.append(rval*rval)
                pvals.append(pval)
                stds.append(std)
                mean_errors.append(tsmath.mean_error(y_series, x_series))
                nmean_errors.append(tsmath.nmean_error(y_series, x_series))
                mses.append(tsmath.mse(y_series, x_series))
                nmses.append(tsmath.nmse(y_series, x_series))
                rmses.append(tsmath.rmse(y_series, x_series))
                nrmses.append(tsmath.nrmse(y_series, x_series))
                percent_biases.append(tsmath.percent_bias(y_series, x_series))
                nses.append(tsmath.nash_sutcliffe(y_series, x_series))
                rsrs.append(tsmath.rsr(y_series, x_series))
                metrics_calculated = True
            else:
                errmsg = 'calibplot.calculate_metrics: no y_series data found. Metrics can not be calculated.\n'
                print(errmsg)
                logging.info(errmsg)

    else:
        errmsg = 'calibplot.calculate_metrics: no x_series data found. Metrics can not be calculated.\n'
        print(errmsg)
        logging.info(errmsg)
    dfmetrics = None

    if metrics_calculated:
        dfmetrics = pd.concat([pd.DataFrame(arr)
                            for arr in (slopes, interceps, equations, r2s, pvals, stds, mean_errors, nmean_errors, mses, nmses, rmses, nrmses, \
                                percent_biases, nses, rsrs)], axis=1)
        dfmetrics.columns = ['regression_slope', 'regression_intercep', 'regression_equation',
                            'r2', 'pval', 'std', 'mean_error', 'nmean_error', 
                            'mse', 'nmse', 'rmse', 'nrmse', 'percent_bias', 'nash_sutcliffe', 'rsr']

        dfmetrics.index = names
    return dfmetrics


def regression_line_plots(dfmetrics, flow_in_thousands):
    """Create Slope from the metrics DataFrame (calculate_metrics function)

    Args:
        dfmetrics (List): DataFrame list

    Returns:
        tuple: Slope list, equations(str) list
    """
    slope_plots = None
    for i, row in dfmetrics.iterrows():
        slope = row['regression_slope']
        intercep = row['regression_intercep']
        intercep = intercep/1000.0 if flow_in_thousands else intercep

        slope_plot = hv.Slope(slope, y_intercept=intercep)
        slope_plots = slope_plot if slope_plots == None else slope_plots*slope_plot
    return slope_plots


def shift_cycle(cycle):
    """Shift cycle to left

    Args:
        cycle (Cycle): Holoview Cycle

    Returns:
        Cycle: Shifted to left with value shifted at right end
    """
    v = cycle.values
    v.append(v.pop(0))
    return hv.Cycle(v)


def tidalplot(df, high, low, name):
    """Tidal plot of series as Curve with high and low as Scatter with markers

    Args:
        df (DataFrame): Tidal signal
        high (DataFrame): Tidal highs
        low (DataFrame): Tidal lows
        name (str): label

    Returns:
        Overlay: Overlay of Curve and 2 Scatter
    """
    h = high.hvplot.scatter(label='high').opts(marker='^')
    l = low.hvplot.scatter(label='low').opts(marker='v')
    o = df.hvplot.line(label=name)
    plts = [h, l, o]
    plts = [c.redim(**{c.vdims[0].name:c.label, c.kdims[0].name: 'Time'}) for c in plts]
    return hv.Overlay(plts)


def kdeplot(dflist, names, xlabel):
    """Kernel Density Estimate (KDE) plots

    Args:
        dflist (List): DataFrame list
        names (List): str list (same size as dflist)
        xlabel (str): label for x axis

    Returns:
        Overlay : Overlay of Distribution
    """
    kdes = [df.hvplot.kde(label=name, xlabel=xlabel) for df, name in zip(dflist, names)]
    for kde in kdes:
        kde.opts(toolbar=None)
    return hv.Overlay(kdes)


# - Customized functions for calibration / validation templates
# Needed because of name with . https://github.com/holoviz/holoviews/issues/4714
def sanitize_name(name):
    return name.replace('.', ' ')

# class DataMaskingTimeSeries:
#     '''
#     This class is not working yet. It is intended to be used to mask data based on DSS gate data.
#     Maybe easier to just use the data exclusion time windows in the input files.
#     '''
#     def __init__(self, gate_studies, gate_location, gate_vartype, timewindow):
#         #     ----------------------------------------------------------
#         # gate_studies, gate_locations,gate_vartype=
#         # ----------------------------------------------------------
#         # [Study(name='Gate', dssfile='../../../timeseries2019/gates-v8-201912.dss')]
#         # [Location(name='DLC', bpart='DLC', description='Delta Cross-Channel Gate')]
#         # VarType(name='POS', units='')
#         # ----------------------------------------------------------
#         self.dssfile = gate_studies[0].dssfile
#         self.location = gate_location
#         self.bpart = self.location.bpart.upper()
#         self.vartype = gate_vartype.name
#         self.timewindow = timewindow
#         self.gate_time_series_tuple = None
#         self.gate_time_series_tuple = next(pyhecdss.get_ts(self.dssfile, '//%s/%s////' % (self.bpart, self.vartype)))
#         # try:
#         #     # self.gate_time_series_tuple = next(pyhecdss.get_ts(self.dssfile, '//%s/%s/%s///' % (self.bpart, self.vartype, self.timewindow)))
#         #     self.gate_time_series_tuple = next(pyhecdss.get_ts(self.dssfile, '//%s/%s////' % (self.bpart, self.vartype)))
#         #     print('DataMaskingTimeSeries constructor: type of df='+str(type(self.gate_time_series_tuple)))

#         # except StopIteration as e:
#         #     print('no data found for ' + self.dssfile + ',//%s/%s/%s///' % (self.bpart, self.vartype, self.timewindow))
#         #     logging.exception('pydsm.postpro.PostProCache.load: no data found')
#         self.time_series_df = self.get_time_series_df()

#     def get_time_series_df(self):
#         '''
#         for now, assume only one data set is being read
#         self.gate_time_series_tuple has 3 elements:
#         1. the dataframe
#         2. string = 'UNSPECIF' (probably units)
#         3. string = 'INST-VAL' (averaging)
#         '''
#         return_df = None
#         for t in self.gate_time_series_tuple:
#             if isinstance(t, pd.DataFrame):
#                 return_df = t
#         return return_df

    # def get_gate_value(self, location, datetime):
    #     '''
    #     returns the value of the gate time series at or before given date.
    #     reference: https://kanoki.org/2022/02/09/how-to-find-closest-date-in-dataframe-for-a-given-date/
    #     '''
    #     gate_value_index = self.gate_time_series_df.get_loc(datetime) - 1
    #     return self.gate_time_series_df[[gate_value_index]]['POS']

def build_calib_plot_template(studies, location, vartype, timewindow, include_inst_plot, tidal_template=False, manuscript_layout=False, flow_in_thousands=False, \
    units=None, inst_plot_timewindow=None, layout_nash_sutcliffe=False, obs_data_included=True, include_kde_plots=False, \
        zoom_inst_plot=False, gate_studies=None, gate_locations=None, gate_vartype=None, invert_timewindow_exclusion=False, \
            remove_data_above_threshold=True, mask_data=True, tech_memo_validation_metrics=False):
    """Builds calibration plot template

    Args:
        studies (List): Studies (name,dssfile)
        location (Location): name,bpart,description
        vartype (VarType): name,units
        timewindow (str): timewindow as start_date_str "-" end_date_str or "" for full availability
        include_inst_plot (bool): include instantaneous plot in layout
        tidal_template (bool, optional): If True include tidal plots. Defaults to False.
        manuscript_layout (bool, optional): If True, no header, no scatter plot, no metrics table
        flow_in_thousands (bool, optional): If True, template is for flow data, and
            1) y axis title will include the string '(1000 CFS)', and
            2) all flow values in the inst, godin, and scatter plots will be divided by 1000.
        units (str, optional): a string representing the units of the data. examples: CFS, FEET, UMHOS/CM.
            Included in axis titles if specified.
        inst_plot_timewindow (str, optional): Defines a separate timewindow to use for the instantaneous plot.
            Must be in format 'YYYY-MM-DD:YYYY-MM-DD'
        layout_nash_sutcliffe (bool, optional): if true, include Nash-Sutcliffe Efficiency in tables that are
            included in plot layouts. NSE will be included in summary tables--separate files containing only
            the equations and statistics for all locations.
        obs_data_included (bool, optional): If true, first study in studies list is assumed to be observed data.
            calibration metrics will be calculated.
        include_kde_plots (bool): If true, kde plots will be included. This is temporary for debugging
        zoom_inst_plot (bool): If true, instantaneous plots will display on data in the inst_plot_timewindow
        time_window_exclusion_list (list of time window strings in format yyyy-mm-dd hh:mm:ss_yyyy-mm-dd hh:mm:ss)
    Returns:
        dict of holoviews Column objects. Keys=['with', 'without'], meaning with toolbars and without toolbars. 
            values are holoviews Column objects, which are templates ready for rendering by display or save.
        dataframe: equations and statistics for all locations
    """
    all_data_found, pp = load_data_for_plotting(studies, location, vartype, timewindow)
    if not all_data_found:
        return None, None

    # data masking using gate data is not fully implemented yet.
    # gate_pp = []
    # data_masking_time_series_dict= {}
    # data_masking_df_dict = {}
    # if gate_studies is not None and gate_locations is not None and gate_vartype is not None:
    #     for gate_location in gate_locations:
    #         dmts = DataMaskingTimeSeries(gate_studies, gate_location, gate_vartype, timewindow)
    #         data_masking_time_series_dict.update({gate_location.name: dmts})
    #         data_masking_df_dict.update({gate_location.name: dmts.get_time_series_df()})
    # else:
    #     print('Not using gate information for plots/metrics data masking because insufficient information provided.')

    tsp = build_inst_plot(pp, location, vartype, flow_in_thousands=flow_in_thousands, manuscript_layout=manuscript_layout, units=units, inst_plot_timewindow=inst_plot_timewindow, zoom_inst_plot=zoom_inst_plot, mask_data=mask_data)
    gtsp = build_godin_plot(pp, location, vartype, flow_in_thousands=flow_in_thousands,  manuscript_layout=manuscript_layout, units=units, 
        time_window_exclusion_list=location.time_window_exclusion_list, invert_timewindow_exclusion=invert_timewindow_exclusion,
        threshold_value=location.threshold_value, remove_data_above_threshold=remove_data_above_threshold, mask_data=mask_data)

    # trying to add labels to beginning and end of x axis, but this isn't working.
    # print('gtsp='+str(gtsp))

    # # print('p='+str(p))
    # gtsp.xaxis.axis_label = "X-axis"

    # # create left and right labels
    # left_label = Label(x=10, y=0, text="Left Label", render_mode="css", text_font_size="10pt")
    # right_label = Label(x=p.plot_width-10, y=0, text="Right Label", render_mode="css", text_font_size="10pt")

    # # add labels to the plot
    # gtsp.add_layout(left_label, 'below')
    # gtsp.add_layout(right_label, 'below')


    scatter_plot_with_toolbar = None
    scatter_plot_without_toolbar = None
    dfdisplayed_metrics = None
    metrics_table = None
    kdeplots_with_toolbar = None
    kdeplots_without_toolbar = None
    # metrics_table_name = ''
 
    if obs_data_included:
        time_window_exclusion_list = location.time_window_exclusion_list
        scatter_plot_with_toolbar = build_scatter_plots(pp, flow_in_thousands=flow_in_thousands, units=units,
            time_window_exclusion_list = time_window_exclusion_list, invert_timewindow_exclusion=invert_timewindow_exclusion,
            threshold_value=location.threshold_value, remove_data_above_threshold=remove_data_above_threshold,toolbar_option='right', mask_data=mask_data)
        scatter_plot_without_toolbar = build_scatter_plots(pp, flow_in_thousands=flow_in_thousands, units=units,
            time_window_exclusion_list = time_window_exclusion_list, invert_timewindow_exclusion=invert_timewindow_exclusion,
            threshold_value=location.threshold_value, remove_data_above_threshold=remove_data_above_threshold, toolbar_option=None, mask_data=mask_data)

        df_displayed_metrics_dict = {}
        metrics_table_dict = {}
        # if gate_studies is not None and gate_locations is not None and gate_vartype is not None:
        #     dfdisplayed_metrics_open, metrics_table_open = build_metrics_table(studies, pp, location, vartype, tidal_template=tidal_template, \
        #         flow_in_thousands=flow_in_thousands, units=units, layout_nash_sutcliffe=False, data_masking_df_dict=data_masking_df_dict, gate_open=True,
        #         time_window_exclusion_list = time_window_exclusion_list)
        #     dfdisplayed_metrics_closed, metrics_table_closed = build_metrics_table(studies, pp, location, vartype, tidal_template=tidal_template, \
        #         flow_in_thousands=flow_in_thousands, units=units, layout_nash_sutcliffe=False, data_masking_df_dict=data_masking_df_dict, gate_open=False,
        #         time_window_exclusion_list=time_window_exclusion_list)
        #     df_displayed_metrics_dict.update({'open': dfdisplayed_metrics_open})
        #     df_displayed_metrics_dict.update({'closed': dfdisplayed_metrics_closed})
        #     metrics_table_dict.update({'open': metrics_table_open})
        #     metrics_table_dict.update({'closed': metrics_table_closed})
        # else:
        manuscript_metrics = manuscript_layout and flow_in_thousands

        dfdisplayed_metrics, metrics_table = build_metrics_table(studies, pp, location, vartype, tidal_template=tidal_template, flow_in_thousands=flow_in_thousands, \
            units=units, layout_nash_sutcliffe=False, time_window_exclusion_list=time_window_exclusion_list, \
                invert_timewindow_exclusion=invert_timewindow_exclusion,\
                    threshold_value=location.threshold_value, remove_data_above_threshold=remove_data_above_threshold, mask_data=mask_data, \
                        tech_memo_validation_metrics=tech_memo_validation_metrics, manuscript_metrics=manuscript_metrics)
        # df_displayed_metrics_dict.update({'all': dfdisplayed_metrics})
        # metrics_table_dict.update({'all': metrics_table})


        if include_kde_plots: 
            kdeplots_with_toolbar = build_kde_plots(pp, include_toolbar=True)
            kdeplots_without_toolbar = build_kde_plots(pp, include_toolbar=False)
            # kdeplots = build_kde_plots(pp)
    
    # # create plot/metrics template
    header_panel = pn.panel(f'## {location.description} ({location.name}/{vartype.name})')
    if manuscript_layout:
        header_panel = pn.panel('')
    # # do this if you want to link the axes
    # # tsplots2 = (tsp.opts(width=900)+gtsp.opts(show_legend=False, width=900)).cols(1)
    # # start_dt = dflist[0].index.min()
    # # end_dt = dflist[0].index.max()

    # temporary fix to add toolbar to all plots. eventually need to only inlucde toolbar if creating html file
    add_toolbars = True
    column_with_toolbar = create_layout(include_inst_plot, scatter_plot_with_toolbar, dfdisplayed_metrics, metrics_table, location, vartype, tsp, gtsp, kdeplots_with_toolbar, \
        tidal_template, add_toolbars, obs_data_included, include_kde_plots, header_panel, manuscript_layout=manuscript_layout)
    add_toolbars = False
    column_without_toolbar = create_layout(include_inst_plot, scatter_plot_without_toolbar, dfdisplayed_metrics, metrics_table, location, vartype, tsp, gtsp, kdeplots_without_toolbar, \
        tidal_template, add_toolbars, obs_data_included, include_kde_plots, header_panel, manuscript_layout=manuscript_layout)
    column_dict = {'with': column_with_toolbar, 'without': column_without_toolbar}

    # now merge all metrics dataframes, adding a column identifying the gate status
    # return_metrics_df = None
    # df_index = 0
    # for metrics_df_name in df_displayed_metrics_dict:
    #     metrics_df_name_list = []
    #     metrics_df = df_displayed_metrics_dict[metrics_df_name]
    #     for r in range(metrics_df.shape[0]):
    #         metrics_df_name_list.append(metrics_df_name)
    #     metrics_df['Gate Pos'] = metrics_df_name_list
    #     # move Gate Pos column to beginning
    #     cols = list(metrics_df)
    #     cols.insert(0, cols.pop(cols.index('Gate Pos')))
    #     metrics_df = metrics_df.loc[:, cols]
    #     # merge df into return_metrics_df
    #     if df_index == 0:
    #         return_metrics_df = metrics_df
    #     else:
    #         return_metrics_df.append(metrics_df)
    #     df_index += 1

    return column_dict, dfdisplayed_metrics

def create_layout(include_inst_plot, scatter_plot, dfdisplayed_metrics, metrics_table, location, vartype, tsp, gtsp, kdeplot_list, \
    tidal_template, add_toolbars, obs_data_included, include_kde_plots, header_panel, manuscript_layout=False):
    '''
    Creates Holoviews Column object with plots and metrics.

    manuscript_layout (bool, optional): if true, only first two items included in layout
    '''
    # Need to set clone=True when changing options below. This prevents changing the original objects.
    column = None
    index_to_title_dict = {1: '(a)', 2: '(b)', 3: '(c)', 4: '(d)', 5: '(e)', 6: '(f)', 7: '(g)', 8: '(h)', 9: '(i)', 10: '(j)'}
    if scatter_plot is None and dfdisplayed_metrics is None and metrics_table is None:
        print('build_calib_plot_template: cplot, dfdisplayedmetrics, metrics_table, and kdeplot are all None for location, vartype='+location.name+','+str(vartype))
    else:
        scatter_and_metrics_row = None
        col_row_index = 1
        column = pn.Column(header_panel)
        if include_inst_plot:
            if not add_toolbars:
                column.append(tsp.opts(width=900, toolbar=None, title=index_to_title_dict[col_row_index], legend_position='right', clone=True))
            else:
                column.append(tsp.opts(width=900, title=index_to_title_dict[col_row_index], legend_position='right', clone=True))
            col_row_index += 1
        if not add_toolbars:
            column.append(gtsp.opts(width=900, toolbar=None, title=index_to_title_dict[col_row_index], legend_position='right', clone=True))
        else:
            column.append(gtsp.opts(width=900, title=index_to_title_dict[col_row_index], legend_position='right', clone=True))
        col_row_index += 1

        if tidal_template:
            if obs_data_included and not manuscript_layout:
                if not add_toolbars:
                    scatter_and_metrics_row = pn.Row(scatter_plot.opts(shared_axes=False, toolbar=None, 
                                                                       title=index_to_title_dict[col_row_index], clone=True))
                else: 
                    scatter_and_metrics_row = pn.Row(scatter_plot.opts(shared_axes=False, 
                                                                       title=index_to_title_dict[col_row_index], clone=True))
                col_row_index += 1
                if metrics_table is not None:
                    # metrics_table_row = pn.Row(metrics_table.opts(title='(d)'))
                    scatter_and_metrics_row.append(metrics_table.opts(title=index_to_title_dict[col_row_index], 
                                                                      fontscale=1, clone=True))
                    col_row_index += 1
                column.append(scatter_and_metrics_row)

                if include_kde_plots:
                    kdeplot_list[0].opts(title=index_to_title_dict[col_row_index])
                    col_row_index+=1
                    kdeplot_list[1].opts(title=index_to_title_dict[col_row_index])
                    col_row_index+=1
                    kdeplot_row = kdeplot_list[0] + kdeplot_list[1]
                    if add_toolbars:
                        kdeplot_row = kdeplot_row.cols(2).opts(shared_axes=False).opts(
                                opts.Distribution(height=200, width=300))
                    else:
                        kdeplot_row = kdeplot_row.cols(2).opts(shared_axes=False, toolbar=None).opts(
                                opts.Distribution(height=200, width=300))
                    column.append(pn.Row(kdeplot_row))
        else:
            if obs_data_included and not manuscript_layout:
                scatter_and_metrics_row = pn.Row(scatter_plot.opts(shared_axes=False, title=index_to_title_dict[col_row_index], clone=True))
                col_row_index += 1
                if metrics_table is not None:
                    scatter_and_metrics_row.append(metrics_table.opts(title=index_to_title_dict[col_row_index], fontscale=1, clone=True))
                    col_row_index += 1
                column.append(scatter_and_metrics_row)
    return column

def load_data_for_plotting(studies, location, vartype, timewindow):
    """Loads data used for creating plots and metrics
    """
    # pp = [postpro.PostProcessor(study, location, vartype) for study in studies]
    pp = []
    all_data_found = True
    for study in studies:
        p = postpro.PostProcessor(study, location, vartype)
        pp.append(p)
    # this was commented out before
    # for p in pp:ed
        success = p.load_processed(timewindow=timewindow)
        if not success:
            errmsg = 'unable to load data for study|location %s|%s' % (str(study), str(location))
            print(errmsg)
            logging.info(errmsg)
            all_data_found = False
    if not all_data_found:
        errmsg = 'Not creating plots because data not found for location, vartype, timewindow = ' + str(location) +','+ str(vartype)+','+str(timewindow)+'\n'
        print(errmsg)
        logging.info(errmsg)
        return None, None
    return all_data_found, pp


def get_units(flow_in_thousands=False, units=None):
    """ create axis titles with units (if specified), and modify titles and data if displaying flow data in 1000 CFS 
    """
    unit_string = ''
    if flow_in_thousands and units is not None:
        unit_string = '(1000 %s)' % units
    elif units is not None:
        unit_string = '(%s)' % units
    return unit_string


def build_inst_plot(pp, location, vartype, flow_in_thousands=False, manuscript_layout=False, units=None, inst_plot_timewindow=None, \
    zoom_inst_plot=False, mask_data=True):
    """Builds calibration plot template

    Args:
        pp (List): postpro.PostProcessor objects created for each study
        location (Location): name,bpart,description
        vartype (VarType): name,units
        flow_in_thousands (bool, optional): If True, template is for flow data, and
            1) y axis title will include the string '(1000 CFS)', and
            2) all flow values in the inst, godin, and scatter plots will be divided by 1000.
        csdp_manuscript (bool, optional): If True, a few things will be different, such as cms vs cfs
        units (str, optional): a string representing the units of the data. examples: CFS, FEET, UMHOS/CM.
            Included in axis titles if specified.
        inst_plot_timewindow (str, optional): Defines a separate timewindow to use for the instantaneous plot.
            Must be in format 'YYYY-MM-DD:YYYY-MM-DD'

    Returns:
        tsp: A plot
    """
    gridstyle = {'grid_line_alpha': 1, 'grid_line_color': 'lightgrey'}
    unit_string = get_units(flow_in_thousands, units)
    y_axis_label = f'{vartype.name} @ {location.name} {unit_string}'
    # plot_data are scaled, if flow_in_thousands == True
    tsp_plot_data = [p.df for p in pp]
    cfs_to_cms = 0.028316847

    if flow_in_thousands:
        if manuscript_layout: 
            y_axis_label = 'Flow (CMS)'
            tsp_plot_data = [p.df * cfs_to_cms if p.df is not None else None for p in pp]
        else:
            tsp_plot_data = [p.df/1000.0 if p.df is not None else None for p in pp]

    # create plots: instantaneous, godin, and scatter

    i=0
    for tpd in tsp_plot_data:
        datatype = None
        if i==0:
            datatype = 'obs'
        elif i==1:
            datatype = 'model'
        # tpd.to_csv('plot_df_files/'+location.name+'_'+y_axis_label+'_'+datatype+'.csv') 
        i+=1


    tsp = tsplot(tsp_plot_data, [p.study.name for p in pp], timewindow=inst_plot_timewindow, zoom_inst_plot=zoom_inst_plot).opts(
        ylabel=y_axis_label, show_grid=True, gridstyle=gridstyle, shared_axes=False)
    tsp = tsp.opts(opts.Curve(color=hv.Cycle('Category10')))
    return tsp

def build_godin_plot(pp, location, vartype, flow_in_thousands=False, manuscript_layout=False, units=None, time_window_exclusion_list=None, \
    invert_timewindow_exclusion=False, threshold_value=None, remove_data_above_threshold=True, mask_data=True):
    """Builds calibration plot template

    Args:
        pp (List): postpro.PostProcessor objects created for each study
        location (Location): name,bpart,description
        vartype (VarType): name,units
        flow_in_thousands (bool, optional): If True, template is for flow data, and
            1) y axis title will include the string '(1000 CFS)', and
            2) all flow values in the inst, godin, and scatter plots will be divided by 1000.
        units (str, optional): a string representing the units of the data. examples: CFS, FEET, UMHOS/CM.
            Included in axis titles if specified.
        time_window_exclusion_list (str): a string containing timewindows separated by commas.
        invert_timewindow_exclusion (bool): if true, data in time_window_exclusion_list will be kept and all other data removed.
        remove_data_above_threshold (bool): if true, data above specified threshold value will be removed
    Returns:
        gtsp: A plot
    """
    gridstyle = {'grid_line_alpha': 1, 'grid_line_color': 'lightgrey'}
    unit_string = get_units(flow_in_thousands, units)
    y_axis_label = f'{vartype.name} @ {location.name} {unit_string}'
    godin_y_axis_label = 'Godin '+y_axis_label
    if manuscript_layout:
        godin_y_axis_label = 'Tidal Avg. Flow (CMS)'
    # plot_data are scaled, if flow_in_thousands == True
    # gtsp_plot_data = [p.gdf for p in pp]

    # if p.gdf is not None:

    # if flow_in_thousands:
    #     gtsp_plot_data = [p.gdf/1000.0 if p.gdf is not None else None for p in pp]

    # zoom in to desired timewindow: works, but doesn't zoom y axis, so need to fix later
    # if inst_plot_timewindow is not None:
    #     start_end_times = parse_time_window(inst_plot_timewindow)
    #     s = start_end_times[0]
    #     e = start_end_times[1]
    #     tsp.opts(xlim=(datetime.datetime(s[0], s[1], s[1]), datetime.datetime(e[0],e[1],e[2])))

    # remove data for specified time window for all time series
    # gtsp_plot_data = [remove_data_for_time_windows(p, time_window_exclusion_list_str=time_window_exclusion_list, 
    #     invert_selection=invert_timewindow_exclusion) if p is not None else None for p in gtsp_plot_data]

    # gtsp_plot_data = [remove_data_above_below_threshold(p, threshold_value, data_in_thousands=flow_in_thousands, remove_above=remove_data_above_threshold) \
    #     if p is not None else None for p in gtsp_plot_data]
    cfs_to_cms = 0.028316847
    gtsp_plot_data = []
    for p in pp:
        if p.gdf is not None:
            if mask_data:
                new_p = remove_data_for_time_windows_thresholds(p.gdf, time_window_exclusion_list_str=time_window_exclusion_list, invert_selection=invert_timewindow_exclusion, upper_threshold=threshold_value)
            else:
                new_p = p.gdf
            # new_p = remove_data_for_time_windows(p.gdf, time_window_exclusion_list, invert_selection=invert_timewindow_exclusion)
            # new_p = remove_data_above_below_threshold(new_p, threshold_value, data_in_thousands=flow_in_thousands, remove_above=remove_data_above_threshold)
            if manuscript_layout:
                new_p = new_p * cfs_to_cms
            elif flow_in_thousands:
                new_p = new_p/1000.0 if new_p is not None else None
            gtsp_plot_data.append(new_p)
        else:
            gtsp_plot_data.append(None)

    i=0
    for gpd in gtsp_plot_data:
        datatype = None
        if i==0:
            datatype = 'obs'
        elif i==1:
            datatype = 'model'

        # gpd.to_csv('plot_df_files/'+location.name+'_godin_'+y_axis_label+'_'+datatype+'.csv') 
        i+=1


    gtsp = tsplot(gtsp_plot_data, [p.study.name for p in pp]).opts(
        ylabel=godin_y_axis_label, show_grid=True, gridstyle=gridstyle)
    gtsp = gtsp.opts(opts.Curve(color=hv.Cycle('Category10')))

    # trying to add labels to beginning and end of x axis, but this isn't working.
    # print('gtsp='+str(gtsp))

    # # print('p='+str(p))
    # gtsp.xaxis.axis_label = "X-axis"

    # # create left and right labels
    # left_label = Label(x=10, y=0, text="Left Label", render_mode="css", text_font_size="10pt")
    # right_label = Label(x=p.plot_width-10, y=0, text="Right Label", render_mode="css", text_font_size="10pt")

    # # add labels to the plot
    # gtsp.add_layout(left_label, 'below')
    # gtsp.add_layout(right_label, 'below')



    return gtsp

# def build_scatter_plots(pp, location, vartype, flow_in_thousands=False, units=None, gate_pp=None, time_window_exclusion_list=None):
def build_scatter_plots(pp, flow_in_thousands=False, units=None, gate_pp=None, time_window_exclusion_list=None, \
    invert_timewindow_exclusion=False, threshold_value=None, remove_data_above_threshold=True, toolbar_option='right', mask_data=True):
    """Builds calibration plot template

    Args:
        pp (List): postpro.PostProcessor objects created for each study
        location (Location): name,bpart,description
        vartype (VarType): name,units
        flow_in_thousands (bool, optional): If True, template is for flow data, and
            1) y axis title will include the string '(1000 CFS)', and
            2) all flow values in the inst, godin, and scatter plots will be divided by 1000.
        units (str, optional): a string representing the units of the data. examples: CFS, FEET, UMHOS/CM.
            Included in axis titles if specified.
        time_window_exclusion_list (str): a string containing timewindows separated by commas.
        invert_timewindow_exclusion (bool): if true, data in time_window_exclusion_list will be kept and all other data removed.
    Returns:
        a plot object
    """
    gridstyle = {'grid_line_alpha': 1, 'grid_line_color': 'lightgrey'}
    unit_string = get_units(flow_in_thousands, units)

    # y_axis_label = f'{vartype.name} @ {location.name} {unit_string}'
    # godin_y_axis_label = 'Godin '+y_axis_label
    # plot_data are scaled, if flow_in_thousands == True
    # This is the old way:
    # gtsp_plot_data = [remove_data_for_time_windows(p.gdf, time_window_exclusion_list) for p in pp]
    # For some reason, this does not work, so the solution is below:
    # gtsp_plot_data = [remove_data_for_time_windows(p.gdf, time_window_exclusion_list).dropna(inplace=True) for p in pp]

    gtsp_plot_data = []
    splot_plot_data = []
    splot_metrics_data = []

    # splot_plot_data = [p.gdf.resample('D').mean() if p.gdf is not None else None for p in pp ]
    # splot_metrics_data = [p.gdf.resample('D').mean() if p.gdf is not None else None for p in pp]
    # if flow_in_thousands:
    #     gtsp_plot_data = [p.gdf/1000.0 if p.gdf is not None else None for p in pp]
    #     splot_plot_data = [p.gdf.resample('D').mean()/1000.0 if p.gdf is not None else None for p in pp]

    # if False, return None
    # this will happen if there are no data in the specified time period, or
    # we're trying to create the right hand side plot (to show plots and metrics for masked data),
    # and there are no data that have been masked. For example, this could happen if only one masking time window is specified,
    # and it's outside the time window of the data.
    any_data_left = True

    for p in pp:
        if mask_data:
            gpd = remove_data_for_time_windows_thresholds(p.gdf, time_window_exclusion_list, invert_selection=invert_timewindow_exclusion, upper_threshold=threshold_value)
        else:
            gpd = p.gdf
        # gpd = remove_data_for_time_windows(p.gdf, time_window_exclusion_list_str=time_window_exclusion_list, invert_selection=invert_timewindow_exclusion)
        # gpd = remove_data_above_below_threshold(gpd, threshold_value, data_in_thousands=flow_in_thousands, remove_above=remove_data_above_threshold)
        gpd.dropna(inplace=True)
        if gpd.notnull().sum()[0] <= 0:
            any_data_left = False
        else:
            spd_plot = None
            spd_metrics = gpd.resample('D').mean() if gpd is not None else None
            if flow_in_thousands:
                gpd = gpd/1000.0 if gpd is not None else None
            spd_plot = gpd.resample('D').mean() if gpd is not None else None

            gtsp_plot_data.append(gpd)
            splot_plot_data.append(spd_plot)
            splot_metrics_data.append(spd_metrics)

    # data have been removed; no need to pass time_window_exclusion_list to calculate_metrics calls
    if any_data_left:
        splot = None
        if splot_plot_data is not None and splot_plot_data[0] is not None:
            splot = scatterplot(splot_plot_data, [p.study.name for p in pp])\
                .opts(opts.Scatter(color=shift_cycle(hv.Cycle('Category10'))))\
                .opts(ylabel='Model', legend_position="top_left")\
                .opts(show_grid=True, frame_height=250, frame_width=250, data_aspect=1)\
                    .opts(toolbar=toolbar_option)

        dfdisplayed_metrics = None
        # calculate calibration metrics
        # slope_plots_dfmetrics = None
        # if gtsp_plot_data is not None and len(gtsp_plot_data) > 0 and gtsp_plot_data[0] is not None:
        #     slope_plots_dfmetrics = calculate_metrics(gtsp_plot_data, [p.study.name for p in pp])
        # dfmetrics = calculate_metrics([p.gdf for p in pp], [p.study.name for p in pp])

        # not using this any more
        dfmetrics = None
        if splot_metrics_data is not None:
            dfmetrics = calculate_metrics(splot_metrics_data, [p.study.name for p in pp])

        # dfmetrics_monthly = None
        # # if p.gdf is not None:
        # dfmetrics_monthly = calculate_metrics(
        #     [p.gdf.resample('M').mean() if p.gdf is not None else None for p in pp], [p.study.name for p in pp])

        # add regression lines to scatter plot, and set x and y axis titles
        slope_plots = None
        scatter_plot = None


        if dfmetrics is not None:
            slope_plots = regression_line_plots(dfmetrics, flow_in_thousands)
            scatter_plot = slope_plots.opts(opts.Slope(color=shift_cycle(hv.Cycle('Category10'))))*splot
            scatter_plot = scatter_plot.opts(xlabel='Observed ' + unit_string, ylabel='Model ' + unit_string, legend_position="top_left")\
                .opts(show_grid=True, frame_height=250, frame_width=250, data_aspect=1, show_legend=False)
        return scatter_plot
    else:
        return None

def create_hv_metrics_table(study_list, metrics_list_dict, metrics_list, width=580, fontscale=8):
    '''
    Create a Holoviews table displaying calibration metrics.
    '''
    metrics_list_list = [study_list.copy()]
    for m in metrics_list:
        if m is not 'Study':
            metrics_list_list.append(metrics_list_dict[m])
    metrics_list_tuple = tuple(metrics_list_list)
    metrics_table = hv.Table(metrics_list_tuple, metrics_list). opts(width=width, fontscale=fontscale)
    return metrics_table

def create_metrics_table_and_metrics_df(study_list, dfmetrics, location, vartype, gtsp_plot_data, pp, tidal_template, amp_avg_pct_errors, \
    amp_avg_phase_errors, layout_nash_sutcliffe, format_dict, tech_memo_validation_metrics=False, manuscript_metrics=False):
    '''
    Create dataframe and holoviews metrics table for calibration metrics.
    Metrics are selected based on options passed to method 
    '''
    dfdisplayed_metrics = None
    metrics_table = None
    # this is used for renaming column headers that come from the method that calculates metrics.
    # Don't make them too long, because they need to be displayed in a holoviews table with no wrapping.
    col_rename_dict = {'regression_equation': 'Equation', 'r2': 'R Squared', 'mean_error': 'Mean Error', 'nmean_error': 'NMean Error', 'nmse': 'NMSE', 
                        'nrmse': 'NRMSE', 'nash_sutcliffe': 'NSE', 'percent_bias': 'PBIAS', 'rsr': 'RSR', 'rmse': 'RMSE',
                        'mnly_regression_equation': 'Mnly Equation', 'mnly_r2': 'Mnly R Squared', 'mnly_mean_err': 'Mnly Mean Err', 'mnly_mean_error': 'Mnly Mean Err', 
                        'mnly_nmean_error': 'Mnly NMean Err', 'mnly_nmse': 'Mnly NMSE', 
                        'mnly_nrmse': 'Mnly NRMSE', 'mnly_nash_sutcliffe': 'Mnly NSE', 'mnly_percent_bias': 'Mnly PBIAS', 'mnly_rsr': 'Mnly RSR', 
                        'mnly_rmse': 'Mnly RMSE', 'Study': 'Study', 'Amp Avg %Err': 'Amp Avg %Err', 'Avg Phase Err': 'Avg Phase Err'}

    if dfmetrics is not None:
        if tidal_template:
            # ok to include things you don't need here, but don't exclude anything you might need later
            cols = ['regression_equation', 'r2', 'mean_error', 'nmean_error', 'nmse', 'nrmse', 'nash_sutcliffe', 'percent_bias', 'rsr']
            if tech_memo_validation_metrics:
                cols.append('rmse')
            if manuscript_metrics:
                cols = ['r2', 'percent_bias', 'rsr']

            dfdisplayed_metrics = dfmetrics.loc[:, cols]
            dfdisplayed_metrics['Amp Avg %Err'] = amp_avg_pct_errors
            dfdisplayed_metrics['Avg Phase Err'] = amp_avg_phase_errors
            dfdisplayed_metrics.index.name = 'DSM2 Run'

            # now create a holoviews table object displaying the metrics
            metrics_list_dict = {}
            for m in dfdisplayed_metrics.columns:
                if m is 'Equation':
                    metrics_list_dict.update({m: dfdisplayed_metrics[m].to_list()})
                else:
                    # metrics_list_dict.update({m: ['{:.2f}'.format(item) for item in dfdisplayed_metrics[m].to_list()] })
                    metrics_list_dict.update({m: [format_dict[m].format(item) for item in dfdisplayed_metrics[m].to_list()] })
            # if you need to change the list of metrics that will be displayed in the table, do it here.
            # every column name in metrics_list_for_hv_table must match a column name in dfdisplayed_metrics
            metrics_list_for_hv_table = None
            if layout_nash_sutcliffe:
                metrics_list_for_hv_table = ['Study', 'regression_equation', 'r2', 'mean_error', 'nmean_error', 'nmse', 'nrmse', 'nse', 'percent_bias', 'rsr']
            else:
                metrics_list_for_hv_table = ['Study', 'regression_equation', 'r2', 'mean_error', 'nmean_error', 'nmse', 'nrmse', 'percent_bias', 'rsr']

            if tech_memo_validation_metrics:
                metrics_list_for_hv_table.append('rmse')
            metrics_list_for_hv_table.append('Amp Avg %Err')
            metrics_list_for_hv_table.append('Avg Phase Err')
            # we now have metrics_list_for_hv_table, which is a list of metrics that we want to use
            # and metrics_list_dict, with key = metric name, value = list of values. Included is a list of Study names.
            # rename all the metrics names in the list, and in the dict (the keys)
            # rename list elements; see https://stackoverflow.com/questions/17295776/how-to-replace-elements-in-a-list-using-dictionary-lookup
            metrics_list_for_hv_table_renamed = list((pd.Series(metrics_list_for_hv_table)).map(col_rename_dict))
            # rename dictionary keys; see https://codereview.stackexchange.com/questions/263904/efficient-renaming-of-dict-keys-from-another-dicts-values-python
            # metrics_list_dict_renamed = {col_rename_dict[k]: v for k, v in metrics_list_dict.items()}
            metrics_list_dict_renamed = {col_rename_dict.get(k, k): v for k, v in metrics_list_dict.items()}

            # if manuscript_metrics:
            #     metrics_list_for_hv_table = ['Study', 'R Squared', 'PBIAS', 'RSR']

            metrics_table = create_hv_metrics_table(study_list, metrics_list_dict_renamed, metrics_list_for_hv_table_renamed)
        else:
            # not tidal: EC data
            dfmetrics_monthly = calculate_metrics(
                [g.resample('M').mean() if g is not None else None for g in gtsp_plot_data], [p.study.name for p in pp])
            # rename columns to include mnly (monthly)
            dfmetrics_monthly = dfmetrics_monthly.add_prefix('mnly_')

            # template for nontidal (EC) data
            # ok to include things you don't need here, but don't exclude anything you might need later
            cols = ['regression_equation', 'r2', 'mean_error', 'nmean_error', 'nmse', 'nrmse', 'rmse', 'nash_sutcliffe', 'percent_bias', 'rsr']

            dfdisplayed_metrics = dfmetrics.loc[:, cols]
            dfdisplayed_metrics = pd.concat(
                [dfdisplayed_metrics, dfmetrics_monthly.loc[:, ['mnly_mean_error', 'mnly_rmse', 'mnly_nmean_error', 'mnly_nrmse']]], axis=1)
            dfdisplayed_metrics.index.name = 'DSM2 Run'

            dfdisplayed_metrics.style.format(format_dict)
            # Ideally, the columns should be sized to fit the data. This doesn't work properly--replaces some values with blanks
            # metrics_table = pn.widgets.DataFrame(dfdisplayed_metrics, autosize_mode='fit_columns')
            metrics_list_dict = {}

            for m in dfdisplayed_metrics.columns:
                if m is 'Equation':
                    metrics_list_dict.update({m: dfdisplayed_metrics[m].to_list()})
                else:
                    metrics_list_dict.update({m: [format_dict[m].format(item) for item in dfdisplayed_metrics[m].to_list()] })
            # now create a holoviews table object displaying the metrics
            # if you need to change the list of metrics that will be displayed in the table, do it here.
            # every column name in metrics_list_for_hv_table must match a column name in dfdisplayed_metrics
            metrics_list_for_hv_table = None

            if layout_nash_sutcliffe:
                metrics_list_for_hv_table = ['Study', 'regression_equation', 'r2', 'mean_error', 'nmean_error', 'nmse', 'nrmse', 'nse', 'percent_bias', 'rsr']
            else:
                metrics_list_for_hv_table = ['Study', 'regression_equation', 'r2', 'mean_error', 'nmean_error', 'nmse', 'nrmse', 'percent_bias', 'rsr']

            if tech_memo_validation_metrics:
                metrics_list_for_hv_table = ['Study', 'regression_equation', 'r2', 'mean_error', 'rmse', 'mnly_mean_error', 'mnly_rmse']

            # we now have metrics_list_for_hv_table, which is a list of metrics that we want to use
            # and metrics_list_dict, with key = metric name, value = list of values. Included is a list of Study names.
            # rename all the metrics names in the list, and in the dict (the keys)
            # rename list elements; see https://stackoverflow.com/questions/17295776/how-to-replace-elements-in-a-list-using-dictionary-lookup
            metrics_list_for_hv_table_renamed = list((pd.Series(metrics_list_for_hv_table)).map(col_rename_dict))
            # rename dictionary keys; see https://codereview.stackexchange.com/questions/263904/efficient-renaming-of-dict-keys-from-another-dicts-values-python
            # metrics_list_dict_renamed = {col_rename_dict[k]: v for k, v in metrics_list_dict.items()}
            metrics_list_dict_renamed = {col_rename_dict.get(k, k): v for k, v in metrics_list_dict.items()}

            metrics_table = create_hv_metrics_table(study_list, metrics_list_dict_renamed, metrics_list_for_hv_table_renamed)
    else:
        print('build_metrics_table: dfmetrics is none, so not creating metrics table for location.name, vartype: '+location.name+','+str(vartype))

    return dfdisplayed_metrics, metrics_table

def build_metrics_table(studies, pp, location, vartype, tidal_template=False, flow_in_thousands=False, units=None,
                              layout_nash_sutcliffe=False, gate_pp=None, data_masking_df_dict=None, gate_open=True, 
                              time_window_exclusion_list=None, invert_timewindow_exclusion=False, threshold_value=None,
                              remove_data_above_threshold=True, mask_data=True, tech_memo_validation_metrics=False, manuscript_metrics=False):
    """Builds calibration plot template

    Args:
        studies (List): Studies (name,dssfile)
        pp (List): postpro.PostProcessor objects created for each study
        location (Location): name,bpart,description
        vartype (VarType): name,units
        tidal_template (bool, optional): If True include tidal plots. Defaults to False.
        flow_in_thousands (bool, optional): If True, template is for flow data, and
            1) y axis title will include the string '(1000 CFS)', and
            2) all flow values in the inst, godin, and scatter plots will be divided by 1000.
        units (str, optional): a string representing the units of the data. examples: CFS, FEET, UMHOS/CM.
            Included in axis titles if specified.
        layout_nash_sutcliffe (bool, optional): if true, include Nash-Sutcliffe Efficiency in tables that are
            included in plot layouts. NSE will be included in summary tables--separate files containing only
            the equations and statistics for all locations.
        gate_dss_file (str): path to DSS file with gate data, to be used for creating metrics tables separately 
            for gate open/closed conditions. This will only be done for a given location if a DSS path is
            specified in the location file in the gate_time_series field.
        data_masking_df_dict (dict of df): contains gate time series used for data masking.
        gate_open (bool): if true, calculate metrics for gate open condition (gate pos > 0) only.
        time_window_exclusion_list (list of strings): contains a list of time windows. Data within these time windows
            will not be used to calculate metrics or scatter plots.

    Returns:
        a list containing one or more table object(s). Will contain more then one object if a DSS path is specified
            in the location file in the gate_time_series field.
    """
    gridstyle = {'grid_line_alpha': 1, 'grid_line_color': 'lightgrey'}
    unit_string = get_units(flow_in_thousands, units)
    y_axis_label = f'{vartype.name} @ {location.name} {unit_string}'
    godin_y_axis_label = 'Godin '+y_axis_label
    # plot_data are scaled, if flow_in_thousands == True
    # gtsp_plot_data = [p.gdf for p in pp]
    gtsp_plot_data = []
    gpd = None
    for p in pp:
        if mask_data:
            gpd = remove_data_for_time_windows_thresholds(p.gdf, time_window_exclusion_list, invert_selection=invert_timewindow_exclusion, upper_threshold=threshold_value)
        else:
            gpd=p.gdf
        if flow_in_thousands:
            gpd = gpd/1000.0 if gpd is not None else None
        gpd.dropna(inplace=True)
        gtsp_plot_data.append(gpd)
    # data have been removed; no need to pass time_window_exclusion_list to calculate_metrics calls

    cfs_to_cms = 0.028316847
    if flow_in_thousands:
        if manuscript_metrics:
            splot_metrics_data = [g.resample('D').mean()*1000.0*cfs_to_cms if g is not None else None for g in gtsp_plot_data]
        else:
            splot_metrics_data = [g.resample('D').mean()*1000.0 if g is not None else None for g in gtsp_plot_data]
    else:
        splot_metrics_data = [g.resample('D').mean() if g is not None else None for g in gtsp_plot_data]

    dfdisplayed_metrics = None
    column = None
    dfmetrics = None
    if splot_metrics_data is not None:
        dfmetrics = calculate_metrics(splot_metrics_data, [p.study.name for p in pp])
    # display calibration metrics
    # create a list containing study names, excluding observed.
    dfdisplayed_metrics = None
    study_list = [study.name.replace('DSM2', '')
                for study in studies if study.name.lower() != 'observed']

    # calculate amp diff, amp % diff, and phase diff
    amp_avg_pct_errors = []
    amp_avg_phase_errors = []
    for p in pp[1:]:  # TODO: move this out of here. Nothing to do with plotting!
        p.process_diff(pp[0])
        amp_avg_pct_errors.append(float(p.amp_diff_pct.mean(axis=0)))
        amp_avg_phase_errors.append(float(p.phase_diff.mean(axis=0)))

    # using a Table object because the dataframe object, when added to a layout, doesn't always display all the values.
    # This could have something to do with inconsistent types.
    metrics_table = None
    format_dict = {'Equation': '{:s}', 'R Squared': '{:.2f}', 'Mean Error': '{:.1f}', 'NMean Error': '{:.3f}', 'NMSE': '{:.1}', 'NRMSE': '{:.4}',
            'Amp Avg %Err': '{:.1f}', 'Avg Phase Err': '{:.2f}', 'NSE': '{:.2f}', 'PBIAS': '{:.1f}', 'RSR': '{:.2f}',
            'Mnly Mean Err': '{:.1f}', 'Mnly RMSE': '{:.1f}', 'RMSE': '{:.2f}',
            'Mnly Equation': '{:s}', 'Mnly R Squared': '{:.2f}', 'Mnly Mean Error': '{:.1f}', 'Mnly NMean Error': '{:.3f}', 'Mnly NMSE': '{:.1}', 'Mnly NRMSE': '{:.4}',
            'Mnly Amp Avg %Err': '{:.1f}', 'Mnly Avg Phase Err': '{:.2f}', 'Mnly NSE': '{:.2f}', 'Mnly PBIAS': '{:.1f}', 'Mnly RSR': '{:.2f}',

            'regression_equation': '{:s}', 'r2': '{:.2f}', 'mean_error': '{:.1f}', 'nmean_error': '{:.3f}', 'nmse': '{:.1}', 'nrmse': '{:.4}',
            'rmse': '{:.4}', 'mnly_nmean_error': '{:4}','mnly_mean_err': '{:4}','mnly_mean_error': '{:4}', 'mnly_nrmse': '{:.4}', 'mnly_rmse': '{:.4}',
            'nash_sutcliffe': '{:.2f}', 'percent_bias': '{:.1f}', 'rsr': '{:.2f}'
            
            }



    dfdisplayed_metrics, metrics_table = create_metrics_table_and_metrics_df(study_list, dfmetrics, location, vartype, gtsp_plot_data,\
         pp, tidal_template, amp_avg_pct_errors, amp_avg_phase_errors, layout_nash_sutcliffe, format_dict, tech_memo_validation_metrics,\
            manuscript_metrics=manuscript_metrics)

    return dfdisplayed_metrics, metrics_table

def set_toolbar_autohide(plot, element):
    bokeh_plot = plot.state
    bokeh_plot.toolbar.autohide = True

def build_kde_plots(pp, amp_title='(e)', phase_title='(f)', include_toolbar=True):
    """Builds calibration plot template

    Args:
        pp (List): postpro.PostProcessor objects created for each study
        location (Location): name,bpart,description
        vartype (VarType): name,units
        timewindow (str): timewindow as start_date_str "-" end_date_str or "" for full availability
        flow_in_thousands (bool, optional): If True, template is for flow data, and
            1) y axis title will include the string '(1000 CFS)', and
            2) all flow values in the inst, godin, and scatter plots will be divided by 1000.
        units (str, optional): a string representing the units of the data. examples: CFS, FEET, UMHOS/CM.
            Included in axis titles if specified.

    Returns:
        a plot object
    """
    # plot_data are scaled, if flow_in_thousands == True
    # calculate amp diff, amp % diff, and phase diff
    amp_avg_pct_errors = []
    amp_avg_phase_errors = []
    for p in pp[1:]:  # TODO: move this out of here. Nothing to do with plotting!
        p.process_diff(pp[0])
        amp_avg_pct_errors.append(float(p.amp_diff_pct.mean(axis=0)))
        amp_avg_phase_errors.append(float(p.phase_diff.mean(axis=0)))

    # create kernel density estimate plots
    # We're currently not including the amplitude diff plot
    # amp_diff_kde = kdeplot([p.amp_diff for p in pp[1:]], [
    #     p.study.name for p in pp[1:]], 'Amplitude Diff')
    # amp_diff_kde = amp_diff_kde.opts(opts.Distribution(
    #     color=shift_cycle(hv.Cycle('Category10'))))

    amp_pdiff_kde = kdeplot([p.amp_diff_pct for p in pp[1:]], [
        p.study.name for p in pp[1:]], 'Amplitude Diff (%)')
    amp_pdiff_kde = amp_pdiff_kde.opts(opts.Distribution(
        line_color=shift_cycle(hv.Cycle('Category10')), filled=False))
    amp_pdiff_kde.opts(opts.Distribution(line_width=5))

    phase_diff_kde = kdeplot([p.phase_diff for p in pp[1:]], [
        p.study.name for p in pp[1:]], 'Phase Diff (minutes)')
    phase_diff_kde = phase_diff_kde.opts(opts.Distribution(
        line_color=shift_cycle(hv.Cycle('Category10')), filled=False))
    phase_diff_kde.opts(opts.Distribution(line_width=5))

    # create panel containing 3 kernel density estimate plots. We currently only want the last two, so commenting this out for now.
    # amp diff, amp % diff, phase diff
    # kdeplots = amp_diff_kde.opts(
    #     show_legend=False)+amp_pdiff_kde.opts(show_legend=False)+phase_diff_kde.opts(show_legend=False)
    # kdeplots = kdeplots.cols(3).opts(shared_axes=False).opts(
    #     opts.Distribution(height=200, width=300))
    # don't use

    # create panel containing amp % diff and phase diff kernel density estimate plots. Excluding amp diff plot
    # kdeplots = amp_pdiff_kde.opts(show_legend=False, title=amp_title) + \
    #     phase_diff_kde.opts(show_legend=False, title=phase_title)
    # if include_toolbar:
    #     kdeplots = kdeplots.cols(2).opts(shared_axes=False).opts(
    #             opts.Distribution(height=200, width=300))
    # else:
    #     kdeplots = kdeplots.cols(2).opts(shared_axes=False, toolbar=None).opts(
    #             opts.Distribution(height=200, width=300))

    # return a list instead so we can change titles later
    kdeplots_list = [amp_pdiff_kde.opts(show_legend=False, title=amp_title), 
        phase_diff_kde.opts(show_legend=False, title=phase_title)]

    return kdeplots_list

def export_svg(plot, fname):
    ''' export holoview object to filename fname '''
    from bokeh.io import export_svgs
    p = hv.render(plot, backend='bokeh')
    p.output_backend = "svg"
    export_svgs(p, filename=fname)

def _process_df_for_validation_bar_charts(vartype, vartype_to_station_list_dict, df):
    '''
    The location field is a string representation of the information in a Location object, which contains
    a lot of text we don't need. all we need is the station abbreviation
    ,Location(,v2022_10,0.9017457956731834,"ANC', bpart='ANC', description='San Joaquin River at Antioch', time_window_exclusion_list='', threshold_value='')"
    These splits must return 2 items, otherwise you get a "SystemError: tile cannot extend outside image"
    '''
    df[['Location', 'new loc']] = df['Location'].str.split('name=\'', expand=True)
    df[['Location', 'new loc2']] = df['new loc'].str.split('\', bpart', expand=True)
    df.drop(['new loc', 'new loc2'], axis=1, inplace=True)

    stations_to_keep = vartype_to_station_list_dict[vartype]
    df = df.loc[df['Location'].isin(stations_to_keep)]
    return df


def create_validation_bar_charts(validation_plot_output_folder, validation_metric_csv_filenames_dict, vartype_to_station_list_dict):
    '''
    Create bar charts for selected locations
    '''
    # these could be read from a file, but for now it's easier to hardcode, since we don't expect this info to change
    mu='\u03BC'
    vartype_to_metric_list = {'EC': ['Mean Error', 'RMSE', 'Mnly Mean Err', 'Mnly RMSE'],
                            'Flow': ['Mean Error', 'RMSE', 'Amp Avg %Err', 'Avg Phase Err'],
                            'Stage': ['Mean Error', 'RMSE', 'Amp Avg %Err', 'Avg Phase Err']}
    vartype_to_units_list = {'EC': [mu+'s/cm', mu+'s/cm', mu+'s/cm', mu+'s/cm'],
                            'Flow': ['cfs', 'cfs', '%', 'minute'],
                            'Stage': ['ft', 'ft', '%', 'minute']}

    # used for plot title and axis titles, to convert abbreviations    
    abbreviation_dict = {'Amp': 'Amplitude', 'Avg': 'Average', 'Mnly': 'Monthly'}
    # convert abbreviations only if string ends with the key. 
    abbreviation_end_of_str_dict = {'%Err': 'Err', 'Err': 'Error'}
    for const_name in validation_metric_csv_filenames_dict:
        plot_list = []

        all_loc_metrics_df = pd.read_csv(validation_metric_csv_filenames_dict[const_name])
        metrics_list = vartype_to_metric_list[const_name]
        units_list = vartype_to_units_list[const_name]
        all_loc_metrics_df.to_csv('all_loc_metrics_df.csv')
        
        df_list = []
        document_plot_title_list = ['(a)', '(b)', '(c)', '(d)']
        i = 0
        for m, u in zip(metrics_list, units_list):
            df = all_loc_metrics_df[['Location', 'DSM2 Run', m]]
            df=_process_df_for_validation_bar_charts(const_name, vartype_to_station_list_dict, df)
            df_list.append(df)
            grid_style={'grid_line_color': 'black', 'xgrid_line_alpha': 0}

            full_title = title='%s, %s,\nDSM2 v 8.3 vs DSM2 v8.2.1' % (m, const_name) # not used for technical memo
            document_plot_title = document_plot_title_list[i]
            for abbrev in abbreviation_dict:
                m = m.replace(abbrev, abbreviation_dict[abbrev])
            for abbrev in abbreviation_end_of_str_dict:
                m = re.sub(abbrev+'$', abbreviation_end_of_str_dict[abbrev], m)

            m_bars = hv.Bars(df, kdims=['Location', 'DSM2 Run']).opts(title=document_plot_title, width=350, height=350, 
                                                                    xrotation=45, multi_level=False, 
                                                                    legend_position='right', ylabel=m+' ('+u+')', 
                                                                    xlabel='', color=shift_cycle(hv.Cycle('Category10')), 
                                                                    gridstyle=grid_style, show_grid=True, show_legend=False,
                                                                    shared_axes=False)
            plot_list.append(m_bars)
        
            hv.save(m_bars, '%s_%s' % (m, const_name), fmt='png')
            i+=1
        layout = hv.Layout(plot_list).cols(2)
        layout.opts(shared_axes=False)
        hv.save(layout, validation_plot_output_folder+'/validation_metrics_layout'+const_name+'.png', fmt='png')
        hv.save(layout, validation_plot_output_folder+'/validation_metrics_layout'+const_name+'.html', fmt='html')