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
## - Generic Plotting Functions ##


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
        print('error in calibplot.parse_time_window, while parsing timewindow. Timewindow must be in format 2011-09-01:2011-09-30. Ignoring timewindow')
    return return_list


def tsplot(dflist, names, timewindow=None):
    """Time series overlay plots

    Handles missing DataFrame, just put None in the list

    Args:
        dflist (List): Time-indexed DataFrame list
        names (List): Names list (same size as dflist)
        timewindow (str, optional): time window for plot. Must be in format: 'YYYY-MM-DD:YYYY-MM-DD'

    Returns:
        Overlay: Overlay of Curve
    """
    start_dt = dflist[0].index.min()
    end_dt = dflist[0].index.max()
    if timewindow is not None:
        try:
            parts = timewindow.split(':')
            start_dt = parts[0]
            end_dt = parts[1]
        except:
            print("error in calibplot.tsplot")
    plt = [df[start_dt:end_dt].hvplot(label=name) if df is not None else hv.Curve(None, label=name)
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


def calculate_metrics(dflist, names, index_x=0):
    """Calculate metrics between the index_x column and other columns


    Args:
        dflist (List): DataFrame list
        names (List): Names list
        index_x (int, optional): Index of base DataFrame. Defaults to 0.

    Returns:
        DataFrame: DataFrame of metrics
    """
    dfa = pd.concat(dflist, axis=1)
    dfa = dfa.dropna()
    # x_series contains observed data
    # y_series contains model output for each of the studies
    x_series = dfa.iloc[:, index_x]
    dfr = dfa.drop(columns=dfa.columns[index_x])
    names.remove(names[index_x])
    slopes, interceps, equations, r2s, pvals, stds = [], [], [], [], [], []
    mean_errors, mses, rmses, percent_biases, nses, rsrs = [], [], [], [], [], []

    for col in dfr.columns:
        y_series = dfr.loc[:, col]
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
        mses.append(tsmath.mse(y_series, x_series))
        rmses.append(tsmath.rmse(y_series, x_series))
        percent_biases.append(tsmath.percent_bias(y_series, x_series))
        nses.append(tsmath.nash_sutcliffe(y_series, x_series))
        rsrs.append(tsmath.rsr(y_series, x_series))
    dfmetrics = pd.concat([pd.DataFrame(arr)
                           for arr in (slopes, interceps, equations, r2s, pvals, stds, mean_errors, mses, rmses, percent_biases, \
                               nses, rsrs)], axis=1)
    dfmetrics.columns = ['regression_slope', 'regression_intercep', 'regression_equation',
                         'r2', 'pval', 'std', 'mean_error',
                         'mse', 'rmse', 'percent_bias', 'nash_sutcliffe', 'rsr']
    dfmetrics.index = names
    return dfmetrics


def regression_line_plots(dfmetrics):
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
    return hv.Overlay(kdes)


# - Customized functions for calibration / validation templates
# Needed because of name with . https://github.com/holoviz/holoviews/issues/4714
def sanitize_name(name):
    return name.replace('.', ' ')


def build_calib_plot_template(studies, location, vartype, timewindow, tidal_template=False, flow_in_thousands=False, units=None,
                              inst_plot_timewindow=None, layout_nash_sutcliffe=False):
    """Builds calibration plot template

    Args:
        studies (List): Studies (name,dssfile)
        location (Location): name,bpart,description
        vartype (VarType): name,units
        timewindow (str): timewindow as start_date_str "-" end_date_str or "" for full availability
        tidal_template (bool, optional): If True include tidal plots. Defaults to False.
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

    Returns:
        panel: A template ready for rendering by display or save
        dataframe: equations and statistics for all locations
    """
    pp = [postpro.PostProcessor(study, location, vartype) for study in studies]
    for p in pp:
        p.load_processed(timewindow=timewindow)
    gridstyle = {'grid_line_alpha': 1, 'grid_line_color': 'lightgrey'}

    # create axis titles with units (if specified), and modify titles and data if displaying flow data in 1000 CFS
    unit_string = ''
    if flow_in_thousands and units is not None:
        unit_string = '(1000 %s)' % units
    elif units is not None:
        unit_string = '(%s)' % units
    y_axis_label = f'{vartype.name} @ {location.name} {unit_string}'
    godin_y_axis_label = 'Godin '+y_axis_label
    # plot_data are scaled, if flow_in_thousands == True
    tsp_plot_data = [p.df for p in pp]
    gtsp_plot_data = [p.gdf for p in pp]
    splot_plot_data = [p.gdf.resample('D').mean() for p in pp]
    splot_metrics_data = [p.gdf.resample('D').mean() for p in pp]
    if flow_in_thousands:
        tsp_plot_data = [p.df/1000.0 for p in pp]
        gtsp_plot_data = [p.gdf/1000.0 for p in pp]
        splot_plot_data = [p.gdf.resample('D').mean()/1000.0 for p in pp]

    # create plots: instantaneous, godin, and scatter
    tsp = tsplot(tsp_plot_data, [p.study.name for p in pp], timewindow=inst_plot_timewindow).opts(
        ylabel=y_axis_label, show_grid=True, gridstyle=gridstyle, shared_axes=False)
    # zoom in to desired timewindow: works, but doesn't zoom y axis, so need to fix later
    # if inst_plot_timewindow is not None:
    #     start_end_times = parse_time_window(inst_plot_timewindow)
    #     s = start_end_times[0]
    #     e = start_end_times[1]
    #     tsp.opts(xlim=(datetime.datetime(s[0], s[1], s[1]), datetime.datetime(e[0],e[1],e[2])))
    gtsp = tsplot(gtsp_plot_data, [p.study.name for p in pp]).opts(
        ylabel=godin_y_axis_label, show_grid=True, gridstyle=gridstyle)
    splot = scatterplot(splot_plot_data, [p.study.name for p in pp])\
        .opts(opts.Scatter(color=shift_cycle(hv.Cycle('Category10'))))\
        .opts(ylabel='Model', legend_position="top_left")\
        .opts(show_grid=True, frame_height=250, frame_width=250, data_aspect=1)

    # calculate calibration metrics
    slope_plots_dfmetrics = calculate_metrics(gtsp_plot_data, [p.study.name for p in pp])
    # dfmetrics = calculate_metrics([p.gdf for p in pp], [p.study.name for p in pp])
    dfmetrics = calculate_metrics(splot_metrics_data, [p.study.name for p in pp])

    dfmetrics_monthly = calculate_metrics(
        [p.gdf.resample('M').mean() for p in pp], [p.study.name for p in pp])

    # add regression lines to scatter plot, and set x and y axis titles
    slope_plots = regression_line_plots(slope_plots_dfmetrics)
    cplot = slope_plots.opts(opts.Slope(color=shift_cycle(hv.Cycle('Category10'))))*splot
    cplot = cplot.opts(xlabel='Observed ' + unit_string, ylabel='Model ' + unit_string, legend_position="top_left")\
        .opts(show_grid=True, frame_height=250, frame_width=250, data_aspect=1, show_legend=False)

    # calculate amp diff, amp % diff, and phase diff
    amp_avg_pct_errors = []
    amp_avg_phase_errors = []
    for p in pp[1:]:  # TODO: move this out of here. Nothing to do with plotting!
        p.process_diff(pp[0])
        amp_avg_pct_errors.append(float(p.amp_diff_pct.mean(axis=0)))
        amp_avg_phase_errors.append(float(p.phase_diff.mean(axis=0)))

    # display calibration metrics
    # create a list containing study names, excluding observed.
    dfdisplayed_metrics = None
    study_list = [study.name.replace('DSM2', '')
                  for study in studies if study.name.lower() != 'observed']
    # using a Table object because the dataframe object, when added to a layout, doesn't always display all the values.
    # This could have something to do with inconsistent types.
    metrics_table = None
    if tidal_template:
        dfdisplayed_metrics = dfmetrics.loc[:, [
            'regression_equation', 'r2', 'mean_error', 'rmse', 'nash_sutcliffe', 'percent_bias', 'rsr']]
        dfdisplayed_metrics['Amp Avg pct Err'] = amp_avg_pct_errors
        dfdisplayed_metrics['Avg Phase Err'] = amp_avg_phase_errors

        dfdisplayed_metrics.index.name = 'DSM2 Run'
        dfdisplayed_metrics.columns = ['Equation', 'R Squared',
                                       'Mean Error', 'RMSE', 'NSE', 'PBIAS', 'RSR', 'Amp Avg %Err', 'Avg Phase Err']

        a = dfdisplayed_metrics['Equation'].to_list()
        b = ['{:.2f}'.format(item) for item in dfdisplayed_metrics['R Squared'].to_list()]
        c = ['{:.2f}'.format(item) for item in dfdisplayed_metrics['Mean Error'].to_list()]
        d = ['{:.2E}'.format(item) for item in dfdisplayed_metrics['RMSE'].to_list()]
        e = ['{:.2E}'.format(item) for item in dfdisplayed_metrics['NSE'].to_list()]
        f = ['{:.2E}'.format(item) for item in dfdisplayed_metrics['PBIAS'].to_list()]
        g = ['{:.2E}'.format(item) for item in dfdisplayed_metrics['RSR'].to_list()]
        h = ['{:.2f}'.format(item) for item in dfdisplayed_metrics['Amp Avg %Err'].to_list()]
        i = ['{:.2f}'.format(item) for item in dfdisplayed_metrics['Avg Phase Err'].to_list()]
        if layout_nash_sutcliffe:
            metrics_table = hv.Table((study_list, a, b, c, d, e, f, g, h, i), [
                                     'Study', 'Equation', 'R Squared', 'Mean Error', 'RMSE', 'NSE', 'PBIAS', 'RSR', \
                                        'Amp Avg %Err', 'Avg Phase Err']).opts(width=580, fontscale=.8)
        else:
            metrics_table = hv.Table((study_list, a, b, c, d, h, i), [
                                     'Study', 'Equation', 'R Squared', 'Mean Error', 'RMSE', 'Amp Avg %Err', 'Avg Phase Err']).opts(width=580, fontscale=.8)
    else:
        dfdisplayed_metrics = dfmetrics.loc[:, [
            'regression_equation', 'r2', 'mean_error', 'rmse', 'nash_sutcliffe', 'percent_bias', 'rsr']]
        dfdisplayed_metrics = pd.concat(
            [dfdisplayed_metrics, dfmetrics_monthly.loc[:, ['mean_error', 'rmse']]], axis=1)
        dfdisplayed_metrics.index.name = 'DSM2 Run'
        dfdisplayed_metrics.columns = ['Equation', 'R Squared',
                                       'Mean Error', 'RMSE', 'NSE', 'PBIAS', 'RSR', 'Mnly Mean Err', 'Mnly RMSE']
        format_dict = {'Equation': '{:,.2f}', 'R Squared': '{:,.2f}', 'Mean Error': '{:,.2f}', 'RMSE': '{:,.2}',
                       'Amp Avg %Err': '{:,.2f}', 'Avg Phase Err': '{:,.2f}', 'NSE': '{:,.2f}', 'PBIAS': '{:,.2f}', 'RSR': '{:,.2f}'}
        dfdisplayed_metrics.style.format(format_dict)
        # Ideally, the columns should be sized to fit the data. This doesn't work properly--replaces some values with blanks
        # metrics_table = pn.widgets.DataFrame(dfdisplayed_metrics, autosize_mode='fit_columns')
        a = dfdisplayed_metrics['Equation'].to_list()
        b = ['{:.2f}'.format(item) for item in dfdisplayed_metrics['R Squared'].to_list()]
        c = ['{:.2f}'.format(item) for item in dfdisplayed_metrics['Mean Error'].to_list()]
        d = ['{:.2E}'.format(item) for item in dfdisplayed_metrics['RMSE'].to_list()]
        e = ['{:.2E}'.format(item) for item in dfdisplayed_metrics['NSE'].to_list()]
        f = ['{:.2E}'.format(item) for item in dfdisplayed_metrics['PBIAS'].to_list()]
        g = ['{:.2E}'.format(item) for item in dfdisplayed_metrics['RSR'].to_list()]
        h = ['{:.2f}'.format(item) for item in dfdisplayed_metrics['Mnly Mean Err'].to_list()]
        i = ['{:.2f}'.format(item) for item in dfdisplayed_metrics['Mnly RMSE'].to_list()]
        if layout_nash_sutcliffe:
            metrics_table = hv.Table((study_list, a, b, c, d, e, f, g, h, i), [
                                     'Study', 'Equation', 'R Squared', 'Mean Error', 'RMSE', 'NSE', 'PBIAS', 'RSR', \
                                        'Mnly Mean Err', 'Mnly RMSE']).opts(width=580, fontscale=.8)
        else:
            metrics_table = hv.Table((study_list, a, b, c, d, h, i), [
                                     'Study', 'Equation', 'R Squared', 'Mean Error', 'RMSE', 'Mnly Mean Err', \
                                        'Mnly RMSE']).opts(width=580, fontscale=.8)
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
    kdeplots = amp_pdiff_kde.opts(show_legend=False, title='(e)') + \
        phase_diff_kde.opts(show_legend=False, title='(f)')
    kdeplots = kdeplots.cols(2).opts(shared_axes=False).opts(
        opts.Distribution(height=200, width=300))

    # create plot/metrics template
    header_panel = pn.panel(f'## {location.description} ({location.name}/{vartype.name})')
    # do this if you want to link the axes
    # tsplots2 = (tsp.opts(width=900)+gtsp.opts(show_legend=False, width=900)).cols(1)
    # start_dt = dflist[0].index.min()
    # end_dt = dflist[0].index.max()

    column = None
    if tidal_template:
        column = pn.Column(
            header_panel,
            # tsp.opts(width=900, legend_position='right'),
            tsp.opts(width=900, toolbar=None, title='(a)'),
            gtsp.opts(width=900, toolbar=None, title='(b)'),
            # pn.Row(tsplots2),
            pn.Row(cplot.opts(shared_axes=False, toolbar=None, title='(c)'), metrics_table.opts(title='(d)')), \
            pn.Row(kdeplots.opts(toolbar=None)))
    else:
        column = pn.Column(
            header_panel,
            pn.Row(gtsp.opts(width=900, show_legend=True, toolbar=None, title='(a)')),
            pn.Row(cplot.opts(shared_axes=False, toolbar=None, title='(b)'), metrics_table.opts(title='(c)')))
    return column, dfdisplayed_metrics


def export_svg(plot, fname):
    ''' export holoview object to filename fname '''
    from bokeh.io import export_svgs
    p = hv.render(plot, backend='bokeh')
    p.output_backend = "svg"
    export_svgs(p, filename=fname)
