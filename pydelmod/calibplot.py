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
## - Generic Plotting Functions ##


def tsplot(dflist, names):
    """Time series overlay plots

    Handles missing DataFrame, just put None in the list

    Args:
        dflist (List): Time-indexed DataFrame list
        names (List): Names list (same size as dflist)

    Returns:
        Overlay: Overlay of Curve
    """
    plt = [df.hvplot(label=name) if df is not None else hv.Curve(None, label=name)
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
    x_series = dfa.iloc[:, index_x]
    dfr = dfa.drop(columns=dfa.columns[index_x])
    names.remove(names[index_x])
    slopes, interceps, equations, r2s, pvals, stds = [], [], [], [], [], []
    mean_errors, mses, rmses, percent_biases, nses = [], [], [], [], []
    for col in dfr.columns:
        y_series = dfr.loc[:, col]
        slope, intercep, rval, pval, std = stats.linregress(x_series, y_series)
        slopes.append(slope)
        interceps.append(intercep)
        sign = '-' if intercep <= 0 else '+'
        equation = 'y = %.4fx %s %.4f' % (slope, sign, abs(intercep))
        equations.append(equation)
        r2s.append(rval*rval)
        pvals.append(pval)
        stds.append(std)
        mean_errors.append(tsmath.mean_error(y_series, x_series))
        mses.append(tsmath.mse(y_series, x_series))
        rmses.append(tsmath.rmse(y_series, x_series))
        percent_biases.append(tsmath.percent_bias(y_series, x_series))
        nses.append(tsmath.nash_sutcliffe(y_series, x_series))
    dfmetrics = pd.concat([pd.DataFrame(arr)
                           for arr in (slopes, interceps, equations, r2s, pvals, stds, mean_errors, mses, rmses, percent_biases, nses)], axis=1)
    dfmetrics.columns = ['regression_slope', 'regression_intercep', 'regression_equation',
                         'r2', 'pval', 'std', 'mean_error',
                         'mse', 'rmse', 'percent_bias', 'nash_sutcliffe']
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


def build_calib_plot_template(studies, location, vartype, timewindow, tidal_template=False):
    """Builds calibration plot template

    Args:
        studies (List): Studies (name,dssfile)
        location (Location): name,bpart,description
        vartype (VarType): name,units
        timewindow (str): timewindow as start_date_str "-" end_date_str or "" for full availability
        tidal_template (bool, optional): If True include tidal plots. Defaults to False.

    Returns:
        panel: A template ready for rendering by display or save
    """
    pp = [postpro.PostProcessor(study, location, vartype) for study in studies]
    for p in pp:
        p.load_processed(timewindow=timewindow)
    gridstyle = {'grid_line_alpha': 1, 'grid_line_color': 'lightgrey'}
    tsp = tsplot([p.df for p in pp], [p.study.name for p in pp]).opts(
        ylabel=f'{vartype.name} @ {location.name}', show_grid=True, gridstyle=gridstyle)
    gtsp = tsplot([p.gdf for p in pp], [p.study.name for p in pp]).opts(
        ylabel=f'{vartype.name} @ {location.name}', show_grid=True, gridstyle=gridstyle)
    splot = scatterplot([p.gdf.resample('D').mean() for p in pp], [p.study.name for p in pp])\
        .opts(opts.Scatter(color=shift_cycle(hv.Cycle('Category10'))))\
        .opts(ylabel='Model', legend_position="top_left")\
        .opts(show_grid=True, frame_height=250, frame_width=250, data_aspect=1)

    dfmetrics = calculate_metrics([p.gdf for p in pp], [p.study.name for p in pp])
    dfmetrics_monthly = calculate_metrics(
        [p.gdf.resample('M').mean() for p in pp], [p.study.name for p in pp])

    slope_plots = regression_line_plots(dfmetrics)
    cplot = slope_plots.opts(opts.Slope(color=shift_cycle(hv.Cycle('Category10'))))*splot
    cplot = cplot.opts(xlabel='Observed', ylabel='Model', legend_position="top_left")\
        .opts(show_grid=True, frame_height=250, frame_width=250, data_aspect=1, show_legend=False)

    dfdisplayed_metrics = dfmetrics.loc[:, ['regression_equation', 'r2', 'mean_error', 'rmse']]
    dfdisplayed_metrics=pd.concat([dfdisplayed_metrics,dfmetrics_monthly.loc[:,['mean_error','rmse']]],axis=1)
    dfdisplayed_metrics.index.name = 'DSM2 Run'
    dfdisplayed_metrics.columns=['Equation','R Squared','Mean Error','RMSE','Monthly Mean Error','Monthly RMSE']
    metrics_panel = pn.widgets.DataFrame(dfdisplayed_metrics)

    for p in pp[1:]:  # TODO: move this out of here. Nothing to do with plotting!
        p.process_diff(pp[0])

    amp_diff_kde = kdeplot([p.amp_diff for p in pp[1:]], [
        p.study.name for p in pp[1:]], 'Amplitude Diff')
    amp_diff_kde = amp_diff_kde.opts(opts.Distribution(
        color=shift_cycle(hv.Cycle('Category10'))))

    amp_pdiff_kde = kdeplot([p.amp_diff_pct for p in pp[1:]], [
        p.study.name for p in pp[1:]], 'Amplitude Diff (%)')
    amp_pdiff_kde = amp_pdiff_kde.opts(opts.Distribution(
        color=shift_cycle(hv.Cycle('Category10'))))

    phase_diff_kde = kdeplot([p.phase_diff for p in pp[1:]], [
        p.study.name for p in pp[1:]], 'Phase Diff (minutes)')
    phase_diff_kde = phase_diff_kde.opts(opts.Distribution(
        line_color=shift_cycle(hv.Cycle('Category10')), filled=True))

    kdeplots = amp_diff_kde.opts(
        show_legend=False)+amp_pdiff_kde.opts(show_legend=False)+phase_diff_kde.opts(show_legend=False)
    kdeplots = kdeplots.cols(3).opts(shared_axes=False).opts(
        opts.Distribution(height=200, width=300))

    header_panel = pn.panel(f'## {location.description} ({location.name}/{vartype.name})')

    tsplots2 = (tsp.opts(width=900)+gtsp.opts(show_legend=False, width=900)).cols(1)

    if tidal_template:
        return pn.Column(
            header_panel,
            pn.Row(tsplots2),
            pn.Row(cplot.opts(shared_axes=False, toolbar=None), metrics_panel), pn.Row(kdeplots.opts(toolbar=None)))
    else:
        return pn.Column(
            header_panel,
            pn.Row(gtsp.opts(width=900, show_legend=True)),
            pn.Row(cplot.opts(shared_axes=False), metrics_panel))
