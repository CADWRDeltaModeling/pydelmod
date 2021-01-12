from scipy import stats
import pandas as pd
# display stuff
import hvplot.pandas
import holoviews as hv
from holoviews import opts
# styling plots
from bokeh.themes.theme import Theme
from bokeh.themes import built_in_themes

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
    dfa=pd.concat(dflist, axis=1)
    dfa.columns=names
    dfa=dfa.resample('D').mean()
    return dfa.hvplot.scatter(x=dfa.columns[index_x])


def regression_line_plots(dflist, index_x=0):
    """Fit linear regressions and return equations (str) and Slope

    Args:
        dflist (List): DataFrame list
        index_x (int, optional): Index to independent variable. Defaults to 0.

    Returns:
        tuple: Slope list, equations(str) list
    """    
    dfa = pd.concat(dflist, axis=1)
    dfa = dfa.dropna()
    x_series = dfa.iloc[:, index_x]
    slope_plots = None
    equations = []
    dfr = dfa.drop(columns=dfa.columns[index_x])
    for col in range(0, len(dfr.columns)):
        y_series = dfr.iloc[:, col]
        slope, intercep, rval, pval, std = stats.linregress(x_series, y_series)
        sign = '-' if intercep <= 0 else '+'
        equation = 'y = %.4fx %s %.4f' % (slope, sign, abs(intercep))
        equations.append(equation)
        slope_plot = hv.Slope(slope, y_intercept=intercep)
        slope_plots = slope_plot if slope_plots == None else slope_plots*slope_plot
    return slope_plots, equations


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

##- Customized functions for calibration / validation templates
