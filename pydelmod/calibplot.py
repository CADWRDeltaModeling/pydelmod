from scipy import stats
import pandas as pd
#
from pydsm import postpro
from pydsm.postpro import Location, VarType
# display stuff
import hvplot.pandas
import holoviews as hv
from holoviews import opts
from bokeh.themes.theme import Theme
from bokeh.themes import built_in_themes


def tsplot(dflist, names):
    plt = [df.hvplot(label=name) if df is not None else hv.Curve(None, label=name)
           for df, name in zip(dflist, names)]
    plt = [c.redim(**{c.vdims[0].name:c.label, c.kdims[0].name: 'Time'})
           if c.name != '' else c for c in plt]
    return hv.Overlay(plt)


def scatterplot(dflist, names, index_x=0):
    dfa = pd.concat(dflist, axis=0)
    dfa.columns = names
    dfa = dfa.resample('D').mean()
    return dfa.hvplot.scatter(x=dfa.columns[index_x])


def regression_line_plots(dflist, index_x=0):
    if index_x != 0:
        raise "Non zero index_x not implemented yet! Sorry."
    dfa = pd.concat(dflist, axis=1)
    dfa = dfa.dropna()
    x_series = dfa.iloc[:, index_x]
    slope_plots = None
    equations = []
    for col in range(1, len(dflist)):
        y_series = dfa.iloc[:, col]
        slope, intercep, rval, pval, std = stats.linregress(x_series, y_series)
        sign = '-' if intercep <= 0 else '+'
        equation = 'y = %.4fx %s %.4f' % (slope, sign, abs(intercep))
        equations.append(equation)
        slope_plot = hv.Slope(slope, y_intercept=intercep)
        slope_plots = slope_plot if slope_plots == None else slope_plots*slope_plot
    return slope_plots, equations


def shift_cycle(cycle):
    v = cycle.values
    v.append(v.pop(0))
    return hv.Cycle(v)


def tidalplot(df, high, low, name):
    h = high.hvplot.scatter(label='high').opts(marker='^')
    l = low.hvplot.scatter(label='low').opts(marker='v')
    o = df.hvplot.line(label=name)
    plts = [h, l, o]
    plts = [c.redim(**{c.vdims[0].name:c.label, c.kdims[0].name: 'Time'}) for c in plts]
    return hv.Overlay(plts)


def kdeplot(dflist, names, xlabel):
    kdes = [df.hvplot.kde(label=name, xlabel=xlabel) for df, name in zip(dflist, names)]
    return hv.Overlay(kdes)
