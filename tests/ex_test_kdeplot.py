# %%
import numpy as np
import pandas as pd
import hvplot.pandas
import holoviews as hv
from holoviews import opts, dim, streams

hv.extension("bokeh")
# %%
# create a time series of a sinusoidal wave
n = 100
step = 0.1
time = pd.date_range("2020-01-01", periods=n / step, freq="h")
amplitude = np.sin(np.arange(0, n, 0.1))
dfsine = pd.DataFrame({"time": time, "amplitude": amplitude})
amplitude = np.cos(np.arange(0, n, 0.1))
dfcosine = pd.DataFrame({"time": time, "amplitude": amplitude})
# %%
tsplot = dfcosine.hvplot(x="time") * dfsine.hvplot(x="time")
tsplot = tsplot.opts(opts.Curve(height=300, width=500))
# %%
kdeplot = dfcosine.hvplot.kde(y="amplitude") * dfsine.hvplot.kde(y="amplitude")
kdeplot = kdeplot.opts(shared_axes=False).opts(opts.Distribution(height=200, width=300))
kkplot = kdeplot + kdeplot
# %%
import panel as pn

pn.extension()
# %%
pn.Row(pn.Column(pn.Row(tsplot), pn.Row(kkplot)))
# %%
pn.Row(pn.Row(pn.Column(pn.Row(tsplot), pn.Row(kkplot))))
# %%
