# %%
dir = "x:/Share/Hans/mss/outputs/run_0"
flux_file = dir + "/flux.out"
flux_prop_file = dir + "/fluxflag.prop"
# %%
import pandas as pd

# %%
# pd.read_csv(dir+'/flux.out')
# %%
from schimpy import station

# %%
names = station.station_names_from_file(flux_prop_file)
reftime = "2020-01-01"
# %%
flux = station.read_flux_out(flux_file, names, reftime)

# %%
snames = pd.Series(names).str.lower()
# %%
sunique_names = snames.unique()
# %%
assert len(sunique_names) == len(snames)
# %%
dflux = pd.read_csv(
    flux_file,
    sep="\s+",
    index_col=0,
    header=None,
    names=["time"] + names,
    dtype="d",
    engine="c",
)
# %%
