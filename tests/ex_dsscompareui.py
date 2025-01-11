# %%
import pyhecdss as dss
import pandas as pd

# %%
dssfiles = [
    "D:/delta/dsm2_studies_fc_mss_merge/studies/historical/output/hist_fc_mss_hydro.dss",
    "D:/delta/mss2/dsm2_mss/mss/output/historical_v82_mss2_extran_hydro.dss",
]
# %%
dssfh = {}
dsscats = {}
for dssfile in dssfiles:
    dssfh[dssfile] = dss.DSSFile(dssfile)
    dfcat = dssfh[dssfile].read_catalog()
    dsscats[dssfile] = dfcat

# %%
