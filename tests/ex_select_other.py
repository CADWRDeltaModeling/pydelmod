# %%
import pandas as pd
import numpy as np
import hvplot.pandas

# %%
# create a pandas data frame with monthly frequency
values = np.arange(12)
dfm = pd.DataFrame(
    {"values": values}, index=pd.date_range("2018-01-01", "2018-12-31", freq="ME")
)
# create another pandas data frame with 15 min frequency
dates15 = pd.date_range("2018-01-01", "2018-12-31", freq="15min")
values15 = np.random.uniform(0, 100, len(dates15))
df15 = pd.DataFrame({"values": values15}, index=dates15)
# %%
# create aconditional to select only the indices of the first data that have values 4 and 9
cond = dfm["values"].isin([4, 9])
# %%
# select the rows of the first data frame that satisfy the condition
dfm.loc[cond]
# %%
# select the rows of the second data frame that satisfy the condition from the first dataframe
df15.loc[dfm.index[cond]]
# %%
dfmp = dfm.to_period("M")
# %%
condition = dfmp["values"].isin([4, 9])
selected_periods = dfmp[condition].index
selected_periods_list = [pd.Period(p, freq="M") for p in selected_periods]
# %%
mask = df15.index.to_series().apply(
    lambda d: pd.Period(d, freq="M") in selected_periods_list
)
# %%
df15_selected = df15[mask]
# %%
df15_selected.hvplot.line()


# %%
def select_other(df1, df2, condition, freq="M"):
    df1p = df1.to_period(freq)
    selected_periods = df1p[condition].index
    selected_periods_list = [pd.Period(p, freq=freq) for p in selected_periods]
    mask = df2.index.to_series().apply(
        lambda d: pd.Period(d, freq=freq) in selected_periods_list
    )
    return df2[mask]


# %%
select_other(dfm, df15, dfmp["values"].isin([3, 8])).hvplot.line()
# %%
