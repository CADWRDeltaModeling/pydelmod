#%%
import numpy as np
import pandas as pd
#%% Create a dataframe of a time series
np.random.seed(0)
n = 100
dates = pd.date_range('20210101', periods=n)
df = pd.DataFrame(np.random.randn(n, 4), index=dates, columns=list('ABCD'))
df.head()
# %%
from pydelmod import calibplot
# %%
