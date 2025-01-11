# %%
import pandas as pd
from pydelmod import datastore

repo_dir = "Y:/repo/continuous/screened"
inventory_file = "Y:/repo/continuous/inventory_datasets_screened_20240326.csv"
ds = datastore.StationDatastore(repo_dir=repo_dir, inventory_file=inventory_file)
# %%
