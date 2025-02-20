# %%
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# Sample data
data = {
    "station_id": ["1", "2", "3"],
    "station_name": ["Station A", "Station B", "Station C"],
    "Latitude": [34.05, 36.16, 37.77],
    "Longitude": [-118.25, -115.15, -122.42],
    "variable": ["temperature"] * 3,
    "unit": ["deg C", "mm", "m/s"],
    "interval": ["hourly", "hourly", "hourly"],
    "start_year": ["2020", "2021", "2022"],
    "max_year": ["2023", "2024", "2025"],
    "source": ["", "", ""],
}
variables = ["precipitation", "wind_speed", "flow"]
# Create a DataFrame
df = pd.DataFrame(data)

MANY_TO_ONE = True
if MANY_TO_ONE:
    expanded_data = []
    for _, row in df.iterrows():
        for variable in variables:
            new_row = row.copy()
            new_row["variable"] = variable
            expanded_data.append(new_row)

    df = pd.DataFrame(expanded_data).reset_index(drop=True)

# Set the coordinate r
# Convert to a GeoDataFrame
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude))

# Set the coordinate reference system (CRS)
gdf.set_crs(epsg=4326, inplace=True)

# Save the GeoDataFrame to a file (optional)
# gdf.to_file("sample_geodata.gpkg", layer='stations', driver="GPKG")

# Print the GeoDataFrame
print(gdf)
# %%
# create a time series dataframe for hourly data

time_range = pd.date_range(start="1/1/2020", end="1/1/2021", freq="h")


def create_tsdf(n=1000):
    date_rng = pd.date_range(start="1/1/2020", end="1/1/2021", freq="h")
    df = pd.DataFrame(date_rng, columns=["date"])
    df["data"] = np.random.randint(0, 100, size=(len(date_rng)))
    return df


tsdfs = {sname: create_tsdf() for sname in gdf.station_name}
# %%
from pydelmod import dataui, tsdataui
import holoviews as hv


class ExampleTimeSeriesDataUIManager(tsdataui.TimeSeriesDataUIManager):

    def __init__(self, gdf):
        self.gdf = gdf
        super().__init__(filename_column="source")

    def get_data_catalog(self):
        return self.gdf

    def _get_station_ids(self, df):
        return df["station_id"]

    def get_time_range(self, dfcat):
        return pd.to_datetime("1/1/2020"), pd.to_datetime("1/1/2021")

    def _get_table_column_width_map(self):
        """only columns to be displayed in the table should be included in the map"""
        column_width_map = {
            "station_id": "5%",
            "station_name": "15%",
            "variable": "10%",
            "unit": "5%",
            "interval": "5%",
            "start_year": "15%",
            "max_year": "15%",
        }
        return column_width_map

    def get_table_filters(self):
        table_filters = {
            "station_name": {
                "type": "input",
                "func": "like",
                "placeholder": "Enter match",
            },
            "station_id": {
                "type": "input",
                "func": "like",
                "placeholder": "Enter match",
            },
            "variable": {"type": "input", "func": "like", "placeholder": "Enter match"},
            "unit": {"type": "input", "func": "like", "placeholder": "Enter match"},
            "interval": {"type": "input", "func": "like", "placeholder": "Enter match"},
            "start_year": {
                "type": "input",
                "func": "like",
                "placeholder": "Enter match",
            },
            "max_year": {"type": "input", "func": "like", "placeholder": "Enter match"},
        }
        return table_filters

    def is_irregular(self, r):
        return False  # only regular time series data in example

    def _get_data_for_time_range(self, r, time_range):
        return tsdfs[r["station_name"]], "dummy_unit", "instantaneous"

    # methods below if geolocation data is available
    def get_tooltips(self):
        return [
            ("station_id", "@station_id"),
            ("station_name", "@station_name"),
        ]

    def get_map_color_columns(self):
        """return the columns that can be used to color the map"""
        return ["max_year"]

    def get_map_marker_columns(self):
        """return the columns that can be used to color the map"""
        return ["max_year"]


# %%
exmgr = ExampleTimeSeriesDataUIManager(gdf)
ui = dataui.DataUI(exmgr, station_id_column="station_id" if MANY_TO_ONE else None)
ui.create_view().show()
# %%
