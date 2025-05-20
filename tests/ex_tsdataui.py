# %%
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from pydelmod.dvue import dataui

# Sample data
variables = ["precipitation", "wind_speed", "flow"]
units = ["mm", "m/s", "cfs"]
data = {
    "station_id": ["1", "2", "3"],
    "station_name": ["Station A", "Station B", "Station C"],
    "Latitude": [34.05, 36.16, 37.77],
    "Longitude": [-118.25, -115.15, -122.42],
    "variable": variables,
    "unit": units,
    "interval": ["hourly", "hourly", "hourly"],
    "start_year": ["2020", "2021", "2022"],
    "max_year": ["2023", "2024", "2025"],
    "source": ["", "", ""],
}
# Create a DataFrame
df = pd.DataFrame(data)

MANY_TO_ONE = True
if MANY_TO_ONE:
    expanded_data = []
    for _, row in df.iterrows():
        for variable, unit in zip(variables, units):
            new_row = row.copy()
            new_row["variable"] = variable
            new_row["unit"] = unit
            expanded_data.append(new_row)
            if row["station_id"] == "1":
                new_row = row.copy()
                new_row["variable"] = variable
                new_row["unit"] = unit
                new_row["interval"] = "daily"
                expanded_data.append(new_row)
            if row["station_id"] == "2":
                new_row = row.copy()
                new_row["variable"] = variable + "_x"
                new_row["unit"] = unit
                expanded_data.append(new_row)

    df = pd.DataFrame(expanded_data).reset_index(drop=True)

# Expand the DataFrame to include multiple intervals
#intervals = ["hourly", "daily", "monthly"]

#expanded_data = []
#for interval in intervals:
#    for _, row in df.iterrows():
#        new_row = row.copy()
#        new_row["interval"] = interval
#        expanded_data.append(new_row)
#df = pd.DataFrame(expanded_data).reset_index(drop=True)
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

def create_tsdf(interval="hourly", n=10):
    if interval == "hourly":
        freq = "h"
    elif interval == "daily":
        freq = "d"
    elif interval == "monthly":
        freq = "M"
    else:
        raise ValueError("Interval must be 'hourly' or 'daily'")
    date_rng = pd.date_range(start="1/1/2020", end="1/1/2021", freq=freq)
    df = pd.DataFrame(
        np.random.randint(0, n, size=(len(date_rng))),
        index=date_rng,
        columns=["value"],
    )
    return df

def create_smooth_tsdf(interval="hourly", n=10, noise_scale=1.0):
    """
    Create a smooth random time series dataframe by cumulative sum of random noise.
    """
    if interval == "hourly":
        freq = "h"
    elif interval == "daily":
        freq = "d"
    elif interval == "monthly":
        freq = "M"
    else:
        raise ValueError("Interval must be 'hourly', 'daily', or 'monthly'")
    date_rng = pd.date_range(start="1/1/2020", end="1/1/2021", freq=freq)
    # Generate smooth random walk
    values = np.cumsum(np.random.randn(len(date_rng)) * noise_scale)
    df = pd.DataFrame(values, index=date_rng, columns=["value"])
    return df

def create_sin_cos_tsdf(interval="hourly", amplitude=1.0, freq_scale=1.0, phase=0.0, kind="sin"):
    """
    Create a time series DataFrame with sine, cosine, or both.
    kind: "sin", "cos", or "both"
    """
    if interval == "hourly":
        freq = "h"
    elif interval == "daily":
        freq = "d"
    elif interval == "monthly":
        freq = "M"
    else:
        raise ValueError("Interval must be 'hourly', 'daily', or 'monthly'")
    date_rng = pd.date_range(start="1/1/2020", end="1/1/2021", freq=freq)
    t = np.arange(len(date_rng))
    if kind == "sin":
        values = amplitude * np.sin(2 * np.pi * freq_scale * t / len(t) + phase)
    elif kind == "cos":
        values = amplitude * np.cos(2 * np.pi * freq_scale * t / len(t) + phase)
    elif kind == "both":
        values = amplitude * (
            np.sin(2 * np.pi * freq_scale * t / len(t) + phase)
            + np.cos(2 * np.pi * freq_scale * t / len(t) + phase)
        )
    else:
        raise ValueError("kind must be 'sin', 'cos', or 'both'")
    df = pd.DataFrame(values, index=date_rng, columns=["value"])
    return df

def freq_to_period(freq):
    if freq == "hourly":
        return "h"
    elif freq == "daily":
        return "d"
    elif freq == "monthly":
        return "M"
    else:
        raise ValueError(f"Unsupported frequency: {freq}")

tsdfs = {
    row["station_name"]
    + row["unit"]
    + row["variable"]
    + row["interval"]: create_tsdf(interval=row["interval"])
    for _, row in gdf.iterrows()
}

smooth_tsdfs = {
    row["station_name"]
    + row["unit"]
    + row["variable"]
    + row["interval"]: create_smooth_tsdf(interval=row["interval"])
    for _, row in gdf.iterrows()
}

sin_df = create_sin_cos_tsdf(interval="hourly", amplitude=5, freq_scale=3, kind="sin")
cos_df = create_sin_cos_tsdf(interval="daily", amplitude=2, freq_scale=1, kind="cos")
both_df = create_sin_cos_tsdf(interval="hourly", amplitude=3, freq_scale=2, kind="both")
# %%
from pydelmod.dvue import tsdataui
import holoviews as hv


class ExampleTimeSeriesDataUIManager(tsdataui.TimeSeriesDataUIManager):

    def __init__(self, gdf):
        self.gdf = gdf
        super().__init__(filename_column="source")
        self.color_cycle_column = "station_name"
        self.dashed_line_cycle_column = "interval"
        self.marker_cycle_column = "variable"
    
    def get_data_catalog(self):
        return self.gdf

    def get_time_range(self, dfcat):
        return pd.to_datetime("1/1/2020"), pd.to_datetime("1/1/2021")

    def build_station_name(self, r):
        if "FILE_NUM" not in r:
            return f"{r['station_name']}"
        else:
            return f'{r["FILE_NUM"]}:{r["station_name"]}'

    def get_table_column_width_map(self):
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

    def get_data_for_time_range(self, r, time_range):
        return (
            smooth_tsdfs[r["station_name"] + r["unit"] + r["variable"] + r["interval"]],
            r["unit"],
            "instantaneous",
        )

    # methods below if geolocation data is available
    def get_tooltips(self):
        return [
            ("station_id", "@station_id"),
            ("station_name", "@station_name"),
        ]

    def get_map_color_columns(self):
        """return the columns that can be used to color the map"""
        return ["variable"]

    def get_map_marker_columns(self):
        """return the columns that can be used to color the map"""
        return ["variable", "unit"]

    def create_curve(self, df, r, unit, file_index=None):
        file_index_label = f"{file_index}:" if file_index is not None else ""
        crvlabel = (
            f'{file_index_label}{r["station_id"]}/{r["variable"]}/{r["interval"]}'
        )
        ylabel = f'{r["variable"]} ({unit})'
        title = f'{r["variable"]} @ {r["station_id"]}'
        crv = hv.Curve(df.iloc[:, [0]], label=crvlabel).redim(value=crvlabel)
        return crv.opts(
            xlabel="Time",
            ylabel=ylabel,
            title=title,
            responsive=True,
            active_tools=["wheel_zoom"],
            tools=["hover"],
        )

    def _append_value(self, new_value, value):
        if new_value not in value:
            value += f'{", " if value else ""}{new_value}'
        return value

    def append_to_title_map(self, title_map, unit, r):
        if unit in title_map:
            value = title_map[unit]
        else:
            value = ["", ""]
        value[0] = self._append_value(r["variable"], value[0])
        value[1] = self._append_value(r["station_id"], value[1])
        title_map[unit] = value

    def create_title(self, v):
        title = f"{v[1]}({v[0]})"
        return title


# %%
exmgr = ExampleTimeSeriesDataUIManager(gdf)
ui = dataui.DataUI(exmgr, station_id_column="station_id" if MANY_TO_ONE else None)
ui.create_view(title="Example Time Series Data UI").servable()
# %%
