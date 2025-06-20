import logging

# Configure logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
import pandas as pd
import xarray as xr
import geopandas as gpd
import numpy as np
import holoviews as hv
from pydelmod.dvue import tsdataui

class DeltaCDUIManager(tsdataui.TimeSeriesDataUIManager):
    """
    UI Manager for DeltaCD netCDF data files.
    Handles data catalog creation and time series extraction for area_id and crop combinations.
    """

    def __init__(self, *nc_file_paths, **kwargs):
        """
        Initialize the DeltaCD UI Manager.
        
        Parameters:
        -----------
        nc_file_paths : str or list of str
            Paths to the netCDF files containing DeltaCD data. Can be a single path or multiple paths.
        geojson_file : str, optional
            Path to the GeoJSON file containing geographical information for area_ids
        """
        self.nc_file_paths = nc_file_paths
        self.geojson_file_path = kwargs.pop("geojson_file", None)
        self.datasets = {}
        dfcats = []
        for nc_file_path in self.nc_file_paths:
            if not nc_file_path.endswith('.nc'):
                raise ValueError(f"Invalid file type: {nc_file_path}. Expected a netCDF file (.nc).")
            self.datasets[nc_file_path] = xr.open_dataset(nc_file_path)
            dfcat = self.get_data_catalog_for_dataset(self.datasets[nc_file_path], nc_file_path)
            dfcats.append(dfcat)
        self.gdf = None
        if self.geojson_file_path:
            self.gdf = gpd.read_file(self.geojson_file_path)
            self.gdf.rename(columns={"NEW_SUB": "area_id"}, inplace=True)

        # concatenate all data catalogs
        dfcat = pd.concat(dfcats, ignore_index=True)
        # merge with GeoDataFrame if available
        # If geojson is available, convert to GeoDataFrame
        if self.gdf is not None:
            # Convert area_id to string in both DataFrames before merging
            dfcat['area_id'] = dfcat['area_id'].astype(str)
            gdf_copy = self.gdf.copy()
            gdf_copy['area_id'] = gdf_copy['area_id'].astype(str)
            
            # Merge with geometry information based on area_id
            merged_df = pd.merge(dfcat, gdf_copy, on="area_id", how="left")
            
            # Create GeoDataFrame
            catalog = gpd.GeoDataFrame(merged_df, geometry="geometry")
            
            # Handle CRS properly
            if self.gdf.crs is not None:
                catalog.crs = self.gdf.crs
            else:
                catalog.set_crs(epsg=4326, inplace=True)
        else:
            # If no GeoDataFrame, just use the DataFrame
            catalog = dfcat
        self.dfcat = catalog
        # Initialize data cache
        self.data_cache = {}
        
        kwargs['filename_column'] = "source"
        super().__init__(**kwargs)
        # Set up columns for visualization
        self.color_cycle_column = "area_id"
        self.dashed_line_cycle_column = "variable"
        self.marker_cycle_column = "area_id"
        

    def get_data_catalog(self):
        return self.dfcat

    def get_data_catalog_for_dataset(self, ds, nc_file_path):
        """
        Create a data catalog from the netCDF file.
        Each row represents a time series for a variable for an area_id and crop combination.
        Includes all possible combinations of area_id, crop, and variable.
        """
        # Get available variables, excluding coordinates
        variables = list(ds.data_vars)
        dims = list(ds.dims)
        if "time" not in dims:
            raise ValueError(f"Dataset must contain a 'time' dimension. Dimensions available: {dims}")
        if "area_id" not in dims:
            raise ValueError(f"Dataset must contain an 'area_id' dimension. Dimensions available: {dims}")
        area_ids = ds.area_id.values
        other_dims = [dim for dim in dims if dim not in ["time", "area_id"]]
        
        # Create all combinations using pandas products
        combinations = []
        columns = ['area_id']+other_dims
        for area_id in area_ids:
            if 'crop' in dims:
                for crop in ds.crop.values:
                    # Create combinations for each variable
                    for var in variables:
                        combinations.append({
                            'area_id': area_id,
                            'crop': crop,
                            'variable': var
                        })
            else:
                for var in variables:
                    combinations.append({
                        'area_id': area_id,
                        'variable': var
                    })
        
        # Create base DataFrame with all combinations
        df = pd.DataFrame(combinations)
        
        # Add additional columns
        # More robust way to get units
        variable_units = {}
        for var in variables:
            try:
                # Try to get unit from the variable's attributes
                unit = ds[var].attrs.get("units", "")
                variable_units[var] = unit
                logger.debug(f"Found unit '{unit}' for variable '{var}'")
            except Exception as e:
                logger.debug(f"Error getting unit for {var}: {e}")
                variable_units[var] = ""  # Default to empty string
        df['unit'] = df['variable'].map(variable_units)
        df['interval'] = 'daily'  # Assuming all data is daily, adjust as needed
        df['source'] = nc_file_path
        
        # Add time range information
        times = pd.to_datetime(ds.time.values)
        df['start_year'] = str(times.min().year)
        df['max_year'] = str(times.max().year)
        return df        

    def get_time_range(self, dfcat):
        """Return the min and max time from the dataset"""
        # return the min and max time for the dfcat
        starttime = pd.to_datetime(dfcat['start_year'].min())
        endtime = pd.to_datetime(dfcat['max_year'].max())
        return starttime, endtime

    def build_station_name(self, r):
        """Build a display name for the area_id and crop combination"""
        if 'crop' not in r or not r['crop'] or pd.isna(r['crop']):
            return f"Area {r['area_id']}"
        return f"Area {r['area_id']} - {r['crop']}"

    def _get_table_column_width_map(self):
        """Define column widths for the data catalog table"""
        column_width_map = {
            "area_id": "8%",
            "variable": "12%",
            "unit": "8%",
            "interval": "10%",
            "start_year": "10%",
            "max_year": "10%",
        }
        if 'crop' in self.get_data_catalog().columns:
            column_width_map["crop"] = "15%"
        return column_width_map

    def get_table_filters(self):
        """Define filters for the data catalog table"""
        table_filters = {
            "area_id": {
                "type": "input",
                "func": "like",
                "placeholder": "Enter area ID",
            },
            "crop": {
                "type": "input",
                "func": "like",
                "placeholder": "Enter crop type",
            },
            "variable": {
                "type": "input", 
                "func": "like", 
                "placeholder": "Enter variable",
            },
            "unit": {
                "type": "input", 
                "func": "like", 
                "placeholder": "Enter unit",
            },
            "interval": {
                "type": "input", 
                "func": "like", 
                "placeholder": "Enter interval",
            },
            "start_year": {
                "type": "input",
                "func": "like",
                "placeholder": "Enter start year",
            },
            "max_year": {
                "type": "input", 
                "func": "like", 
                "placeholder": "Enter end year",
            },
        }
        return table_filters

    def is_irregular(self, r):
        """Check if time series is irregular"""
        return False  # Assuming all time series are regular

    def get_data_for_time_range(self, r, time_range):
        """
        Extract time series data for a specific area_id, crop, and variable combination
        within the specified time range.
        
        Parameters:
        -----------
        r : pandas.Series
            Row from data catalog containing area_id and variable (crop may be optional)
        time_range : tuple
            Start and end time for data extraction
    
        Returns:
        --------
        tuple
            (time series DataFrame, unit, data type)
        """
        area_id = r["area_id"]
        variable = r["variable"]
        unit = r["unit"]
        filename = r["source"]
        ds = self.datasets[filename]
        try:
            # Check if 'crop' exists in the row before trying to access it
            if 'crop' in r and not pd.isna(r['crop']):
                crop = r["crop"]
                # Extract data from xarray for the specific area_id, crop, and variable
                data = ds[variable].sel(area_id=area_id, crop=crop)
            else:
                # Handle case where crop is not in the data catalog
                data = ds[variable].sel(area_id=area_id)
        
            # Convert to pandas Series and then DataFrame
            df = data.to_pandas().to_frame()
            # Ensure the index is a datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                # Try to convert the index to a datetime index
                df.index = pd.to_datetime(df.index)
            
            # Filter by time range if specified
            if time_range and len(time_range) == 2:
                start_time, end_time = time_range
                df = df.loc[start_time:end_time]
                
            return df, unit, "instantaneous"
        
        except Exception as e:
            # Handle any exception that occurs during data extraction
            logger.debug(f"Error extracting data for area_id={area_id}, variable={variable}: {e}")
            return pd.DataFrame(), unit, "instantaneous"

    def get_tooltips(self):
        """Define tooltips for map visualization"""
        return [
            ("Area ID", "@area_id"),
            ("Crop", "@crop"),
            ("Variable", "@variable"),
            ("Unit", "@unit")
        ]

    def get_map_color_columns(self):
        """Return columns that can be used to color the map"""
        return ["crop", "variable"]

    def get_map_marker_columns(self):
        """Return columns that can be used as markers on the map"""
        return ["variable", "crop"]

    def create_curve(self, df, r, unit, file_index=None):
        """Create a holoviews curve for plotting"""
        file_index_label = f"{file_index}:" if file_index is not None else ""
        
        # Handle case where crop is missing or blank
        if 'crop' not in r or not r['crop'] or pd.isna(r['crop']):
            crvlabel = f'{file_index_label}Area {r["area_id"]}: {r["variable"]}'
            title = f'{r["variable"]} @ Area {r["area_id"]}'
        else:
            crvlabel = f'{file_index_label}Area {r["area_id"]} - {r["crop"]}: {r["variable"]}'
            title = f'{r["variable"]} for {r["crop"]} @ Area {r["area_id"]}'
        
        ylabel = f'{r["variable"]} ({unit})'
        
        # Create curve with appropriate data
        if df.empty:
            crv = hv.Curve(pd.DataFrame({'x': [], 'y': []}), kdims=['x'], vdims=['y'], label=crvlabel).redim(y=crvlabel)
        else:
            crv = hv.Curve(df, label=crvlabel).redim(value=crvlabel)
        
        return crv.opts(
            xlabel="Time",
            ylabel=ylabel,
            title=title,
            responsive=True,
            active_tools=["wheel_zoom"],
            tools=["hover"]
        )

    def _append_value(self, new_value, value):
        """Helper method for title creation"""
        if new_value not in value:
            value += f'{", " if value else ""}{new_value}'
        return value

    def append_to_title_map(self, title_map, unit, r):
        """Append information to the title map for plot titles"""
        if unit in title_map:
            value = title_map[unit]
        else:
            value = ["", ""]
        value[0] = self._append_value(r["variable"], value[0])
        
        # Handle case where crop is missing or blank
        if 'crop' not in r or not r['crop'] or pd.isna(r['crop']):
            location_str = f'Area {r["area_id"]}'
        else:
            location_str = f'Area {r["area_id"]} - {r["crop"]}'
        
        value[1] = self._append_value(location_str, value[1])
        title_map[unit] = value

    def create_title(self, v):
        """Create plot title from values"""
        title = f"{v[1]} ({v[0]})"
        return title
    
import click
@click.command()
@click.argument(
    "nc_files",
    nargs=-1,
    type=click.Path(exists=True, dir_okay=False),
    required=True,
)
@click.option(
    "--geojson_file",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Path to the GeoJSON file containing area geometries",
)
def show_deltacd_ui(nc_files, geojson_file=None):
    """
    Show the DeltaCD UI Manager for the specified netCDF file and GeoJSON file.
    """
    dcd_ui = DeltaCDUIManager(*nc_files, geojson_file=geojson_file)
    from pydelmod.dvue import dataui
    import cartopy.crs as ccrs
    dui=dataui.DataUI(dcd_ui, station_id_column="area_id", crs=ccrs.epsg(26910))
    dui.create_view().servable(title="DeltaCD UI Manager").show()

