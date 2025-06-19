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

    def __init__(self, nc_file_path, geojson_file_path=None):
        """
        Initialize the DeltaCD UI Manager.
        
        Parameters:
        -----------
        nc_file_path : str
            Path to the netCDF file containing DeltaCD data
        geojson_file_path : str, optional
            Path to the GeoJSON file containing geographical information for area_ids
        """
        self.nc_file_path = nc_file_path
        self.geojson_file_path = geojson_file_path
        self.ds = xr.open_dataset(nc_file_path)
        self.gdf = None
        
        if geojson_file_path:
            self.gdf = gpd.read_file(geojson_file_path)
            self.gdf.rename(columns={"OBJECTID": "area_id"}, inplace=True)
        
        # Initialize data cache
        self.data_cache = {}
        
        # Set up columns for visualization
        self.color_cycle_column = "crop"
        self.dashed_line_cycle_column = "variable"
        self.marker_cycle_column = "area_id"
        
        super().__init__(filename_column="source")

    def get_data_catalog(self):
        """
        Create a data catalog from the netCDF file.
        Each row represents a time series for a variable for an area_id and crop combination.
        Includes all possible combinations of area_id, crop, and variable.
        """
        # Get available variables, excluding coordinates
        variables = [var for var in self.ds.data_vars]
        area_ids = self.ds.area_id.values
        crops = self.ds.crop.values
        
        # Create all combinations using pandas products
        combinations = []
        for area_id in area_ids:
            for crop in crops:
                for var in variables:
                    combinations.append({
                        'area_id': int(area_id),
                        'crop': str(crop),
                        'variable': var
                    })
        
        # Create base DataFrame with all combinations
        df = pd.DataFrame(combinations)
        
        # Get time range and other metadata for each combination
        variable_units = {var: self.ds[var].attrs.get("units", "") for var in variables}
        
        # Add additional columns
        df['unit'] = df['variable'].map(variable_units)
        df['interval'] = 'daily'  # Assuming all data is daily, adjust as needed
        df['source'] = self.nc_file_path
        
        # Add time range information
        times = pd.to_datetime(self.ds.time.values)
        df['start_year'] = str(times.min().year)
        df['max_year'] = str(times.max().year)
        
        # If geojson is available, convert to GeoDataFrame
        if self.gdf is not None:
            # Merge with geometry information based on area_id
            merged_df = pd.merge(df, self.gdf, on="area_id", how="left")
            
            # Create GeoDataFrame
            catalog = gpd.GeoDataFrame(merged_df, geometry="geometry")
            
            # Handle CRS properly
            # Check if the GeoDataFrame already has a CRS
            if self.gdf.crs is not None:
                # GDF already has a CRS, no need to set it
                pass
            else:
                # Set a default CRS if none exists
                catalog.set_crs(epsg=4326, inplace=True)
            
            return catalog
        else:
            return df

    def get_time_range(self, dfcat):
        """Return the min and max time from the dataset"""
        times = pd.to_datetime(self.ds.time.values)
        return times.min(), times.max()

    def build_station_name(self, r):
        """Build a display name for the area_id and crop combination"""
        return f"Area {r['area_id']} - {r['crop']}"

    def get_table_column_width_map(self):
        """Define column widths for the data catalog table"""
        column_width_map = {
            "area_id": "8%",
            "crop": "15%",
            "variable": "12%",
            "unit": "8%",
            "interval": "10%",
            "start_year": "10%",
            "max_year": "10%",
        }
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
            Row from data catalog containing area_id, crop, and variable
        time_range : tuple
            Start and end time for data extraction
        
        Returns:
        --------
        tuple
            (time series DataFrame, unit, data type)
        """
        area_id = r["area_id"]
        crop = r["crop"]
        variable = r["variable"]
        unit = r["unit"]
        
        # Extract data from xarray for the specific area_id, crop, and variable
        data = self.ds[variable].sel(area_id=area_id, crop=crop)
        
        # Convert to pandas Series and then DataFrame
        df = data.to_pandas().to_frame()
            
        # Filter by time range if specified
        if time_range and len(time_range) == 2:
            start_time, end_time = time_range
            df = df.loc[start_time:end_time]
            
        return df, unit, "instantaneous"

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
        crvlabel = f'{file_index_label}Area {r["area_id"]} - {r["crop"]}: {r["variable"]}'
        ylabel = f'{r["variable"]} ({unit})'
        title = f'{r["variable"]} for {r["crop"]} @ Area {r["area_id"]}'
        
        crv = hv.Curve(df.iloc[:, [0]], label=crvlabel).redim(value=crvlabel)
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
        value[1] = self._append_value(f'Area {r["area_id"]} - {r["crop"]}', value[1])
        title_map[unit] = value

    def create_title(self, v):
        """Create plot title from values"""
        title = f"{v[1]} ({v[0]})"
        return title
    
import click
@click.command()
@click.argument("detaw_output_file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--geojson_file_path",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Path to the GeoJSON file containing area geometries",
)
def show_deltacd_ui(detaw_output_file, geojson_file_path=None):
    """
    Show the DeltaCD UI Manager for the specified netCDF file and GeoJSON file.
    """
    dcd_ui = DeltaCDUIManager(detaw_output_file, geojson_file_path=geojson_file_path)
    from pydelmod.dvue import dataui
    import cartopy.crs as ccrs
    dui=dataui.DataUI(dcd_ui, station_id_column="area_id", crs=ccrs.epsg(26910))
    dui.create_view().servable(title="DeltaCD UI Manager").show()
