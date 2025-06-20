import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
from functools import partial
import click
import logging

# Configure logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

# viz imports
import geoviews as gv
import hvplot.pandas
import hvplot.xarray
import holoviews as hv
from holoviews import opts, dim
import cartopy.crs as ccrs

hv.extension("bokeh")
#
import panel as pn

pn.extension()

from pydelmod.dvue import tsdataui

class DeltaCDNodesUIManager(tsdataui.TimeSeriesDataUIManager):
    """
    UI Manager for DeltaCD netCDF data files with node as the station dimension.
    Handles data catalog creation and time series extraction for node and variable combinations.
    """

    def __init__(self, *nc_file_paths, **kwargs):
        """
        Initialize the DeltaCD Nodes UI Manager.
        
        Parameters:
        -----------
        nc_file_paths : str or list of str
            Paths to the netCDF files containing DeltaCD data. Can be a single path or multiple paths.
        nodes_file : str, optional
            Path to the GeoJSON file containing geographical information for nodes
        """
        self.nc_file_paths = nc_file_paths
        self.nodes_file_path = kwargs.pop("nodes_file", None)
        self.datasets = {}
        dfcats = []
        for nc_file_path in self.nc_file_paths:
            if not nc_file_path.endswith('.nc'):
                raise ValueError(f"Invalid file type: {nc_file_path}. Expected a netCDF file (.nc).")
            self.datasets[nc_file_path] = xr.open_dataset(nc_file_path)
            dfcat = self.get_data_catalog_for_dataset(self.datasets[nc_file_path], nc_file_path)
            dfcats.append(dfcat)
        self.gdf = None
        if self.nodes_file_path:
            self.gdf = gpd.read_file(self.nodes_file_path)
            # Make sure the node ID column is named 'id' to match with data
            if 'id' not in self.gdf.columns:
                raise ValueError(f"GeoJSON file must contain an 'id' column for node IDs. Available columns: {self.gdf.columns}")

        # concatenate all data catalogs
        dfcat = pd.concat(dfcats, ignore_index=True)
        
        # Merge with GeoDataFrame if available
        if self.gdf is not None:
            # Convert node to string in both DataFrames before merging
            dfcat['node'] = dfcat['node'].astype(str)
            gdf_copy = self.gdf.copy()
            gdf_copy['id'] = gdf_copy['id'].astype(str)
            
            # Merge with geometry information based on node id
            merged_df = pd.merge(dfcat, gdf_copy, left_on="node", right_on="id", how="left")
            
            # Create GeoDataFrame
            catalog = gpd.GeoDataFrame(merged_df, geometry="geometry")
            
            # Handle CRS properly
            if self.gdf.crs is not None:
                catalog.crs = self.gdf.crs
            else:
                catalog.set_crs(epsg=26910, inplace=True)
        else:
            # If no GeoDataFrame, just use the DataFrame
            catalog = dfcat
            
        self.dfcat = catalog
        # Initialize data cache
        self.data_cache = {}
        
        kwargs['filename_column'] = "source"
        super().__init__(**kwargs)
        # Set up columns for visualization
        self.color_cycle_column = "node"
        self.dashed_line_cycle_column = "variable"
        self.marker_cycle_column = "node"

    def get_data_catalog(self):
        return self.dfcat

    def get_data_catalog_for_dataset(self, ds, nc_file_path):
        """
        Create a data catalog from the netCDF file.
        Each row represents a time series for a variable for a node combination.
        Includes all possible combinations of node and variable.
        """
        # Get available variables, excluding coordinates
        variables = list(ds.data_vars)
        dims = list(ds.dims)
        if "time" not in dims:
            raise ValueError(f"Dataset must contain a 'time' dimension. Dimensions available: {dims}")
        if "node" not in dims:
            raise ValueError(f"Dataset must contain a 'node' dimension. Dimensions available: {dims}")
        nodes = ds.node.values
        
        # Create all combinations
        combinations = []
        for node in nodes:
            for var in variables:
                combinations.append({
                    'node': node,
                    'variable': var
                })
        
        # Create base DataFrame with all combinations
        df = pd.DataFrame(combinations)
        
        # Add additional columns
        # Try to get units from variables
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
        df['interval'] = 'daily'  # Assuming all data is daily
        df['source'] = nc_file_path
        
        # Add time range information
        times = pd.to_datetime(ds.time.values)
        df['start_year'] = str(times.min().year)
        df['max_year'] = str(times.max().year)
        return df

    def get_time_range(self, dfcat):
        """Return the min and max time from the dataset"""
        starttime = pd.to_datetime(dfcat['start_year'].min())
        endtime = pd.to_datetime(dfcat['max_year'].max())
        return starttime, endtime

    def build_station_name(self, r):
        """Build a display name for the node"""
        return f"Node {r['node']}"

    def _get_table_column_width_map(self):
        """Define column widths for the data catalog table"""
        return {
            "node": "8%",
            "variable": "12%",
            "unit": "8%",
            "interval": "10%",
            "start_year": "10%",
            "max_year": "10%",
        }

    def get_table_filters(self):
        """Define filters for the data catalog table"""
        return {
            "node": {
                "type": "input",
                "func": "like",
                "placeholder": "Enter node ID",
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

    def is_irregular(self, r):
        """Check if time series is irregular"""
        return False  # Assuming all time series are regular

    def get_data_for_time_range(self, r, time_range):
        """
        Extract time series data for a specific node and variable combination
        within the specified time range.
        
        Parameters:
        -----------
        r : pandas.Series
            Row from data catalog containing node and variable
        time_range : tuple
            Start and end time for data extraction
    
        Returns:
        --------
        tuple
            (time series DataFrame, unit, data type)
        """
        node = r["node"]
        variable = r["variable"]
        unit = r["unit"]
        filename = r["source"]
        ds = self.datasets[filename]
        try:
            # Extract data from xarray for the specific node and variable
            data = ds[variable].sel(node=node)
            
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
            logger.error(f"Error extracting data for node={node}, variable={variable}: {e}")
            return pd.DataFrame(), unit, "instantaneous"

    def get_tooltips(self):
        """Define tooltips for map visualization"""
        return [
            ("Node ID", "@node"),
            ("Variable", "@variable"),
            ("Unit", "@unit")
        ]

    def get_map_color_columns(self):
        """Return columns that can be used to color the map"""
        return ["variable"]

    def get_map_marker_columns(self):
        """Return columns that can be used as markers on the map"""
        return ["variable"]

    def create_curve(self, df, r, unit, file_index=None):
        """Create a holoviews curve for plotting"""
        file_index_label = f"{file_index}:" if file_index is not None else ""
        
        crvlabel = f'{file_index_label}Node {r["node"]}: {r["variable"]}'
        title = f'{r["variable"]} @ Node {r["node"]}'
        
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
        location_str = f'Node {r["node"]}'
        value[1] = self._append_value(location_str, value[1])
        title_map[unit] = value

    def create_title(self, v):
        """Create plot title from values"""
        title = f"{v[1]} ({v[0]})"
        return title


def build_map(time, gdf, df=None, var=""):
    if var == "diversion":
        color = "blue"
    elif var == "seepage":
        color = "green"
    else:
        color = "red"
    dft = df[df["time"] == time].melt(id_vars="time", value_vars=df.columns)
    dft["node"] = dft["node"].astype(int)
    gdfm = gdf.merge(dft, left_on="id", right_on="node")
    return (
        gdfm.hvplot.points(
            geo=True,
            crs=26910,
            hover_cols="all",
            alpha=0.35,
            width=300,
            tiles="CartoLight",
            c=color,
            responsive=False,
            legend=True,
        )
        .opts(opts.Points(size=0.5 * dim("value")))
        .opts(title=var.upper() + " " + pd.to_datetime(time).strftime("%Y-%m"))
    )


@click.command()
@click.argument(
    "nc_files",
    nargs=-1,
    type=click.Path(exists=True, dir_okay=False),
    required=True,
)
@click.option(
    "--nodes_file",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Path to the GeoJSON file containing node geometries",
)
def show_deltacd_nodes_ui(nc_files, nodes_file=None):
    """
    Show the DeltaCD Nodes UI Manager for the specified netCDF file(s) and nodes GeoJSON file.
    
    This UI is designed for netCDF files that use 'node' as the station dimension.
    """
    dcd_ui = DeltaCDNodesUIManager(*nc_files, nodes_file=nodes_file)
    from pydelmod.dvue import dataui
    dui = dataui.DataUI(dcd_ui, station_id_column="node", crs=ccrs.epsg(26910))
    dui.create_view().servable(title="DeltaCD Nodes UI Manager").show()


@click.command()
@click.option(
    "--ncfile",
    help="dcd netcdf file",
)
@click.option(
    "--nodes_file",
    help="geojson file with nodes from dsm2 model gis info",
)
def dcd_geomap(
    ncfile,
    nodes_file,
):
    """
    Create a map of the nodes with sizes based on the diversion amount

    Parameters
    ----------
    ncfile : str, optional
        dcd netcdf file
    nodes_file : str, optional
        geojson file with nodes from dsm2 model gis info
    """
    # Create a map of the nodes with sizes based on the diversion amount
    # read the nc data
    xrdata = xr.open_dataset(ncfile)
    dflist = {}
    for var in ["diversion", "seepage", "drainage"]:
        monthly = xrdata[var].resample(time="M").mean()
        monthly = monthly.to_dataframe().reset_index()
        df = monthly.pivot(index="time", columns="node", values=var)
        df = df.reset_index()
        df.melt(id_vars="time", value_vars=df.columns)
        # BBID mapped on to node 94 (There was no value from the diversion data for that node so substituting with BBID)
        df = df.rename(columns={"BBID": "94"})
        dflist[var] = df
    gdf = gpd.read_file(nodes_file).to_crs(epsg=26910)
    #
    times = dflist["diversion"]["time"].unique()
    dmaplist = {
        var: hv.DynamicMap(
            partial(build_map, gdf=gdf, df=dflist[var], var=var), kdims="time"
        ).redim.values(time=times)
        for var in dflist.keys()
    }
    maplist = pn.Row(
        dmaplist["diversion"], dmaplist["drainage"], dmaplist["seepage"]
    ).servable()
    maplist.show()
