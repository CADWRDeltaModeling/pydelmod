# User interface components for DSM2 related information
import panel as pn
import param
import colorcet as cc

import cartopy.crs as ccrs

# viz imports
import geoviews as gv
import holoviews as hv
from holoviews import opts

hv.extension("bokeh")
import hvplot.pandas

#
import panel as pn

# Try to load VTK extension, but make it optional
HAS_VTK = False
if HAS_VTK:
    try:
        pn.extension("vtk")
        import pyvista

        HAS_VTK = True
    except (ImportError, Exception):
        HAS_VTK = False


# Define a function to safely import PyVista-related modules
def import_vtk_modules():
    """Import PyVista and VTK-related modules if available."""
    if not HAS_VTK:
        return None, None

    try:
        import pyvista as pv
        from panel.pane import VTK, VTKVolume

        return pv, VTK
    except ImportError:
        return None, None


import pyhecdss as dss
from vtools.functions.filter import cosine_lanczos
from .dsm2study import *
from .dvue.dataui import full_stack
from .dvue.tsdataui import TimeSeriesDataUIManager


class DSM2DataUIManager(TimeSeriesDataUIManager):

    def __init__(self, output_channels, **kwargs):
        """
        output_channels is a geopandas dataframe with columns:
        NAME  CHAN_NO  DISTANCE  VARIABLE  INTERVAL  PERIOD_OP  FILE
        """
        self.time_range = kwargs.pop("time_range", None)
        self.output_channels = output_channels
        self.display_fileno = False
        filename_column = "FILE"
        unique_files = self.output_channels[filename_column].unique()
        if len(unique_files) > 1:
            output_channels["FILE_NO"] = output_channels[filename_column].apply(
                lambda x: unique_files.tolist().index(x)
            )
        self.station_id_column = "NAME"
        super().__init__(file_number_column_name="FILE_NO", **kwargs)
        self.color_cycle_column = "NAME"
        self.dashed_line_cycle_column = "FILE"
        self.marker_cycle_column = "VARIABLE"

    def build_station_name(self, r):
        if self.display_fileno:
            return f'{r["FILE_NO"]}:{r["NAME"]}'
        else:
            return f'{r["NAME"]}'

    # data related methods
    def get_data_catalog(self):
        return self.output_channels

    def get_time_range(self, dfcat):
        return self.time_range

    def _get_table_column_width_map(self):
        """only columns to be displayed in the table should be included in the map"""
        column_width_map = {
            "NAME": "15%",
            "CHAN_NO": "10%",
            "DISTANCE": "10%",
            "VARIABLE": "10%",
            "INTERVAL": "5%",
            "PERIOD_OP": "5%",
        }
        return column_width_map

    def get_table_filters(self):
        table_filters = {
            "NAME": {"type": "input", "func": "like", "placeholder": "Enter match"},
            "CHAN_NO": {"type": "input", "func": "like", "placeholder": "Enter match"},
            "DISTANCE": {"type": "input", "func": "like", "placeholder": "Enter match"},
            "VARIABLE": {"type": "input", "func": "like", "placeholder": "Enter match"},
            "INTERVAL": {"type": "input", "func": "like", "placeholder": "Enter match"},
        }
        return table_filters

    def _append_value(self, new_value, value):
        if new_value not in value:
            value += f'{", " if value else ""}{new_value}'
        return value

    def append_to_title_map(self, title_map, unit, r):
        if unit in title_map:
            value = title_map[unit]
        else:
            value = ["", "", "", ""]
        value[0] = self._append_value(r["VARIABLE"], value[0])
        value[1] = self._append_value(r["NAME"], value[1])
        value[2] = self._append_value(str(r["CHAN_NO"]), value[2])
        value[3] = self._append_value(str(r["DISTANCE"]), value[3])
        title_map[unit] = value

    def create_title(self, v):
        title = f"{v[1]} @ {v[2]} ({v[3]}::{v[0]})"
        return title

    def is_irregular(self, r):
        return False

    def create_curve(self, df, r, unit, file_index=None):
        file_index_label = f"{file_index}:" if file_index is not None else ""
        crvlabel = f'{file_index_label}{r["NAME"]}/{r["VARIABLE"]}'
        ylabel = f'{r["VARIABLE"]} ({unit})'
        title = f'{r["VARIABLE"]} @ {r["NAME"]} ({r["CHAN_NO"]}/{r["DISTANCE"]})'
        irreg = self.is_irregular(r)
        if irreg:
            crv = hv.Scatter(df.iloc[:, [0]], label=crvlabel).redim(value=crvlabel)
        else:
            crv = hv.Curve(df.iloc[:, [0]], label=crvlabel).redim(value=crvlabel)
        return crv.opts(
            xlabel="Time",
            ylabel=ylabel,
            title=title,
            responsive=True,
            active_tools=["wheel_zoom"],
            tools=["hover"],
        )

    def get_data_for_time_range(self, r, time_range):
        dssfile = r[self.filename_column]
        pathname = f'//{r["NAME"]}/{r["VARIABLE"]}////'
        df, unit, ptype = next(
            lru_cache(maxsize=32)(dss.get_matching_ts)(dssfile, pathname)
        )  # get first matching time series
        df = df[slice(*time_range)]
        return df, unit, ptype

    # methods below if geolocation data is available
    def get_tooltips(self):
        return [
            ("NAME", "@NAME"),
            ("CHAN_NO", "@CHAN_NO"),
            ("DISTANCE", "@DISTANCE"),
            ("VARIABLE", "@VARIABLE"),
        ]

    def get_map_color_category(self):
        return "VARIABLE"

    def get_map_color_columns(self):
        """return the columns that can be used to color the map"""
        return ["VARIABLE"]

    def get_map_marker_columns(self):
        """return the columns that can be used to color the map"""
        return ["VARIABLE"]


import numpy as np
from pydsm.hydroh5 import HydroH5
from pydsm.qualh5 import QualH5


class DSM2TidefileXsectUIManager(param.Parameterized):
    selected_channel = param.Selector(default=None, doc="Selected channel number")
    selected_xsect = param.Selector(default=None, doc="Selected cross-section number")
    show_all_xsects = param.Boolean(default=False, label="Show all cross sections")
    table_xsect = param.Selector(
        default=None, doc="Cross-section to show in data table"
    )
    show_3d_view = param.Boolean(default=False, label="Show 3D channel volume")

    def __init__(self, tidefile, **kwargs):
        """
        Create a UI for examining cross-sections from a DSM2 tidefile

        Parameters
        ----------
        tidefile: str
            Path to the tidefile
        """
        super().__init__(**kwargs)

        # Load the tidefile
        self.tidefile = tidefile
        self.hydro = HydroH5(tidefile)

        # Get the virtual cross-section data
        self.vx = self.hydro.get_geometry_table("/hydro/geometry/virtual_xsect")

        # Get unique channel numbers
        self.channel_numbers = np.sort(self.vx["chan_no"].unique())
        self.param.selected_channel.objects = list(self.channel_numbers)

        if len(self.channel_numbers) > 0:
            # Default to the first channel
            self.selected_channel = self.channel_numbers[0]

            # Get cross-sections for the selected channel
            self.update_xsections()

    def update_xsections(self):
        """Update the available cross-sections for the selected channel"""
        if self.selected_channel is None:
            return

        # Filter for the selected channel
        chan_xsects = self.vx[self.vx["chan_no"] == self.selected_channel]

        # Get unique vsecno values for the channel
        self.unique_vsecno = np.sort(chan_xsects["vsecno"].unique())
        self.param.selected_xsect.objects = list(self.unique_vsecno)
        self.param.table_xsect.objects = list(self.unique_vsecno)

        if len(self.unique_vsecno) > 0:
            self.selected_xsect = self.unique_vsecno[0]
            self.table_xsect = self.unique_vsecno[0]

    @param.depends("selected_channel", watch=True)
    def _update_xsections_callback(self):
        self.update_xsections()

    def get_xsection_data(self, channel, vsecno=None):
        """
        Get cross-section data for the specified channel and cross-section

        Parameters
        ----------
        channel : int
            Channel number
        vsecno : int, optional
            Cross-section number. If None, returns data for all cross-sections in the channel

        Returns
        -------
        dict
            Dictionary of dataframes containing cross-section data, keyed by vsecno
        """
        # Filter for the specified channel
        chan_xsects = self.vx[self.vx["chan_no"] == channel]

        if vsecno is not None:
            # Filter for specific cross-section if provided
            chan_xsects = chan_xsects[chan_xsects["vsecno"] == vsecno]

        # Get unique vsecno values
        unique_vsecs = chan_xsects["vsecno"].unique()

        # Create a dictionary to hold dataframes for each vsecno
        xsect_dfs = {}

        for vsec in unique_vsecs:
            # Filter for this specific cross section
            xsect_df = chan_xsects[chan_xsects["vsecno"] == vsec]

            # Create a new dataframe with just the required columns
            xsect_dfs[vsec] = xsect_df[
                ["min_elev", "elevation", "area", "wet_p", "width"]
            ].reset_index(drop=True)

        return xsect_dfs

    def create_symmetric_df(self, df):
        """
        Create a symmetrical cross-section dataframe for plotting

        Parameters
        ----------
        df : DataFrame
            Cross-section data

        Returns
        -------
        DataFrame
            Dataframe with symmetrical width points for plotting
        """
        import pandas as pd

        # Create a new dataframe for the symmetrical cross-section
        symmetric_df = pd.DataFrame()

        # For each unique elevation, create symmetrical width points
        for elev in df["elevation"].unique():
            elev_rows = df["elevation"] == elev
            width = df.loc[elev_rows, "width"].max()  # Get the width at this elevation

            # Create two points: one at -width/2 and one at +width/2
            new_rows = pd.DataFrame(
                {
                    "elevation": [elev, elev],
                    "width": [-width / 2, width / 2],
                    "min_elev": df.loc[elev_rows, "min_elev"].iloc[0],
                    "area": df.loc[elev_rows, "area"].iloc[0],
                    "wet_p": df.loc[elev_rows, "wet_p"].iloc[0],
                }
            )

            symmetric_df = pd.concat([symmetric_df, new_rows])

        # Sort by width for proper plotting
        symmetric_df = symmetric_df.sort_values("width")

        return symmetric_df

    def create_3d_channel_volume(self, channel):
        """
        Create a 3D volume representation of a channel from its cross-sections

        Parameters
        ----------
        channel : int
            Channel number

        Returns
        -------
        tuple
            (volume_data, spacing, origin) for VTKVolume
        """
        import numpy as np
        import pandas as pd

        # Get all cross-sections for this channel
        xsect_dfs = self.get_xsection_data(channel)

        if not xsect_dfs:
            return None

        # Sort cross-sections by their number to ensure proper order along channel
        xsect_nums = sorted(list(xsect_dfs.keys()))

        # Determine the dimensions of our volume
        n_xsects = len(xsect_nums)

        # For each cross-section, standardize the elevations and widths for interpolation
        # Find the min/max elevation and width across all cross-sections
        min_elev = float("inf")
        max_elev = float("-inf")
        max_width = 0

        for xsect in xsect_nums:
            df = xsect_dfs[xsect]
            symmetric_df = self.create_symmetric_df(df)
            min_elev = min(min_elev, symmetric_df["elevation"].min())
            max_elev = max(max_elev, symmetric_df["elevation"].max())
            max_width = max(max_width, symmetric_df["width"].abs().max() * 2)

        # Create a uniform grid for all cross-sections
        n_height = 50  # Number of points in vertical direction
        n_width = 50  # Number of points in horizontal direction

        # Create 3D array for the channel volume
        volume = np.zeros((n_xsects, n_height, n_width), dtype=np.float32)

        # For each cross-section, interpolate onto the uniform grid
        for i, xsect in enumerate(xsect_nums):
            df = xsect_dfs[xsect]
            symmetric_df = self.create_symmetric_df(df)

            # Sort by elevation for proper interpolation
            symmetric_df = symmetric_df.sort_values("elevation")

            # Create a regular grid for this cross-section
            elev_points = np.linspace(min_elev, max_elev, n_height)
            width_points = np.linspace(-max_width / 2, max_width / 2, n_width)

            # For each elevation, interpolate the width profile
            for j, elev in enumerate(elev_points):
                # Find closest elevations in the data
                elev_idx = np.searchsorted(symmetric_df["elevation"].values, elev)

                # If we're at or beyond the highest elevation, fill with zeros (air)
                if (
                    elev_idx >= len(symmetric_df["elevation"])
                    or elev > symmetric_df["elevation"].max()
                ):
                    volume[i, j, :] = 0
                    continue

                # If we're below the lowest elevation, fill with ones (solid ground)
                if elev_idx == 0 or elev < symmetric_df["elevation"].min():
                    volume[i, j, :] = 1
                    continue

                # Get the widths at this elevation
                elev_df = symmetric_df[
                    symmetric_df["elevation"]
                    == symmetric_df["elevation"].iloc[elev_idx]
                ]

                if len(elev_df) >= 2:  # We have both left and right points
                    left_width = elev_df["width"].min()
                    right_width = elev_df["width"].max()

                    # For each width point, check if it's inside the channel
                    for k, width in enumerate(width_points):
                        if left_width <= width <= right_width:
                            volume[i, j, k] = 1  # Inside channel (water)
                        else:
                            volume[i, j, k] = 0  # Outside channel

        # Calculate spacing (assuming units are in feet)
        # For the channel length, we use an arbitrary spacing since we don't have actual distances
        channel_length = 1000  # Arbitrary channel length in feet
        x_spacing = channel_length / (n_xsects - 1 if n_xsects > 1 else 1)
        y_spacing = (max_elev - min_elev) / (n_height - 1)
        z_spacing = max_width / (n_width - 1)

        spacing = (x_spacing, y_spacing, z_spacing)
        origin = (0, min_elev, -max_width / 2)

        return volume, spacing, origin

    def create_3d_channel_surface(self, channel):
        """
        Create a 3D surface representation of a channel from its cross-sections
        using PyVista to create a structured surface from the cross-section points.

        Parameters
        ----------
        channel : int
            Channel number

        Returns
        -------
        pyvista.PolyData
            A PyVista mesh representing the channel surface
        """
        import numpy as np
        import pandas as pd
        import os
        import time

        # Check if VTK extension is available
        if not HAS_VTK:
            print("VTK extension not available. Cannot create 3D visualization.")
            return None

        # Try to import PyVista - this is optional
        try:
            import pyvista as pv
        except ImportError:
            print("PyVista package not found. Please install with: pip install pyvista")
            return None

        # Get all cross-sections for this channel
        xsect_dfs = self.get_xsection_data(channel)

        if not xsect_dfs:
            return None

        # Sort cross-sections by their number to ensure proper order along channel
        xsect_nums = sorted(list(xsect_dfs.keys()))

        if len(xsect_nums) < 2:
            return None  # Need at least 2 cross sections to build a surface

        # Channel length (arbitrary, for visualization)
        channel_length = 1000  # feet
        station_spacing = channel_length / (len(xsect_nums) - 1)

        # Create lists to store points and connectivity information for each cross-section
        cross_section_points = []
        n_points_per_section = []

        # Process each cross-section to get points
        for i, xsect_num in enumerate(xsect_nums):
            df = xsect_dfs[xsect_num]
            symmetric_df = self.create_symmetric_df(df)

            # Sort by elevation for consistent ordering
            symmetric_df = symmetric_df.sort_values(["elevation", "width"])

            # Station position along the channel
            station = i * station_spacing

            # Create points for this cross-section: (station, elevation, width)
            section_points = np.column_stack(
                [
                    np.full(len(symmetric_df), station),  # x: station along channel
                    symmetric_df["elevation"].values,  # y: elevation
                    symmetric_df["width"].values,  # z: width (centered)
                ]
            )

            # Store points and count
            cross_section_points.append(section_points)
            n_points_per_section.append(len(section_points))

        # Create a PyVista PolyData object for the channel surface
        surface = pv.PolyData()

        # Combine all points into a single array
        all_points = np.vstack(cross_section_points)
        surface.points = all_points

        # Create faces for the surface (triangulated quads between cross-sections)
        faces = []
        point_offset = 0

        # First, let's verify our cross-section data
        print(f"Creating surface with {len(xsect_nums)} cross-sections")
        for i, n_pts in enumerate(n_points_per_section):
            print(f"  Cross-section {i}: {n_pts} points")

        # Try a different approach to create faces - use PyVista's StructuredGrid
        if len(cross_section_points) >= 2:
            try:
                # Instead of manual face creation, try to use PyVista's structured grid
                # and then extract the surface
                print("Attempting to create structured grid from cross-sections")

                # Make sure all cross-sections have the same number of points
                min_points = min(n_points_per_section)
                print(f"Normalizing cross-sections to {min_points} points each")

                # Create a structured grid with normalized cross-sections
                grid_points = np.zeros((len(xsect_nums), min_points, 3))

                for i, points in enumerate(cross_section_points):
                    # If needed, interpolate to get the same number of points per cross-section
                    if len(points) > min_points:
                        # Simple subsampling for now - ideally this would be interpolation
                        indices = np.linspace(0, len(points) - 1, min_points, dtype=int)
                        grid_points[i] = points[indices]
                    elif len(points) == min_points:
                        grid_points[i] = points

                # Create a structured grid
                print(f"Creating structured grid with shape {grid_points.shape}")
                grid = pv.StructuredGrid(grid_points)

                # Extract the surface
                surface = grid.extract_surface()
                print(
                    f"Extracted surface with {surface.n_points} points and {surface.n_faces} faces"
                )

                # Add the elevation data
                surface.point_data["elevation"] = surface.points[:, 1]

                # Skip the manual face creation below
                return surface

            except Exception as e:
                print(f"Structured grid approach failed: {str(e)}")
                print("Falling back to manual face creation")

        # If the structured grid approach failed, fall back to manual face creation
        for i in range(len(xsect_nums) - 1):
            n_current = n_points_per_section[i]
            n_next = n_points_per_section[i + 1]

            # Determine how to connect points between cross-sections
            # We'll use the smaller number of points to avoid index errors
            n_connect = min(n_current, n_next)

            if n_connect < 2:
                print(
                    f"Warning: Not enough points to create faces between cross-sections {i} and {i+1}"
                )
                continue

            # Create triangular faces between the cross-sections
            for j in range(n_connect - 1):
                # First triangle: current, next, current+1
                faces.append(
                    [
                        3,
                        point_offset + j,
                        point_offset + n_current + j,
                        point_offset + j + 1,
                    ]
                )

                # Second triangle: current+1, next, next+1
                faces.append(
                    [
                        3,
                        point_offset + j + 1,
                        point_offset + n_current + j,
                        point_offset + n_current + j + 1,
                    ]
                )

            # Update point offset for the next cross-section pair
            point_offset += n_current

        # Add faces to the surface
        if faces:
            surface.faces = np.hstack(faces)

            # Add elevation as a scalar field for coloring
            surface.point_data["elevation"] = all_points[:, 1]

            # Compute normals for better rendering
            surface.compute_normals(inplace=True)

            # Save the surface to a file for external inspection
            import os

            output_dir = os.path.join(os.path.expanduser("~"), "dsm2_surfaces")
            os.makedirs(output_dir, exist_ok=True)

            # Create a filename with channel number and timestamp
            import time

            timestamp = int(time.time())
            filename = os.path.join(
                output_dir, f"channel_{channel}_surface_{timestamp}.vtk"
            )

            try:
                surface.save(filename)
                print(f"Surface saved to: {filename}")
            except Exception as e:
                print(f"Error saving surface to file: {str(e)}")

            # Check if the surface is valid
            print(
                f"Surface info - Points: {surface.n_points}, Cells: {surface.n_cells}, Faces: {len(faces)}"
            )
            if surface.n_points == 0 or surface.n_cells == 0:
                print("Warning: Empty surface detected!")

            return surface
        else:
            return None

    @param.depends("selected_channel", "selected_xsect", "show_all_xsects")
    def create_xsection_plot(self):
        """Create cross-section plot for the selected channel and cross-section"""
        import holoviews as hv
        import hvplot.pandas
        from holoviews import opts

        if self.selected_channel is None:
            return hv.Div("No channel selected")

        if self.show_all_xsects:
            # Show all cross-sections for the selected channel
            xsect_dfs = self.get_xsection_data(self.selected_channel)

            # Create a dictionary to store the plots
            xsect_plots = {}

            # Generate plots for each cross-section
            for vsec, df in xsect_dfs.items():
                symmetric_df = self.create_symmetric_df(df)

                # Create a scatter plot of elevation vs centered width
                plot = symmetric_df.hvplot.scatter(
                    x="width",
                    y="elevation",
                    title=f"Cross-Section {vsec}",
                    xlabel="Centered Width (ft)",
                    ylabel="Elevation (ft)",
                    height=400,
                    width=600,
                )

                # Store the plot in the dictionary
                xsect_plots[vsec] = plot

            # Create an overlay of all plots
            all_plots = hv.NdOverlay(
                {f"XS {vsec}": xsect_plots[vsec] for vsec in xsect_dfs.keys()}
            )
            all_plots = all_plots.opts(
                opts.NdOverlay(
                    legend_position="right",
                    width=800,
                    height=500,
                    title=f"All Cross-Sections for Channel {self.selected_channel}",
                )
            )

            return all_plots
        else:
            # Show only the selected cross-section
            if self.selected_xsect is None:
                return hv.Div("No cross-section selected")

            xsect_dfs = self.get_xsection_data(
                self.selected_channel, self.selected_xsect
            )

            if not xsect_dfs:
                return hv.Div(
                    f"No data for Channel {self.selected_channel}, Cross-section {self.selected_xsect}"
                )

            df = xsect_dfs[self.selected_xsect]
            symmetric_df = self.create_symmetric_df(df)

            # Create a scatter plot of elevation vs centered width
            plot = symmetric_df.hvplot.scatter(
                x="width",
                y="elevation",
                title=f"Channel {self.selected_channel}, Cross-Section {self.selected_xsect}",
                xlabel="Centered Width (ft)",
                ylabel="Elevation (ft)",
                height=500,
                width=800,
            )

            return plot

    @param.depends("selected_channel", "show_3d_view")
    def create_vtk_volume(self):
        """Create VTK volume visualization for the channel"""
        import panel as pn
        import os
        import time

        if not self.show_3d_view or self.selected_channel is None:
            return pn.pane.Str("")

        # Check if VTK extension is available
        if not HAS_VTK:
            return pn.pane.Markdown(
                """
                ### 3D Visualization Not Available

                The VTK extension for Panel is not available.
                Please install it with:
                ```
                pip install pyvista panel vtk
                ```
                Then restart your application.
                """
            )

        try:
            # Import the necessary libraries - these are optional
            try:
                import pyvista as pv
                from panel.pane import VTK
            except ImportError as e:
                return pn.pane.Markdown(
                    f"""
                    ### 3D Visualization requires additional libraries

                    To use the 3D visualization feature, please install:
                    ```
                    pip install pyvista panel vtk
                    ```

                    Error: {str(e)}
                    """
                )

            # Create 3D channel surface using the new surface method
            surface = self.create_3d_channel_surface(self.selected_channel)

            if surface is None:
                print("Surface creation failed, falling back to legacy volume approach")
                return self.create_vtk_volume_legacy()

            # Create debug information
            debug_info = f"""
            Surface Details:
            - Points: {surface.n_points}
            - Faces: {surface.n_faces}
            - Cells: {surface.n_cells}
            - Has Elevation Data: {"elevation" in surface.point_data}
            - Bounds: {surface.bounds}
            """
            print(debug_info)

            # Create a plotter for better control
            plotter = pv.Plotter()
            plotter.add_mesh(
                surface,
                scalars="elevation",
                cmap="terrain",  # Terrain colormap works well for elevation data
                show_edges=True,  # Show edges for better debugging
                line_width=1,
                smooth_shading=True,
            )

            # Save the rendered scene to a file
            output_dir = os.path.join(os.path.expanduser("~"), "dsm2_surfaces")
            os.makedirs(output_dir, exist_ok=True)
            timestamp = int(time.time())
            screenshot_path = os.path.join(
                output_dir, f"channel_{self.selected_channel}_render_{timestamp}.png"
            )

            try:
                # Take a screenshot of the scene
                plotter.show(auto_close=False)  # Render the scene but don't close
                plotter.screenshot(screenshot_path)
                print(f"Rendering screenshot saved to: {screenshot_path}")
                plotter.close()

                # Create fresh plotter for Panel
                plotter = pv.Plotter()
                plotter.add_mesh(
                    surface,
                    scalars="elevation",
                    cmap="terrain",
                    show_edges=True,
                    line_width=1,
                    smooth_shading=True,
                )
            except Exception as e:
                print(f"Error saving screenshot: {str(e)}")

            # Set up a VTK pane with the surface
            debug_message = f"Surface has {surface.n_points} points and {surface.n_faces} faces. View saved to: {os.path.basename(screenshot_path)}"

            return pn.Column(
                pn.pane.Markdown(
                    f"### 3D Channel Bathymetry for Channel {self.selected_channel}"
                ),
                pn.pane.Markdown(debug_message),
                VTK(
                    plotter,
                    height=500,
                    width=800,
                ),
            )
        except ImportError as e:
            return pn.pane.Markdown(
                f"""
                #### 3D Visualization requires additional libraries

                To use the 3D visualization feature, please install:
                ```
                pip install pyvista panel vtk
                ```

                Error: {str(e)}
                """
            )
        except Exception as e:
            # If there's an error with the surface approach, fall back to volume approach
            try:
                return self.create_vtk_volume_legacy()
            except Exception as inner_e:
                return pn.pane.Markdown(
                    f"Error creating 3D visualization: {str(e)}\nFallback error: {str(inner_e)}"
                )

    def create_vtk_volume_legacy(self):
        """Legacy method for creating VTK volume visualization using voxel-based approach"""
        import panel as pn
        import numpy as np

        # Check if VTK extension is available
        if not HAS_VTK:
            return pn.pane.Markdown(
                """
                ### 3D Visualization Not Available

                The VTK extension for Panel is not available.
                Please install it with:
                ```
                pip install pyvista panel vtk
                ```
                Then restart your application.
                """
            )

        try:
            # Import the necessary libraries - these are optional
            try:
                import pyvista as pv
                from panel.pane import VTKVolume
            except ImportError as e:
                return pn.pane.Markdown(
                    f"""
                    ### 3D Visualization requires additional libraries

                    To use the 3D visualization feature, please install:
                    ```
                    pip install pyvista panel vtk
                    ```

                    Error: {str(e)}
                    """
                )

            # Create 3D channel volume
            volume_data = self.create_3d_channel_volume(self.selected_channel)

            if volume_data is None:
                return pn.pane.Markdown(
                    f"No data available for 3D visualization of Channel {self.selected_channel}"
                )

            volume, spacing, origin = volume_data

            # Create a PyVista uniform grid - handle different PyVista versions
            try:
                # For newer PyVista versions
                grid = pv.UniformGrid()
            except AttributeError:
                try:
                    # For older PyVista versions
                    from pyvista.core.grid import UniformGrid

                    grid = UniformGrid()
                except (ImportError, AttributeError):
                    # Fall back to using ImageData which is similar to UniformGrid
                    # but has been available in earlier PyVista versions
                    grid = pv.ImageData(
                        dimensions=np.array(volume.shape) + 1,
                        spacing=spacing,
                        origin=origin,
                    )

            # Set dimensions, spacing, and origin if not already set (for UniformGrid)
            if not hasattr(grid, "dimensions") or grid.dimensions is None:
                grid.dimensions = np.array(volume.shape) + 1
            if not hasattr(grid, "spacing") or grid.spacing is None:
                grid.spacing = spacing
            if not hasattr(grid, "origin") or grid.origin is None:
                grid.origin = origin

            # Create a properly sized array for the grid points
            # The grid dimensions are volume.shape + 1, so we need to resize our data
            # to match the number of points in the grid
            dims = np.array(volume.shape) + 1
            num_points = dims[0] * dims[1] * dims[2]

            # Create a new volume array that's compatible with the grid dimensions
            grid_values = np.zeros(num_points, dtype=np.float32)

            # Add the volume data to the grid
            grid.cell_data["values"] = volume.flatten(order="F")

            # We need to convert cell data to point data for proper volume rendering
            grid.cell_data_to_point_data()

            # Make sure we have data in point_data after conversion
            if "values" not in grid.point_data:
                if "values" in grid.cell_data:
                    # If the conversion didn't work, create a simpler volume directly
                    new_grid = pv.ImageData(
                        dimensions=volume.shape,  # Not +1 for direct mapping
                        spacing=spacing,
                        origin=origin,
                    )
                    new_grid.point_data["values"] = volume.flatten(order="F")
                    grid = new_grid

            # Set up a VTK volume with the data
            return pn.Column(
                pn.pane.Markdown(
                    f"### 3D Channel Volume for Channel {self.selected_channel}"
                ),
                VTKVolume(
                    grid,
                    colormap="Rainbow Desaturated",  # More widely available colormap in VTK
                    shadow=True,
                    height=500,
                    width=800,
                    orientation_widget=True,
                ),
            )
        except Exception as e:
            return pn.pane.Markdown(
                f"Error creating 3D visualization (legacy method): {str(e)}"
            )
            return pn.pane.Markdown(f"Error creating 3D visualization: {str(e)}")

    @param.depends(
        "selected_channel", "selected_xsect", "show_all_xsects", "table_xsect"
    )
    def get_data_table(self):
        """Create a data table for the selected cross-section"""
        import panel as pn

        if self.selected_channel is None:
            return pn.pane.Markdown("No channel selected")

        # For table display, we'll use table_xsect if showing all, otherwise selected_xsect
        display_xsect = (
            self.table_xsect if self.show_all_xsects else self.selected_xsect
        )

        if display_xsect is None:
            return pn.pane.Markdown("No cross-section selected")

        xsect_dfs = self.get_xsection_data(self.selected_channel, display_xsect)
        if not xsect_dfs or display_xsect not in xsect_dfs:
            return pn.pane.Markdown(
                f"No data for Channel {self.selected_channel}, Cross-section {display_xsect}"
            )

        df = xsect_dfs[display_xsect]
        return pn.Column(
            pn.pane.Markdown(f"#### Data for Cross-Section {display_xsect}"),
            pn.widgets.Tabulator(
                df,
                pagination="local",
                page_size=10,
                sizing_mode="stretch_width",
                height=250,
            ),
        )

    def get_panel(self):
        """Create a panel interface for the cross-section viewer"""
        import panel as pn

        # Create widgets for channel and cross-section selection
        channel_select = pn.Param(
            self.param.selected_channel,
            widgets={
                "selected_channel": pn.widgets.Select(
                    name="Channel", options=list(self.channel_numbers)
                )
            },
        )

        xsect_select = pn.Param(
            self.param.selected_xsect,
            widgets={
                "selected_xsect": pn.widgets.Select(
                    name="Cross-Section",
                    options=(
                        list(self.unique_vsecno)
                        if hasattr(self, "unique_vsecno")
                        else []
                    ),
                )
            },
        )

        show_all_checkbox = pn.Param(
            self.param.show_all_xsects,
            widgets={
                "show_all_xsects": pn.widgets.Checkbox(name="Show all cross-sections")
            },
        )

        # Only show 3D checkbox if VTK is available
        if HAS_VTK:
            show_3d_checkbox = pn.Param(
                self.param.show_3d_view,
                widgets={
                    "show_3d_view": pn.widgets.Checkbox(name="Show 3D channel volume")
                },
            )
        else:
            show_3d_checkbox = pn.pane.Markdown(
                "*3D visualization not available - install PyVista and VTK*"
            )

        # Create table cross-section selector for when showing all cross-sections
        table_xsect_select = pn.Param(
            self.param.table_xsect,
            widgets={
                "table_xsect": pn.widgets.Select(
                    name="Table Cross-Section",
                    options=(
                        list(self.unique_vsecno)
                        if hasattr(self, "unique_vsecno")
                        else []
                    ),
                )
            },
        )

        # The table section will show the selector only when "show all" is checked
        table_controls = pn.Column(
            pn.pane.Markdown("### Cross-Section Data"),
            pn.bind(
                lambda show_all: table_xsect_select if show_all else pn.pane.Str(""),
                self.param.show_all_xsects,
            ),
        )

        # Create a tabulator widget for the cross-section data
        data_table_pane = pn.Column(table_controls, self.get_data_table)

        # Create the panel layout
        return pn.Column(
            pn.pane.Markdown(
                f"# DSM2 Tidefile Cross-Section Viewer\n## File: {self.tidefile}"
            ),
            pn.Row(
                pn.Column(
                    channel_select,
                    xsect_select,
                    show_all_checkbox,
                    show_3d_checkbox,
                    width=300,
                ),
                pn.Column(self.create_xsection_plot, width=800),
            ),
            data_table_pane,
            self.create_vtk_volume,
        )


class DSM2TidefileUIManager(TimeSeriesDataUIManager):

    def __init__(self, tidefiles, **kwargs):
        """
        tidefiles: A list of tide files
        """
        self.time_range = kwargs.pop("time_range", None)
        self.channels = kwargs.pop("channels", None)
        self.tidefiles = tidefiles
        self.display_fileno = False
        self.tidefile_map = {
            f: DSM2TidefileUIManager.read_tidefile(f) for _, f in enumerate(tidefiles)
        }
        self.dfcat = pd.concat(
            [f.create_catalog() for k, f in self.tidefile_map.items()]
        )
        self.dfcat.reset_index(drop=True, inplace=True)
        self.dfcat["geoid"] = self.dfcat.id.str.split("_", expand=True).iloc[:, 1]
        if self.channels is not None:
            self.channels.id = self.channels.id.astype("str")
            self.channels.rename(columns={"id": "geoid"}, inplace=True)
            self.dfcat = pd.merge(
                self.channels,
                self.dfcat,
                left_on="geoid",
                right_on="geoid",
                how="right",
            )
        self.station_id_column = "geoid"
        time_ranges = [f.get_start_end_dates() for k, f in self.tidefile_map.items()]
        self.time_range = (
            min([pd.to_datetime(t[0]) for t in time_ranges]),
            max([pd.to_datetime(t[1]) for t in time_ranges]),
        )
        super().__init__(
            filename_column="filename",
            file_number_column_name="FILE_NUM",
            time_range=self.time_range,
            **kwargs,
        )
        self.color_cycle_column = "id"
        self.dashed_line_cycle_column = "filename"
        self.marker_cycle_column = "variable"

    def read_tidefile(tidefile, guess="hydro"):
        try:
            if guess == "hydro":
                return HydroH5(tidefile)
            else:
                return QualH5(tidefile)
        except:
            if guess == "hydro":
                return QualH5(tidefile)
            else:
                return HydroH5(tidefile)

    def build_station_name(self, r):
        if self.display_fileno:
            return f'{r["FILE_NUM"]}:{r[self.station_id_column]}'
        else:
            return f"{r[self.station_id_column]}"

    # data related methods
    def get_data_catalog(self):
        return self.dfcat

    def get_time_range(self, dfcat):
        return self.time_range

    def _get_table_column_width_map(self):
        """only columns to be displayed in the table should be included in the map"""
        column_width_map = {
            "geoid": "10%",
            "id": "15%",
            "variable": "10%",
            "unit": "10%",
        }
        return column_width_map

    def get_table_filters(self):
        table_filters = {
            "geoid": {"type": "input", "func": "like", "placeholder": "Enter match"},
            "id": {"type": "input", "func": "like", "placeholder": "Enter match"},
            "variable": {"type": "input", "func": "like", "placeholder": "Enter match"},
            "unit": {"type": "input", "func": "like", "placeholder": "Enter match"},
        }
        return table_filters

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
        value[1] = self._append_value(r["id"], value[1])
        title_map[unit] = value

    def create_title(self, v):
        title = f"{v[0]} @ {v[1]}"
        return title

    def is_irregular(self, r):
        return False

    def create_curve(self, df, r, unit, file_index=None):
        file_index_label = f"{file_index}:" if file_index is not None else ""
        crvlabel = f'{file_index_label}{r["id"]}/{r["variable"]}'
        ylabel = f'{r["variable"]} ({unit})'
        title = f'{r["variable"]} @ {r["id"]}'
        irreg = self.is_irregular(r)
        crv = hv.Curve(df.iloc[:, [0]], label=crvlabel).redim(value=crvlabel)
        return crv.opts(
            xlabel="Time",
            ylabel=ylabel,
            title=title,
            responsive=True,
            active_tools=["wheel_zoom"],
            tools=["hover"],
        )

    def _get_timewindow_for_time_range(self, time_range):
        # extend time range to include whole days
        xtime_range = (
            pd.to_datetime(time_range[0]).floor("D"),
            pd.to_datetime(time_range[1]).ceil("D"),
        )
        return "-".join(map(lambda x: x.strftime("%d%b%Y"), xtime_range))

    @lru_cache(maxsize=32)
    def _get_data_for_catalog_entry(self, filename, variable, id, time_window):
        entry = {self.filename_column: filename, "variable": variable, "id": id}
        return self.tidefile_map[filename].get_data_for_catalog_entry(
            entry, time_window
        )

    def get_data_for_time_range(self, r, time_range):
        var = r["variable"]
        time_window = self._get_timewindow_for_time_range(time_range)
        df = self._get_data_for_catalog_entry(
            r[self.filename_column], r["variable"], r["id"], time_window
        )
        unit = r["unit"]
        ptype = "INST-VAL"
        df = df[slice(*time_range)]
        return df, unit, ptype

    # methods below if geolocation data is available
    def get_tooltips(self):
        return [
            ("id", "@id"),
            ("variable", "@variable"),
            ("unit", "@unit"),
        ]

    def get_map_color_category(self):
        return "variable"

    def get_map_color_columns(self):
        """return the columns that can be used to color the map"""
        return ["variable"]

    def get_map_marker_columns(self):
        """return the columns that can be used to color the map"""
        return ["variable"]


class DSM2FlowlineMap:

    def __init__(self, shapefile, hydro_echo_file, hydro_echo_file_base=None):
        self.shapefile = shapefile
        self.hydro_echo_file = hydro_echo_file
        self.hydro_echo_file_base = hydro_echo_file_base
        self.dsm2_chans = load_dsm2_flowline_shapefile(self.shapefile)
        self.dsm2_chans.geometry = self.dsm2_chans.geometry.buffer(
            250, cap_style=1, join_style=1
        )
        self.tables = load_echo_file(self.hydro_echo_file)
        if self.hydro_echo_file_base:
            self.tables_base = load_echo_file(self.hydro_echo_file_base)
            # assumption that there is a match on the index of the tables
            for column in ["MANNING", "LENGTH", "DISPERSION"]:
                self.tables["CHANNEL"].loc[:, column] = (
                    self.tables["CHANNEL"].loc[:, column]
                    - self.tables_base["CHANNEL"].loc[:, column]
                )
        self.dsm2_chans_joined = self._join_channels_info_with_shapefile(
            self.dsm2_chans, self.tables
        )
        self.map = hv.element.tiles.CartoLight().opts(width=800, height=600, alpha=0.5)

    def _join_channels_info_with_shapefile(self, dsm2_chans, tables):
        return dsm2_chans.merge(tables["CHANNEL"], right_on="CHAN_NO", left_on="id")

    def show_map_colored_by_length_matplotlib(self):
        return self.dsm2_chans.plot(figsize=(10, 10), column="length_ft", legend=True)

    def show_map_colored_by_mannings_matplotlib(self):
        return self.dsm2_chans_joined.plot(
            figsize=(10, 10), column="MANNING", legend=True
        )

    def show_map_colored_by_column(self, column_name="MANNING"):
        titlestr = column_name
        cmap = cc.b_rainbow_bgyrm_35_85_c71
        if self.hydro_echo_file_base:
            titlestr = titlestr + " Difference from base"
            cmap = cc.b_diverging_bwr_20_95_c54
            # make diffs range centered on 0 difference
            amin = abs(self.dsm2_chans_joined[column_name].min())
            amax = abs(self.dsm2_chans_joined[column_name].max())
            val = max(amin, amax)
            clim = (-val, val)

        plot = self.dsm2_chans_joined.hvplot(
            c=column_name,
            hover_cols=["CHAN_NO", column_name, "UPNODE", "DOWNNODE"],
            title=titlestr,
        ).opts(
            opts.Polygons(
                color_index=column_name, colorbar=True, line_alpha=0, cmap=cmap
            )
        )
        if self.hydro_echo_file_base:
            plot = plot.opts(clim=clim)
        return self.map * plot

    def show_map_colored_by_manning(self):
        return self.show_map_colored_by_column("MANNING")

    def show_map_colored_by_dispersion(self):
        return self.show_map_colored_by_column("DISPERSION")

    def show_map_colored_by_length(self):
        return self.show_map_colored_by_column("LENGTH")


class DSM2GraphNetworkMap(param.Parameterized):
    selected = param.List(default=[0], doc="Selected node indices to display in plot")
    date_range = param.DateRange()  # filter by date range
    godin = param.Boolean()  # godin filter and display
    percent_ratios = param.Boolean()  # show percent ratios instead of total flows

    def __init__(self, node_shapefile, hydro_echo_file, **kwargs):
        super().__init__(**kwargs)

        nodes = load_dsm2_node_shapefile(node_shapefile)
        nodes["x"] = nodes.geometry.x
        nodes["y"] = nodes.geometry.y
        node_map = to_node_tuple_map(nodes)

        self.study = DSM2Study(hydro_echo_file)
        stime, etime = self.study.get_runtime()
        # tuple(map(pd.Timestamp,time_window.split('-')))
        self.param.set_param("date_range", (etime - pd.Timedelta("10 days"), etime))
        # self.param.set_default('date_range', (stime, etime)) # need to set bounds

        # should work but doesn't yet
        tiled_network = hv.element.tiles.CartoLight() * hv.Graph.from_networkx(
            self.study.gc, node_map
        ).opts(
            opts.Graph(
                directed=True,
                arrowhead_length=0.001,
                labelled=["index"],
                node_alpha=0.5,
                node_size=10,
            )
        )

        selector = hv.streams.Selection1D(source=tiled_network.Graph.I.nodes)
        selector.add_subscriber(self.set_selected)

        self.nodes = nodes
        self.tiled_network = tiled_network
        # this second part of overlay needed only because of issue.
        # see https://discourse.holoviz.org/t/selection-on-graph-nodes-doesnt-work/3437
        self.map_pane = self.tiled_network * (
            self.tiled_network.Graph.I.nodes.opts(alpha=0)
        )

    def set_selected(self, index):
        if index is None or len(index) == 0:
            pass  # keep the previous selections
        else:
            self.selected = index

    def display_node_map(self):
        return hv.element.tiles.CartoLight() * self.nodes.hvplot()

    def _date_range_to_twstr(self):
        return "-".join(map(lambda x: x.strftime("%d%b%Y %H%M"), self.date_range))

    @param.depends("selected", "date_range", "percent_ratios")
    def show_sankey(self):
        nodeid = int(
            self.tiled_network.Graph.I.nodes.data.iloc[self.selected].values[0][2]
        )

        inflows, outflows = self.study.get_inflows_outflows(
            nodeid, self._date_range_to_twstr()
        )
        mean_inflows = [df.mean() for df in inflows]
        mean_outflows = [df.mean() for df in outflows]
        if self.percent_ratios:
            total_inflows = sum([f.values[0] for f in mean_inflows])
            total_outflows = sum([f.values[0] for f in mean_outflows])
            mean_inflows = [df / total_inflows * 100 for df in mean_inflows]
            mean_outflows = [df / total_outflows * 100 for df in mean_outflows]
        inlist = [[x.index[0], str(nodeid), x[0]] for x in mean_inflows]
        outlist = [[str(nodeid), x.index[0], x[0]] for x in mean_outflows]
        edges = pd.DataFrame(inlist + outlist, columns=["from", "to", "value"])
        sankey = hv.Sankey(edges, label=f"Flows in/out of {nodeid}")
        sankey = sankey.opts(
            label_position="left",
            edge_fill_alpha=0.75,
            edge_fill_color="value",
            node_alpha=0.5,
            node_color="index",
            cmap="blues",
            colorbar=True,
        )
        return sankey.opts(frame_width=300, frame_height=300)

    @param.depends("selected", "date_range", "godin")
    def show_ts(self):
        nodeid = int(
            self.tiled_network.Graph.I.nodes.data.iloc[self.selected].values[0][2]
        )
        inflows, outflows = self.study.get_inflows_outflows(
            nodeid, self._date_range_to_twstr()
        )
        if godin:
            inflows = [godin(df) for df in inflows]
            outflows = [godin(df) for df in outflows]
        tsin = [df.hvplot(label=df.columns[0]) for df in inflows]
        tsout = [df.hvplot(label=df.columns[0]) for df in outflows]
        return (
            hv.Overlay(tsin).opts(title="Inflows")
            + hv.Overlay(tsout).opts(title="Outflows")
        ).cols(1)

    def get_panel(self):
        slider = pn.Param(
            self.param.date_range,
            widgets={"date_range": pn.widgets.DatetimeRangePicker},
        )
        godin_box = pn.Param(self.param.godin, widgets={"godin": pn.widgets.Checkbox})
        percent_ratios_box = pn.Param(
            self.param.percent_ratios, widgets={"percent_ratios": pn.widgets.Checkbox}
        )
        self.sankey_pane = pn.Row(self.show_sankey)
        self.ts_pane = pn.Row(self.show_ts)
        return pn.Column(
            pn.Row(
                pn.Column(
                    pn.pane.HoloViews(self.map_pane, linked_axes=False),
                    slider,
                    godin_box,
                    percent_ratios_box,
                ),
                self.sankey_pane,
            ),
            self.ts_pane,
        )


#### functions for cli
def merge_dsm2_channel_info_with_shapefile(*echo_files, channel_shapefile=None):
    channels_table = None
    for file in echo_files:
        if not os.path.isfile(file):
            raise FileNotFoundError(f"File {file} not found")
        tables = load_echo_file(file)
        if "CHANNEL" in tables:
            channels_table = tables["CHANNEL"]
    if channels_table is None:
        raise ValueError("No CHANNEL table found in any of the echo files")
    if channel_shapefile is not None:
        dsm2_chan_lines = load_dsm2_channelline_shapefile(channel_shapefile)
        dsm2_chan_lines = join_channels_info_with_dsm2_channel_line(
            dsm2_chan_lines, {"CHANNEL": channels_table}
        )
    return dsm2_chan_lines


def build_output_plotter(*echo_files, channel_shapefile=None):
    output_channels = {}
    time_range = None
    channels_table = None
    for file in echo_files:
        if not os.path.isfile(file):
            raise FileNotFoundError(f"File {file} not found")
        tables = load_echo_file(file)
        try:
            current_time_range = get_runtime(tables)
        except Exception as exc:
            print("Error getting runtime for file:", file)
            raise exc
        print("Time range:", current_time_range, "for file:", file)
        if time_range is None:
            time_range = current_time_range
        else:
            time_range = (
                min(time_range[0], current_time_range[0]),
                max(time_range[1], current_time_range[1]),
            )
        if "OUTPUT_CHANNEL" in tables:
            output_channel = tables["OUTPUT_CHANNEL"]
            output_dir = os.path.dirname(
                file
            )  # assume that location of echo file is the output directory
            output_channel["FILE"] = output_channel["FILE"].str.replace(
                "./output", output_dir, regex=False
            )
            output_channels[file] = output_channel
        if "CHANNEL" in tables:
            channels_table = tables["CHANNEL"]
    if channels_table is None:
        raise ValueError("No CHANNEL table found in any of the echo files")
    output_channels = pd.concat(output_channels.values())
    output_channels.reset_index(drop=True, inplace=True)
    if channel_shapefile is not None:
        dsm2_chan_lines = load_dsm2_channelline_shapefile(channel_shapefile)
        dsm2_chan_lines = join_channels_info_with_dsm2_channel_line(
            dsm2_chan_lines, {"CHANNEL": channels_table}
        )
        pts = output_channels.apply(
            lambda row: get_location_on_channel_line(
                row["CHAN_NO"], row["DISTANCE"], dsm2_chan_lines
            ),
            axis=1,
            result_type="reduce",
        )
        output_channels = gpd.GeoDataFrame(
            output_channels, geometry=pts, crs={"init": "epsg:26910"}
        )
    # output_channels = output_channels.dropna(subset=["geometry"])
    # convert CHAN_NO to string
    output_channels["CHAN_NO"] = output_channels["CHAN_NO"].astype(str)
    plotter = DSM2DataUIManager(output_channels, time_range=time_range)

    return plotter


from .dvue import dataui
import click


@click.command()
@click.argument("echo_files", nargs=-1)
@click.option(
    "--channel-shapefile",
    help="GeoJSON file for channel centerlines with DSM2 channel information",
)
def show_dsm2_output_ui(echo_files, channel_shapefile=None):
    """
    Show a user interface for viewing DSM2 output data

    The channel centerlines are used with the hydro_echo file to display the output data at the output locations
    CHAN_NO is assumed to be the channel number in the hydro_echo file and the in the channel centerlines file
    DISTANCE is projected in a normalized way to the channel length (LENGTH keyword is converted to 1)

    Parameters
    ----------

    dsm2_echo_files : list of strings atlease one of which should be a echo file containing 'CHANNELS' table (hydro echo file)

    channel_shapefile : GeoJSON file for channel centerlines with DSM2 channel information

    """
    import cartopy.crs as ccrs

    plotter = build_output_plotter(*echo_files, channel_shapefile=channel_shapefile)
    ui = dataui.DataUI(plotter, crs=ccrs.UTM(10))
    ui.create_view(title="DSM2 Output UI").show()


@click.command()
@click.argument("tidefiles", nargs=-1)
@click.option(
    "--channel-file",
    help="GeoJSON file for channel centerlines with DSM2 channel information",
    required=False,
)
def show_dsm2_tidefile_ui(tidefiles, channel_file=None):
    """
    Show a user interface for viewing DSM2 tide files

    Parameters
    ----------

    tidefiles : list of strings atlease one of which should be a tide file
    --channel-file : GeoJSON file for channel centerlines with DSM2 channel information

    """
    import cartopy.crs as ccrs

    channels = None
    if channel_file is not None:
        channels = gpd.read_file(channel_file)

    tidefile_manager = DSM2TidefileUIManager(tidefiles, channels=channels)
    ui = dataui.DataUI(
        tidefile_manager, crs=ccrs.epsg("26910"), station_id_column="geoid"
    )
    ui.create_view(title="DSM2 Tidefile UI").show()


@click.command()
@click.argument("tidefile", type=str)
def show_dsm2_tidefile_xsect_ui(tidefile):
    """
    Show a user interface for viewing cross-sections from DSM2 tide files

    Parameters
    ----------

    tidefile : string path to a DSM2 tide file (HDF5 format)
    """
    import panel as pn

    # Extension is needed for serving the panel app
    pn.extension()

    # Create the UI manager
    xsect_manager = DSM2TidefileXsectUIManager(tidefile)

    # Create the panel and serve it
    panel = xsect_manager.get_panel()
    panel.servable(title="DSM2 Tidefile Cross-Section UI")

    # Show the panel
    panel.show()
