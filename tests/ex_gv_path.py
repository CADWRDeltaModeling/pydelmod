# %%
import geoviews as gv
import geoviews.tile_sources as gvts
from shapely.geometry import LineString
import pandas as pd
import geopandas as gpd
import param
from holoviews.streams import Selection1D
import panel as pn

pn.extension()
# Enable notebook extension
gv.extension("bokeh")

# Create a sample line string
coords = [
    (-122.4194, 37.7749),  # San Francisco
    (-122.2712, 37.8044),  # Berkeley
    (-122.2729, 37.8716),  # Richmond
]
line = LineString(coords)

# Convert to GeoDataFrame-like structure
df = gpd.GeoDataFrame({"geometry": [line], "name": ["Bay Area Route"], "id": [0]})


# Create a dynamic map with selection capability
class InteractiveMap(param.Parameterized):
    def __init__(self, df, **params):
        super().__init__(**params)
        self.df = df
        # Create selection stream
        self.selection = Selection1D()

    def show_selection(self, index):
        print(index)
        print(self.df.iloc[index])

    def view(self):
        # Create base map
        tiles = gvts.CartoLight

        # Create line string layer
        lines = gv.Path(data=self.df, vdims=["name", "id"]).opts(
            color="blue",
            line_width=3,
            tools=["tap"],
            nonselection_alpha=0.3,
            selection_color="red",
        )
        self.selection.source = lines
        pn.bind(self.show_selection, index=self.selection.param.index)
        # Combine layers
        map_view = tiles * lines

        return map_view.opts(width=800, height=500, title="Interactive Route Map")


# %%

# Create and display the interactive map
interactive_map = InteractiveMap(df)
map_display = interactive_map.view()

# %%
# Display the map
pn.panel(map_display).show()

# %%
