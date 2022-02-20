# User interface components for DSM2 related information
import colorcet as cc
import panel as pn
from turtle import shape
import pydsm
from pydsm.input import parser
from pydsm import hydroh5
import numpy as np
import pandas as pd
# viz imports
import shapely
import geoviews as gv
import geopandas as gpd
import hvplot.pandas
import holoviews as hv
from holoviews import opts
hv.extension('bokeh')
pn.extension()


class DSM2FlowlineMap:

    def __init__(self, shapefile, hydro_echo_file):
        self.shapefile = shapefile
        self.hydro_echo_file = hydro_echo_file
        self.dsm2_chans = self.load_dsm2_flowline_shapefile(self.shapefile)
        self.tables = self.load_echo_file(self.hydro_echo_file)
        self.dsm2_chans_joined = self._join_channels_info_with_shapefile(
            self.dsm2_chans, self.tables)
        self.map = hv.element.tiles.CartoLight().opts(width=800, height=600, alpha=0.5)

    def load_dsm2_flowline_shapefile(self, shapefile):
        dsm2_chans = gpd.read_file(shapefile).to_crs(epsg=3857)
        # dsm2_chans.geometry=dsm2_chans.geometry.simplify(tolerance=50)
        dsm2_chans.geometry = dsm2_chans.geometry.buffer(250, cap_style=1, join_style=1)
        return dsm2_chans

    def load_echo_file(self, fname):
        with open(fname, 'r') as file:
            df = parser.parse(file.read())
        return df

    def _join_channels_info_with_shapefile(self, dsm2_chans, tables):
        return dsm2_chans.merge(tables['CHANNEL'], right_on='CHAN_NO', left_on='id')

    def show_map_colored_by_length_matplotlib(self):
        return self.dsm2_chans.plot(figsize=(10, 10), column='length_ft', legend=True)

    def show_map_colored_by_mannings_matplotlib(self):
        return self.dsm2_chans_joined.plot(figsize=(10, 10), column='MANNING', legend=True)

    def show_map_colored_by_column(self, column_name='MANNING'):
        return self.map*self.dsm2_chans_joined.hvplot(c=column_name, hover_cols=['CHAN_NO', column_name, 'UPNODE', 'DOWNNODE'],
                                                      title=column_name).opts(opts.Polygons(color_index=column_name, colorbar=True, line_alpha=0, cmap=cc.b_rainbow_bgyrm_35_85_c71))

    def show_map_colored_by_manning(self):
        return self.show_map_colored_by_column('MANNING')

    def show_map_colored_by_dispersion(self):
        return self.show_map_colored_by_column('DISPERSION')

    def show_map_colored_by_length(self):
        return self.show_map_colored_by_column('LENGTH')
