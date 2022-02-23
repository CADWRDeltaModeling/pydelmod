# User interface components for DSM2 related information
import os
from functools import lru_cache
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
# our imports
import pyhecdss
import pydsm
from pydsm.input import parser
from pydsm import hydroh5
from vtools.functions.filter import godin
# viz imports
import geoviews as gv
import hvplot.pandas
import holoviews as hv
from holoviews import opts
hv.extension('bokeh')
import colorcet as cc
#
import param
import panel as pn
pn.extension()


def load_dsm2_channelline_shapefile(channel_shapefile):
    return gpd.read_file(channel_shapefile).to_crs(epsg=3857)

def join_channels_info_with_dsm2_channel_line(dsm2_chan_lines, tables):
    return dsm2_chan_lines.merge(tables['CHANNEL'], right_on='CHAN_NO', left_on='id')

def load_echo_file(fname):
    with open(fname, 'r') as file:
        df = parser.parse(file.read())
    return df

def load_dsm2_flowline_shapefile(shapefile):
    dsm2_chans = gpd.read_file(shapefile).to_crs(epsg=3857)
    # dsm2_chans.geometry=dsm2_chans.geometry.simplify(tolerance=50)
    dsm2_chans.geometry = dsm2_chans.geometry.buffer(250, cap_style=1, join_style=1)
    return dsm2_chans

def join_channels_info_with_shapefile(dsm2_chans, tables):
    return dsm2_chans.merge(tables['CHANNEL'], right_on='CHAN_NO', left_on='id')

def get_location_on_channel_line(channel_id, distance, dsm2_chan_lines):
    chan = dsm2_chan_lines[dsm2_chan_lines.CHAN_NO == channel_id]
    # chan_line = chan.boundary # chan is from a polygon
    try:
        pt = chan.interpolate(distance/chan.LENGTH, normalized=True)
    except: # if not a number always default to assuming its length
        pt = chan.interpolate(1, normalized=True)
    # chan.hvplot()*gpd.GeoDataFrame(geometry=pt).hvplot() # to check plot of point and line
    return pt

def get_runtime(tables):
    scalars = tables['SCALAR']
    rs=scalars[scalars['NAME'].str.contains('run')]
    tmap = dict(zip(rs['NAME'],rs['VALUE']))
    stime = tmap['run_start_date']+' '+tmap['run_start_time']
    etime = tmap['run_end_date']+' '+tmap['run_end_time']
    return pd.to_datetime(stime), pd.to_datetime(etime)

def build_output_plotter(channel_shapefile, hydro_echo_file, variable='flow'):
    hydro_tables = load_echo_file(hydro_echo_file)
    dsm2_chan_lines = load_dsm2_channelline_shapefile(channel_shapefile)
    dsm2_chan_lines = join_channels_info_with_dsm2_channel_line(dsm2_chan_lines, hydro_tables)
    output_channels=hydro_tables['OUTPUT_CHANNEL']
    output_dir = os.path.dirname(hydro_echo_file)
    output_channels['FILE']=output_channels['FILE'].str.replace('./output',output_dir,regex=False)
    pts = output_channels.apply(lambda row: get_location_on_channel_line(row['CHAN_NO'],row['DISTANCE'], dsm2_chan_lines).values[0], 
                                axis=1, result_type='reduce')
    output_channels = gpd.GeoDataFrame(output_channels, geometry=pts, crs = {'init': 'epsg:26910'})
    time_range = get_runtime(hydro_tables)
    plotter = DSM2StationPlotter(output_channels[output_channels.VARIABLE==variable],time_range)
    return plotter
class DSM2StationPlotter(param.Parameterized):
    """Plots all data for single selected station

    """
    selected = param.List(
        default=[0], doc='Selected node indices to display in plot')
    date_range = param.DateRange() # filter by date range
    godin = param.Boolean() # godin filter and display
    
    def __init__(self, stations, time_range, **kwargs):
        super().__init__(**kwargs)
        self.date_range = time_range
        self.godin = False
        self.stations = stations
        self.points_map = self.stations.hvplot.points('easting', 'northing',
                                                       geo=True, tiles='CartoLight', crs='EPSG:3857',
                                                       project=True,
                                                       frame_height=400, frame_width=300,
                                                       fill_alpha=0.9, line_alpha=0.4,
                                                       hover_cols=['CHAN_NO', 'NAME', 'VARIABLE'])
        self.points_map = self.points_map.opts(opts.Points(tools=['tap', 'hover'], size=5,
                                                           nonselection_color='red', nonselection_alpha=0.3,
                                                           active_tools=['wheel_zoom']))
        self.map_pane = pn.Row(self.points_map)
        # create a selection and add it to a dynamic map calling back show_ts
        self.select_stream = hv.streams.Selection1D(
            source=self.points_map, index=[0])
        self.select_stream.add_subscriber(self.set_selected)
        self.meta_pane = pn.Row()
        self.ts_pane = pn.Row()

    def set_selected(self, index):
        if index is None or len(index) == 0:
            pass  # keep the previous selections
        else:
            self.selected = index

    @lru_cache(maxsize=25)
    def _get_all_sensor_data(self, name, var, intvl, file):
        # get data for location
        return [next(pyhecdss.get_ts(file, f'//{name}/{var}//{intvl}//'))[0]]

    def _get_selected_data_row(self):
        index = self.selected
        if index is None or len(index) == 0:
            index = self.selected
        # Use only the first index in the array
        first_index = index[0]
        return self.stations.iloc[first_index, :]

    ##@@ callback to get data for index
    def get_selected_data(self):
        dfselected = self._get_selected_data_row()
        # NAME	CHAN_NO	DISTANCE	VARIABLE	INTERVAL	PERIOD_OP	FILE
        stn_name = dfselected['NAME']
        chan_id = dfselected['CHAN_NO']
        dist = dfselected['DISTANCE']
        var = dfselected['VARIABLE']
        intvl = dfselected['INTERVAL']
        file = dfselected['FILE']
        data_array = self._get_all_sensor_data(stn_name, var, intvl, file)
        stn_id = f'{chan_id}-{dist}_{var}'
        return data_array, stn_id, stn_name

    ##@@ callback to display ..
    @param.depends('selected')
    def show_meta(self):
        dfselected = self._get_selected_data_row()
        dfselected = dfselected.drop('geometry') # Points is not serializable to JSON https://github.com/bokeh/bokeh/issues/8423
        self.data_frame = pn.widgets.DataFrame(dfselected.to_frame())
        return self.data_frame

    ##@ callback to display ..
    @param.depends('selected', 'date_range', 'godin')
    def show_ts(self):
        data_array, stn_id, stn_name = self.get_selected_data()
        crv_list = []  # left here for multi curve later
        for data in data_array:
            if self.godin:
                el = hv.Curve(godin(data),label='godin filtered')
            else:
                el = hv.Curve(data)
            el = el.opts(title=f'Station: {stn_id} :: {stn_name}',  xlim=self.date_range, ylabel='Time')
            crv_list.append(el)
        layout = hv.Layout(crv_list).cols(1).opts(opts.Curve(width=900))
        return layout.opts(title=f'{stn_id}: {stn_name}')

    def get_panel(self):
        slider = pn.Param(self.param.date_range, widgets={
                          'date_range': pn.widgets.DatetimeRangePicker})
        godin_box = pn.Param(self.param.godin, widgets={'godin': pn.widgets.Checkbox})
        self.meta_pane = pn.Row(self.show_meta)
        self.ts_pane = pn.Row(self.show_ts)
        return pn.Column(pn.Row(pn.Column(self.map_pane, slider, godin_box), self.meta_pane), self.ts_pane)

class DSM2FlowlineMap:

    def __init__(self, shapefile, hydro_echo_file):
        self.shapefile = shapefile
        self.hydro_echo_file = hydro_echo_file
        self.dsm2_chans = load_dsm2_flowline_shapefile(self.shapefile)
        self.dsm2_chans.geometry = self.dsm2_chans.geometry.buffer(250, cap_style=1, join_style=1)
        self.tables = load_echo_file(self.hydro_echo_file)
        self.dsm2_chans_joined = self._join_channels_info_with_shapefile(
            self.dsm2_chans, self.tables)
        self.map = hv.element.tiles.CartoLight().opts(width=800, height=600, alpha=0.5)

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
