# User interface components for DSM2 related information
import panel as pn
import param
import colorcet as cc

# viz imports
import geoviews as gv
import holoviews as hv
from holoviews import opts

hv.extension("bokeh")
import hvplot.pandas

#
import panel as pn

pn.extension()


import pyhecdss as dss
from vtools.functions.filter import cosine_lanczos
from .dsm2study import *
from .tsdataui import TimeSeriesDataUIManager, full_stack

LINE_DASH_MAP = ["solid", "dashed", "dotted", "dotdash", "dashdot"]


class DSM2DataUIManager(TimeSeriesDataUIManager):
    do_tidal_filter = param.Boolean(default=False, doc="Apply tidal filter")

    def __init__(self, output_channels, **kwargs):
        """
        output_channels is a geopandas dataframe with columns:
        NAME  CHAN_NO  DISTANCE  VARIABLE  INTERVAL  PERIOD_OP  FILE
        """
        self.time_range = kwargs.pop("time_range", None)
        self.output_channels = output_channels
        self.display_fileno = False
        unique_files = self.output_channels["FILE"].unique()
        if len(unique_files) > 1:
            output_channels["FILE_NO"] = output_channels["FILE"].apply(
                lambda x: unique_files.tolist().index(x)
            )
            self.display_fileno = True
        self.station_id_column = "NAME"
        super().__init__(**kwargs)

    def get_widgets(self):
        super_widgets = super().get_widgets()
        return pn.Column(super_widgets, self.param.do_tidal_filter)

    # data related methods
    def get_data_catalog(self):
        return self.output_channels

    def get_time_range(self, dfcat):
        return self.time_range

    def get_station_ids(self, df):
        return df["NAME"].unique()

    def get_table_column_width_map(self):
        """only columns to be displayed in the table should be included in the map"""
        column_width_map = {
            "NAME": "15%",
            "CHAN_NO": "10%",
            "DISTANCE": "10%",
            "VARIABLE": "10%",
            "INTERVAL": "5%",
            "PERIOD_OP": "5%",
            "FILE": "25%",
        }
        if self.display_fileno:
            column_width_map["FILE_NO"] = "5%"
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

    def _append_to_title_map(self, title_map, unit, r):
        value = title_map[unit]
        if r["VARIABLE"] not in value[0]:
            value[0] += f',{r["VARIABLE"]}'
        if r["NAME"] not in value[1]:
            value[1] += f',{r["NAME"]}'
        if str(r["CHAN_NO"]) not in value[2]:
            value[2] += f',{r["CHAN_NO"]}'
        if str(r["DISTANCE"]) not in value[3]:
            value[3] += f',{r["DISTANCE"]}'
        title_map[unit] = value

    def _create_title(self, v):
        title = f"{v[1]} @ {v[2]} ({v[3]}::{v[0]})"
        return title

    def _create_crv(self, df, crvlabel, ylabel, title, irreg=False, file_index=None):
        if irreg:
            crv = hv.Scatter(df.iloc[:, [0]], label=crvlabel).redim(value=crvlabel)
        else:
            crv = hv.Curve(df.iloc[:, [0]], label=crvlabel).redim(value=crvlabel)
        if file_index:
            crv = crv.opts(line_dash=LINE_DASH_MAP[file_index % len(LINE_DASH_MAP)])
        return crv.opts(
            xlabel="Time",
            ylabel=ylabel,
            title=title,
            responsive=True,
            active_tools=["wheel_zoom"],
            tools=["hover"],
        )

    def get_data_for_time_range(self, dssfile, r, time_range):
        try:
            dssfile = r["FILE"]
            pathname = f'//{r["NAME"]}/{r["VARIABLE"]}////'
            df, unit, ptype = next(
                dss.get_matching_ts(dssfile, pathname)
            )  # get first matching time series
        except Exception as e:
            print(full_stack())
            if pn.state.notifications:
                pn.state.notifications.error(
                    f"Error while fetching data for {dssfile}/{pathname}: {e}"
                )
            df = pd.DataFrame(columns=["value"], dtype=float)
            unit = "X"
            ptype = "INST-VAL"
        df = df[slice(*time_range)]
        if self.do_tidal_filter:
            df = cosine_lanczos(df, "40h")
        return df, unit, ptype

    def build_station_name(self, r):
        return f'{r["NAME"]}'  # unique within units...

    def create_layout(self, df, time_range):
        layout_map = {}
        title_map = {}
        range_map = {}
        station_map = {}  # list of stations for each unit
        local_unique_files = df["FILE"].unique()
        for _, r in df.iterrows():
            data, unit, _ = self.get_data_for_time_range(r["FILE"], r, time_range)
            if len(local_unique_files) > 0:
                file_index = local_unique_files.tolist().index(r["FILE"])
            else:
                file_index = ""
            file_index_label = (
                str(r["FILE_NO"]) + ":" if len(local_unique_files) > 1 else ""
            )
            crv = self._create_crv(
                data,
                f'{file_index_label}{r["NAME"]}/{r["VARIABLE"]}',
                f'{r["VARIABLE"]} ({unit})',
                f'{r["VARIABLE"]} @ {r["NAME"]} ({r["CHAN_NO"]}/{r["DISTANCE"]})',
                file_index=file_index,
            )
            if unit not in layout_map:
                layout_map[unit] = []
                title_map[unit] = [
                    r["VARIABLE"],
                    r["NAME"],
                    str(r["CHAN_NO"]),
                    str(r["DISTANCE"]),
                ]
                range_map[unit] = None
                station_map[unit] = []
            layout_map[unit].append(crv)
            station_map[unit].append(self.build_station_name(r))
            self._append_to_title_map(title_map, unit, r)
        title_map = {k: self._create_title(v) for k, v in title_map.items()}
        return layout_map, station_map, range_map, title_map

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
            ).values[0],
            axis=1,
            result_type="reduce",
        )
        output_channels = gpd.GeoDataFrame(
            output_channels, geometry=pts, crs={"init": "epsg:26910"}
        )
    # output_channels = output_channels.dropna(subset=["geometry"])
    plotter = DSM2DataUIManager(output_channels, time_range=time_range)

    return plotter


from . import dataui
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
    ui.create_view().show()
