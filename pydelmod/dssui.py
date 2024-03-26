# %%
# organize imports by category
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")
#
import pandas as pd

# viz and ui
import holoviews as hv
from holoviews import opts

hv.extension("bokeh")
import geoviews as gv

gv.extension("bokeh")
import param
import panel as pn

pn.extension("tabulator", notifications=True, design="native")
#
import pyhecdss as dss
from pydelmod import fullscreen


def get_colors(stations, dfc):
    """
    Create a dictionary with station names and colors
    """
    return hv.Cycle(list(dfc.loc[stations].values.flatten()))


# from stackoverflow.com https://stackoverflow.com/questions/6086976/how-to-get-a-complete-exception-stack-trace-in-python
def full_stack():
    import traceback, sys

    exc = sys.exc_info()[0]
    stack = traceback.extract_stack()[:-1]  # last one would be full_stack()
    if exc is not None:  # i.e. an exception is present
        del stack[-1]  # remove call of full_stack, the printed exception
        # will contain the caught exception caller instead
    trc = "Traceback (most recent call last):\n"
    stackstr = trc + "".join(traceback.format_list(stack))
    if exc is not None:
        stackstr += "  " + traceback.format_exc().lstrip(trc)
    return stackstr


def get_color_dataframe(stations, color_cycle=hv.Cycle()):
    """
    Create a dataframe with station names and colors
    """
    cc = color_cycle.values
    # extend cc to the size of stations
    while len(cc) < len(stations):
        cc = cc + cc
    dfc = pd.DataFrame({"stations": stations, "color": cc[: len(stations)]})
    dfc.set_index("stations", inplace=True)
    return dfc


def get_colors(stations, dfc):
    """
    Create a dictionary with station names and colors
    """
    return hv.Cycle(list(dfc.loc[stations].values.flatten()))


class DSSUI(param.Parameterized):
    """
    Show table of data from DSS file
    Furthermore select the data rows and click on button to display plots for selected rows
    """

    time_range = param.CalendarDateRange(
        default=(datetime.now() - timedelta(days=10), datetime.now()),
        doc="Time window for data. Default is last 10 days",
    )
    show_legend = param.Boolean(default=True, doc="Show legend")
    legend_position = param.Selector(
        objects=["top_right", "top_left", "bottom_right", "bottom_left"],
        default="top_right",
        doc="Legend position",
    )

    def __init__(self, *dssfiles, **kwargs):
        super().__init__(**kwargs)
        dfcats = []
        dssfh = {}
        dsscats = {}
        for dssfile in dssfiles:
            dssfh[dssfile] = dss.DSSFile(dssfile)
            dfcat = dssfh[dssfile].read_catalog()
            dsscats[dssfile] = self._build_map_pathname_to_catalog(dfcat)
            dfcat = dfcat.drop(columns=["T"])
            dfcat["filename"] = dssfile
            dfcats.append(dfcat)
        self.dssfh = dssfh
        self.dsscats = dsscats
        self.dfcat = pd.concat(dfcats)
        self.dfcat = self.dfcat.drop_duplicates().reset_index(drop=True)
        self.dssfiles = dssfiles
        self.time_range = self.calculate_time_range(self.dfcat)
        self.dfcatpath = self._build_map_pathname_to_catalog(self.dfcat)

    def __del__(self):
        for dssfile in self.dssfiles:
            self.dssfh[dssfile].close()

    def calculate_time_range(self, dfcat):
        """
        Calculate time range from the data catalog
        """
        dftw = dfcat.D.str.split("-", expand=True)
        dftw.columns = ["Tmin", "Tmax"]
        dftw["Tmin"] = pd.to_datetime(dftw["Tmin"])
        dftw["Tmax"] = pd.to_datetime(dftw["Tmax"])
        tmin = dftw["Tmin"].min()
        tmax = dftw["Tmax"].max()
        return tmin, tmax

    def show_data_catalog(self):
        dfs = self.dfcat.iloc[:]  # FIXME: later add filters
        # return a UI with controls to plot and show data
        return self.update_data_table(dfs)

    def update_data_table(self, dfs):
        if not hasattr(self, "display_table"):
            column_width_map = {
                "A": "15%",
                "B": "15%",
                "C": "15%",
                "E": "10%",
                "F": "15%",
                "D": "20%",
            }
            table_filters = {
                "A": {"type": "input", "func": "like", "placeholder": "Enter match"},
                "B": {"type": "input", "func": "like", "placeholder": "Enter match"},
                "C": {"type": "input", "func": "like", "placeholder": "Enter match"},
                "E": {"type": "input", "func": "like", "placeholder": "Enter match"},
                "F": {"type": "input", "func": "like", "placeholder": "Enter match"},
            }
            self.display_table = pn.widgets.Tabulator(
                dfs,
                disabled=True,
                widths=column_width_map,
                show_index=False,
                sizing_mode="stretch_width",
                header_filters=table_filters,
            )

            self.plot_button = pn.widgets.Button(
                name="Plot", button_type="primary", icon="chart-line"
            )
            self.plot_button.on_click(self.update_plots)
            self.plot_panel = pn.panel(
                hv.Div("<h3>Select rows from table and click on button</h3>"),
                sizing_mode="stretch_both",
            )
            gspec = pn.GridStack(
                sizing_mode="stretch_both", allow_resize=True, allow_drag=False
            )  # ,
            gspec[0, 0:5] = pn.Row(
                self.plot_button,
                pn.layout.HSpacer(),
            )
            gspec[1:5, 0:10] = fullscreen.FullScreen(pn.Row(self.display_table))
            gspec[6:15, 0:10] = fullscreen.FullScreen(pn.Row(self.plot_panel))
            self.plots_panel = pn.Row(
                gspec
            )  # fails with object of type 'GridSpec' has no len()

        else:
            self.display_table.value = dfs

        return self.plots_panel

    def update_plots(self, event):
        self.plot_panel.loading = True
        self.plot_panel.object = self.create_plots(event)
        self.plot_panel.loading = False

    def _slice_df(self, df, time_range):
        sdf = df.loc[slice(*time_range), :]
        if sdf.empty:
            return pd.DataFrame(
                columns=["value"],
                index=pd.date_range(*time_range, freq="D"),
                dtype=float,
            )
        else:
            return sdf

    def _build_pathname(self, r):
        return f'/{r["A"]}/{r["B"]}/{r["C"]}//{r["E"]}/{r["F"]}/'

    def _build_map_pathname_to_catalog(self, dfcat):
        dfcatpath = dfcat.copy()
        dfcatpath["pathname"] = dfcatpath.apply(self._build_pathname, axis=1)
        return dfcatpath

    def get_data_for_time_range(self, dssfile, r):
        try:
            if r["E"].startswith("IR-"):
                raise ValueError(f"IR- data not supported yet!")
            dssfh = self.dssfh[dssfile]
            dfcatp = self.dsscats[dssfile]
            dfcatp = dfcatp[dfcatp["pathname"] == self._build_pathname(r)]
            pathname = dssfh.get_pathnames(dfcatp)[0]
            df, unit, ptype = dssfh.read_rts(
                pathname,
                self.time_range[0].strftime("%Y-%m-%d"),
                self.time_range[1].strftime("%Y-%m-%d"),
            )
        except Exception as e:
            print(full_stack())
            if pn.state.notifications:
                pn.state.notifications.error(
                    f"Error while fetching data for {dssfile}/{pathname}: {e}"
                )
            df = pd.DataFrame(columns=["value"], dtype=float)
            unit = "X"
            ptype = "INST-VAL"
        df = df[slice(df.first_valid_index(), df.last_valid_index())]
        return df, unit, ptype

    def _append_to_title_map(self, title_map, unit, r):
        value = title_map[unit]
        if r["C"] not in value[0]:
            value[0] += f',{r["C"]}'
        if r["B"] not in value[1]:
            value[1] += f',{r["B"]}'
        if r["A"] not in value[2]:
            value[2] += f',{r["A"]}'
        if r["F"] not in value[3]:
            value[3] += f',{r["F"]}'
        title_map[unit] = value

    def _create_title(self, v):
        title = f"{v[1]} @ {v[2]} ({v[3]}::{v[0]})"
        return title

    def create_plots(self, event):
        df = self.display_table.value.iloc[self.display_table.selection]
        try:
            layout_map = {}
            title_map = {}
            range_map = {}
            station_map = {}  # list of stations for each unit
            stationids = list(
                (df.apply(self._build_pathname, axis=1).astype(str).unique())
            )
            color_df = get_color_dataframe(stationids, hv.Cycle())
            for _, r in df.iterrows():
                data, unit, _ = self.get_data_for_time_range(r["filename"], r)
                crv = self._create_crv(
                    data,
                    f'{r["B"]}/{r["C"]}',
                    f'{r["C"]} ({unit})',
                    f'{r["C"]} @ {r["B"]} ({r["A"]}/{r["F"]})',
                )
                if unit not in layout_map:
                    layout_map[unit] = []
                    title_map[unit] = [
                        r["C"],
                        r["B"],
                        r["A"],
                        r["F"],
                    ]
                    range_map[unit] = None
                    station_map[unit] = []
                layout_map[unit].append(crv)
                station_map[unit].append(self._build_pathname(r))
                self._append_to_title_map(title_map, unit, r)
            if len(layout_map) == 0:
                return hv.Div("<h3>Select rows from table and click on button</h3>")
            else:
                return (
                    hv.Layout(
                        [
                            hv.Overlay(layout_map[k])
                            .opts(
                                opts.Curve(color=get_colors(station_map[k], color_df))
                            )
                            .opts(
                                show_legend=self.show_legend,
                                legend_position=self.legend_position,
                                ylim=(
                                    tuple(range_map[k])
                                    if range_map[k] is not None
                                    else (None, None)
                                ),
                                title=self._create_title(title_map[k]),
                            )
                            for k in layout_map
                        ]
                    )
                    .cols(1)
                    .opts(axiswise=True, sizing_mode="stretch_both")
                )
        except Exception as e:
            stackmsg = full_stack()
            print(stackmsg)
            pn.state.notifications.error(f"Error while fetching data for {e}")
            return hv.Div(f"<h3> Exception while fetching data </h3> <pre>{e}</pre>")

    def _create_crv(self, df, crvlabel, ylabel, title):
        crv = hv.Curve(df.iloc[:, [0]], label=crvlabel).redim(value=crvlabel)
        return crv.opts(
            xlabel="Time",
            ylabel=ylabel,
            title=title,
            responsive=True,
            active_tools=["wheel_zoom"],
            tools=["hover"],
        )

    def get_about_text(self):
        version = "0.1.0"

        # insert app version with date time of last commit and commit id
        version_string = f"DSS UI: {version}"
        about_text = f"""
        ## App version:
        ### {version}

        ## An App to view DSS data using Holoviews and Panel
        """
        return about_text

    def create_about_button(self, template):
        about_btn = pn.widgets.Button(
            name="About App", button_type="primary", icon="info-circle"
        )

        def about_callback(event):
            template.open_modal()

        about_btn.on_click(about_callback)
        return about_btn

    def create_view(self):
        control_widgets = pn.Row(
            pn.Column(
                pn.Param(
                    self.param.time_range,
                    widgets={
                        "time_range": {
                            "widget_type": pn.widgets.DatetimeRangeInput,
                            "format": "%Y-%m-%d %H:%M",
                        }
                    },
                ),
                pn.WidgetBox(
                    self.param.show_legend,
                    self.param.legend_position,
                ),
            ),
        )
        sidebar_view = pn.Column(control_widgets)
        main_view = pn.Column(pn.bind(self.show_data_catalog))

        template = pn.template.VanillaTemplate(
            title="DSS User Interface",
            sidebar=[sidebar_view],
            sidebar_width=450,
            header_color="lightgray",
        )
        template.main.append(main_view)
        # Adding about button
        template.modal.append(self.get_about_text())
        control_widgets[0].append(self.create_about_button(template))
        return template


# %%
import glob
import click


@click.command()
@click.argument("dssfiles", nargs=-1)
def show_dss_ui(dssfiles):
    """
    Show DSS UI for the given DSS files
    """
    ui = DSSUI(*dssfiles)
    ui.create_view().show()
