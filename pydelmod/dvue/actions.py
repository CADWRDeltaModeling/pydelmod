import panel as pn

pn.extension()
import pandas as pd
from io import StringIO
import logging
from .utils import full_stack

logger = logging.getLogger(__name__)


class PlotAction:
    def callback(self, event, dataui):
        try:
            dataui._display_panel.loading = True
            dfselected = dataui.display_table.value.iloc[dataui.display_table.selection]
            plot_panel = dataui._dataui_manager.create_panel(dfselected)
            if len(dataui._display_panel.objects) > 0 and isinstance(
                dataui._display_panel.objects[0], pn.Tabs
            ):
                tabs = dataui._display_panel.objects[0]
                dataui._tab_count += 1
                tabs.append((str(dataui._tab_count), plot_panel))
                tabs.active = len(tabs) - 1
            else:
                dataui._tab_count = 0
                dataui._display_panel.objects = [
                    pn.Tabs((str(dataui._tab_count), plot_panel), closable=True)
                ]
        except Exception as e:
            stack_str = full_stack()
            logger.error(stack_str)
            dataui._display_panel.objects = [pn.pane.Markdown(stack_str)]
            pn.state.notifications.error(
                "Error updating plots: " + str(stack_str), duration=0
            )
        finally:
            dataui._display_panel.loading = False


class DownloadDataAction:

    def callback(self, event, dataui):
        dataui._display_panel.loading = True
        try:
            dfselected = dataui.display_table.value.iloc[dataui.display_table.selection]
            dfdata = pd.concat(
                [df for df in dataui._dataui_manager.get_data(dfselected)], axis=1
            )
            sio = StringIO()
            dfdata.to_csv(sio)
            sio.seek(0)
            return sio
        except Exception as e:
            pn.state.notifications.error(
                "Error downloading data: " + str(e), duration=0
            )
        finally:
            dataui._display_panel.loading = False


class DownloadDataCatalogAction:
    def callback(self, event, dataui):
        """Callback to download the currently displayed catalog as a CSV file."""
        dataui._display_panel.loading = True
        try:
            df = dataui._dataui_manager.get_data_catalog()
            sio = StringIO()
            df.to_csv(sio, index=False)
            sio.seek(0)
            return sio
        except Exception as e:
            logger.error(f"Error downloading catalog: {e}")
            pn.state.notifications.error("Failed to download catalog")
        finally:
            dataui._display_panel.loading = False


class PermalinkAction:
    def callback(self, event, dataui):
        # Implement permalink action callback here
        pass
