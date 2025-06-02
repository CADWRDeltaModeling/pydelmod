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
            dataui.set_progress(-1)  # Start indeterminate progress

            # Check if there's a selection
            if not dataui.display_table.selection or len(dataui.display_table.selection) == 0:
                if pn.state.notifications is not None:
                    pn.state.notifications.warning('Please select at least one row from the table.', duration=3000)
                logger.warning('No rows selected for plotting')
                return

            # Get selected data
            dfselected = dataui.display_table.value.iloc[dataui.display_table.selection]

            # Show 20% progress
            dataui.set_progress(20)

            plot_panel = dataui._dataui_manager.create_panel(dfselected)

            # Show 90% progress
            dataui.set_progress(90)

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
                    pn.Tabs((str(dataui._tab_count), plot_panel), closable=True, dynamic=True)
                ]

            # Complete the progress
            dataui.set_progress(100)
        except Exception as e:
            stack_str = full_stack()
            logger.error(stack_str)
            dataui._display_panel.objects = [pn.pane.Markdown("```"+stack_str+"```")]
            # Handle the case where notifications might be None
            if pn.state.notifications is not None:
                pn.state.notifications.error(
                    "Error updating plots: " + str(stack_str), duration=0
                )
            else:
                # Log error when notifications is not available
                logger.error(f"Could not display notification: {str(stack_str)}")
        finally:
            dataui._display_panel.loading = False
            # Hide progress after a short delay to show completion
            import asyncio

            # Hide the progress bar immediately when no selection
            if not dataui.display_table.selection or len(dataui.display_table.selection) == 0:
                dataui.hide_progress()
            else:
                pn.state.curdoc.add_next_tick_callback(
                    lambda: asyncio.create_task(self._hide_progress_after_delay(dataui))
                )

    async def _hide_progress_after_delay(self, dataui):
        """Hide the progress bar after a short delay to show completion"""
        import asyncio

        await asyncio.sleep(0.5)
        dataui.hide_progress()


class DownloadDataAction:
    def callback(self, event, dataui):
        dataui._display_panel.loading = True
        try:
            # Show indeterminate progress initially
            dataui.set_progress(-1)

            # Check if there's a selection
            if not dataui.display_table.selection or len(dataui.display_table.selection) == 0:
                if pn.state.notifications is not None:
                    pn.state.notifications.warning('Please select at least one row from the table.', duration=3000)
                logger.warning('No rows selected for download')
                return None

            dfselected = dataui.display_table.value.iloc[dataui.display_table.selection]

            # Update progress to 30%
            dataui.set_progress(30)

            dfdata = pd.concat(
                [df for df in dataui._dataui_manager.get_data(dfselected)], axis=1
            )

            # Update progress to 70%
            dataui.set_progress(70)

            sio = StringIO()
            dfdata.to_csv(sio)
            sio.seek(0)

            # Indicate completion
            dataui.set_progress(100)

            return sio
        except Exception as e:
            logger.error(f"Error downloading data: {e}")
            if pn.state.notifications is not None:
                pn.state.notifications.error(
                    "Error downloading data: " + str(e), duration=0
                )
            return None
        finally:
            dataui._display_panel.loading = False
            # Hide the progress bar if we returned early due to no selection
            if not dataui.display_table.selection or len(dataui.display_table.selection) == 0:
                dataui.hide_progress()
            # For regular cases, the progress bar will be hidden when the download is complete


class DownloadDataCatalogAction:
    def callback(self, event, dataui):
        """Callback to download the currently displayed catalog as a CSV file."""
        dataui._display_panel.loading = True
        try:
            # Show indeterminate progress initially
            dataui.set_progress(-1)

            df = dataui._dataui_manager.get_data_catalog()

            # Update progress to 50%
            dataui.set_progress(50)

            sio = StringIO()
            df.to_csv(sio, index=False)
            sio.seek(0)

            # Indicate completion
            dataui.set_progress(100)

            return sio
        except Exception as e:
            logger.error(f"Error downloading catalog: {e}")
            if pn.state.notifications is not None:
                pn.state.notifications.error("Failed to download catalog")
            return None
        finally:
            dataui._display_panel.loading = False
            # We don't hide the progress bar here as the download might still be in progress


class PermalinkAction:
    def callback(self, event, dataui):
        # Implement permalink action callback here
        pass
