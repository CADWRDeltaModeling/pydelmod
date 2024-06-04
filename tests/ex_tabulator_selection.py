# %%
import pandas as pd
import panel as pn
import logging

#
df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}, index=["x", "y", "z"])
b = pn.widgets.Button(name="Show Selection")
t = pn.widgets.Tabulator(df)
#
logger = logging.getLogger("panel.tabulator_select")

debug_info = pn.widgets.Debugger(
    name="Debugger info level",
    level=logging.INFO,
    sizing_mode="stretch_both",
    logger_names=[
        "panel.tabulator_select"
    ],  # comment this line out to get all panel errors
)


#
def show_selection(event):
    logger.info(f"Selection: {t.selection}")
    logger.info(f"Selected Dataframe: {t.selected_dataframe}")
    logger.info(f"_processed: {t._processed}")
    logger.info(f"current_view: {t.current_view}")


b.on_click(show_selection)

pn.Column(t, b, debug_info).show()

# %%
