import numpy as np
import pandas as pd

import holoviews as hv
import hvplot.pandas

def read_summary_stats(fname):
    '''

    Reads the 1_summary*csv file format and extracts the name from the Location column to us as index.add()

    The table is pivoted with the DSM2 Run column. 

    The resulting pivoted data frame has nested columns where the first level is the metric e.g. NRMSE and the second level is the study

    Example
    -------

    So to view the metrics simply ask for the column NRMSE
    >>> df = read_summary_stats('1_summary_FLOW.csv')
    >>> df['NRMSE']
    '''
    df = pd.read_csv(fname)
    dfp = df.pivot('Location', columns='DSM2 Run')
    names = dfp.index.to_series().str.split('name=', expand=True).iloc[:, 1].str.split(
        ',', expand=True).iloc[:, 0].str.replace("'", "").values
    dfp['names'] = names
    return dfp.set_index('names')

def heatmap(df, metric, title, base_column=None, base_diff_type='abs'):
    """

    heatmap by selecting the column metric (for multi indexed data frame this could result in multiple columns)
    if base_column is specified, uses that to do an absolute or percent diff calculation
    if base_diff_type is abs then subtract absolute value otherwise calculate percent diffs
    """
    df = df[metric]
    if base_column is not None:
        if base_diff_type == 'abs':
            df = df.sub(df[base_column],axis=0)
        else:
            df = df.sub(df[base_column],axis=0).div(df[base_column],axis=0)*100
    mm = max(abs(df.max().max()),abs(df.min().min()))
    heatmap = df.hvplot.heatmap(title=title+' Metric: '+metric,
                                cmap='RdBu',
                                #cnorm='eq_hist',
                                grid=True,
                                xaxis='top',
                                clim=(mm,-mm),
                                rot=0).opts(margin=10)
    return heatmap*(hv.Labels(heatmap).opts(text_color='black', text_font_size='8pt'))