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

def heatmap_for_alt_run(df, title, alt_column, base_column, base_diff_type='diff-abs'):
    """

    heatmap by selecting the alternative run (for multi indexed data frame this could result in multiple columns)
    if base_column is specified, uses that to do an absolute or percent diff calculation
    if base_diff_type is abs then subtract absolute value otherwise calculate percent diffs
    """
    df = df[(base_column, alt_column)]
    if base_column is not None:
        if base_diff_type == 'diff-abs':
            # df = df.sub(df[base_column],axis=0)
            df = abs(df).sub(abs(df[base_column]))
        else:
            df = df.sub(df[base_column],axis=0).div(df[base_column],axis=0)*100
    df = df[alt_column]
    mm = max(abs(df.max().max()),abs(df.min().min()))
    heatmap = df.hvplot.heatmap(title=title+' Diff-Abs',
                                cmap='RdBu',
                                #cnorm='eq_hist',
                                grid=True,
                                xaxis='top',
                                clim=(mm,-mm),
                                rot=0).opts(margin=10)
    return heatmap*(hv.Labels(heatmap).opts(text_color='black', text_font_size='8pt'))

def heatmap_for_metric(df, metric, title, base_column=None, base_diff_type='abs'):
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
    if base_column is not None and base_diff_type != 'abs':
        mm = min(mm,50) # for percent more than 50% difference is too much ...   
    heatmap = df.hvplot.heatmap(title=title+' Metric: '+metric,
                                cmap='RdBu',
                                #cnorm='eq_hist',
                                grid=True,
                                xaxis='top',
                                clim=(mm,-mm),
                                rot=0).opts(margin=10)
    return heatmap*(hv.Labels(heatmap).opts(text_color='black', text_font_size='8pt'))

folder_name = 'run22'
run_name = 'v8_3_run22'
plots_folder = 'X:/Share/DSM2/full_calibration_8_3/delta/dsm2v8.3/studies/' + folder_name + '/postprocessing/plots/'
location_info_folder = 'X:/Share/DSM2/full_calibration_8_3/delta/dsm2v8.3/postprocessing/location_info/'

def format_location(x):
    parts = x.split("'")
    return parts[1]+'/'+parts[3]


def main():
    station_order_file = 'd:/documents/calibrationHeatMapStationOrderCombined.csv'

    location_files_dict = {'Flow': location_info_folder + 'calibration_flow_stations.csv',
        'Stage': location_info_folder + 'calibration_stage_stations.csv',
        'EC': location_info_folder + 'calibration_ec_stations.csv'
    }
    calib_metric_csv_filenames_dict = {'Flow':plots_folder + '1_summary_statistics_all_FLOW.csv',
        'Stage': plots_folder + '1_summary_statistics_all_STAGE.csv',
        'EC': plots_folder + '1_summary_statistics_all_EC.csv'
    }

    base_run_name = 'v8_2_1'
    alt_run_name_list = ['v8_2_0', run_name]

    dfp_dict = {}
    heatmap_dict = {}
    for constituent in calib_metric_csv_filenames_dict:
        l=location_files_dict[constituent]
        f = calib_metric_csv_filenames_dict[constituent]
        df = read_summary_stats(f)
        df.to_csv('e:/temp/'+constituent+'_test_df.csv')
        dfp_dict.update({constituent: df})
        # heatmap = heatmap(df, metric, title, base_column=None, base_diff_type='abs')
        # heatmap_for_alt_run(df, title, alt_column, base_column, base_diff_type='diff-abs'):
        heatmap = heatmap_for_alt_run(df, 'Run ' + run_name +'- v8_2_1', run_name, 'v8_2_1', 'diff-abs')
        heatmap_dict.update({constituent: heatmap})

        # create_heatmaps(constituent, l, f, base_run_name, alt_run_name_list, station_order_file)

if __name__ == "__main__":
    main()