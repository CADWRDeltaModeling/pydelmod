from asyncio.proactor_events import BaseProactorEventLoop
from itertools import dropwhile
import numpy as np
import pandas as pd

import holoviews as hv
import hvplot.pandas
import panel as pn

# yes, I know this is a bad way to do this. Someone should fix this so this information is read from the location files
central=['RSAC101','RVB','SRV','STI','RSMKL008','MOK','SAL','RSAN032','TSL','SLTRM004','EMM','RSAC092',
        'PRI','RSAN037','FAL','JER','RSAN018','HOL','DSJ','SLDUT007','HLT','TRN','RSL','SLRCK005','OBI',
        'ROLD024','BAC','RRI','RSAN058','OH4','ROLD034']
north= ['FPT','RSAC155','SUT','HWB','SSS','BKS','SLBAR002','RSAC128','RSAC128-RSAC123','SDC','CHDCC000','DLC',
        'SDC-GES','RSAC123','GES','GSS','Georg_SL']
south=['SJG','RSAN063','CHVCT000','VCU','BDT','RSAN072','CHSWP003','CLC','UNI','OLD_MID','GCT','CHGRL009','CHDMC006',
       'DMC','OH1','ROLD074','OAD','ROLD047','ROLD059','OLD','MSD','RSAN087','VER','RSAN112']
west=['BDL','SLMZU011','SNC','SLCBN002','VOL','SLSUS012','NSL','SLMZU025','CLL','RSAC081','PCT','RSAC064','RSAC075',
      'MAL','MRZ','RSAC054','ANH','RSAN007','ANC']

def format_location(x):
    parts = x.split("'")
    return parts[1]+'/'+parts[3]

def read_summary_stats(fname, station_order_df):
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
    # # This should sort the rows using station_order_df, but it doesn't work with the pivot table for some reason.
    # df['Location'] = df.apply(lambda x: format_location(x['Location']), axis=1)
    # # add index to use for sorting by region, northing, easting
    # df = df.merge(station_order_df, on=["Location"], how='left')
    # df = df.sort_values('station_index')
    # df=df[['Region']].reset_index().drop_duplicates()

    dfp = df.pivot('Location', columns='DSM2 Run')

    #original
    # names = dfp.index.to_series().str.split('name=', expand=True).iloc[:, 1].str.split(
    #     ',', expand=True).iloc[:, 0].str.replace("'", "").values
    names = dfp.index.to_series().str.split('name=', expand=True).iloc[:,1].str.split(',',expand=True).iloc[:, 0].str.replace("'", "").values
    bparts = dfp.index.to_series().str.split('bpart=', expand=True).iloc[:,1].str.split(',',expand=True).iloc[:, 0].str.replace("'", "").values


    dfp['names'] = names
    dfp['Location']=names+'/'+bparts
    dfp = dfp.set_index('Location')
    station_order_df = station_order_df.set_index('Location')
    dfp_ordered = dfp.loc[station_order_df.index.intersection(dfp.index),:]
    dfp_ordered = dfp_ordered.set_index('names')
    return dfp_ordered
    # return dfp

def heatmap_for_metric(df, metric, region, vartype, title, base_column=None, base_diff_type='abs-diff'):
    """
    heatmap by selecting the column metric (for multi indexed data frame this could result in multiple columns)
    if base_column is specified, uses that to do an absolute or percent diff calculation
    if base_diff_type is abs then subtract absolute value otherwise calculate percent diffs
    """
    df = df[metric]
    if base_column is not None:
        if base_diff_type == 'abs':
            df = df.sub(df[base_column],axis=0)
        elif base_diff_type == 'abs-diff':
            df = abs(df).sub(abs(df[base_column]),axis=0)
        else:
            df = df.sub(df[base_column],axis=0).div(df[base_column],axis=0)*100
    else:
        print('calib_heatmap.heatmap_for_metric: base_column not specified, not subtracting.')

    mm = max(abs(df.max().max()),abs(df.min().min()))
    if base_column is not None and base_diff_type != 'abs':
        mm = min(mm,50) # for percent more than 50% difference is too much ...   
    
    # drop columns we don't want to display
    drop_col_list = ['v8_2_0', 'v8_2_1']
    drop_col_list.append(base_column)
    # df.drop(drop_col_list, inplace=True, axis=1)
    for c in drop_col_list:
        try:
            df.drop(c, inplace=True, axis=1)
        except:
            print('exception caught in calib_heatmap.read_summary_stats while trying to drop column '+c)
        

    heatmap = df.hvplot.heatmap(title=title+' Metric: '+metric,
                                cmap='RdBu',
                                #cnorm='eq_hist',
                                grid=True,
                                xaxis='top',
                                clim=(mm,-mm),
                                rot=0).opts(margin=10)
    return heatmap*(hv.Labels(heatmap).opts(text_color='black', text_font_size='8pt'))


def create_save_heatmaps(calib_metric_csv_filenames_dict, station_order_file, base_run_name, run_name, metrics_list, \
    heatmap_width = 800, heatmap_height=1000,process_vartype_dict=None, base_diff_type='abs-diff'):
    station_order_df = pd.read_csv(station_order_file)

    for vartype in calib_metric_csv_filenames_dict:
        f = calib_metric_csv_filenames_dict[vartype]
        df = read_summary_stats(f, station_order_df)

        title = 'Run ' + run_name +'- '+base_run_name
        heatmap_list = []
        if process_vartype_dict is None or vartype.upper() in process_vartype_dict:
            for metric in metrics_list:
                df_central = df[df.index.isin(central)]
                df_north = df[df.index.isin(north)]
                df_south = df[df.index.isin(south)]
                df_west = df[df.index.isin(west)]
                len_central = len(df_central.index)
                len_north = len(df_north.index)
                len_south = len(df_south.index)
                len_west = len(df_west.index)

                h_central = heatmap_for_metric(df_central, metric, 'Central Delta', vartype, 'Central Delta ' + vartype + \
                    ' summary status from run '+run_name +' :: ', base_column=base_run_name, base_diff_type=base_diff_type).opts(\
                        width=heatmap_width, height=heatmap_height).opts(invert_yaxis=True)
                h_north = heatmap_for_metric(df_north, metric, 'North Delta', vartype, 'North Delta ' + vartype + \
                    ' summary status from run '+run_name +' :: ', base_column=base_run_name, base_diff_type=base_diff_type).opts(\
                        width=heatmap_width, height=heatmap_height).opts(invert_yaxis=True)
                h_south = heatmap_for_metric(df_south, metric, 'South Delta', vartype, 'South Delta ' + vartype + \
                    ' summary status from run '+run_name +' :: ', base_column=base_run_name, base_diff_type=base_diff_type).opts(\
                        width=heatmap_width, height=heatmap_height).opts(invert_yaxis=True)
                h_west = heatmap_for_metric(df_west, metric, 'Western Delta', vartype, 'Western Delta ' + vartype + \
                    ' summary status from run '+run_name +' :: ', base_column=base_run_name, base_diff_type=base_diff_type).opts(\
                        width=heatmap_width, height=heatmap_height).opts(invert_yaxis=True)
                h_all = heatmap_for_metric(df, metric, 'All Regions', vartype, vartype + ' summary stats from run '+ \
                    run_name + ':: ', base_column=base_run_name, base_diff_type=base_diff_type).opts(width=heatmap_width, height=heatmap_height).opts(\
                        invert_yaxis=True)
                heatmap_layout = (h_north+h_central+h_south+h_west).opts(shared_axes=False)
                heatmap_column = pn.Column()
                heatmap_column.append(heatmap_layout)
                heatmap_column.save(f'plots/HeatMaps/heatmap_{vartype}_{metric}.png')
                
                heatmap_all_column = pn.Column()
                heatmap_all_column.append(h_all)
                heatmap_all_column.save(f'plots/HeatMaps/heatmap_all_{vartype}_{metric}.png')

def main():
    folder_name = 'run22'
    run_name = 'v8_3_run22'
    cal_folder = 'X:/Share/DSM2/full_calibration_8_3/delta/dsm2v8.3/'
    plots_folder = cal_folder + 'studies/' + folder_name + '/postprocessing/plots/'
    location_info_folder = cal_folder + 'postprocessing/location_info/'
    station_order_file = location_info_folder + 'calibrationHeatMapStationOrderCombined.csv'
    calib_metric_csv_filenames_dict = {'Flow':plots_folder + '1_summary_statistics_all_FLOW.csv',
        'Stage': plots_folder + '1_summary_statistics_all_STAGE.csv',
        'EC': plots_folder + '1_summary_statistics_all_EC.csv'
    }

    base_run_name = 'v8_2_1'
    metrics_list = ['NRMSE', 'NMSE']
    create_save_heatmaps(calib_metric_csv_filenames_dict, station_order_file, base_run_name, run_name, metrics_list)

if __name__ == "__main__":
    main()