# from matplotlib import figure
import pandas as pd
import numpy as np
import os
from pydsm import postpro
from pyparsing import col
from sklearn import metrics

# import matplotlib.pyplot as plt
# import seaborn as sns


import holoviews as hv
from holoviews import opts
hv.extension('bokeh')

def format_location(x):
    parts = x.split("'")
    return parts[1]+'/'+parts[3]


def create_heatmaps(constituent, location_file, calib_metric_csv_filename, base_run_name, alt_run_name_list):
    '''
    Read a file that looks like this, and create heatmap for selected metrics, with station on y axis, and run num on x axis
    DSM2 Run,Location,Gate Pos,Equation,R Squared,Mean Error,NMean Error,NMSE,NRMSE,NSE,PBIAS,RSR,Mnly Mean Err,Mnly RMSE
    v8_2_0,"Location(name='CHSWP003', bpart='CLC', description='Banks Pumping Plant /Clifton Court Forebay')",all,y=0.89x+18.75,0.8991475178506713,-32.01976618307047,-0.0709186705975315,10.350886930630224,0.1514118957684851,0.8706125853991984,-7.09186705975314,0.3596297374617811,-0.0676412735704089,0.1318403846764428
    v8_2_1,"Location(name='CHSWP003', bpart='CLC', description='Banks Pumping Plant /Clifton Court Forebay')",all,y=0.89x+18.69,0.8991669226060319,-32.04213236012419,-0.070968208109265,10.35237044401123,0.1514227457336705,0.8705940412916084,-7.096820810926483,0.3596555080269754,-0.0676903370676067,0.1318512421929696
    v8_3_run14,"Location(name='CHSWP003', bpart='CLC', description='Banks Pumping Plant /Clifton Court Forebay')",all,y=1.02x+5.10,0.8510049101846094,13.257005336400274,0.0293621505287248,14.932263424235996,0.1818586807448959,0.8133447914610373,2.936215052872494,0.4319461775409954,0.0318920011005336,0.1607514784747907
    v8_3_run15,"Location(name='CHSWP003', bpart='CLC', description='Banks Pumping Plant /Clifton Court Forebay')",all,y=1.02x+5.15,0.8508815748766786,13.36075882813884,0.0295919479501639,14.956651245418035,0.1820071286873817,0.8130399405674242,2.9591947950163826,0.4322987673709555,0.0321358894781639,0.1608655522290293
    v8_3_run16,"Location(name='CHSWP003', bpart='CLC', description='Banks Pumping Plant /Clifton Court Forebay')",all,y=1.02x+5.23,0.8511493585568933,13.173752906541168,0.0291777596866473,14.896213310845,0.181639022558475,0.8137954225034771,2.9177759686647384,0.4314244509255738,0.0317187955489635,0.1606419519184552
    v8_3_run17,"Location(name='CHSWP003', bpart='CLC', description='Banks Pumping Plant /Clifton Court Forebay')",all,y=1.02x+5.23,0.8511578648479327,13.173322549275804,0.0291768065140049,14.89541002699278,0.1816341250164691,0.8138054636546872,2.9176806514005107,0.4314128184066303,0.0317168997748543,0.1606459220617164
    v8_3_run18,"Location(name='CHSWP003', bpart='CLC', description='Banks Pumping Plant /Clifton Court Forebay')",all,y=1.02x+5.24,0.8511566257160627,13.176849406860876,0.0291846179405434,14.895289316078586,0.1816333890425798,0.8138069725566013,2.918461794054359,0.4314110703399064,0.0317230708568137,0.1606541298779595
    v8_3_run19,"Location(name='CHSWP003', bpart='CLC', description='Banks Pumping Plant /Clifton Court Forebay')",all,y=1.02x+5.23,0.8511684731333812,13.175540476127017,0.02918171886792,14.894364555743628,0.1816277506841286,0.8138185321794305,2.918171886792008,0.4313976782523216,0.031717748711075,0.1606643565385912
    v8_3_run_merged_s19_c3_wBase,"Location(name='CHSWP003', bpart='CLC', description='Banks Pumping Plant /Clifton Court Forebay')",all,y=1.01x+6.89,0.8502843335349171,13.467883280015672,0.0298292115101836,14.91825529313119,0.1817733588512759,0.8135198948970217,2.982921151018357,0.4317435231191182,0.0323871582133587,0.1611401305953851
    v8_2_0,"Location(name='CLC', bpart='CLC', description='Banks Pumping Plant /Clifton Court Forebay')",all,y=0.89x+18.75,0.8991475178506713,-32.01976618307047,-0.0709186705975315,10.350886930630224,0.1514118957684851,0.8706125853991984,-7.09186705975314,0.3596297374617811,-0.0676412735704089,0.1318403846764428
    v8_2_1,"Location(name='CLC', bpart='CLC', description='Banks Pumping Plant /Clifton Court Forebay')",all,y=0.89x+18.69,0.8991669226060319,-32.04213236012419,-0.070968208109265,10.35237044401123,0.1514227457336705,0.8705940412916084,-7.096820810926483,0.3596555080269754,-0.0676903370676067,0.1318512421929696
    '''
    metrics_df = pd.read_csv(calib_metric_csv_filename)
    # now subtract metrics: 8.2.1 - 8.2, run 19 - run8.2.1
    location_file_df = pd.read_csv(location_file)
    columns_to_keep = []
    if constituent.lower() == 'flow' or constituent.lower() == 'stage':
        columns_to_keep =  ['Location', 'N Mean Error','NMSE','NRMSE','PBIAS','RSR']
    else:
        columns_to_keep =  ['Location', 'NMean Error','NMSE','NRMSE','PBIAS','RSR']

    metrics_df['Location'] = metrics_df.apply(lambda x: format_location(x['Location']), axis=1)

    # create new dataframe, and rename columns
    print('about to apply columns to keep: constituent='+constituent)
    metrics_diff_df = metrics_df.loc[metrics_df['DSM2 Run'] == "v8_2_1"][columns_to_keep]
    metrics_diff_df = metrics_diff_df.rename({'Mean Error': base_run_name+'_Mean_Error', 'N Mean Error': base_run_name+'_NMean_Error', \
        'NMean Error': base_run_name+'_NMean_Error', \
        'NMSE': base_run_name+'_NMSE', 'NRMSE': base_run_name+'_NRMSE', 'PBIAS': base_run_name+'_PBIAS', 'RSR': base_run_name+'_RSR'}, axis='columns')

    # add runs, rename columns
    for r in alt_run_name_list:
        metrics_diff_df = metrics_diff_df.merge(metrics_df.loc[metrics_df['DSM2 Run'] == r][columns_to_keep], on=["Location"], how='left')
        metrics_diff_df = metrics_diff_df.rename({'Mean Error': r+'_Mean_Error', 'N Mean Error': r+'_NMean_Error', \
            'NMean Error': r+'_NMean_Error', \
            'NMSE': r+'_NMSE', 'NRMSE': r+'_NRMSE', 'PBIAS': r+'_PBIAS', 'RSR': r+'_RSR'}, axis='columns')


    # now do subtraction
    # metrics_to_subtract_list = ['Mean_Error', 'NMean_Error', 'NMSE', 'NRMSE', 'PBIAS', 'RSR']
    metrics_to_subtract_list = ['NMean_Error', 'NMSE', 'NRMSE', 'PBIAS', 'RSR']
    for r in alt_run_name_list:
        for m in metrics_to_subtract_list:
            metrics_diff_df[r+'_'+m+'_diff'] = metrics_diff_df[r+'_'+m] - metrics_diff_df[base_run_name+'_'+m]

    # now get the data we need, and rename the columns to remove the run number from column names
    for r in alt_run_name_list:
        metrics_names_list = metrics_to_subtract_list
        column_names_list = ['Location']
        for mn in metrics_names_list:
            column_names_list.append(r+'_'+mn+'_diff')
        # column_names_list = [r+'_'+x for x in metrics_names_list]
        stations_list = list(metrics_diff_df['Location'])
        df = metrics_diff_df[column_names_list]
        df.set_index('Location')
        col_headers_list = metrics_names_list.copy()
        col_headers_list.insert(0, 'Location')
        df.columns = col_headers_list

        # now create the heatmap (not working)
        # p = figure(title='title', x_range=metrics_names_list, yrange=stations_list, x_axis_location="below", width=900, height=800,
        #     toolbar_location='above')
        # p.grid.grid_line_color = None
        # p.axis.axis_line_color = None
        # p.axis.major_tick_line_color = None
        # p.axis.major_label_text_font_size = "7px"
        # p.axis.major_label_standoff = 0
        # p.xaxis.major_label_orientation = 3.14159 / 3
        # p.rect(x="Metric", y="Station", width=1, height=1,source=df)
        # # hm = p.rect(metrics_diff_df, x='metric', y='players',values='score', title=m, stat=None)
        # # show(hm)
        # show(p)
    metrics_diff_df.to_csv('E:/full_calibration_8_3/delta/dsm2v8.3/studies/run_merged_s19_c3_wBase/postprocessing/plots/Heat Maps/v2/%s_metric_diff.csv' % constituent)


location_files_dict = {'Flow':'E:/full_calibration_8_3/delta/dsm2v8.3/postprocessing/location_info/calibration_flow_stations.csv',
    'Stage': 'E:/full_calibration_8_3/delta/dsm2v8.3/postprocessing/location_info/calibration_stage_stations.csv',
    'EC': 'E:/full_calibration_8_3/delta/dsm2v8.3/postprocessing/location_info/calibration_ec_stations.csv'
}
calib_metric_csv_filenames_dict = {'Flow':'E:/full_calibration_8_3/delta/dsm2v8.3/studies/run_merged_s19_c3_wBase/postprocessing/plots/1_summary_statistics_all_FLOW.csv',
    'Stage':'E:/full_calibration_8_3/delta/dsm2v8.3/studies/run_merged_s19_c3_wBase/postprocessing/plots/1_summary_statistics_all_STAGE.csv',
    'EC':'E:/full_calibration_8_3/delta/dsm2v8.3/studies/run_merged_s19_c3_wBase/postprocessing/plots/1_summary_statistics_all_EC.csv',
}

base_run_name = 'v8_2_1'
alt_run_name_list = ['v8_2_0', 'v8_3_run_merged_s19_c3_wBase']

i=0
for constituent in location_files_dict:
    l=location_files_dict[constituent]
    f = calib_metric_csv_filenames_dict[constituent]
    create_heatmaps(constituent, l, f, base_run_name, alt_run_name_list)
    i+=1


