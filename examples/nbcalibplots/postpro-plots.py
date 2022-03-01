# Calibration Plot Generation Notebook
import pydsm
from pydsm import postpro

import pydelmod
from pydelmod import calibplot
import panel as pn
import pandas as pd
import holoviews as hv

# Read Config file
import json
config_filename = 'postpro_config.json'
f = open(config_filename)
data = json.load(f)
options_dict = data['options_dict']
vartype_dict = data['vartype_dict']
process_vartype_dict = data['process_vartype_dict']
dask_options_dict = data['dask_options_dict']
location_files_dict = data['location_files_dict']
observed_files_dict = data['observed_files_dict']
study_files_dict = data['study_files_dict']
inst_plot_timewindow_dict = data['inst_plot_timewindow_dict']
timewindow_dict = data['timewindow_dict']

output_folder = options_dict['output_folder']
timewindow = timewindow_dict[timewindow_dict['default_timewindow']]


# Build and save plot for each location
def export_svg(plot, fname):
    ''' export holoview object to filename fname '''
    from bokeh.io import export_svgs
    p =  hv.render(plot, backend='bokeh')
    p.output_backend = "svg"
    export_svgs(p, filename=fname)


def save_to_graphics_format(calib_plot_template,fname):
    #     hvobj=calib_plot_template[1][0]
    #     hvobj.object=hvobj.object.opts(toolbar=None) # remove the toolbar from the second row plot
    calib_plot_template.save(fname)


def build_and_save_plot(studies, location, vartype, timewindow, output_plot_dir, write_html=False,
                        write_graphics=True, output_format='png'):
    print(str(location))
    flow_or_stage = (vartype.name == 'FLOW') or (vartype.name == 'STAGE')
    if location=='RSAC128-RSAC123':
        print('cross-delta flow')
        flow_or_stage = False
    flow_in_thousands = (vartype == 'FLOW')

    units = vartype.units
    include_kde_plots = options_dict['include_kde_plots']
    calib_plot_template, metrics_df = calibplot.build_calib_plot_template(studies, location, vartype, timewindow,
                                                            tidal_template=flow_or_stage,
                                                                          flow_in_thousands=flow_in_thousands,
                                                                          units=units,
                                                                          inst_plot_timewindow=inst_plot_timewindow,
                                                                          include_kde_plots=include_kde_plots)
    if calib_plot_template is None:
        print('failed to create plots')
    if metrics_df is None:
        print('failed to create metrics')
    if calib_plot_template is not None and metrics_df is not None:
        if write_html: calib_plot_template.save(f'{output_plot_dir}{location.name}_{vartype.name}.html')
        if write_graphics:
            save_to_graphics_format(calib_plot_template,f'{output_plot_dir}{location}_{vartype.name}.png')
    #         export_svg(calib_plot_template,f'{output_plot_dir}{location.name}_{vartype.name}.svg')
    if metrics_df is not None:
        location_list = []
        for r in range(metrics_df.shape[0]):
            location_list.append(location)
        metrics_df['Location'] = location_list
        # move Location column to beginning
        cols = list(metrics_df)
        cols.insert(0, cols.pop(cols.index('Location')))
        metrics_df = metrics_df.loc[:, cols]

        # files for individual studies
        for study in study_files_dict:
            # print(str(location))
            metrics_df[metrics_df.index == study].to_csv(
                output_plot_dir + '0_summary_statistics_' + study + '_' + vartype.name + '_' + location.name + '.csv')
            # metrics_df[metrics_df.index==study].to_html(output_plot_dir+'0_summary_statistics_'+study+'_'+vartype.name+'_'+location.name+'.html')

    return calib_plot_template, metrics_df


# merge study statistics files
def merge_statistics_files(vartype):
    import glob, os
    print('merging statistics files')
    output_dir = options_dict['output_folder']
    files = glob.glob(output_dir + '0_summary_statistics_*'+vartype.name+'*.csv')
    frames = []
    for f in files:
        frames.append(pd.read_csv(f))
    result_df = pd.concat(frames)
    result_df.sort_values(by=['Location', 'DSM2 Run'], inplace=True, ascending=True)
    result_df.to_csv(output_dir + '1_summary_statistics_all_'+vartype.name+'.csv', index=False)
    for f in files:
        os.remove(f)


# Start Dask Cluster
# Using 8 workers here, each with a limit of 4GB
import dask
from dask.distributed import Client, LocalCluster


class DaskCluster:
    def __init__(self):
        self.client=None

    def start_local_cluster(self):
        cluster = LocalCluster(n_workers=8, threads_per_worker=1, memory_limit='4G') # threads_per_worker=1 needed if using numba :(
        self.client = Client(cluster)

    def stop_local_cluster(self):
        self.client.shutdown()
        self.client=None

# Create cluster if using dask
cluster = None
c_link = None
if dask_options_dict['use_dask']:
    cluster = DaskCluster()
    cluster.start_local_cluster()
    c_link=cluster.client
    print(c_link)
c_link

## Set options and run processes. If using dask, create delayed tasks
try:
    for var_name in vartype_dict:
        vartype = postpro.VarType(var_name, vartype_dict[var_name])
        print('vartype='+str(vartype))
        if process_vartype_dict[vartype.name]:
            # set a separate timewindow for instantaneous plots
            inst_plot_timewindow = inst_plot_timewindow_dict[vartype.name]
            ## Load locations from a .csv file, and create a list of postpro.Location objects
            #The .csv file should have atleast 'Name','BPart' and 'Description' columns
            locationfile=location_files_dict[vartype.name]
            dfloc = postpro.load_location_file(locationfile)
            locations = [postpro.Location(r['Name'],r['BPart'],r['Description']) for i,r in dfloc.iterrows()]
            # create list of postpro.Study objects, with observed Study followed by model Study objects
            obs_study=postpro.Study('Observed',observed_files_dict[vartype.name])
            model_studies=[postpro.Study(name,study_files_dict[name]) for name in study_files_dict]
            studies=[obs_study]+model_studies
            # now run the processes
            if dask_options_dict['use_dask']:
                print('using dask')
                tasks = [dask.delayed(build_and_save_plot)(studies, location, vartype, timewindow, output_folder,
                                                 write_html=True,write_graphics=False) for location in locations]
                dask.compute(tasks)
            else:
                print('not using dask')
                for location in locations:
                    build_and_save_plot(studies, location, vartype, timewindow, output_folder,
                                                 write_html=True,write_graphics=False)
            merge_statistics_files(vartype)
finally:
    if dask_options_dict['use_dask']:
        cluster.stop_local_cluster()
