from .postpro_dsm2 import merge_statistics_files
from pydsm import postpro
import sys

def build_checklist_plot(config_data, studies, location, vartype):
# def build_plot(config_data, studies, location, vartype):

    print("build_checklist_plot")

    options_dict = config_data['options_dict']
    inst_plot_timewindow_dict = config_data['inst_plot_timewindow_dict']
    inst_plot_timewindow = inst_plot_timewindow_dict[vartype.name]
    timewindow_dict = config_data['timewindow_dict']
    timewindow = timewindow_dict[timewindow_dict['default_timewindow']]
    zoom_inst_plot = options_dict['zoom_inst_plot']
    gate_file_dict = config_data['gate_file_dict'] if 'gate_file_dict' in config_data else None
    flow_or_stage = (vartype.name == 'FLOW') or (vartype.name == 'STAGE')
    if location=='RSAC128-RSAC123':
        print('cross-delta flow')
        flow_or_stage = False
    flow_in_thousands = (vartype.name == 'FLOW')
    units = vartype.units
    include_kde_plots = options_dict['include_kde_plots']

    calib_plot_template, metrics_df = \
        calibplot.build_calib_plot_template(studies, location, vartype, timewindow, \
            tidal_template=flow_or_stage, flow_in_thousands=flow_in_thousands, units=units,inst_plot_timewindow=inst_plot_timewindow, include_kde_plots=include_kde_plots,
            zoom_inst_plot=zoom_inst_plot, gate_studies=gate_studies, gate_locations=gate_locations, gate_vartype=gate_vartype)
    # calib_plot_template, metrics_df = \
    #     calibplot.build_calib_plot_template(studies, location, vartype, timewindow, \
    #         tidal_template=flow_or_stage, flow_in_thousands=flow_in_thousands, units=units,inst_plot_timewindow=inst_plot_timewindow, include_kde_plots=include_kde_plots,
    #         zoom_inst_plot=zoom_inst_plot)
    if calib_plot_template is None:
        print('failed to create plots')
    if metrics_df is None:
        print('failed to create metrics')
    else:
        location_list = []
        for r in range(metrics_df.shape[0]):
            location_list.append(location)
        metrics_df['Location'] = location_list
        # move Location column to beginning
        cols = list(metrics_df)
        cols.insert(0, cols.pop(cols.index('Location')))
        metrics_df = metrics_df.loc[:, cols]
    return calib_plot_template, metrics_df

def build_and_save_checklist_plot(config_data, studies, location, vartype,
                                  write_html=False, write_graphics=True,
                                  output_format='png'):

    study_files_dict = config_data['study_files_dict']
    output_plot_dir = config_data['options_dict']['output_folder']
    print('build and save plot: output_plot_dir = ' + output_plot_dir)
    print('Building plot template for location: ' + str(location))

    checklist_plot_template, metrics_df = build_checklist_plot(config_data, studies, location, vartype)

    sys.exit()
    # if calib_plot_template is None:
    #     print('failed to create plots')
    # if metrics_df is None:
    #     print('failed to create metrics')
    # os.makedirs(output_plot_dir, exist_ok=True)
    # # save plot to html and/or png file
    # if calib_plot_template is not None and metrics_df is not None:
    #     if write_html:
    #         print('writing to html: 'f'{output_plot_dir}{location.name}_{vartype.name}.html')
    #         calib_plot_template.save(f'{output_plot_dir}{location.name}_{vartype.name}.html')
    #     if write_graphics:
    #         save_to_graphics_format(calib_plot_template,f'{output_plot_dir}{location}_{vartype.name}.png')
    # #         export_svg(calib_plot_template,f'{output_plot_dir}{location.name}_{vartype.name}.svg')
    # if metrics_df is not None:
    #     location_list = []
    #     for r in range(metrics_df.shape[0]):
    #         location_list.append(location)
    #     metrics_df['Location'] = location_list
    #     # move Location column to beginning
    #     cols = list(metrics_df)
    #     cols.insert(0, cols.pop(cols.index('Location')))
    #     metrics_df = metrics_df.loc[:, cols]

    #     # files for individual studies
    #     for study in study_files_dict:
    #         metrics_df[metrics_df.index == study].to_csv(
    #             output_plot_dir + '0_summary_statistics_' + study + '_' + vartype.name + '_' + location.name + '.csv')
    #         # metrics_df[metrics_df.index==study].to_html(output_plot_dir+'0_summary_statistics_'+study+'_'+vartype.name+'_'+location.name+'.html')
    return

def checklist_plots(cluster, config_data, use_dask):
    vartype_dict = config_data['vartype_dict']
    process_vartype_dict = config_data['process_vartype_dict']
    location_files_dict = config_data['location_files_dict']
    observed_files_dict = config_data['observed_files_dict']
    study_files_dict = config_data['study_files_dict']
    ## Set options and run processes. If using dask, create delayed tasks
    try:
        for var_name in vartype_dict:
            vartype = postpro.VarType(var_name, vartype_dict[var_name])
            print('vartype='+str(vartype))
            if process_vartype_dict[vartype.name]:
                ## Load locations from a .csv file, and create a list of postpro.Location objects
                #The .csv file should have atleast 'Name','BPart' and 'Description' columns
                locationfile=location_files_dict[vartype.name]
                dfloc = postpro.load_location_file(locationfile)
                print('about to read location file: '+ locationfile)
                locations = [postpro.Location(r['Name'],r['BPart'],r['Description'],r['time_window_exclusion_list']) for i,r in dfloc.iterrows()]

                # create list of postpro.Study objects, with observed Study followed by model Study objects
                obs_study=postpro.Study('Observed',observed_files_dict[vartype.name])
                model_studies=[postpro.Study(name,study_files_dict[name]) for name in study_files_dict]
                studies=[obs_study]+model_studies


                # now run the processes
                if use_dask:
                    print('using dask')
                    tasks = [dask.delayed(build_and_save_checklist_plot)(config_data, studies, location, vartype, 
                                                    write_html=True,write_graphics=False,
                                                    dask_key_name=f'build_and_save::{location}:{vartype}') for location in locations]
                    # tasks = [dask.delayed(build_and_save_plot)(config_data, studies, location, vartype, 
                    #                                 write_html=True,write_graphics=False,                                                     
                    #                                 dask_key_name=f'build_and_save::{location}:{vartype}') for location in locations]
                    dask.compute(tasks)
                else:
                    print('not using dask')
                    for location in locations:
                        build_and_save_checklist_plot(config_data, studies, location, vartype, write_html=True,write_graphics=False)
                merge_statistics_files(vartype, config_data)
    finally:
        if use_dask:
            cluster.stop_local_cluster()