import panel as pn
from .postpro_dsm2 import merge_statistics_files
from .calibplot import load_data_for_plotting,build_inst_plot, \
                       build_godin_plot, build_scatter_plots, \
                       build_metrics_table
from pydsm import postpro
import sys
import os

def build_checklist_plot_template(studies, location, vartype, 
                                  timewindow,
                                  tidal_template=False,
                                  flow_in_thousands=False,
                                  units=None,
                                  inst_plot_timewindow=None,
                                  layout_nash_sutcliffe=False,
                                  obs_data_included=False,
                                  zoom_inst_plot=False):
    """Builds calibration plot template

    Args:
        studies (List): Studies (name,dssfile)
        location (Location): name,bpart,description
        vartype (VarType): name,units
        timewindow (str): timewindow as start_date_str "-" end_date_str or "" for full availability
        tidal_template (bool, optional): If True include tidal plots. Defaults to False.
        flow_in_thousands (bool, optional): If True, template is for flow data, and
            1) y axis title will include the string '(1000 CFS)', and
            2) all flow values in the inst, godin, and scatter plots will be divided by 1000.
        units (str, optional): a string representing the units of the data. examples: CFS, FEET, UMHOS/CM.
            Included in axis titles if specified.
        inst_plot_timewindow (str, optional): Defines a separate timewindow to use for the instantaneous plot.
            Must be in format 'YYYY-MM-DD:YYYY-MM-DD'
        layout_nash_sutcliffe (bool, optional): if true, include Nash-Sutcliffe Efficiency in tables that are
            included in plot layouts. NSE will be included in summary tables--separate files containing only
            the equations and statistics for all locations.
        obs_data_included (bool, optional): If true, first study in studies list is assumed to be observed data.
            calibration metrics will be calculated.
        include_kde_plots (bool): If true, kde plots will be included. This is temporary for debugging
        zoom_inst_plot (bool): If true, instantaneous plots will display on data in the inst_plot_timewindow
        time_window_exclusion_list (list of time window strings in format yyyy-mm-dd hh:mm:ss_yyyy-mm-dd hh:mm:ss)
    Returns:
        panel: A template ready for rendering by display or save
        dataframe: equations and statistics for all locations
    """
    all_data_found, pp = load_data_for_plotting(studies, location, vartype, timewindow)
    if not all_data_found:
        return None, None
    print('build_checklist_plot_template')

    data_masking_time_series_dict= {}
    data_masking_df_dict = {}

    tsp = build_inst_plot(pp, location, vartype, flow_in_thousands=flow_in_thousands, units=units, inst_plot_timewindow=inst_plot_timewindow, zoom_inst_plot=zoom_inst_plot)
    gtsp = build_godin_plot(pp, location, vartype, flow_in_thousands=flow_in_thousands, units=units, 
        time_window_exclusion_list=location.time_window_exclusion_list)
    cplot = None
    dfdisplayed_metrics = None
    metrics_table = None

    if obs_data_included:
        time_window_exclusion_list = location.time_window_exclusion_list

        cplot = build_scatter_plots(pp, location, vartype, flow_in_thousands=flow_in_thousands, units=units,
            time_window_exclusion_list = time_window_exclusion_list)

        df_displayed_metrics_dict = {}
        metrics_table_dict = {}

        dfdisplayed_metrics, metrics_table = build_metrics_table(studies, pp, location, vartype, tidal_template=tidal_template, flow_in_thousands=flow_in_thousands, units=units,
                            layout_nash_sutcliffe=False, time_window_exclusion_list=time_window_exclusion_list)
        df_displayed_metrics_dict.update({'all': dfdisplayed_metrics})
        metrics_table_dict.update({'all': metrics_table})

    # # create plot/metrics template
    header_panel = pn.panel(f'## {location.description} ({location.name}/{vartype.name})')
    # # do this if you want to link the axes
    # # tsplots2 = (tsp.opts(width=900)+gtsp.opts(show_legend=False, width=900)).cols(1)
    # # start_dt = dflist[0].index.min()
    # # end_dt = dflist[0].index.max()

    column = None
    # temporary fix to add toolbar to all plots. eventually need to only inlucde toolbar if creating html file
    add_toolbar = True
    print('before creating column object (plot layout) for returning')
    if tidal_template:
        if not add_toolbar:
            if obs_data_included:
                column = pn.Column(
                    header_panel,
                    # tsp.opts(width=900, legend_position='right'),
                    tsp.opts(width=900, toolbar=None, title='(a)', legend_position='right'),
                    gtsp.opts(width=900, toolbar=None, title='(b)', legend_position='right'))
                    # pn.Row(tsplots2),
                    # pn.Row(cplot.opts(shared_axes=False, toolbar=None, title='(c)')))
                metrics_table_column = pn.Column()
                for metrics_table_name in metrics_table_dict:
                    metrics_table_column.append(metrics_table_dict[metrics_table_name].opts(title='(d) ' + metrics_table_name))

                scatter_and_metrics_row = pn.Row(cplot.opts(shared_axes=False, toolbar=None, title='(c)'))
                scatter_and_metrics_row.append(metrics_table_column)
                column.append(scatter_and_metrics_row)
            else:
                column = pn.Column(
                    header_panel,
                    # tsp.opts(width=900, legend_position='right'),
                    tsp.opts(width=900, toolbar=None, title='(a)', legend_position='right'),
                    gtsp.opts(width=900, toolbar=None, title='(b)', legend_position='right'))
        else:
            if obs_data_included:
                column = pn.Column(
                    header_panel,
                    # tsp.opts(width=900, legend_position='right'),
                    tsp.opts(width=900, title='(a)', legend_position='right'),
                    gtsp.opts(width=900, title='(b)', legend_position='right'))
                    # pn.Row(tsplots2),
                    # pn.Row(cplot.opts(shared_axes=False, title='(c)')))
                metrics_table_column = pn.Column()
                for metrics_table_name in metrics_table_dict:
                    metrics_table_column.append(metrics_table_dict[metrics_table_name].opts(title='(d) ' + metrics_table_name))
                scatter_and_metrics_row = pn.Row(cplot.opts(shared_axes=False, title='(c)'))
                scatter_and_metrics_row.append(metrics_table_column)
                column.append(scatter_and_metrics_row)
            else:
                column = pn.Column(
                    header_panel,
                    # tsp.opts(width=900, legend_position='right'),
                    tsp.opts(width=900, title='(a)', legend_position='right'),
                    gtsp.opts(width=900, title='(b)', legend_position='right'))
    else:
        if not add_toolbar:
            if obs_data_included:
                column = pn.Column(
                    header_panel,
                    pn.Row(gtsp.opts(width=900, show_legend=True, toolbar=None, title='(a)', legend_position='right')))
                    # pn.Row(cplot.opts(shared_axes=False, toolbar=None, title='(b)')))
                metrics_table_column = pn.Column()
                for metrics_table_name in metrics_table_dict:
                    metrics_table_column.append(metrics_table_dict[metrics_table_name].opts(title='(c) ' + metrics_table_name))
                scatter_and_metrics_row = pn.Row(cplot.opts(shared_axes=False, toolbar=None, title='(b)'))
                scatter_and_metrics_row.append(metrics_table_column)
                column.append(scatter_and_metrics_row)
            else:
                column = pn.Column(
                    header_panel,
                    pn.Row(gtsp.opts(width=900, show_legend=True, toolbar=None, title='(a)', legend_position='right')))

        else:
            if obs_data_included:
                column = pn.Column(
                    header_panel,
                    pn.Row(gtsp.opts(width=900, show_legend=True, title='(a)')))
                    # pn.Row(cplot.opts(shared_axes=False, title='(b)')))
                metrics_table_column = pn.Column()
                for metrics_table_name in metrics_table_dict:
                    metrics_table_column.append(metrics_table_dict[metrics_table_name].opts(title='(c) ' + metrics_table_name))
                scatter_and_metrics_row = pn.Row(cplot.opts(shared_axes=False, title='(b)'))
                scatter_and_metrics_row.append(metrics_table_column)
                column.append(scatter_and_metrics_row)
            else:
                column = pn.Column(
                    header_panel,
                    pn.Row(gtsp.opts(width=900, show_legend=True, title='(a)')))

    # now merge all metrics dataframes, adding a column identifying the gate status
    return_metrics_df = None

    if obs_data_included:
        df_index = 0
        for metrics_df_name in df_displayed_metrics_dict:
            metrics_df_name_list = []
            metrics_df = df_displayed_metrics_dict[metrics_df_name]
            for r in range(metrics_df.shape[0]):
                metrics_df_name_list.append(metrics_df_name)

            # merge df into return_metrics_df
            if df_index == 0:
                return_metrics_df = metrics_df
            else:
                return_metrics_df.append(metrics_df)
            df_index += 1
    return column, return_metrics_df


def build_checklist_plot(config_data, studies, location, vartype, obs_data_included):
# def build_plot(config_data, studies, location, vartype):

    print("build_checklist_plot")

    options_dict = config_data['options_dict']
    inst_plot_timewindow_dict = config_data['inst_plot_timewindow_dict']
    inst_plot_timewindow = inst_plot_timewindow_dict[vartype.name]
    timewindow_dict = config_data['timewindow_dict']
    timewindow = timewindow_dict[timewindow_dict['default_timewindow']]
    zoom_inst_plot = options_dict['zoom_inst_plot']

    units = vartype.units

    checklist_plot_template, metrics_df = \
        build_checklist_plot_template(studies, location, vartype, timewindow, \
            tidal_template=True, flow_in_thousands=True, units=units,inst_plot_timewindow=inst_plot_timewindow,
            obs_data_included=obs_data_included, zoom_inst_plot=zoom_inst_plot)

    if checklist_plot_template is None:
        print('failed to create plots')
    if metrics_df is None:
        print('failed to create metrics: build_checklist_plot')
    else:
        location_list = []
        for r in range(metrics_df.shape[0]):
            location_list.append(location)
        metrics_df['Location'] = location_list
        # move Location column to beginning
        cols = list(metrics_df)
        cols.insert(0, cols.pop(cols.index('Location')))
        metrics_df = metrics_df.loc[:, cols]
    return checklist_plot_template, metrics_df

def build_and_save_checklist_plot(config_data, studies, location, vartype,
                                  obs_data_included,
                                  write_html=False, write_graphics=True,
                                  output_format='png'):

    study_files_dict = config_data['study_files_dict']
    output_plot_dir = config_data['options_dict']['output_folder']
    print('build and save plot: output_plot_dir = ' + output_plot_dir)
    print('Building plot template for location: ' + str(location))

    checklist_plot_template, metrics_df = build_checklist_plot(config_data, studies, location, vartype, obs_data_included)

    if checklist_plot_template is None:
        print('failed to create plots')
    if metrics_df is None:
        print('failed to create metrics: build_and_save_checklist_plot')
    os.makedirs(output_plot_dir, exist_ok=True)
    # save plot to html and/or png file
    if checklist_plot_template is not None:
        if write_html:
            print('writing to html: 'f'{output_plot_dir}{location.name}_{vartype.name}.html')
            checklist_plot_template.save(f'{output_plot_dir}{location.name}_{vartype.name}.html')
        if write_graphics:
            save_to_graphics_format(checklist_plot_template,f'{output_plot_dir}{location}_{vartype.name}.png')
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
            metrics_df[metrics_df.index == study].to_csv(
                output_plot_dir + '0_summary_statistics_' + study + '_' + vartype.name + '_' + location.name + '.csv')
            # metrics_df[metrics_df.index==study].to_html(output_plot_dir+'0_summary_statistics_'+study+'_'+vartype.name+'_'+location.name+'.html')
    return

def checklist_plots(cluster, config_data, use_dask):
    vartype_dict = config_data['vartype_dict']
    process_vartype_dict = config_data['process_vartype_dict']
    location_files_dict = config_data['location_files_dict']
    observed_files_dict = config_data['observed_files_dict']
    study_files_dict = config_data['study_files_dict']

    checklist_dict = config_data['checklist_dict']
    checklist_vartype_dict = config_data['checklist_vartype_dict']
    compare_with_obs_dict = config_data["compare_with_obs_dict"]
    
    ## Set options and run processes. If using dask, create delayed tasks
    try:
        for checklist_item in checklist_dict:
            var_name = checklist_vartype_dict [checklist_item]
            vartype = postpro.VarType(var_name, vartype_dict[var_name])

            # Compare with observation            
            if compare_with_obs_dict[checklist_item]:
                if os.path.exists(observed_files_dict[checklist_item]):
                    obs_data_included = True
                else:
                    print("Observation file does not exist.")
                    sys.exit()
            else:
                obs_data_included = False

            print('vartype='+str(vartype))
            if process_vartype_dict[vartype.name]:
                ## Load locations from a .csv file, and create a list of postpro.Location objects
                #The .csv file should have atleast 'Name','BPart' and 'Description' columns
                locationfile=location_files_dict[checklist_item]
                dfloc = postpro.load_location_file(locationfile)
                print('about to read location file: '+ locationfile)
                locations = [postpro.Location(r['Name'],r['BPart'],r['Description'],r['time_window_exclusion_list']) for i,r in dfloc.iterrows()]

                # create list of postpro.Study objects, with observed Study followed by model Study objects
                studies = []
                for name in study_files_dict:
                    studies = studies + [postpro.Study(name,study_files_dict[name])]

                if (obs_data_included == True):
                    obs_study = [postpro.Study('Observed',observed_files_dict[checklist_item])]
                    studies = obs_study + studies

                # now run the processes
                if use_dask:
                    print('using dask')
                    tasks = [dask.delayed(build_and_save_checklist_plot)(config_data, studies, location, vartype,
                                                    obs_data_included=obs_data_included,
                                                    write_html=True,write_graphics=False,
                                                    dask_key_name=f'build_and_save::{location}:{vartype}') for location in locations]
                    # tasks = [dask.delayed(build_and_save_plot)(config_data, studies, location, vartype, 
                    #                                 write_html=True,write_graphics=False,                                                     
                    #                                 dask_key_name=f'build_and_save::{location}:{vartype}') for location in locations]
                    dask.compute(tasks)
                else:
                    print('not using dask')
                    for location in locations:
                        build_and_save_checklist_plot(config_data, studies, location, vartype, 
                                                      obs_data_included=obs_data_included,
                                                      write_html=True,write_graphics=False)
                        
                if obs_data_included == True:
                    merge_statistics_files(vartype, config_data)
    finally:
        if use_dask:
            cluster.stop_local_cluster()