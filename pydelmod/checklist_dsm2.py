from .postpro_dsm2 import merge_statistics_files, \
                          build_and_save_plot

from pydsm import postpro
import sys
import os

def checklist_plots(cluster, config_data, use_dask):
    vartype_dict = config_data['vartype_dict']
    study_files_dict = config_data['study_files_dict']

    checklist_dict = config_data['checklist_dict']
    checklist_vartype_dict = config_data['checklist_vartype_dict']
    checklist_location_files_dict = config_data['checklist_location_files_dict']
    checklist_observed_files_dict = config_data['checklist_observed_files_dict']
    
    # Store main output folder
    output_folder_main = config_data["options_dict"]["output_folder"]

    ## Set options and run processes. If using dask, create delayed tasks
    try:
        for checklist_item in checklist_dict:
            print("checklist_item:", checklist_item)

            var_name = checklist_vartype_dict [checklist_item]
            vartype = postpro.VarType(var_name, vartype_dict[var_name])

            print('vartype='+str(vartype))

            # Specify output subfolder for each checklist item
            output_folder_sub = os.path.join(output_folder_main, checklist_item+"/")
            config_data["options_dict"]["output_folder"] = output_folder_sub

            ## Load locations from a .csv file, and create a list of postpro.Location objects
            #The .csv file should have atleast 'Name','BPart' and 'Description' columns
            locationfile=checklist_location_files_dict[checklist_item]
            dfloc = postpro.load_location_file(locationfile)
            print('about to read location file: '+ locationfile)
            locations = [postpro.Location(r['Name'],r['BPart'],r['Description'],r['time_window_exclusion_list']) for i,r in dfloc.iterrows()]

            # create list of postpro.Study objects, with observed Study followed by model Study objects
            obs_study = postpro.Study('Observed',checklist_observed_files_dict[checklist_item])
            model_studies = [postpro.Study(name,study_files_dict[name]) for name in study_files_dict]
            studies = [obs_study] + model_studies

            # now run the processes
            if use_dask:
                # print('using dask')
                tasks = [dask.delayed(build_and_save_checklist_plot)(config_data, studies, location, vartype,
                                                                     write_html=True,write_graphics=False,
                                                                     gate_studies=None, gate_locations=None, gate_vartype=None,
                                                dask_key_name=f'build_and_save::{location}:{vartype}') for location in locations]
                # tasks = [dask.delayed(build_and_save_plot)(config_data, studies, location, vartype,
                #                                 write_html=True,write_graphics=False,
                #                                 dask_key_name=f'build_and_save::{location}:{vartype}') for location in locations]
                dask.compute(tasks)
            else:
                # print('not using dask')
                for location in locations:
                    build_and_save_plot(config_data, studies, location, vartype, write_html=True,write_graphics=False,
                                        gate_studies=None, gate_locations=None, gate_vartype=None)

            merge_statistics_files(vartype, config_data)

            print("Completed:", checklist_item)

    finally:
        if use_dask:
            cluster.stop_local_cluster()