from .postpro_dsm2 import build_and_save_plot, merge_statistics_files
from pydsm import postpro

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
                    tasks = [dask.delayed(build_and_save_plot)(config_data, studies, location, vartype, 
                                                    write_html=True,write_graphics=False,
                                                    dask_key_name=f'build_and_save::{location}:{vartype}') for location in locations]
                    # tasks = [dask.delayed(build_and_save_plot)(config_data, studies, location, vartype, 
                    #                                 write_html=True,write_graphics=False,                                                     
                    #                                 dask_key_name=f'build_and_save::{location}:{vartype}') for location in locations]
                    dask.compute(tasks)
                else:
                    print('not using dask')
                    for location in locations:
                        build_and_save_plot(config_data, studies, location, vartype, write_html=True,write_graphics=False)
                merge_statistics_files(vartype, config_data)
    finally:
        if use_dask:
            cluster.stop_local_cluster()