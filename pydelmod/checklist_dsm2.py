from .postpro_dsm2 import merge_statistics_files, \
                          build_and_save_plot

from pydsm import postpro
import sys
import os
import pyhecdss
import pandas as pd
import json
import shutil

def resample_to_15min(config_data):
    resample_switch_dict = config_data["resample_switch_dict"]
    resample_file_dict = config_data["resample_file_dict"]

    for data_source in resample_switch_dict:
        if resample_switch_dict[data_source] == True:

            print("resampling for: ", data_source)

            fpath_in = resample_file_dict[data_source]

            # create new file for resampled data
            # close the file immediately to avoid any memory issues
            [fname, ext] = os.path.splitext(fpath_in)
            fpath_out = fname + "_resampled" + ext
            newdss = pyhecdss.DSSFile(fpath_out, create_new = True)
            newdss.close()

            # select time series with desired location and variable
            with pyhecdss.DSSFile(fpath_in) as dss_in:
                catdf = dss_in.read_catalog()
                paths = dss_in.get_pathnames()

                for p in paths:

                    print("processing:", p)

                    # extract time series, whether regular or irregular
                    try:
                        dfr, unit0, type0 = dss_in.read_rts(p)

                    except:
                        dfr, unit0, type0 = dss_in.read_its(p)

                    # resample to 15min interval
                    dfr = dfr.resample(rule="15min").ffill()

                    # write to new file
                    with pyhecdss.DSSFile(fpath_out, create_new = False) as newdss:
                        newdss.write_rts(p, dfr, unit0, type0)

            print("Resampled time series saved to " + fpath_out)

    print("Done\n")

def checklist_station_extract(config_data):

    extract_switch_dict = config_data["extract_switch_dict"]
    extract_file_dict = config_data["extract_file_dict"]
    extract_station_dict = config_data["extract_station_dict"]
    extract_copy_all_dict = config_data["extract_copy_all_dict"]

    for data_source in extract_switch_dict:
        if extract_switch_dict[data_source] == True:

            # obtain path of file containing original time series data
            #   (e.g., DSM2 output or observation)
            fpath_in = extract_file_dict[data_source]

            print("extracting station data for:", data_source)
            print("source:", fpath_in)

            # Path of the new file to be generated.
            # Remove existing file if exists.
            [fname, ext] = os.path.splitext(fpath_in)
            fpath_out = fname + "_" + data_source + ext
            try:
                os.remove(fpath_out)
            except:
                pass

            if extract_copy_all_dict == True:
                # The resulting file will contain all time series of original.
                # Instead of copying time series one by one,
                #   simply make a copy of the original.
                shutil.copy(fpath_in, fpath_out)
                newdss = pyhecdss.DSSFile(fpath_out, create_new = False)
                newdss.close()

            else:
                # create new file for extracted data
                # close the file immediately to avoid any memory issues
                newdss = pyhecdss.DSSFile(fpath_out, create_new = True)
                newdss.close()

            # obtain dictionary of checklist station name (e.g., SAC)
            station_checklist = extract_station_dict[data_source]

            for station_out in station_checklist:

                print("   checklist station:", station_out)
                print("   origin station(s):", station_checklist[station_out])

                for vartype in ["FLOW", "STAGE", "EC"]:

                    print("      processing ", vartype)

                    try:
                        # list of time series
                        list_dfr = []

                        # for given checklist station (e.g., SWP),
                        #   loop through the constituting stations
                        for station_source in station_checklist[station_out]:

                            # time series values are reversed when specified by "-"
                            sign = +1.
                            if (station_source[0] == "-"):
                                sign = -1.
                                station_source = station_source[1:]

                            # select time series with desired location and variable
                            with pyhecdss.DSSFile(fpath_in) as dss_in:
                                catdf = dss_in.read_catalog()
                                pathi = dss_in.get_pathnames(
                                        catdf[(catdf.B == station_source) &
                                              (catdf.C == vartype)])

                            # extract time series
                            try:
                                dfr, unit0, type0 = dss_in.read_rts(pathi[0])

                            except:
                                dfr, unit0, type0 = dss_in.read_its(pathi[0])

                            # write original data to new file for sanity check
                            # if sign is reversed, the sign-reversed data is saved.
                            if sign == -1.:
                                substr = pathi[0].split("/")
                                substr[2] = "-" + station_source
                                pathi[0] = "/".join(substr)

                            with pyhecdss.DSSFile(fpath_out, create_new = False) as newdss:
                                newdss.write_rts(pathi[0], dfr*sign, unit0, type0)

                            # if the original and new station names are same
                            #   with exception of the sign, the original time series
                            #   is indicated accordingly. Otherwise, it is confusing
                            #   what the "raw" data contained.
                            if station_source == station_out:
                                substr = pathi[0].split("/")
                                substr[2] = station_source + "(original)"
                                pathi[0] = "/".join(substr)

                            with pyhecdss.DSSFile(fpath_out, create_new = False) as newdss:
                                newdss.write_rts(pathi[0], dfr, unit0, type0)

                            # append to the list of dfr
                            list_dfr.append(dfr*sign)

                            dfr_checklist = pd.concat(list_dfr, axis=1)

                        # replace the station name to checklist convention
                        #   using the last known path is acceptable,
                        #   because only Part B (location) is different.
                        substr = pathi[0].split("/")
                        substr[2] = station_out
                        path_new = "/".join(substr)

                        # write to new file using checklist alias
                        dfr_checklist = dfr_checklist.sum(axis=1)
                        with pyhecdss.DSSFile(fpath_out, create_new = False) as newdss:
                            newdss.write_rts(path_new, dfr_checklist, unit0, type0)

                        print("         success.")

                    except:
                        print("         variable does not exist.")

            print("Extracted time series saved to " + fpath_out)
    print("Done\n")

def checklist_plots(cluster, config_data, use_dask):

    vartype_dict = config_data['vartype_dict']

    checklist_dict = config_data['checklist_dict']
    checklist_vartype_dict = config_data['checklist_vartype_dict']
    checklist_location_files_dict = config_data['checklist_location_files_dict']
    checklist_observed_files_dict = config_data['checklist_observed_files_dict']

    # Store main output folder
    output_folder_main = config_data["options_dict"]["output_folder"]

    ## Set options and run processes. If using dask, create delayed tasks
    try:
        for checklist_item in checklist_dict:
            if (checklist_dict[checklist_item] == True):
                print("checklist_item:", checklist_item)
                for var_name in checklist_vartype_dict[checklist_item]:
                    vartype = postpro.VarType(var_name, vartype_dict[var_name])

                    print('vartype='+str(vartype))

                    checklist_study_files_dict = config_data['checklist_study_files_dict'][checklist_item]

                    # store checklist dictionary as postprocessor dictionary for compatibility
                    config_data["study_files_dict"] = checklist_study_files_dict

                    # Specify output subfolder for each checklist item
                    output_folder_sub = os.path.join(output_folder_main, checklist_item+"/")
                    config_data["options_dict"]["output_folder"] = output_folder_sub

                    ## Load locations from a .csv file, and create a list of postpro.Location objects
                    #The .csv file should have atleast 'Name','BPart' and 'Description' columns
                    locationfile=checklist_location_files_dict[checklist_item]
                    dfloc = postpro.load_location_file(locationfile)
                    print('about to read location file: '+ locationfile)
                    locations = [postpro.Location(r['Name'],r['BPart'],r['Description'],r['time_window_exclusion_list'],r['threshold_value']) for i,r in dfloc.iterrows()]

                    # create list of postpro.
                    obs_study = postpro.Study('Observed',checklist_observed_files_dict[checklist_item])
                    model_studies = [postpro.Study(name,checklist_study_files_dict[name]) for name in checklist_study_files_dict]
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
                            build_and_save_plot(config_data, studies, location, vartype, write_html=True,write_graphics=True,
                                                gate_studies=None, gate_locations=None, gate_vartype=None)

                    merge_statistics_files(vartype, config_data)

                    print("Completed:", checklist_item, "\n")

    finally:
        if use_dask:
            cluster.stop_local_cluster()

def run_checklist(process_name, config_filename):
    '''
    process_name (str): should be 'resample', 'extract', or 'plot'
    config_filename (str): filename of config (json) file
    '''
    # Read Config file
    with open(config_filename) as f:
        config_data = json.load(f)

    # Run checklist process
    if process_name.lower() == 'resample':
        resample_to_15min(config_data)
    elif process_name.lower() == 'extract':
        checklist_station_extract(config_data)
    elif process_name.lower() == 'plot':
        checklist_plots(cluster=None, config_data=config_data, use_dask=False)
    else:
        print('Error in pydelmod.checklist_dsm2: process_name unrecognized: '+process_name)