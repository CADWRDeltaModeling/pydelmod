# Postpro-Model
from cmath import e
from distutils.command.config import config
import os
import pydsm
from pydsm import postpro
import json
import yaml
import sys


# dask is a parallel processing library. Using it will save a lot of time, but error
# messages will not be as helpful, so set to False for debugging.
# Another option is to set scheduler (below) to single_threaded

import dask
from dask.distributed import Client, LocalCluster
from pydelmod import calibplot
from pydelmod import calib_heatmap
import pyhecdss
import panel as pn
import pandas as pd
import holoviews as hv
import json
import sys


class DaskCluster:
    def __init__(self, config_data):
        self.client = None
        self.dask_options_dict = config_data["dask_options_dict"]

    def start_local_cluster(self):
        cluster = LocalCluster(
            n_workers=self.dask_options_dict["n_workers"],
            threads_per_worker=self.dask_options_dict["threads_per_worker"],
            memory_limit=self.dask_options_dict["memory_limit"],
        )  # threads_per_worker=1 needed if using numba :(
        self.client = Client(cluster)

    def stop_local_cluster(self):
        self.client.shutdown()
        self.client = None


def run_all(processors):
    tasks = [
        dask.delayed(postpro.run_processor)(
            processor,
            dask_key_name=f"{processor.study.name}::{processor.location.name}/{processor.vartype.name}",
        )
        for processor in processors
    ]
    dask.compute(tasks)
    # to use only one processor. Also prints more helpful messages
    #     dask.compute(tasks, scheduler='single-threaded')


def postpro_model(cluster, config_data, use_dask):
    # Setup for EC, FLOW, STAGE
    # import logging
    # logging.basicConfig(filename='postpro-model.log', level=logging.DEBUG)
    vartype_dict = config_data["vartype_dict"]
    # this specifies the files that are to be post-processed. Using study_files_dict resulted in all study files being post-processed.
    postpro_model_dict = config_data["postpro_model_dict"]
    location_files_dict = config_data["location_files_dict"]
    vartype_timewindow_dict = config_data["vartype_timewindow_dict"]

    try:
        for var_name in vartype_dict:
            vartype = postpro.VarType(var_name, vartype_dict[var_name])
            if vartype_timewindow_dict[vartype.name] is not None:
                print("processing model " + vartype.name + " data")
                for study_name in postpro_model_dict:
                    dssfile = postpro_model_dict[study_name]
                    # catalog the DSS file. If you don't do this, processes are likely to fail the first time you run them with an
                    # uncataloged DSS File, if you are using dask.
                    pyhecdss.DSSFile(dssfile).catalog()
                    locationfile = location_files_dict[vartype.name]
                    units = vartype.units
                    observed = False
                    processors = postpro.build_processors(
                        dssfile, locationfile, vartype.name, units, study_name, observed
                    )
                    print(f"Processing {vartype.name} for study: {study_name}")
                    if use_dask:
                        run_all(processors)
                    else:
                        for p in processors:
                            postpro.run_processor(p)
    except e:
        print("exception caught in postpro-model.py.run_processes. exiting.")
    finally:
        # Always shut down the cluster when done.
        if use_dask:
            cluster.stop_local_cluster()


def postpro_observed(cluster, config_data, use_dask):
    # Setup for EC, FLOW, STAGE
    vartype_dict = config_data["vartype_dict"]
    observed_files_dict = config_data["observed_files_dict"]
    location_files_dict = config_data["location_files_dict"]
    vartype_timewindow_dict = config_data["vartype_timewindow_dict"]

    try:
        for vartype in vartype_dict:
            if vartype_timewindow_dict[vartype] is not None:
                print("processing observed " + vartype + " data")
                dssfile = observed_files_dict[vartype]
                # catalog the DSS file. If you don't do this, processes are likely to fail the first time you run them with an
                # uncataloged DSS File, if you are using dask.
                pyhecdss.DSSFile(dssfile).catalog()
                location_file = location_files_dict[vartype]
                units = vartype_dict[vartype]
                study_name = "Observed"
                observed = True
                processors = postpro.build_processors(
                    dssfile, location_file, vartype, units, study_name, observed
                )
                if use_dask:
                    run_all(processors)
                else:
                    for p in processors:
                        postpro.run_processor(p)
    finally:
        # Always shut down the cluster when done.
        if use_dask:
            cluster.stop_local_cluster()


# Build and save plot for each location
def export_svg(plot, fname):
    """This doesn't work. It would be very useful."""
    """ export holoview object to filename fname """
    from bokeh.io import export_svgs

    p = hv.render(plot, backend="bokeh")
    p.output_backend = "svg"
    export_svgs(p, filename=fname)


def save_to_graphics_format(calib_plot_template, fname):
    #     hvobj=calib_plot_template[1][0]
    #     hvobj.object=hvobj.object.opts(toolbar=None) # remove the toolbar from the second row plot
    # saved=False
    # while not saved:
    #     try:
    #         calib_plot_template.save(fname)
    #         saved=True
    #     except RuntimeError as e:
    #         print('runtime error trying to save plot '+fname)
    #         print('will try again until it works.')
    try:
        calib_plot_template.save(fname)
    except:
        print('unable to create plot for ' + fname)


def build_plot(
    config_data,
    studies,
    location,
    vartype,
    gate_studies=None,
    gate_locations=None,
    gate_vartype=None,
    invert_timewindow_exclusion=False,
    remove_data_above_threshold=True,
    metrics_table_list=None,
):
    # def build_plot(config_data, studies, location, vartype):
    options_dict = config_data["options_dict"]
    inst_plot_timewindow_dict = config_data["inst_plot_timewindow_dict"]
    inst_plot_timewindow = inst_plot_timewindow_dict[vartype.name]
    timewindow_dict = config_data["timewindow_dict"]
    vartype_timewindow_dict = config_data["vartype_timewindow_dict"]
    timewindow = timewindow_dict[vartype_timewindow_dict[vartype.name]]
    tech_memo_validation_metrics = (
        options_dict["tech_memo_validation_metrics"]
        if "tech_memo_validation_metrics" in options_dict
        else False
    )
    manuscript_layout = (
        options_dict["manuscript_layout"]
        if "manuscript_layout" in options_dict
        else False
    )
    include_inst_plot_dict = (
        config_data["include_inst_plot_dict"]
        if "include_inst_plot_dict" in config_data
        else None
    )
    zoom_inst_plot = options_dict["zoom_inst_plot"]
    gate_file_dict = (
        config_data["gate_file_dict"] if "gate_file_dict" in config_data else None
    )
    mask_plot_metric_data = (
        options_dict["mask_plot_metric_data"]
        if "mask_plot_metric_data" in options_dict
        else True
    )
    # Flow and stage are tidal (also certain water quality constituents)
    tidal_data = vartype.name != "EC"
    if location == "RSAC128-RSAC123":
        print("cross-delta flow")
        tidal_data = False
    flow_in_thousands = vartype.name == "FLOW"
    units = vartype.units
    include_kde_plots = options_dict["include_kde_plots"]

    include_inst_plot = True
    if include_inst_plot_dict is None:
        if vartype.name == "EC":
            include_inst_plot = False
    else:
        if vartype.name in include_inst_plot_dict:
            include_inst_plot = include_inst_plot_dict[vartype.name]
        else:
            if vartype.name == "EC":
                include_inst_plot = False

    calib_plot_template_dict, metrics_df = calibplot.build_calib_plot_template(
        studies,
        location,
        vartype,
        timewindow,
        include_inst_plot,
        tidal_template=tidal_data,
        flow_in_thousands=flow_in_thousands,
        units=units,
        inst_plot_timewindow=inst_plot_timewindow,
        include_kde_plots=include_kde_plots,
        zoom_inst_plot=zoom_inst_plot,
        gate_studies=gate_studies,
        gate_locations=gate_locations,
        gate_vartype=gate_vartype,
        invert_timewindow_exclusion=invert_timewindow_exclusion,
        remove_data_above_threshold=remove_data_above_threshold,
        mask_data=mask_plot_metric_data,
        tech_memo_validation_metrics=tech_memo_validation_metrics,
        manuscript_layout=manuscript_layout,
        metrics_table_list=metrics_table_list,
    )

    # calib_plot_template, metrics_df = \
    #     calibplot.build_calib_plot_template(studies, location, vartype, timewindow, \
    #         tidal_template=flow_or_stage, flow_in_thousands=flow_in_thousands, units=units,inst_plot_timewindow=inst_plot_timewindow, include_kde_plots=include_kde_plots,
    #         zoom_inst_plot=zoom_inst_plot)
    if calib_plot_template_dict is None:
        print("failed to create plots")
    if metrics_df is None:
        print("failed to create metrics")
    else:
        location_list = []
        for r in range(metrics_df.shape[0]):
            location_list.append(location)
        metrics_df["Location"] = location_list
        # move Location column to beginning
        cols = list(metrics_df)
        cols.insert(0, cols.pop(cols.index("Location")))
        metrics_df = metrics_df.loc[:, cols]
    return calib_plot_template_dict, metrics_df


def build_and_save_plot(
    config_data,
    studies,
    location,
    vartype,
    gate_studies=None,
    gate_locations=None,
    gate_vartype=None,
    write_html=False,
    write_graphics=True,
    output_format="png",
    metrics_table_list=None,
):
    print('build_and_save_plot: location='+str(location))
    # def build_and_save_plot(config_data, studies, location, vartype, write_html=False, write_graphics=True, output_format='png'):
    study_files_dict = config_data["study_files_dict"]
    options_dict = config_data["options_dict"]
    output_plot_dir = options_dict["output_folder"]
    print("build and save plot: output_plot_dir = " + output_plot_dir)
    print("Building plot template for location: " + str(location))
    mask_data = (
        options_dict["mask_plot_metric_data"]
        if "mask_plot_metric_data" in options_dict
        else True
    )

    calib_plot_template_dict, metrics_df = build_plot(
        config_data,
        studies,
        location,
        vartype,
        gate_studies=gate_studies,
        gate_locations=gate_locations,
        gate_vartype=gate_vartype,
        metrics_table_list=metrics_table_list,
    )

    metrics_df_masked_time_period = None
    if calib_plot_template_dict is not None:
        calib_plot_template_with_toolbar = calib_plot_template_dict["with"]
        calib_plot_template_without_toolbar = calib_plot_template_dict["without"]
        # calib_plot_template, metrics_df = build_plot(config_data, studies, location, vartype)
        if calib_plot_template_dict is None:
            print("failed to create plots")
        if metrics_df is None:
            print("failed to create metrics")
        output_template_with_toolbar = calib_plot_template_with_toolbar
        output_template_without_toolbar = calib_plot_template_without_toolbar

        time_window_exclusion_list = location.time_window_exclusion_list
        threshold_value = location.threshold_value
        # calib_plot_template_masked_time_period = None
        create_second_panel = (
            True
            if (
                mask_data
                and (
                    (
                        time_window_exclusion_list is not None
                        and len(time_window_exclusion_list) > 0
                    )
                    or (threshold_value is not None and len(str(threshold_value)) > 0)
                )
            )
            else False
        )
        if create_second_panel:
            (
                calib_plot_template_masked_time_period_dict,
                metrics_df_masked_time_period,
            ) = build_plot(
                config_data,
                studies,
                location,
                vartype,
                gate_studies=gate_studies,
                gate_locations=gate_locations,
                gate_vartype=gate_vartype,
                invert_timewindow_exclusion=True,
                remove_data_above_threshold=False,
                metrics_table_list=metrics_table_list,
            )
            # calib_plot_template, metrics_df = build_plot(config_data, studies, location, vartype)
            if calib_plot_template_masked_time_period_dict is None:
                print("failed to create plots for masked time period")
            if metrics_df_masked_time_period is None:
                print("failed to create metrics for masked time period")
            calib_plot_template_masked_time_period_with_toolbar = (
                calib_plot_template_masked_time_period_dict["with"]
            )
            calib_plot_template_masked_time_period_without_toolbar = (
                calib_plot_template_masked_time_period_dict["without"]
            )
            # This puts the two calib plots templates side by side, with the
            # data removed from the masked time periods on the right,
            # and the data removed from outside the masked time periods on the left
            output_template_with_toolbar = pn.Row(
                calib_plot_template_with_toolbar,
                calib_plot_template_masked_time_period_with_toolbar,
            )
            output_template_without_toolbar = pn.Row(
                calib_plot_template_without_toolbar,
                calib_plot_template_masked_time_period_without_toolbar,
            )

        os.makedirs(output_plot_dir, exist_ok=True)
        # save plot to html and/or png file

        # if (
        #     calib_plot_template_with_toolbar is not None
        #     and calib_plot_template_without_toolbar is not None
        #     and metrics_df is not None
        # ):
        if (
            calib_plot_template_with_toolbar is not None
            and calib_plot_template_without_toolbar is not None
        ):
            if write_html:
                print(
                    "writing to html: "
                    f"{output_plot_dir}{location.name}_{vartype.name}.html"
                )
                output_template_with_toolbar.save(
                    f"{output_plot_dir}{location.name}_{vartype.name}.html",
                    title=location.name,
                )
            if write_graphics:
                print(
                    "writing to png: "
                    f"{output_plot_dir}{location.name}_{vartype.name}.png"
                )
                save_to_graphics_format(
                    output_template_without_toolbar,
                    f"{output_plot_dir}{location.name}_{vartype.name}.png",
                )
        #         export_svg(calib_plot_template,f'{output_plot_dir}{location.name}_{vartype.name}.svg')
    else:
        print(
            "***************************************************************************************************************************************"
        )
        print(
            "not creating output for location (there may be no data in specified time period): "
            + location.name
        )
        print(
            "***************************************************************************************************************************************"
        )

    if metrics_df is not None:
        location_list = []
        for r in range(metrics_df.shape[0]):
            location_list.append(location)
        metrics_df["Location"] = location_list
        # move Location column to beginning
        cols = list(metrics_df)
        cols.insert(0, cols.pop(cols.index("Location")))
        metrics_df = metrics_df.loc[:, cols]

        # files for individual studies
        for study in study_files_dict:
            metrics_df[metrics_df.index == study].to_csv(
                output_plot_dir
                + "0_summary_statistics_unmasked_"
                + study
                + "_"
                + vartype.name
                + "_"
                + location.name
                + ".csv"
            )
            # metrics_df[metrics_df.index==study].to_html(output_plot_dir+'0_summary_statistics_'+study+'_'+vartype.name+'_'+location.name+'.html')

    if metrics_df_masked_time_period is not None:
        location_list = []
        for r in range(metrics_df_masked_time_period.shape[0]):
            location_list.append(location)
        metrics_df_masked_time_period["Location"] = location_list
        # move Location column to beginning
        cols = list(metrics_df_masked_time_period)
        cols.insert(0, cols.pop(cols.index("Location")))
        metrics_df_masked_time_period = metrics_df_masked_time_period.loc[:, cols]

        # files for individual studies
        for study in study_files_dict:
            metrics_df_masked_time_period[
                metrics_df_masked_time_period.index == study
            ].to_csv(
                output_plot_dir
                + "0_summary_statistics_masked_time_period_"
                + study
                + "_"
                + vartype.name
                + "_"
                + location.name
                + ".csv"
            )
    return


# merge study statistics files
def merge_statistics_files(vartype, config_data):
    """Statistics files are written for each individual run, which is necessary (?) with dask
    This method merges all of them.
    """
    options_dict = config_data["options_dict"]
    col_rename_dict = {
        "regression_equation": "Equation",
        "r2": "R Squared",
        "mean_error": "Mean Error",
        "nmean_error": "NMean Error",
        "nmse": "NMSE",
        "nrmse": "NRMSE",
        "nash_sutcliffe": "NSE",
        "percent_bias": "PBIAS",
        "rsr": "RSR",
        "rmse": "RMSE",
        "mnly_regression_equation": "Mnly Equation",
        "mnly_r2": "Mnly R Squared",
        "mnly_mean_err": "Mnly Mean Err",
        "mnly_mean_error": "Mnly Mean Err",
        "mnly_nmean_error": "Mnly NMean Err",
        "mnly_nmse": "Mnly NMSE",
        "mnly_nrmse": "Mnly NRMSE",
        "mnly_nash_sutcliffe": "Mnly NSE",
        "mnly_kge": "Mnly KGE",
        "mnly_percent_bias": "Mnly PBIAS",
        "mnly_rsr": "Mnly RSR",
        "mnly_rmse": "Mnly RMSE",
        "Study": "Study",
        "Amp Avg %Err": "Amp Avg %Err",
        "Avg Phase Err": "Avg Phase Err",
    }

    import glob, os

    print("merging statistics files")
    filename_prefix_list = [
        "summary_statistics_masked_time_period_",
        "summary_statistics_unmasked_",
    ]
    for fp in filename_prefix_list:
        output_dir = options_dict["output_folder"]
        os.makedirs(output_dir, exist_ok=True)
        files = glob.glob(output_dir + "0_" + fp + "*" + vartype.name + "*.csv")
        # files = glob.glob(output_dir + '0_summary_statistics_*'+vartype.name+'*.csv')
        frames = []
        for f in files:
            frames.append(pd.read_csv(f))
        if len(frames) > 0:
            result_df = pd.concat(frames)

            result_df.rename(columns=col_rename_dict, inplace=True)

            result_df.sort_values(
                by=["Location", "DSM2 Run"], inplace=True, ascending=True
            )
            # result_df.to_csv(output_dir + '1_summary_statistics_all_'+vartype.name+'.csv', index=False)
            result_df.to_csv(
                output_dir + "1_" + fp + "all_" + vartype.name + ".csv", index=False
            )
            for f in files:
                os.remove(f)
                print('removed file '+f)


def postpro_heatmaps(cluster, config_data, use_dask):
    options_dict = config_data["options_dict"]
    heatmap_options_dict = config_data["heatmap_options_dict"]
    calib_metric_csv_filenames_dict = config_data["calib_metric_csv_filenames_dict"]
    station_order_file = heatmap_options_dict["station_order_file"]
    base_run_name = heatmap_options_dict["base_run"]
    run_name = heatmap_options_dict["alt_run"]
    metrics_list = heatmap_options_dict["metrics_list"]
    base_diff_type = heatmap_options_dict["base_diff_type"]
    heatmap_width = heatmap_options_dict["heatmap_width"]
    process_heatmap_vartype_dict = config_data["process_heatmap_vartype_dict"]

    calib_heatmap.create_save_heatmaps(
        calib_metric_csv_filenames_dict,
        station_order_file,
        base_run_name,
        run_name,
        metrics_list,
        heatmap_width=heatmap_width,
        process_vartype_dict=process_heatmap_vartype_dict,
        base_diff_type=base_diff_type,
    )


def postpro_validation_bar_charts(
    cluster, config_data, write_graphics=True, write_html=True
):
    """
    Creates bar charts for technical memo validation section
    """
    validation_bar_chart_options_dict = config_data["validation_bar_chart_options_dict"]
    validation_metric_csv_filenames_dict = validation_bar_chart_options_dict[
        "validation_metric_csv_filenames_dict"
    ]
    validation_plot_output_folder = validation_bar_chart_options_dict[
        "validation_plot_output_folder"
    ]
    vartype_to_station_list_dict = validation_bar_chart_options_dict[
        "vartype_to_station_list_dict"
    ]
    calibplot.create_validation_bar_charts(
        validation_plot_output_folder,
        validation_metric_csv_filenames_dict,
        vartype_to_station_list_dict,
    )


def postpro_copy_plot_files(cluster, config_data):
    import shutil

    plot_file_copying_options_dict = config_data["plot_file_copying_options_dict"]
    # plot_type will be 'flow_calibration', 'ec_validation', etc.
    # filenames to copy have filenames such as ANC_EC.png.
    # validation plots are in ./plots_val/ and calibration plots are in ./plots_cal/
    plot_type_to_const_dict = {
        "flow_calibration": "Flow",
        "flow_validation": "Flow",
        "stage_calibration": "Stage",
        "stage_validation": "Stage",
        "ec_calibration": "EC",
        "ec_validation": "EC",
    }
    plot_type_to_dir_dict = {
        "flow_calibration": "./plots_cal/",
        "flow_validation": "./plots_val/",
        "stage_calibration": "./plots_cal/",
        "stage_validation": "./plots_val/",
        "ec_calibration": "./plots_cal/",
        "ec_validation": "./plots_val/",
    }
    plot_types_to_copy_list = plot_file_copying_options_dict["plot_types_to_copy_list"]
    for plot_type in plot_types_to_copy_list:
        # location list is a list of locations, for example, RSAC155
        const = plot_type_to_const_dict[plot_type]
        d = plot_type_to_dir_dict[plot_type]
        location_list = plot_file_copying_options_dict[plot_type]
        #     now copy the files...
        i = 0
        for location in location_list:
            infile = d + location + "_" + const + ".png"
            outfile = d + plot_type + "_" + str(i) + "_" + location + ".png"
            print("infile, outfile=" + infile + "," + outfile)
            try:
                shutil.copy2(infile, outfile)
            except FileNotFoundError as e:
                print(
                    "FileNotFoundError exception caught: infile, outfile: "
                    + infile
                    + ","
                    + outfile
                )
            i += 1


def postpro_plots(cluster, config_data, use_dask):
    vartype_dict = config_data["vartype_dict"]
    location_files_dict = config_data["location_files_dict"]
    observed_files_dict = config_data["observed_files_dict"]
    study_files_dict = config_data["study_files_dict"]
    inst_plot_timewindow_dict = config_data["inst_plot_timewindow_dict"]
    gate_file_dict = (
        config_data["gate_file_dict"] if "gate_file_dict" in config_data else None
    )
    tech_memo_validation_metrics = (
        config_data["tech_memo_validation_metrics"]
        if "tech_memo_validation_metrics" in config_data
        else False
    )
    options_dict = config_data["options_dict"]
    write_graphics = (
        False
        if ("write_graphics" in options_dict and not options_dict["write_graphics"])
        else True
    )
    write_html = (
        False
        if ("write_html" in options_dict and not options_dict["write_html"])
        else True
    )
    gate_location_file_dict = (
        config_data["gate_location_file_dict"]
        if "gate_location_file_dict" in config_data
        else None
    )
    metrics_table_list = (
        options_dict["metrics_table_list"]
        if "metrics_table_list" in options_dict
        else None
    )

    vartype_timewindow_dict = config_data["vartype_timewindow_dict"]

    ## Set options and run processes. If using dask, create delayed tasks
    try:
        gate_vartype = postpro.VarType("POS", "")
        for var_name in vartype_dict:
            vartype = postpro.VarType(var_name, vartype_dict[var_name])
            print("vartype=" + str(vartype))
            if vartype_timewindow_dict[vartype.name] is not None:
                # set a separate timewindow for instantaneous plots
                inst_plot_timewindow = inst_plot_timewindow_dict[vartype.name]
                ## Load locations from a .csv file, and create a list of postpro.Location objects
                # The .csv file should have atleast 'Name','BPart' and 'Description' columns
                locationfile = location_files_dict[vartype.name]
                dfloc = postpro.load_location_file(locationfile)
                print("about to read location file: " + locationfile)
                locations = [
                    postpro.Location(
                        r["Name"],
                        r["BPart"],
                        r["Description"],
                        r["time_window_exclusion_list"],
                        r["threshold_value"],
                    )
                    for i, r in dfloc.iterrows()
                ]

                # now get gate data
                gate_studies = None
                gate_locations = None
                if gate_file_dict is not None and gate_location_file_dict is not None:
                    gate_locationfile = gate_location_file_dict["GATE"]
                    df_gate_loc = postpro.load_location_file(
                        gate_locationfile, gate_data=True
                    )
                    print("df_gate_loc=" + str(df_gate_loc))
                    gate_locations = [
                        postpro.Location(r["Name"], r["BPart"], r["Description"])
                        for i, r in df_gate_loc.iterrows()
                    ]
                    print("gate_locations: " + str(gate_locations))
                    gate_studies = [
                        postpro.Study("Gate", gate_file_dict[name])
                        for name in gate_file_dict
                    ]

                # create list of postpro.Study objects, with observed Study followed by model Study objects
                obs_study = postpro.Study("Observed", observed_files_dict[vartype.name])
                model_studies = [
                    postpro.Study(name, study_files_dict[name])
                    for name in study_files_dict
                ]
                studies = [obs_study] + model_studies

                # now run the processes
                if use_dask:
                    print("using dask")
                    tasks = [
                        dask.delayed(build_and_save_plot)(
                            config_data,
                            studies,
                            location,
                            vartype,
                            write_html=write_html,
                            write_graphics=write_graphics,
                            gate_studies=gate_studies,
                            gate_locations=gate_locations,
                            gate_vartype=gate_vartype,
                            metrics_table_list=metrics_table_list,
                            dask_key_name=f"build_and_save::{location}:{vartype}",
                        )
                        for location in locations
                    ]
                    # tasks = [dask.delayed(build_and_save_plot)(config_data, studies, location, vartype,
                    #                                 write_html=True,write_graphics=False,
                    #                                 dask_key_name=f'build_and_save::{location}:{vartype}') for location in locations]
                    dask.compute(tasks)
                else:
                    print("not using dask")
                    for location in locations:
                        try:
                            build_and_save_plot(
                                config_data,
                                studies,
                                location,
                                vartype,
                                write_html=write_html,
                                write_graphics=write_graphics,
                                gate_studies=gate_studies,
                                gate_locations=gate_locations,
                                gate_vartype=gate_vartype,
                                metrics_table_list=metrics_table_list,
                            )
                        except Exception as e:
                            print('unable to create plots/metrics layout for '+str(location.name))

                merge_statistics_files(vartype, config_data)
    finally:
        if use_dask:
            cluster.stop_local_cluster()


def check_config_data(config_data):
    if (
        "process_vartype_dict" in config_data
        or "vartype_timewindow_dict" not in config_data
    ):
        print(
            """**********************************************************************************************************
        Config file error: process_vartype_dict should be replaced with vartype_timewindow_dict. YAML Example:
        vartype_timewindow_dict:\n  EC: qual_tw\n  FLOW: hydro_tw\n  STAGE: hydro_tw
        Exiting. Fix your config file before re-running processes, by removing process_vartype_dict, and/or adding vartype_timewindow_dict.
        **********************************************************************************************************"""
        )
        exit(0)

    # required_dicts_list = ['options_dict', 'location_files_dict', 'observed_files_dict', 'study_files_dict', 'postpro_model_dict', \
    #     'timewindow_dict', 'vartype_dict', 'vartype_timewindow_dict', 'dask_options_dict']
    # not_found_list = []
    # for d in required_dicts_list:
    #     if d not in config_data:
    #         not_found_list.append(d)
    # if len(not_found_list) > 0:
    #     print('**********************************************************************************************************')
    #     print("Config file error: the following dictionaries are missing from your file: ")
    #     for d in not_found_list:
    #         print(d)
    #     print('Exiting. Fix your config file before re-running processes.')
    #     print('**********************************************************************************************************')


def run_process(process_name, config_filename, use_dask):
    """
    process_name (str): should be 'model', 'observed', 'plots', or 'bars'
    config_filename (str): filename of config (json) file
    use_dask (boolean): if true, dask will be used
    """
    import csv

    config_data = None
    # This enables user to use a json or yaml file. One advantage of yaml is you can add comments.
    # You can use this web page to convert json to yaml: https://codebeautify.org/json-to-yaml
    if ".json" in config_filename:
        # Read Config file
        with open(config_filename) as f:
            config_data = json.load(f)
    elif ".yml" in config_filename or ".yaml" in config_filename:
        # using yaml instead
        with open(config_filename, "r") as stream:
            try:
                config_data = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
    else:
        print("error: config file must be .json, .yml, or .yaml")
        exit(0)

    print('config_data='+str(config_data))
    print('config filename='+config_filename)
    # check data in json or yaml file
    check_config_data(config_data)

    # Create cluster if using dask
    cluster = None
    c_link = None
    if use_dask:
        cluster = DaskCluster(config_data)
        cluster.start_local_cluster()
        print(c_link)
    c_link
    if process_name.lower() == "model":
        postpro_model(cluster, config_data, use_dask)
    elif process_name.lower() == "observed":
        postpro_observed(cluster, config_data, use_dask)
    elif process_name.lower() == "plots":
        postpro_plots(cluster, config_data, use_dask)
    elif process_name.lower() == "validation_bar_charts":
        postpro_validation_bar_charts(cluster, config_data)
    elif process_name.lower() == "heatmaps":
        postpro_heatmaps(cluster, config_data, use_dask)
    elif process_name.lower() == "copy_plot_files":
        postpro_copy_plot_files(cluster, config_data)
    else:
        print("Error in pydelmod.postpro: process_name unrecognized: " + process_name)


# if __name__ == "__main__":
#     main()
