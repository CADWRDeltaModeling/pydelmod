import numpy as np
import pandas as pd
import pyhecdss
import pydsm
from pydsm.functions import tsmath
import tkinter as tk
from tkinter import ttk
import tkinter.filedialog
import os


def get_dss_data(
    primary_pathname_part_dss_filename_dict,
    primary_pathname_part,
    primary_part_c_part_dict=None,
    primary_part_e_part_dict=None,
    primary_part_f_part_dict=None,
    daily_avg=True,
    filter_b_part_numeric=False,
):
    """
    Read each dss time series from specified b part, c part, e part, and filename, and return a
    dataframe containing all time series individual columns

    primary_pathname_part (str):
            The 'primary pathname part' is the pathname part for which we will be extracting one or more
        time series. For example:
        1) if we want data for a specific list of stations, then 'b_part' will
            be the primary pathname part. This is used for getting flow or ec data for specific stations. This
            should be used when we want one time series per b part.
        2) If we want all div-flow, seep-flow, or drain-flow data, then 'e_part' will be the primary
            pathname part. This is typically used when we want multiple time series per c part.
    primary_part_dss_filename_dict (dict): key=b part, value=DSS filename
    primary_part_c_part_dict (dict): key=primary part, value=c_part to use for filtering
    primary_part_e_part_dict (dict): key=primary part, value=e_part to use for filtering
    primary_part_f_part_dict (dict): key=primary part, value=f_part to use for filtering
    daily_avg (bool, optional): if true and data are not daily, only daily averaged data will be returned
    filter_b_part_numeric (bool, optional): if true, remove any columns for which b part in dss path header is not numeric
    """
    print("==============================================================")
    return_df = None
    for pp in primary_pathname_part_dss_filename_dict:
        dss_filename = primary_pathname_part_dss_filename_dict[pp]
        b_part = None
        c_part = None
        e_part = None
        f_part = None
        if primary_pathname_part is "b_part":
            b_part = pp
            c_part = (
                primary_part_c_part_dict[pp]
                if primary_part_c_part_dict is not None
                else None
            )
            e_part = (
                primary_part_e_part_dict[pp]
                if primary_part_e_part_dict is not None
                else None
            )
            f_part = (
                primary_part_f_part_dict[pp]
                if (
                    primary_part_f_part_dict is not None
                    and b_part in primary_part_f_part_dict
                )
                else None
            )
            print(
                "bcef="
                + str(b_part)
                + ","
                + str(c_part)
                + ","
                + str(e_part)
                + ","
                + str(f_part)
            )
        elif primary_pathname_part is "c_part":
            c_part = pp
        else:
            print("FATAL ERROR! Primary pathname part is not b_part or c_part!")
            exit(0)

        with pyhecdss.DSSFile(dss_filename) as d:
            catdf = d.read_catalog()
            dss_file_parts = dss_filename.split("/")
            dfilename = dss_file_parts[len(dss_file_parts) - 1]
            filtered_df = None
            if b_part is not None:
                filtered_df = (
                    filtered_df[(catdf.B == b_part)]
                    if filtered_df is not None
                    else catdf[(catdf.B == b_part)]
                )
            if c_part is not None:
                filtered_df = (
                    filtered_df[(catdf.C == c_part)]
                    if filtered_df is not None
                    else catdf[(catdf.C == c_part)]
                )
            if e_part is not None:
                filtered_df = (
                    filtered_df[(catdf.E == e_part)]
                    if filtered_df is not None
                    else catdf[(catdf.E == e_part)]
                )
            if f_part is not None:
                filtered_df = (
                    filtered_df[(catdf.F == f_part)]
                    if filtered_df is not None
                    else catdf[(catdf.F == f_part)]
                )
            if filter_b_part_numeric:
                filtered_df = filtered_df[(catdf.B.str.isnumeric())]
            path_list = d.get_pathnames(filtered_df)
            for p in path_list:
                df = None
                units = None
                ptype = None
                if d.parse_pathname_epart(p).startswith("IR-"):
                    df, units, ptype = d.read_its(p)
                else:
                    df, units, ptype = d.read_rts(p)
                inst_val = False
                if "inst" in ptype.lower():
                    inst_val = True
                time_interval_str = p.split("/")[5]

                #####################################################################################################
                # See http://msb-confluence/display/DSM2/DSM2+inputs+are+off+by+1DAY                                #
                # If a daily flow input time series is marked inst-val, DSM2 will apply each value to the end of    #
                # the day/beginning of the next day. Each value is applied 1 day later than it would be if it were  #
                # marked per-avg. Therefore, DSM2 input is being created by shifting the data back 1 day.           #
                # This means that to effectively convert the time series to per-aver, we must shift values          #
                # forward by one day.                                                                               #
                # 1) if time series is hourly or 15min (and not mtz stage for tidal energy calculation)             #
                # and we're daily averaging it, then no need to shift by 1 day after averaging.                     #
                # 2) if time series is daily, it's inst-val, then no need to average, but we need to convert        #
                # timestamps to periods and shift values ahead by 1 day. This is done automatically by pandas.      #
                #####################################################################################################
                if daily_avg and "1DAY" not in time_interval_str:
                    print("daily averaging")
                    # df = df.resample('D', inclusive='right').mean()
                    df = tsmath.per_aver(df, "1D")

                if isinstance(df.index, pd.core.indexes.datetimes.DatetimeIndex):
                    print(
                        "timeseries is inst-val, converting to per-aver. b_part, c_part="
                        + str(b_part)
                        + ","
                        + str(c_part)
                    )
                    df.index = df.index.to_period()
                # this shifting doesn't seem to be necessary. It is somehow taken care of automatically
                # if '1day' in time_interval_str.lower() and inst_val:
                #     print('shifting data for b_part='+b_part)
                #     df = df.shift(periods=1)

                # rename columns to b_part if b_part is the primary pathname
                if primary_pathname_part is "b_part":
                    df.columns = [pp]
                print("path=" + p)
                return_df = (
                    df
                    if return_df is None
                    else pd.merge(
                        return_df, df, how="left", left_index=True, right_index=True
                    )
                )
            d.close()
        print("==============================================================")

    return return_df


def process_gate_data(dss_filename, output_file, b_part, c_part):
    """
    Read delta cross-channel gate operation data
    Create daily time series indicating fraction of maximum gate opening (100% means both gates open all day).
    """
    with pyhecdss.DSSFile(dss_filename) as d:
        catdf = d.read_catalog()
        filtered_df = catdf[(catdf.B == b_part) & (catdf.C == c_part)]
        path_list = d.get_pathnames(filtered_df)
        for p in path_list:
            df = None
            units = None
            ptype = None

            if d.parse_pathname_epart(p).startswith("IR-"):
                df, units, ptype = d.read_its(p)
            else:
                df, units, ptype = d.read_rts(p)
            print("path=" + p)

            # resample to 1 minute, then fill forward (with last value)
            df_1min = df.resample("T", inclusive="right").ffill()
            # now find daily averages of one minute data
            df_daily_avg = df_1min.resample("D", inclusive="right").mean()
            df_daily_avg_half = df_daily_avg / 2.0
            df_daily_avg_half.to_csv(output_file)
        d.close()


def get_file_path(
    selected_files_dict,
    row_index,
    text_field_text_var,
    title,
    filetypes,
    initialdir,
    initialfile,
    file_not_folder=True,
):
    """
    Listener for button that user click to load a file or folder.
    brings up a file selector allowing user to select a file or folder.
    """
    file_path = None
    if file_not_folder:
        file_path = tk.filedialog.askopenfilename(
            title=title,
            filetypes=filetypes,
            initialdir=initialdir.get(),
            initialfile=initialfile,
        )
    else:
        # root.directory = tkFileDialog.askdirectory()
        file_path = tk.filedialog.askdirectory()

    text_field_text_var.set(file_path)
    initialdir.set(os.path.dirname(file_path))
    print("updating selected_files_dict: " + str(row_index) + "," + str(file_path))
    selected_files_dict.update({row_index: file_path})


def add_file_or_folder_specifier(
    root,
    selected_files_dict,
    row_index,
    title,
    filetypes,
    initialdir,
    initialfile,
    file_not_folder=True,
):
    """
    Adds a label and a button, which user clicks to specify file or folder.
    """
    text_field_text_var = tk.StringVar()
    text_field_object = tk.Label(root, textvariable=text_field_text_var)

    text_field_text_var.set(initialdir.get() + initialfile)
    button = ttk.Button(
        root,
        text=title,
        command=lambda: get_file_path(
            selected_files_dict,
            row_index,
            text_field_text_var,
            title,
            filetypes,
            initialdir,
            initialfile,
            file_not_folder=file_not_folder,
        ),
    )

    text_field_object.grid(row=row_index, column=0)
    button.grid(row=row_index, column=1)
    selected_files_dict.update({row_index: text_field_text_var.get()})


def submit_listener(root, proceed_booleanvar):
    """
    Called when Create ANN Inputs button clicked
    """
    proceed_booleanvar.set(True)
    root.destroy()


def create_ann_inputs():
    """
    1. Northern flow = Sum(Sac, Yolo, Moke, CSMR, Calaveras, -NBA)
    2. San Joaquin River flow (the model input time series)
    3. Exports: Sum(Banks, Jones, CCC plants(Rock Sl, Middle R (actually old river), Victoria))
    4. DCC gate operation as daily percentage
    5. Net Delta CU, daily (DIV+SEEP-DRAIN) for DCD and SMCD
    6. Tidal Energy: daily max-daily min
    7. SJR inflow salinity at vernalis, daily
    8. Sacramento River EC
    9. EC Output for various locations
    """
    output_folder = "D:/data/ucd_ann/dataProcessing/annInputFeb2023/"
    base_study_folder = "D:/delta/2022HistoricalUpdate/"
    model_input_folder = (
        base_study_folder + "dsm2_2022.01_historical_update/timeseries/"
    )
    model_output_folder = (
        base_study_folder + "dsm2_2022.01_historical_update/studies/historical/output/"
    )

    model_data_version = "2021"
    # if model_data_version == '2021':
    #     # for 2021 historical release
    #     hist_dss_file = model_input_folder + 'hist.dss'
    #     model_ec_file = model_output_folder + 'hist_v2022_01_EC.dss'
    #     gate_dss_file = model_input_folder + 'gates-v8.dss'
    #     dcd_dss_file = model_input_folder + 'DCD_hist_Lch5.dss'
    #     smcd_dss_file = model_input_folder + 'SMCD_hist.dss'
    # else:
    #     # 8.2.1 release and 2022 historical update use output locations in slightly different locations
    #     # for verification with previous versions of the model ec output ANN inputs, we should use 8.2.1
    #     # to crete the 2020-2021 dataset for the dashboard, and presumably

    #     model_ec_file = 'D:/delta/dsm2_v8.2.1_historical/studies_historical/hist19smcd/output/hist_v82_19smcd.dss'
    #     hist_dss_file = 'D:/delta/dsm2_v8.2.1_historical/timeseries2019/hist201912.dss'
    #     gate_dss_file = 'D:/delta/dsm2_v8.2.1_historical/timeseries2019/gates-v8-201912.dss'
    #     dcd_dss_file = 'D:/delta/dsm2_v8.2.1_historical/timeseries2019/delta_DCD_Sep2020_Lch5.dss'
    #     smcd_dss_file = 'D:/delta/dsm2_v8.2.1_historical/timeseries2019/SMCD_hist_Sep2020.dss'

    ##########################################
    # use dialog to get file paths from user #
    ##########################################
    dss_filetypes = (("dss files", "*.dss"), ("DSS Files", "*.*"))
    root = tk.Tk()
    root.geometry("700x350")
    root.title("Specify input dss files below")
    cd_dir = "//cnrastore-bdo/Delta_Mod/Share/DSM2/full_calibration_8_3/delta/dsm2v8.3/studies/"
    cwd = tk.StringVar()

    (
        cwd.set(model_input_folder)
        if model_data_version == "2021"
        else cwd.set("D:/delta/dsm2_v8.2.1_historical/timeseries2019/")
    )

    selected_files_dict = {}

    add_file_or_folder_specifier(
        root, selected_files_dict, 0, "boundary input", dss_filetypes, cwd, "hist.dss"
    )
    add_file_or_folder_specifier(
        root, selected_files_dict, 1, "gate input", dss_filetypes, cwd, "gates-v8.dss"
    )
    add_file_or_folder_specifier(
        root,
        selected_files_dict,
        2,
        "DCD input",
        dss_filetypes,
        cwd,
        "DCD_hist_Lch5.dss",
    )
    add_file_or_folder_specifier(
        root, selected_files_dict, 3, "SMCD input", dss_filetypes, cwd, "SMCD_hist.dss"
    )
    cwd.set(model_output_folder)
    add_file_or_folder_specifier(
        root,
        selected_files_dict,
        4,
        "EC model output",
        dss_filetypes,
        cwd,
        "hist_v2022_01_EC.dss",
    )
    cwd.set(output_folder)
    add_file_or_folder_specifier(
        root,
        selected_files_dict,
        5,
        "output folder",
        None,
        cwd,
        "",
        file_not_folder=False,
    )
    # Button for closing
    proceed_booleanvar = tk.BooleanVar(False)
    submit_button = ttk.Button(
        root,
        text="Create ANN Input",
        command=lambda: submit_listener(root, proceed_booleanvar),
    )
    submit_button.grid(row=7, column=0)
    # submit_button.pack(pady=20)
    cancel_button = ttk.Button(root, text="Cancel", command=root.destroy)
    cancel_button.grid(row=7, column=1)
    # cancel_button.pack(pady=20)
    root.mainloop()

    if proceed_booleanvar.get():
        hist_dss_file = selected_files_dict[0]
        gate_dss_file = selected_files_dict[1]
        dcd_dss_file = selected_files_dict[2]
        smcd_dss_file = selected_files_dict[3]
        model_ec_file = selected_files_dict[4]
        output_folder = selected_files_dict[5] + "/"
        print("hist_dss_file=" + hist_dss_file)
        print("gate_dss_file=" + gate_dss_file)
        print("dcd_dss_file=" + dcd_dss_file)
        print("smcd_dss_file=" + smcd_dss_file)
        print("model_ec_file=" + model_ec_file)
        print("output_folder=" + output_folder)

        #################
        # Northern Flow #
        #################
        print("northern flow: hist_dss_file=" + hist_dss_file)
        b_part_dss_filename_dict = {
            "RSAC155": hist_dss_file,
            "BYOLO040": hist_dss_file,
            "RMKL070": hist_dss_file,
            "RCSM075": hist_dss_file,
            "RCAL009": hist_dss_file,
            "SLBAR002": hist_dss_file,
        }
        # b_part_f_part_dict = {'RMKL070': 'DWR-DMS-201912'} # for 2021
        # df_northern_flow = get_dss_data(b_part_dss_filename_dict, 'b_part', primary_part_f_part_dict=b_part_f_part_dict)
        df_northern_flow = get_dss_data(b_part_dss_filename_dict, "b_part")
        print("northern flow columns=" + str(df_northern_flow.columns))
        df_northern_flow.fillna(0, inplace=True)
        df_northern_flow["northern_flow"] = (
            df_northern_flow["RSAC155"]
            + df_northern_flow["BYOLO040"]
            + df_northern_flow["RMKL070"]
            + df_northern_flow["RCSM075"]
            + df_northern_flow["RCAL009"]
            - df_northern_flow["SLBAR002"]
        )
        df_northern_flow.to_csv(output_folder + "/df_northern_flow.csv")

        #############
        # SJR Flow  #
        #############
        b_part_dss_filename_dict = {"RSAN112": hist_dss_file}
        b_part_c_part_dict = {"RSAN112": "FLOW"}
        df_sjr_flow = get_dss_data(
            b_part_dss_filename_dict, "b_part", b_part_c_part_dict
        )
        df_sjr_flow.to_csv(output_folder + "/df_sjr_flow.csv")

        ###############################################################################################
        # 3. Exports: Sum(Banks, Jones, CCC plants(Rock Sl, Middle R (actually old river), Victoria)) #
        ###############################################################################################
        b_part_dss_filename_dict = {
            "CHSWP003": hist_dss_file,
            "CHDMC004": hist_dss_file,
            "CHCCC006": hist_dss_file,
            "ROLD034": hist_dss_file,
            "CHVCT001": hist_dss_file,
        }
        df_exports_flow = get_dss_data(b_part_dss_filename_dict, "b_part")
        # df_exports_flow.fillna(0, inplace=True)
        df_exports_flow["exports"] = (
            df_exports_flow["CHSWP003"]
            + df_exports_flow["CHDMC004"]
            + df_exports_flow["CHCCC006"]
            + df_exports_flow["ROLD034"]
            + df_exports_flow["CHVCT001"]
        )
        df_exports_flow.to_csv(output_folder + "df_exports_flow.csv")

        #############################################
        # 4. DCC gate operation as daily percentage #
        #############################################
        b_part = "RSAC128"
        c_part = "POS"
        gate_output_file = output_folder + "dcc_gate_op.csv"
        process_gate_data(gate_dss_file, gate_output_file, b_part, c_part)

        ############################################################
        # 5. Net Delta CU, daily (DIV+SEEP-DRAIN) for DCD and SMCD #
        ############################################################
        div_seep_dcd_c_part_dss_filename_dict = {
            "DIV-FLOW": dcd_dss_file,
            "SEEP-FLOW": dcd_dss_file,
        }
        div_seep_smcd_c_part_dss_filename_dict = {
            "DIV-FLOW": smcd_dss_file,
            "SEEP-FLOW": smcd_dss_file,
        }
        drain_dcd_c_part_dss_filename_dict = {"DRAIN-FLOW": dcd_dss_file}
        drain_smcd_c_part_dss_filename_dict = {"DRAIN-FLOW": smcd_dss_file}

        df_div_seep_dcd = get_dss_data(
            div_seep_dcd_c_part_dss_filename_dict, "c_part", filter_b_part_numeric=True
        )
        df_div_seep_smcd = get_dss_data(
            div_seep_smcd_c_part_dss_filename_dict, "c_part", filter_b_part_numeric=True
        )
        df_drain_dcd = get_dss_data(
            drain_dcd_c_part_dss_filename_dict, "c_part", filter_b_part_numeric=True
        )
        df_drain_smcd = get_dss_data(
            drain_smcd_c_part_dss_filename_dict, "c_part", filter_b_part_numeric=True
        )

        df_div_seep_dcd["dcd_divseep_total"] = df_div_seep_dcd[
            df_div_seep_dcd.columns
        ].sum(axis=1)
        df_div_seep_smcd["smcd_divseep_total"] = df_div_seep_smcd[
            df_div_seep_smcd.columns
        ].sum(axis=1)

        df_drain_dcd["dcd_drain_total"] = df_drain_dcd[df_drain_dcd.columns].sum(axis=1)
        df_drain_smcd["smcd_drain_total"] = df_drain_smcd[df_drain_smcd.columns].sum(
            axis=1
        )

        # df_div_seep_dcd.to_csv('d:/temp/df_div_seep_dcd.csv')
        # df_div_seep_smcd.to_csv('d:/temp/df_div_seep_smcd.csv')
        # df_drain_dcd.to_csv('d:/temp/df_drain_dcd.csv')
        # df_drain_smcd.to_csv('d:/temp/df_drain_smcd.csv')

        cu_total_dcd = pd.merge(
            df_div_seep_dcd, df_drain_dcd, how="left", left_index=True, right_index=True
        )
        cu_total_smcd = pd.merge(
            df_div_seep_smcd,
            df_drain_smcd,
            how="left",
            left_index=True,
            right_index=True,
        )
        cu_total = pd.merge(
            cu_total_dcd, cu_total_smcd, how="left", left_index=True, right_index=True
        )

        cu_total["cu_total"] = (
            cu_total["dcd_divseep_total"]
            + cu_total["smcd_divseep_total"]
            - cu_total["dcd_drain_total"]
            - cu_total["smcd_drain_total"]
        )
        # now only save the grand total column to csv
        cu_total[["cu_total"]].to_csv(output_folder + "df_cu_total.csv")

        ########################################
        # 6. Tidal Energy: daily max-daily min #
        ########################################
        b_part_dss_filename_dict = {"RSAC054": hist_dss_file}
        b_part_c_part_dict = {"RSAC054": "STAGE"}
        df_mtz_stage = get_dss_data(
            b_part_dss_filename_dict,
            "b_part",
            primary_part_c_part_dict=b_part_c_part_dict,
            daily_avg=False,
        )
        df_mtz_daily_max = df_mtz_stage.resample("D", inclusive="right").max()
        df_mtz_daily_max.columns = ["max"]
        df_mtz_daily_min = df_mtz_stage.resample("D", inclusive="right").min()
        df_mtz_daily_min.columns = ["min"]

        df_mtz_tidal_energy = pd.merge(
            df_mtz_daily_max,
            df_mtz_daily_min,
            how="outer",
            left_index=True,
            right_index=True,
        )
        df_mtz_tidal_energy["tidal_energy"] = (
            df_mtz_tidal_energy["max"] - df_mtz_tidal_energy["min"]
        )
        df_mtz_stage.to_csv(output_folder + "df_mtz_stage.csv")
        df_mtz_tidal_energy.to_csv(output_folder + "df_mtz_tidal_energy.csv")

        #############################################
        # 7. SJR inflow salinity at vernalis, daily #
        #############################################
        b_part_dss_filename_dict = {"RSAN112": hist_dss_file}
        b_part_c_part_dict = {"RSAN112": "EC"}
        df_sjr_ec = get_dss_data(
            b_part_dss_filename_dict,
            "b_part",
            primary_part_c_part_dict=b_part_c_part_dict,
        )
        df_sjr_ec.to_csv(output_folder + "/df_sjr_ec.csv")

        ##########################
        # 8. Sacramento River EC #
        ##########################
        b_part_dss_filename_dict = {"RSAC139": hist_dss_file}
        b_part_c_part_dict = {"RSAC139": "EC"}
        df_sac_ec = get_dss_data(
            b_part_dss_filename_dict,
            "b_part",
            primary_part_c_part_dict=b_part_c_part_dict,
        )
        df_sac_ec.to_csv(output_folder + "/df_sac_ec.csv")

        ######################################
        # 9. EC Output for various locations #
        ######################################
        b_part_dss_filename_dict = {
            "CHDMC006": model_ec_file,
            "CHSWP003": model_ec_file,
            "CHVCT000": model_ec_file,
            "OLD_MID": model_ec_file,
            "ROLD024": model_ec_file,
            "ROLD059": model_ec_file,
            "RSAC064": model_ec_file,
            "RSAC075": model_ec_file,
            "RSAC081": model_ec_file,
            "RSAC092": model_ec_file,
            "RSAC101": model_ec_file,
            "RSAN007": model_ec_file,
            "RSAN018": model_ec_file,
            "RSAN032": model_ec_file,
            "RSAN037": model_ec_file,
            "RSAN058": model_ec_file,
            "RSAN072": model_ec_file,
            "RSMKL008": model_ec_file,
            "SLCBN002": model_ec_file,
            "SLDUT007": model_ec_file,
            "SLMZU011": model_ec_file,
            "SLMZU025": model_ec_file,
            "SLSUS012": model_ec_file,
            "SLTRM004": model_ec_file,
            "SSS": model_ec_file,
            "RSAC054": hist_dss_file,
        }
        b_part_c_part_dict = {
            "CHDMC006": "EC",
            "CHSWP003": "EC",
            "CHVCT000": "EC",
            "OLD_MID": "EC",
            "ROLD024": "EC",
            "ROLD059": "EC",
            "RSAC064": "EC",
            "RSAC075": "EC",
            "RSAC081": "EC",
            "RSAC092": "EC",
            "RSAC101": "EC",
            "RSAN007": "EC",
            "RSAN018": "EC",
            "RSAN032": "EC",
            "RSAN037": "EC",
            "RSAN058": "EC",
            "RSAN072": "EC",
            "RSMKL008": "EC",
            "SLCBN002": "EC",
            "SLDUT007": "EC",
            "SLMZU011": "EC",
            "SLMZU025": "EC",
            "SLSUS012": "EC",
            "SLTRM004": "EC",
            "SSS": "EC",
            "RSAC054": "EC",
        }
        b_part_e_part_dict = {
            "CHDMC006": "15MIN",
            "CHSWP003": "15MIN",
            "CHVCT000": "15MIN",
            "OLD_MID": "15MIN",
            "ROLD024": "15MIN",
            "ROLD059": "15MIN",
            "RSAC064": "15MIN",
            "RSAC075": "15MIN",
            "RSAC081": "15MIN",
            "RSAC092": "15MIN",
            "RSAC101": "15MIN",
            "RSAN007": "15MIN",
            "RSAN018": "15MIN",
            "RSAN032": "15MIN",
            "RSAN037": "15MIN",
            "RSAN058": "15MIN",
            "RSAN072": "15MIN",
            "RSMKL008": "15MIN",
            "SLCBN002": "15MIN",
            "SLDUT007": "15MIN",
            "SLMZU011": "15MIN",
            "SLMZU025": "15MIN",
            "SLSUS012": "15MIN",
            "SLTRM004": "15MIN",
            "SSS": "15MIN",
            "RSAC054": "1HOUR",
        }
        df_model_ec = get_dss_data(
            b_part_dss_filename_dict,
            "b_part",
            primary_part_c_part_dict=b_part_c_part_dict,
            primary_part_e_part_dict=b_part_e_part_dict,
        )
        # df_model_ec = df_model_ec.resample('D').mean()

        # now add duplicate columns
        duplication_dict = {
            "RSAN007": "Antioch_dup",
            "CHSWP003": "CCFB_Intake_dup",
            "RSAC081": "Collinsville_dup",
            "CHDMC006": "CVP_Intake_dup",
            "RSAC092": "Emmaton_dup",
            "RSAN018": "Jersey_Point_dup",
            "RSAC075": "Mallard_Island_dup",
        }
        for rki in duplication_dict:
            new_name = duplication_dict[rki]
            df_model_ec[new_name] = df_model_ec[rki]

        # print('before error: columns='+str(df_model_ec.columns))
        # now add model output ec near CCC intakes
        b_part_dss_filename_dict = {"ROLD034": model_ec_file, "SLRCK005": model_ec_file}
        b_part_c_part_dict = {"ROLD034": "EC", "SLRCK005": "EC"}
        df_model_ec_2 = get_dss_data(
            b_part_dss_filename_dict, "b_part", b_part_c_part_dict
        )
        df_model_ec = pd.merge(
            df_model_ec, df_model_ec_2, how="outer", left_index=True, right_index=True
        )
        # print('before error: columns='+str(df_model_ec.columns))

        # now add a copy of victoria intake ec
        df_model_ec["CHVCT000_dup"] = df_model_ec["CHVCT000"]
        # now add another copy of Mtz ec
        df_model_ec["Martinez_input"] = df_model_ec["RSAC054"]

        # now rename some of the columns
        col_rename_dict = {
            "CHDMC006": "CHDMC006-CVP INTAKE",
            "CHSWP003": "CHSWP003-CCFB_INTAKE",
            "CHVCT000": "CHVCT000-VICTORIA INTAKE",
            "OLD_MID": "OLD_MID-OLD RIVER NEAR MIDDLE RIVER",
            "ROLD024": "ROLD024-OLD RIVER AT BACON ISLAND",
            "ROLD059": "ROLD059-OLD RIVER AT TRACY BLVD",
            "RSAC064": "RSAC064-SACRAMENTO R AT PORT CHICAGO",
            "RSAC075": "RSAC075-MALLARDISLAND",
            "RSAC081": "RSAC081-COLLINSVILLE",
            "RSAC092": "RSAC092-EMMATON",
            "RSAC101": "RSAC101-SACRAMENTO R AT RIO VISTA",
            "RSAN007": "RSAN007-ANTIOCH",
            "RSAN018": "RSAN018-JERSEYPOINT",
            "RSAN032": "RSAN032-SACRAMENTO R AT SAN ANDREAS LANDING",
            "RSAN037": "RSAN037-SAN JOAQUIN R AT PRISONERS POINT",
            "RSAN058": "RSAN058-ROUGH AND READY ISLAND",
            "RSAN072": "RSAN072-SAN JOAQUIN R AT BRANDT BRIDGE",
            "RSMKL008": "RSMKL008-S FORK MOKELUMNE AT TERMINOUS",
            "SLCBN002": "SLCBN002-CHADBOURNE SLOUGH NR SUNRISE DUCK CLUB",
            "SLDUT007": "SLDUT007-DUTCH SLOUGH",
            "SLMZU011": "SLMZU011-MONTEZUMA SL AT BELDONS LANDING",
            "SLMZU025": "SLMZU025-MONTEZUMA SL AT NATIONAL STEEL",
            "SLSUS012": "SLSUS012-SUISUN SL NEAR VOLANTI SL",
            "SLTRM004": "SLTRM004-THREE MILE SLOUGH NR SAN JOAQUIN R",
            "SSS": "SSS-STEAMBOAT SL",
            "ROLD034": "Old_River_Hwy_4",
            "SLRCK005": "CCWD_Rock",
            "CHVCT000_dup": "CCWD_Victoria_dup",
            "RSAC054": "Martinez_input_dup",
        }

        df_model_ec.rename(columns=col_rename_dict, inplace=True)
        df_model_ec.to_csv(output_folder + "/df_model_ec.csv")
        ########
        # Done #
        ########
        # test('chdmc.dss','.')
        # test_read_dicu(dcd_dss_file, '.')


def test(model_ec_file, output_folder):
    with pyhecdss.DSSFile(model_ec_file) as d:
        catdf = d.read_catalog()
        filtered_df = catdf[(catdf.B == "CHDMC006") & (catdf.C == "EC")]
        path_list = d.get_pathnames(filtered_df)
        for p in path_list:
            df = None
            units = None
            ptype = None
            df, units, ptype = d.read_rts(p)
            # convert per_aver to inst_val
            # if isinstance(df.index,pd.PeriodIndex):
            #     df.index=df.index.to_timestamp()
            df = tsmath.per_aver(df, "1D")
            df.to_csv(output_folder + "/test_chdmc006.csv")
        d.close()
