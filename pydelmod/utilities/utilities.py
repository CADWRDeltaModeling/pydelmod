# -*- conding: utf-8 -*-

import os
import re
import numpy as np
import pandas as pd
import netCDF4 as nc
import pyhecdss

__all__ = ['read_wateryear_types', 'read_regulations',
           'read_dss_to_df', 'generate_regulation_timeseries', ]

MONTHS = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5,
          'JUN': 6, 'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12}


def read_wateryear_types(fpath):
    """ Read a table containing water year types from a text file.
        The text file is the top part of http://cdec.water.ca.gov/reportapp/javareports?name=WSIHIST
        down to the last row of the first table (before min and others).

        Parameters
        ----------
        fpath: string-like
            a text file name to read

        Returns
        -------
        pandas.DataFrame
            A WY table. The column names are:
            'wy', 'sac_oct_mar', 'sac_apr_jul', 'sac_wysum', 'sac_index',
            'sac_yrtype', 'sjr_oct_mar', 'sjr_apr_jul', 'sjr_wysum',
            'sjr_index', 'sjr_yrtype'
    """
    df = pd.read_fwf(fpath, header=None,
                     names=['wy', 'sac_oct_mar', 'sac_apr_jul', 'sac_wysum',
                            'sac_index', 'sac_yrtype', 'sjr_oct_mar',
                            'sjr_apr_jul', 'sjr_wysum', 'sjr_index',
                            'sjr_yrtype'],
                     skiprows=14)
    return df


def read_regulations(fpath, df_wyt):
    """ Read regulations and create irregular time series DataFrame from them.

        Parameters
        ----------
        fpath: string-like
            a CSV file that contains regulations
        df_wyt: Pandas.DataFrame
            a wateryear information DataFrame. Water years and Sac water year
            type from this DataFrame is used to create a time series.
            The expected columns are: location, wyt, month, date, val.
            The first column, 'location,' is the location of the regulation.
            Station names like 'RSAN018' would be easy to use.
            The second column, 'wyt,' is the short name of Sac water year type
            like 'W', 'BN'. The third and fourth columns, 'month' and 'date,'
            are when the regulation starts. The five columns, 'val,' is a
            regulation value.
            Currently no conditional regulation is expected in the CSV.

        Returns
        -------
        pandas.DataFrame
            an irregular time series of regulations in DataFrame.
    """
    # Read a CSV file
    df = pd.read_csv(fpath, comment='#')
    # Filter out years without yeartype information
    df_wy = df_wyt[df_wyt['sac_yrtype'].notnull()]
    locs = df['location'].unique()
    yr_start = df_wy['wy'].min()
    yr_end = df_wy['wy'].max()
    dfs = []
    for loc in locs:
        timestamps = []
        regs = []
        for wy in range(yr_start, yr_end + 1):
            wyt = df_wyt[df_wyt['wy'] == wy]['sac_yrtype'].iloc[0]
            mask = (df['wyt'] == wyt) & (df['location'] == loc)
            for i, row in df[mask].iterrows():
                mo = row['month']
                yr = wy - 1 if mo >= 10 else wy
                t_start = pd.to_datetime(f"{yr:d}-{mo}-{row['date']}")
                val = row['val']
                timestamps.append(t_start)
                regs.append(val)
        dfs.append(pd.DataFrame(
            data={'value': regs, 'location': loc}, index=timestamps))
    return pd.concat(dfs)


def generate_regulation_timeseries(df_reg, df, freq=None):
    time = df['time'].unique()
    t_begin = time.min()
    t_end = time.max()
    if freq is None:
        raise NotImplementedError(
            'Auto interval detection not implemented yet')
    dfs = []
    for station in df_reg['location'].unique():
        df_reg_station = df_reg[df_reg['location'] == station]
        mask = (df_reg_station.index >= (t_begin - pd.Timedelta('365 days'))) & (
            df_reg_station.index <= (t_end + pd.Timedelta('365 days')))
        times = []
        values = []
        for i in range(df_reg_station[mask].shape[0] - 1):
            row_i = df_reg_station[mask].iloc[i]
            row_next = df_reg_station[mask].iloc[i + 1]
            r = pd.date_range(
                start=row_i.name, end=row_next.name, freq=freq, closed='left').to_list()
            v = np.full_like(r, row_i['value'])
            times.extend(r)
            values.extend(v)
        df_ec_reg_ts = pd.DataFrame(
            data={'time': times, 'value': values, 'station': station})
        mask = (df_ec_reg_ts['time'] >= t_begin) & (
            df_ec_reg_ts['time'] <= t_end)
        dfs.append(df_ec_reg_ts[mask])
    return pd.concat(dfs)


def read_dss_to_df(fpath, bparts_to_read=None,
                   cparts_to_read=None,
                   eparts_to_read=None,
                   start_date_str=None, end_date_str=None,
                   with_metadata=False):
    """
        Convert a DSS File into a dataframe.

        Parameters:
        -----------
        fpath : str
            path to the DSS File
        bparts_to_read: list, optional
            list of part B to read in.
            If it is none, all available paths are read.
        start_date_str : str, optional
            this string should be a date in the format '%Y%m%d'
            and should refer to the earliest date to fetch
            (see http://strftime.org/)
        end_date_str : str, optional
            this string should be a date in the format '%Y%m%d'
            and should refer to the last date to fetch
            (see http://strftime.org/)
        with_metadata: boolean, optional
            If true, add two columns for unit and time.

        Returns:
        --------
        pandas.DataFrame :
            This data frame will contain all data contained within
            the entire DSS file given as input.
    """
    pyhecdss.set_message_level(2)
    dssfile = pyhecdss.DSSFile(fpath)  # create DSSFile object
    paths = dssfile.get_pathnames()  # fetch all internal paths
    dfs = []
    for path in paths:
        parts = path.split('/')
        if bparts_to_read is not None and parts[2] not in bparts_to_read:
            continue
        if cparts_to_read is not None and parts[3] not in cparts_to_read:
            continue
        if eparts_to_read is not None and parts[5] not in eparts_to_read:
            continue
        data, cunits, ctype = dssfile.read_rts(path, start_date_str,
                                               end_date_str)
        try:
            data.index = data.index.to_timestamp()
        except:
            pass  # it is probably already a DateTimeIndex?
        data = pd.melt(data.reset_index(), id_vars=[
                       'index'], value_vars=[path], var_name='pathname')
        data.rename(columns={'index': 'time'}, inplace=True)
        if with_metadata:
            data['cunits'] = cunits
            data['ctype'] = ctype
        dfs.append(data)
    if dfs is None:
        raise ValueError('No timeseries is read')
    df = pd.concat(dfs)
    return df
