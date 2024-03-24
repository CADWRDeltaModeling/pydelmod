import os
import pandas as pd
import dms_datastore
from dms_datastore import read_ts
import pyhecdss as dss
from pathlib import Path
import tqdm


# this should be a util function
def find_lastest_fname(pattern, dir="."):
    d = Path(dir)
    fname, mtime = None, 0
    for f in d.glob(pattern):
        fmtime = f.stat().st_mtime
        if fmtime > mtime:
            mtime = fmtime
            fname = f.absolute()
    return fname, mtime


def read_from_datastore_write_to_dss(
    datastore_dir, dssfile, param, repo_level="screened"
):
    """
    Reads datastore timeseries files and writes to a DSS file

    Parameters
    ----------
    datastore_dir : str
        Directory where Datastore files are stored
    repo_level : str
        default is screened
    dssfile : str
        Filename to write to
    param : str
        e.g one of "flow","elev", "ec", etc.
    """
    inventory_file, mtime = find_lastest_fname(
        f"inventory_datasets_{repo_level}*.csv", datastore_dir
    )
    print("Using inventory file:", inventory_file)
    inventory = pd.read_csv(inventory_file)
    param_inventory = inventory[inventory["param"] == param]
    apart = "DMS-DATASTORE"
    fpart = os.path.basename(inventory_file).split("_")[-1].split(".csv")[0]
    with dss.DSSFile(dssfile, create_new=True) as f:
        for idx, row in tqdm.tqdm(
            param_inventory.iterrows(), total=len(param_inventory)
        ):
            filepattern = os.path.join(datastore_dir, repo_level, row["filename"])
            ts = read_ts.read_ts(filepattern)
            print("Reading ", filepattern)
            if pd.isna(row["subloc"]):
                bpart = row["station_id"]
            else:
                bpart = row["station_id"] + row["subloc"]
            epart = ts.index.freqstr
            pathname = f'/{apart}/{bpart}/{row["param"]}///{fpart}/'
            print("Writing to ", pathname)
            f.write_rts(pathname, ts, row["unit"], "INST-VAL")
    print("Done")


def write_station_lat_lng(datastore_dir, station_file, param):
    """
    Writes station_id, latitude, longitude to a csv file

    Parameters
    ----------
    datastore_dir : str
        Directory where Datastore files are stored
    station_file : str
        Filename to write to
    param : str
        e.g one of "flow","elev", "ec", etc.
    """
    inventory_file, mtime = find_lastest_fname(
        f"inventory_datasets_screened*.csv", datastore_dir
    )
    print("Using inventory file:", inventory_file)
    inventory = pd.read_csv(inventory_file)
    inventory = inventory[inventory["param"] == param]
    inventory = inventory.drop_duplicates(subset=["station_id"])
    inventory = inventory[["station_id", "lat", "lon"]]
    inventory.to_csv(station_file, index=False)
    print("Wrote to ", station_file)
    print("Done")
