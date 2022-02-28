# Postpro-observed

import pydsm
from pydsm import postpro

# Dask related functions
# read config file
import json
config_filename = 'postpro_config.json'
f = open(config_filename)
data = json.load(f)
vartype_dict = data['vartype_dict']
process_vartype_dict = data['process_vartype_dict']
dask_options_dict = data['dask_options_dict']
location_files_dict = data['location_files_dict']
observed_files_dict = data['observed_files_dict']
study_files_dict = data['study_files_dict']
use_dask = dask_options_dict['use_dask']
use_dask = False #override what's in the config file for now

# dask is a parallel processing library. Using it will save a lot of time, but error
# messages will not be as helpful, so set to False for debugging.
# Another option is to set scheduler (below) to single_threaded
# use_dask = dask_options_dict['use_dask']

import dask
from dask.distributed import Client, LocalCluster

class DaskCluster:
    def __init__(self):
        self.client=None
    def start_local_cluster(self):
        cluster = LocalCluster(n_workers=dask_options_dict['n_workers'],
                               threads_per_worker=dask_options_dict['threads_per_worker'],
                               memory_limit=dask_options_dict['memory_limit']) # threads_per_worker=1 needed if using numba :(
        self.client = Client(cluster)
    def stop_local_cluster(self):
        self.client.shutdown()
        self.client=None

def run_all(processors):
    tasks=[dask.delayed(postpro.run_processor)(processor,dask_key_name=f'{processor.study.name}::{processor.location.name}/{processor.vartype.name}') for processor in processors]
    dask.compute(tasks)
    # to use only one processor. Also prints more helpful messages
#     dask.compute(tasks, scheduler='single_threaded')

c_link = None
if use_dask:
    cluster = DaskCluster()
    cluster.start_local_cluster()
    c_link = cluster.client
c_link

# Setup for EC, FLOW, STAGE
try:
    for vartype in vartype_dict:
        if process_vartype_dict[vartype]:
            print('processing observed ' + vartype + ' data')
            dssfile = observed_files_dict[vartype]
            location_file = location_files_dict[vartype]
            # dssfile = './observed_data/cdec/ec_merged.dss'
            # locationfile='./location_info/calibration_ec_stations.csv'
            # units='uS/cm'
            units = vartype_dict[vartype]
            study_name='Observed'
            observed=True
            processors=postpro.build_processors(dssfile, location_file, vartype, units, study_name, observed)
            if use_dask:
                run_all(processors)
            else:
                for p in processors:
                    postpro.run_processor(p)
finally:
    # Always shut down the cluster when done.
    if use_dask:
        cluster.stop_local_cluster()
