# Postpro-Model

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

# dask is a parallel processing library. Using it will save a lot of time, but error
# messages will not be as helpful, so set to False for debugging.
# Another option is to set scheduler (below) to single_threaded
# use_dask = dask_options_dict['use_dask']
# until we get dask working with DSS write processes, for now, override the value in the config file
use_dask=False

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
    #     dask.compute(tasks, scheduler='single-threaded')

c_link = None
if use_dask:
    cluster = DaskCluster()
    cluster.start_local_cluster()
    cluster.client
c_link

# Setup for EC, FLOW, STAGE
#import logging
#logging.basicConfig(filename='postpro-model.log', level=logging.DEBUG)
try:
    for var_name in vartype_dict:
        vartype = postpro.VarType(var_name, vartype_dict[var_name])
        if process_vartype_dict[vartype.name]:
            print('processing model ' + vartype.name + ' data')
            for study_name in study_files_dict:
                dssfile=study_files_dict[study_name]
                locationfile = location_files_dict[vartype.name]
                units=vartype.units
                observed=False
                processors=postpro.build_processors(dssfile, locationfile, vartype.name, units, study_name, observed)
                print(f'Processing {vartype.name} for study: {study_name}')
                if use_dask:
                    run_all(processors)
                else:
                    for p in processors:
                       postpro.run_processor(p)
finally:
    # Always shut down the cluster when done.
    if use_dask:
        cluster.stop_local_cluster()
