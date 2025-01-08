import pandas as pd
from pydelmod.dsm2_x2 import *

# First using:
# > pydelmod stations-output-file x2route_bay_sac_1.csv dsm2_v8_2_1_historical_centerline_chan_norest.geojson x2.inp
# x2route_bay_sac.csv is from https://github.com/CADWRDeltaModeling/BayDeltaSCHISM/blob/master/bdschism/bdschism/x2route_bay_sac.csv
# dsm2_v8_2_1_historical_centerline_chan_norest.geojson is from converting the dsm2 shapefiles to geojson
# this creates x2.inp which is used as a qual output file to run DSM2 with
# that needs to be modified to include the remaining headers/footers necessary for an output_channel .inp -> x2_ec_locs.inp
# then after running DSM2 with those specified output locations you can run the calc_x2_from_dss function

model_x2_ec_file = 'EC_X2.dss'
x2_csv_infile = 'x2route_bay_sac.csv'

x2_names = pd.read_csv(x2_csv_infile, comment='#')
x2_names = x2_names['distance'].tolist()[::100] # only use every 100 columns out of the ~1000 for demonstration purposes

print(f'calculating X2 from {model_x2_ec_file}')
x2 = calc_x2_from_dss(model_x2_ec_file, x2_names, epart_int='1HOUR')
print(x2.tail(10))