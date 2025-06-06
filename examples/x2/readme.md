# How to calculate X2 from DSM2

## Generate x2_ec_locs.inp

This step usees a .geojson file to geolocate the x2 stations against the DSM2 channel configurations. 

### Generate .geojson for your geometry

### Specify desired X2 stations

In this folder we use x2route_bay_sac.csv which comes from [BayDeltaSCHISM](https://github.com/CADWRDeltaModeling/BayDeltaSCHISM/blob/master/bdschism/bdschism/x2route_bay_sac.csv), but you can use a different route or increment if desired.

### Calculate

Run the following command in a terminal with an environment which has pydelmod installed:

> pydelmod stations_output_file x2route_bay_sac.csv dsm2_v8_2_1_historical_centerline_chan_norest.geojson x2.inp

### Modify output for proper headers/footers

Add this header:

> OUTPUT_CHANNEL
> NAME	CHAN_NO	DISTANCE	VARIABLE	INTERVAL	PERIOD_OP	FILE

And this footer:

> END

to x2.inp and save. 

## Setup DSM2 qual simulation

Change your qual_ec.inp to include **only** this line under "OUTPUT_TIME_SERIES":

> ./x2_ec_locs.inp                                                           # X2 profile along sac river

And make sure that the config points to that qual_ec.inp.

You'll need to run DSM2 qual with **only** these outputs, because they maximize the number of DSS outputs that are allowed by DSM2.

This produces a DSS file written to "QUALOUTFILE_EC" which is specified by your config.inp file.

## Calculate X2 from DSS Results

You can use this example script (.\calc_x2_from_dsm2.py) to calculate X2 from your DSM2 QUALOUTFILE_EC .DSS file.

That produces a variable "x2" which you can write to a csv file or handle however you'd like in Python.