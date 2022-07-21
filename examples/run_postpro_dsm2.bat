setlocal
call conda activate dev_pydelmod
#pydelmod exec-postpro-dsm2 observed postpro_cal_config.json --dask
#pydelmod exec-postpro-dsm2 model postpro_cal_config.json --dask
#pydelmod exec-postpro-dsm2 plots postpro_cal_config.json --dask
pydelmod exec-postpro-dsm2 heatmaps postpro_cal_config.json --dask
endlocal
