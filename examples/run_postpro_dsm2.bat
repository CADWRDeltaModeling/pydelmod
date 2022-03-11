setlocal 
call conda activate pydelmod
pydelmod exec-postpro-dsm2 observed postpro_config.json --no-dask
pydelmod exec-postpro-dsm2 model postpro_config.json --no-dask
pydelmod exec-postpro-dsm2 plots postpro_config.json --no-dask
endlocal