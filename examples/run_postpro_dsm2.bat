setlocal 
call conda activate pydelmod
pydelmod exec-postpro-dsm2 observed postpro_config.json --dask
pydelmod exec-postpro-dsm2 model postpro_config.json --dask
pydelmod exec-postpro-dsm2 plots postpro_config.json --dask
endlocal