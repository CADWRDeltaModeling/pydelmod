setlocal
call conda activate dev_pydelmod
pydelmod exec-postpro-dsm2 observed postpro_cal_config.json --dask
REM pydelmod exec-postpro-dsm2 model postpro_cal_config.json --dask
REM pydelmod exec-postpro-dsm2 plots postpro_cal_config.json --dask
REM pydelmod exec-postpro-dsm2 heatmaps postpro_cal_config.json --no-dask

REM colored maps for manning's/dispersion/length/all values
REM If you specify a base file, the map will be colored by difference
REM pydelmod map-channels-colored ..\..\..\postprocessing\grid_shapefile\dsm2_channels_straightlines_8_3.shp ..\output\hydro_echo_run_merged_ns28_c14_w52.inp --colored-by "MANNING" --base-file ..\..\hist19smcd\output\hydro_echo_hist_v82_19smcd.inp
REM If you don't specify a base file, then the mannings value (not difference) is displayed
REM pydelmod map-channels-colored ..\..\..\postprocessing\grid_shapefile\dsm2_channels_straightlines_8_3.shp ..\output\hydro_echo_run_merged_ns28_c14_w52.inp --colored-by "MANNING"
REM 
REM pydelmod map-channels-colored ..\..\..\postprocessing\grid_shapefile\dsm2_channels_straightlines_8_3.shp ..\output\hydro_echo_run_merged_ns28_c14_w52.inp --colored-by "DISPERSION"
endlocal
