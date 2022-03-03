setlocal 
call conda activate dev_pydelmod
pydelmod map-channels-colored d:\delta\maps\v8.2-opendata\gisgridmapv8.2channelcenterlines\dsm2_channels_centerlines_8_2.shp d:\delta\DSM2v821\study_templates\historical\output\hydro_echo_hist_v821.inp -c all
endlocal