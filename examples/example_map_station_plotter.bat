setlocal 
call conda activate dev_pydelmod
pydelmod output-map-plotter d:\delta\maps\v8.2-opendata\gisgridmapv8.2channelstraightlines\dsm2_channels_straightlines_8_2.shp d:\delta\DSM2v821\study_templates\historical\output\hydro_echo_hist_v821.inp -v flow
endlocal