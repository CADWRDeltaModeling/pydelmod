setlocal 
call conda activate dev_pydelmod
pydelmod map-channels-colored d:\delta\maps\v8.2-opendata\gisgridmapv8.2channelcenterlines\dsm2_channels_centerlines_8_2.shp \\cnrastore-bdo\Delta_Mod\Share\DSM2\full_calibration_8_3\delta\dsm2v8.3\studies\run0_8_2_mann_disp\output\hydro_echo_run0_8_2_mann_disp.inp -c all -b \\cnrastore-bdo\Delta_Mod\Share\DSM2\full_calibration_8_3\delta\dsm2v8.3\studies\run2\output\hydro_echo_run2.inp
endlocal