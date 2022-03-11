setlocal 
call conda activate dev_pydelmod
pydelmod node-map-flow-splits d:/delta/maps/v8.2-opendata/gisgridmapv8.2nodes/dsm2_nodes_8_2.shp d:\delta\DSM2v821\study_templates\historical\output\hydro_echo_hist_v821.inp
endlocal