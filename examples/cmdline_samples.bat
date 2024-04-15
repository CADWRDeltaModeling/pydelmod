:: For demo of command line commands
:: Calibration Plot UI for a calibration configuration file
pydelmod calib-plot-ui d:/temp/postprocessingTestFolder/studies/1.1.88.2_dsm2_83_calib/postprocessing/postpro_cal_config.yml
:: Show DSS UI for a DSS file
pydelmod show-dss-ui d:\temp\postprocessingTestFolder\studies\1.1.88.2_dsm2_83_calib\output\historical.dss
:: Show DSS UI for a DSS file with a location file to display map of locations
pydelmod show-dss-ui --location-file tests\elev_stations.csv d:\temp\dsm2_FC.2023.02\studies\run2_1\output\runFC2_1.dss
:: Show DSS UI for DCD file with a location file to display map of locations
:: https://github.com/OSGeo/PROJ/issues/2320. Had to disable SSL verification for PROJ
set PROJ_UNSAFE_SSL=True
pydelmod show-dss-ui --location-file examples\dsm2gis\dsm2_nodes_8_2.geojson d:\temp\dsm2_FC.2023.02\timeseries\dcd_dsm2_mss1.dss --location-id-column id
:: Show DSM2 Output UI for a multiple DSM2 echofiles
pydelmod show-dsm2-output-ui d:\temp\postprocessingTestFolder\studies\1.1.88.2_dsm2_83_calib\output\*echo*.inp
:: Show DSM2 Output UI for a DSM2 output file with map
pydelmod show-dsm2-output-ui d:\temp\postprocessingTestFolder\studies\1.1.88.2_dsm2_83_calib\output\hydro_echo_historical.inp --channel-shapefile examples\dsm2gis\dsm2_channels_centerlines_8_2.geojson
:: Show DSM2 Output UI for multiple DSM2 output file with map and location file
pydelmod show-dsm2-output-ui d:\temp\postprocessingTestFolder\studies\1.1.88.2_dsm2_83_calib\output\*echo*.inp --channel-shapefile examples\dsm2gis\dsm2_channels_centerlines_8_2.geojson
:: Show DSM2 Output UI for multiple DSM2 output files from two different runs with map and location file
pydelmod show-dsm2-output-ui d:\temp\postprocessingTestFolder\studies\1.1.88.2_dsm2_83_calib\output\*echo*.inp d:\temp\postprocessingTestFolder\studies\hist19smcd\output\*echo*.inp --channel-shapefile examples\dsm2gis\dsm2_channels_centerlines_8_2.geojson
:: Show metrics geo heatmap for a metrics file
pydelmod show-metrics-geo-heatmap d:\temp\postprocessingTestFolder\studies\1.1.88.2_dsm2_83_calib\postprocessing\plots_cal\1_summary_statistics_unmasked_all_EC.csv d:\temp\postprocessingTestFolder\postprocessing\location_info\calibration_ec_stations.csv
:: Show schism output UI for a SCHISM output file along with datastore data
pydelmod show-schism-output-ui --schism_dir d:/temp/schism_output_sample --repo_dir y:/repo/continuous/screened --inventory_file y:/repo/continuous/inventory_datasets_screened_20240326.csv --flux_xsect_file preprocessed/flow_station_xsects.yaml
:: Show tidefile UI for a tidefile. No map
pydelmod show-dsm2-tidefile-ui d:/dev/pydsm/tests/data/historical_v82_ec.h5
:: Show schism ui for calib plots
pydelmod schism-calib-plot-ui d:\dev\schimpy\tests\example_full_mss\input_compare.yaml