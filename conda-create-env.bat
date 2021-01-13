rem conda install script
call conda create -y -n pydelmod
echo "created env pydelmod"
call conda install -y -n pydelmod python=3.7.*
call conda install -y -n pydelmod -c defaults scipy jupyter matplotlib shapely xarray gdal holoviews dask numba
call conda install -y -n pydelmod -c defaults -c cadwr-dms pyhecdss vtools3 pydsm
call conda install -y -n pydelmod -c defaults -c plotly plotly=3.*
call conda install -y -n pydelmod -c defaults -c conda-forge jupyter_contrib_nbextensions
call conda install -y -n pydelmod -c defaults networkx matplotlib hvplot
call conda install -y -n pydelmod -c defaults -c conda-forge ipywidgets ipyleaflet ipysheet
# versioneer install
call conda install -y -n pydelmod -c defaults - conda-forge versioneer