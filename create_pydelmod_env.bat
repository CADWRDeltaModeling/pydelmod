@ECHO OFF
SETLOCAL
IF [%1]==[] (SET ENV_NAME=pydelmod) ELSE (SET ENV_NAME=%1)
ECHO Creating an environment %ENV_NAME% ...
CALL conda create -n %ENV_NAME%
ECHO Installing packages ...
CALL conda install -y -n %ENV_NAME% python=3.7.* jupyter netcdf4 psutil
CALL conda install -y -n %ENV_NAME% -c defaults -c cadwr-dms pydelmod pyhecdss
CALL conda install -y -n %ENV_NAME% -c defaults -c plotly plotly-orca
CALL conda install -y -n %ENV_NAME% -c defaults -c conda-forge jupyter_contrib_nbextensions

