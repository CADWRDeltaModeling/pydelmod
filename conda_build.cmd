@echo off
REM This script helps build conda package with proper version extraction from git

REM Get the git describe tag (e.g. v0.4.0)
FOR /F "tokens=*" %%g IN ('git describe --tags --abbrev^=0 2^>nul') DO (
    SET GIT_DESCRIBE_TAG=%%g
)

REM Get the commit hash for all builds
FOR /F "tokens=*" %%h IN ('git rev-parse HEAD') DO (
    SET GIT_FULL_HASH=%%h
)
SET SHORT_HASH=%GIT_FULL_HASH:~0,7%

REM Generate a date-based version component (YYYYMMDD)
FOR /F "tokens=2-4 delims=/ " %%a IN ('date /t') DO (
    SET MONTH=%%a
    SET DAY=%%b
    SET YEAR=%%c
)
SET BUILD_DATE=%YEAR%%MONTH%%DAY%

REM Always use a version that includes commit info
IF "%GIT_DESCRIBE_TAG%"=="" (
    REM No tag exists, use only commit hash
    SET VERSION=0.0.0.dev%BUILD_DATE%+%SHORT_HASH%
    echo No git tag found, using version: %VERSION%
) ELSE (
    REM Remove 'v' prefix from tag for conda versioning
    IF "%GIT_DESCRIBE_TAG:~0,1%"=="v" (
        SET GIT_DESCRIBE_TAG=%GIT_DESCRIBE_TAG:~1%
    )
    
    REM Use tag + commit hash
    SET VERSION=%GIT_DESCRIBE_TAG%.dev%BUILD_DATE%+%SHORT_HASH%
    echo Using version: %VERSION%
)

REM Set the VERSION environment variable for conda-build to use
set VERSION=%VERSION%

REM Build the conda package
conda build conda.recipe/ %*
