@echo off
REM This script helps build conda package with proper version extraction from git

REM Get the git describe tag (e.g. v0.4.0)
FOR /F "tokens=*" %%g IN ('git describe --tags --abbrev^=0 2^>nul') DO (
    SET GIT_DESCRIBE_TAG=%%g
)

REM If no tag exists, use the commit hash
IF "%GIT_DESCRIBE_TAG%"=="" (
    FOR /F "tokens=*" %%h IN ('git rev-parse HEAD') DO (
        SET GIT_FULL_HASH=%%h
    )
    echo No git tag found, using commit hash: %GIT_FULL_HASH:~0,7%
) ELSE (
    echo Using git tag: %GIT_DESCRIBE_TAG%
    REM Remove 'v' prefix from tag for conda versioning
    IF "%GIT_DESCRIBE_TAG:~0,1%"=="v" (
        SET GIT_DESCRIBE_TAG=%GIT_DESCRIBE_TAG:~1%
        echo Modified git tag for conda: %GIT_DESCRIBE_TAG%
    )
)

REM Build the conda package
conda build conda.recipe/ %*
