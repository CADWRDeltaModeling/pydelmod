#!/bin/bash
# This script helps build conda package with proper version extraction from git

# Get the git describe tag (e.g. v0.4.0)
export GIT_DESCRIBE_TAG=$(git describe --tags --abbrev=0 2>/dev/null || echo "")
export GIT_FULL_HASH=$(git rev-parse HEAD)
export SHORT_HASH=${GIT_FULL_HASH:0:7}

# Generate a date-based version component
export BUILD_DATE=$(date +"%Y%m%d")

# Always use a version that includes commit info
if [ -z "$GIT_DESCRIBE_TAG" ]; then
    # No tag exists, use only commit hash
    export VERSION="0.0.0.dev${BUILD_DATE}+${SHORT_HASH}"
    echo "No git tag found, using version: $VERSION"
else
    # Remove 'v' prefix from tag for conda versioning
    if [[ $GIT_DESCRIBE_TAG == v* ]]; then
        export GIT_DESCRIBE_TAG="${GIT_DESCRIBE_TAG:1}"
    fi
    
    # Use tag + commit hash
    export VERSION="${GIT_DESCRIBE_TAG}.dev${BUILD_DATE}+${SHORT_HASH}"
    echo "Using version: $VERSION"
fi

# Export the VERSION to environment for conda-build to use
export VERSION

# Build the conda package
conda build conda.recipe/ "$@"
