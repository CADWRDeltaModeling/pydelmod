#!/bin/bash
# This script helps build conda package with proper version extraction from git

# Get the git describe tag (e.g. v0.4.0)
export GIT_DESCRIBE_TAG=$(git describe --tags --abbrev=0 2>/dev/null || echo "")

# If no tag exists, use the commit hash
if [ -z "$GIT_DESCRIBE_TAG" ]; then
    export GIT_FULL_HASH=$(git rev-parse HEAD)
    echo "No git tag found, using commit hash: ${GIT_FULL_HASH:0:7}"
else
    echo "Using git tag: $GIT_DESCRIBE_TAG"
    # Remove 'v' prefix from tag for conda versioning
    if [[ $GIT_DESCRIBE_TAG == v* ]]; then
        export GIT_DESCRIBE_TAG="${GIT_DESCRIBE_TAG:1}"
        echo "Modified git tag for conda: $GIT_DESCRIBE_TAG"
    fi
fi

# Build the conda package
conda build conda.recipe/ "$@"
