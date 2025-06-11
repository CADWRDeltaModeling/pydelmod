#!/bin/bash
# This script helps build conda package with proper version extraction from git

# Get git information for versioning
# Get full git describe output (tag-commits-hash) if on an untagged commit
export GIT_DESCRIBE=$(git describe --tags 2>/dev/null || echo "")

# Get just the most recent tag
export GIT_DESCRIBE_TAG=$(git describe --tags --abbrev=0 2>/dev/null || echo "")

# Get commit info
export GIT_FULL_HASH=$(git rev-parse HEAD)
export SHORT_HASH=${GIT_FULL_HASH:0:7}

# Generate a date-based version component
export BUILD_DATE=$(date +"%Y%m%d")

# Check if we're exactly on a tag
if git describe --exact-match --tags HEAD >/dev/null 2>&1; then
    # We are exactly on a tag, use just the tag version
    TAG_VERSION=$GIT_DESCRIBE_TAG

    # Remove 'v' prefix from tag for conda versioning
    if [[ $TAG_VERSION == v* ]]; then
        TAG_VERSION="${TAG_VERSION:1}"
    fi

    export VERSION="$TAG_VERSION"
    echo "On exact tag commit, using version: $VERSION"
else
    # Not on a tag, use tag-commits-hash format (convert to PEP 440 compatible version)
    if [ -z "$GIT_DESCRIBE" ]; then
        # No tag exists at all, fallback to dev version
        export VERSION="0.0.0.dev${BUILD_DATE}+${SHORT_HASH}"
        echo "No git tag found at all, using version: $VERSION"
    else
        # Process the git describe output to create PEP 440 compatible version
        # git describe format is typically: tag-N-gHASH (e.g., v1.2.3-5-g123abc)
        # Extract parts from git describe output
        TAG=$(echo $GIT_DESCRIBE | sed -E 's/^(v?[0-9]+\.[0-9]+\.[0-9]+)(-.*)?$/\1/')
        # Remove 'v' prefix from tag if present
        if [[ $TAG == v* ]]; then
            TAG="${TAG:1}"
        fi
        # Extract commit count since tag (N from tag-N-gHASH)
        COMMITS=$(echo $GIT_DESCRIBE | sed -E 's/^v?[0-9]+\.[0-9]+\.[0-9]+-([0-9]+)-.*/\1/' 2>/dev/null || echo "0")
        # Format as PEP 440 compatible version: tag.devN+hash
        export VERSION="${TAG}.dev${COMMITS}+${SHORT_HASH}"
        echo "Not on exact tag commit, using version: $VERSION"
    fi
fi

# Export the VERSION to environment for conda-build to use
export VERSION

# Build the conda package
conda build conda.recipe/ "$@"
