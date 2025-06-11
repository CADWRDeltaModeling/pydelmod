# Developer Guide for pydelmod

## Package Setup

This package uses modern Python packaging with `pyproject.toml` and `setuptools_scm` for version management.

### Installation for Development

To install the package in development mode:

```bash
pip install -e .
```

This will install the package in editable mode, allowing you to make changes to the code and see them reflected immediately without reinstalling.

### Version Management

The package uses `setuptools_scm` to automatically generate version numbers from git tags and commits. The version is written to `pydelmod/_version.py` during installation.

Version format:
- If you're on a tagged commit (e.g., v0.4.0), the version will be just that tag (e.g., "0.4.0")
- If you're on a commit after a tag, the version will include dev information and git hash (e.g., "0.4.1.dev17+gca7a018.d20250520")

### Creating a New Release

1. Make sure all tests pass: `pytest`
2. Update HISTORY.rst with the latest changes
3. Commit all changes
4. Tag the release: `git tag v0.x.x`
5. Push the tag: `git push --tags`
6. Create a new release on GitHub using the tag
   - This will trigger the GitHub Actions workflow to publish to PyPI
   - The conda package will be built and published by the push event

Alternatively, you can manually build and upload:

7. Build the package: `python -m build`
8. Upload to PyPI: `python -m twine upload dist/*`
9. Build conda package: `./conda_build.sh`

## Project Structure

- `pydelmod/` - Main package code
- `tests/` - Test files
- `examples/` - Example scripts and notebooks
- `docsrc/` - Documentation source files
- `docs/` - Generated documentation

## Running Tests

To run the tests:

```bash
pytest
```

## Building Documentation

To build the documentation:

```bash
cd docsrc
make html
```

The generated documentation will be in the `docs/` directory.

## Conda Environment

For development, you can use the provided conda environment:

```bash
conda env create -f environment_dev.yml
conda activate dev_pydelmod
```

## Building Conda Package

To build the conda package:

```bash
# Use the helper script to automatically extract version from git
./conda_build.sh

# Or build directly with conda build
conda build conda.recipe/
```

Note: The conda recipe uses git tags for versioning. If no tag exists, it will use the git commit hash.

## Continuous Integration

This package uses GitHub Actions for continuous integration:

1. `python-package-conda.yml` - Builds the conda package and uploads to Anaconda.org when triggered
2. `python-package-pip.yml` - Tests the package installation with pip across different Python versions
3. `python-publish.yml` - Publishes the package to PyPI when a new release is created

You can also manually trigger these workflows from the GitHub Actions tab in the repository.
