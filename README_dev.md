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
6. Build the package: `python -m build`
7. Upload to PyPI: `python -m twine upload dist/*`

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
