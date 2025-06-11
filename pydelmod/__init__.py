# -*- coding: utf-8 -*-

"""Top-level package for Delta Modeling Python Package."""

__author__ = """Kijin Nam"""
__email__ = 'knam@water.ca.gov'

__all__ = ['nbplot', 'utilities']

try:
    from . import _version
    __version__ = _version.__version__
except (ImportError, AttributeError):
    __version__ = '0.0.0+unknown'
