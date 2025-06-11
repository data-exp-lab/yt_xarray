"""Top-level package for yt_xarray."""

__author__ = """Chris Havlin"""
__email__ = "chris.havlin@gmail.com"
__version__ = "0.4.1"


# import the xarray accessor so it is registered with xarray

from .accessor import YtAccessor
from .accessor._xr_to_yt import known_coord_aliases, reset_coordinate_aliases
from .yt_xarray import open_dataset
