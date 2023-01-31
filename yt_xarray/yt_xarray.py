from functools import wraps

import xarray as xr

from yt_xarray.utilities._utilities import _validate_file


@wraps(xr.open_dataset)
@_validate_file
def open_dataset(filename, *args, **kwargs):
    """
    A direct wrapper of xarray.open_dataset with filename validation
    """
    return xr.open_dataset(filename, *args, **kwargs)
