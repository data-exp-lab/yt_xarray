import xarray as xr

import yt_xarray  # noqa: F401
from yt_xarray._utilities import construct_minimal_ds


def test_construct_minimal_ds():
    ds = construct_minimal_ds()
    assert isinstance(ds, xr.Dataset)

    ds = construct_minimal_ds(n_fields=2)
    assert len(ds.variables) - len(ds.coords) == 2
