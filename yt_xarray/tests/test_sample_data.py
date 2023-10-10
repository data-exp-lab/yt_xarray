import pytest
import xarray as xr

from yt_xarray.sample_data import load_random_xr_data


def test_load_random_xr_data():

    fields = {
        "field0": ("x", "y"),
        "field1": ("x", "y", "z"),
        "field2": ("x", "t"),
        "field3": ("x", "y", "t"),
    }
    dims = {"x": (0, 1, 4), "y": (0, 2, 6), "z": (-1, 0.5, 5), "t": (1, 200, 7)}
    ds = load_random_xr_data(fields, dims)
    assert isinstance(ds, xr.Dataset)
    for field in fields.keys():
        assert hasattr(ds, field)

    ds = load_random_xr_data(fields, dims, length_unit="km")
    assert ds.attrs["geospatial_vertical_units"] == "km"


def test_load_random_xr_data_bad():

    fields = {"field0": ("x", "y")}
    dims = {"x": (0, 1, 4)}

    with pytest.raises(KeyError, match="is specified as a dimension for"):
        _ = load_random_xr_data(fields, dims)
