import numpy as np
import pytest

import yt_xarray  # noqa: F401
from yt_xarray._utilities import construct_ds_with_time, construct_minimal_ds


@pytest.fixture()
def ds_xr():
    # a base xarray ds to be used in various places.
    tfield = "a_new_field"
    n_x = 3
    n_y = 4
    n_z = 5
    ds = construct_minimal_ds(
        field_name=tfield,
        n_fields=3,
        n_x=n_x,
        n_y=n_y,
        n_z=n_z,
        z_name="depth",
        coord_order=["z", "y", "x"],
    )
    return ds


def test_accessor():

    tfield = "a_new_field"
    n_x = 3
    n_y = 4
    n_z = 5
    ds = construct_minimal_ds(
        field_name=tfield, n_x=n_x, n_y=n_y, n_z=n_z, coord_order=["x", "y", "z"]
    )
    assert hasattr(ds, "yt")
    assert ds.yt.coord_type == "geodetic"
    assert all([i in ds.yt._coord_list for i in ds.coords.keys()])

    assert tfield in ds.yt.field_list
    assert all([i not in ds.yt.field_list for i in ds.yt._coord_list])

    x, y, z = ds.yt.get_field_coords(tfield)
    assert len(x) == n_x
    assert len(y) == n_y
    assert len(z) == n_z

    ds = construct_minimal_ds(x_name="x", y_name="y", z_name="z")
    assert ds.yt.coord_type == "cartesian"


def test_bbox():
    n_x = 3
    n_y = 4
    n_z = 5
    c_ranges = {"latitude": [-50, 50], "longitude": [10, 300], "depth": [10, 3400]}
    c_ordering = ("longitude", "latitude", "depth")
    ds = construct_minimal_ds(
        n_x=n_x,
        n_y=n_y,
        n_z=n_z,
        field_name="test_field",
        n_fields=3,
        coord_order=["x", "y", "z"],
        min_x=c_ranges[c_ordering[0]][0],
        max_x=c_ranges[c_ordering[0]][1],
        min_y=c_ranges[c_ordering[1]][0],
        max_y=c_ranges[c_ordering[1]][1],
        min_z=c_ranges[c_ordering[2]][0],
        max_z=c_ranges[c_ordering[2]][1],
        z_units="km",
        z_name="depth",
    )

    # normal bbox should return the input ranges
    bbox = ds.yt.get_bbox("test_field_0")
    expected = np.array([[c_ranges[c][0], c_ranges[c][1]] for c in c_ordering])
    assert np.all(expected == bbox)

    bbox = ds.yt.get_bbox("test_field_1")
    assert np.all(expected == bbox)


def test_load_uniform_grid(ds_xr):

    flds = ["a_new_field_0", "a_new_field_1"]
    ds_yt = ds_xr.yt.load_uniform_grid(flds)
    assert ds_yt.coordinates.name == "internal_geographic"
    expected_field_list = [("stream", f) for f in flds]
    assert all([f in expected_field_list] for f in ds_yt.field_list)

    ds_yt = ds_xr.yt.load_uniform_grid()  # should generate a ds with all fields
    flds = flds + [
        "a_new_field_2",
    ]
    expected_field_list = [("stream", f) for f in flds]
    assert all([f in expected_field_list] for f in ds_yt.field_list)

    tfield = "nice_field"
    n_x, n_y, n_z = (7, 5, 17)
    ds = construct_minimal_ds(
        field_name=tfield,
        n_fields=3,
        n_x=n_x,
        n_y=n_y,
        n_z=n_z,
        z_name="altitude",
        coord_order=["z", "y", "x"],
    )
    ds_yt = ds.yt.load_uniform_grid()
    assert ds_yt.coordinates.name == "geographic"
    assert all([f in expected_field_list] for f in ds_yt.field_list)

    ds = construct_minimal_ds(
        field_name=tfield,
        n_fields=3,
        n_x=n_x,
        n_y=n_y,
        n_z=n_z,
        z_name="z",
        x_name="x",
        y_name="y",
        coord_order=["z", "y", "x"],
    )
    flds = [
        tfield + "_0",
    ]
    ds_yt = ds.yt.load_uniform_grid(flds, length_unit="km")
    assert ds_yt.coordinates.name == "cartesian"
    assert all([f in expected_field_list] for f in ds_yt.field_list)


def test_load_grid_from_callable(ds_xr):
    ds = ds_xr.yt.load_grid_from_callable()
    flds = list(ds_xr.data_vars)
    for fld in flds:
        assert ("stream", fld) in ds.field_list

    f = ds.all_data()[flds[0]]
    assert len(f) == ds_xr.data_vars[flds[0]].size


def test_yt_ds_attr(ds_xr):
    ds = ds_xr.yt.ds()
    flds = list(ds_xr.data_vars)
    for fld in flds:
        assert ("stream", fld) in ds.field_list

    f = ds.all_data()[flds[0]]
    assert len(f) == ds_xr.data_vars[flds[0]].size


@pytest.mark.parametrize("coord_set", range(5))
def test_time_reduction(coord_set):
    ds = construct_ds_with_time(coord_set)
    flds = list(ds.data_vars)

    with pytest.raises(ValueError, match=r".* reduce dimensionality .*"):
        _ = ds.yt.load_uniform_grid(flds, length_unit="km")

    ds_yt = ds.yt.load_uniform_grid(flds, length_unit="km", sel_dict={"time": 0})
    f = ds_yt.all_data()[("stream", flds[0])]
    expected = ds.data_vars[flds[0]].isel({"time": 0}).values
    assert len(f) == expected.size
    assert ds_yt.current_time == float(ds.time[0].values)
