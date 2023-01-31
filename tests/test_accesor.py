import numpy as np
import pytest
import xarray as xr

import yt_xarray  # noqa: F401
from yt_xarray.accessor import _xr_to_yt
from yt_xarray.utilities._utilities import (
    _get_test_coord,
    construct_ds_with_extra_dim,
    construct_minimal_ds,
)


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
    assert ds.yt.geometry == "geodetic"
    assert all([i in ds.yt._coord_list for i in ds.coords.keys()])

    ds = construct_minimal_ds(x_name="x", y_name="y", z_name="z")
    assert ds.yt.geometry == "cartesian"
    assert all([i in ds.yt._coord_list for i in ds.coords.keys()])


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


@pytest.mark.parametrize("use_callable", (True, False))
def test_load_grid(ds_xr, use_callable):

    flds = ["a_new_field_0", "a_new_field_1"]
    ds_yt = ds_xr.yt.load_grid(flds, use_callable=use_callable)
    assert ds_yt.coordinates.name == "internal_geographic"
    expected_field_list = [("stream", f) for f in flds]
    assert all([f in expected_field_list] for f in ds_yt.field_list)

    ds_yt = ds_xr.yt.load_grid(use_callable=use_callable)
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
    ds_yt = ds.yt.load_grid(use_callable=use_callable)
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
    ds_yt = ds.yt.load_grid(use_callable=use_callable, length_unit="km")
    assert ds_yt.coordinates.name == "cartesian"
    assert all([f in expected_field_list] for f in ds_yt.field_list)

    f = ds_yt.all_data()[("stream", flds[0])]
    assert len(f) == ds.data_vars[flds[0]].size


@pytest.mark.parametrize("use_callable", (True, False))
@pytest.mark.parametrize("coord_set", range(5))
def test_time_reduction(coord_set, use_callable):
    ds = construct_ds_with_extra_dim(coord_set)
    flds = list(ds.data_vars)

    with pytest.raises(
        NotImplementedError,
        match="Loading data with time as a dimension is not currently",
    ):
        _ = ds.yt.load_grid(flds, length_unit="km", use_callable=use_callable)

    ds_yt = ds.yt.load_grid(
        flds, length_unit="km", use_callable=use_callable, sel_dict={"time": 0}
    )
    f = ds_yt.all_data()[("stream", flds[0])]

    # figure out if this ds will be interpolated to cell centers or not so we
    # know the expected size of the output arrays
    sel = _xr_to_yt.Selection(
        ds,
        fields=flds,
        sel_dict={"time": 0},
        sel_dict_type="isel",
    )
    interpd, _, _ = sel.interp_validation(str(ds_yt.geometry))
    f_ = ds.data_vars[flds[0]].isel({"time": 0})
    if interpd:
        expected = np.prod([n - 1 for n in f_.shape])
    else:
        expected = f_.size
    assert len(f) == expected
    assert ds_yt.current_time == float(ds.time[0].values)


def test_coord_aliasing():
    clist = ("b1", "b2", "b3")
    coords = {c: _get_test_coord(c, 4) for c in clist}
    var_shape = tuple([len(c) for c in coords.values()])
    vals = np.random.random(var_shape)
    da = xr.DataArray(vals, coords=coords, dims=clist)
    fld = "test_field"
    ds = xr.Dataset(data_vars={fld: da})

    yt_xarray.known_coord_aliases["b1"] = "z"
    yt_xarray.known_coord_aliases["b2"] = "y"
    yt_xarray.known_coord_aliases["b3"] = "x"
    ds_yt = ds.yt.load_grid([fld], length_unit="km")
    f = ds_yt.all_data()[("stream", fld)]
    assert len(f) == ds.data_vars[fld].size


def test_geom_kwarg(ds_xr):
    # make sure we can specify the geometry
    flds = ["a_new_field_0", "a_new_field_1"]
    _ = ds_xr.yt.load_grid(fields=flds, geometry="cartesian")


def test_stretched_grid():
    ds = construct_minimal_ds(
        x_stretched=False,
        x_name="x",
        y_stretched=True,
        y_name="y",
        z_stretched=True,
        z_name="z",
    )

    with pytest.raises(NotImplementedError, match="Detected a stretched grid"):
        _ = ds.yt.load_grid(
            fields=[
                "test_field",
            ]
        )

    _ = ds.yt.load_grid(
        fields=[
            "test_field",
        ],
        use_callable=False,
    )
