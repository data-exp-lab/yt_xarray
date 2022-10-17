import numpy as np
import pytest
import xarray as xr

from yt_xarray._utilities import (
    _get_test_coord,
    construct_ds_with_time,
    construct_minimal_ds,
)
from yt_xarray.accessor._xr_to_yt import Selection, _coord_aliases, known_coord_aliases


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


c_m_ds_kwargs = {"latitude": "y_name", "longitude": "x_name", "altitude": "z_name"}


@pytest.mark.parametrize("coord", ("latitude", "longitude", "altitude"))
def test_selection_aliases(coord):

    for othername in _coord_aliases[coord]:

        kwargs = {c_m_ds_kwargs[coord]: othername}
        ds = construct_minimal_ds(**kwargs)
        fields = list(ds.data_vars)
        sel = Selection(ds, fields)
        assert np.all(sel.starting_indices == np.array((0, 0, 0)))
        assert sel.selected_shape == ds.data_vars[fields[0]].shape

        if othername not in ("latitude", "longitude", "altitude"):
            # only check the non-yt names
            assert othername not in sel.yt_coord_names
            assert known_coord_aliases[othername] in sel.yt_coord_names


def _isel_tester(ds_xr, sel, fields, coord, start_index):
    dim_id = ds_xr.data_vars[fields[0]].dims.index(coord)
    expected = np.array((0, 0, 0))
    expected[dim_id] = start_index
    assert np.all(sel.starting_indices == expected)
    assert sel.full_bbox[dim_id][0] < sel.selected_bbox[dim_id][0]
    assert sel.full_shape[dim_id] > sel.selected_shape[dim_id]

    expected = ds_xr.coords[coord].isel({coord: start_index}).values
    assert sel.selected_bbox[dim_id][0] == expected
    assert coord in sel.full_coords

    for idim in range(3):
        if idim != dim_id:
            assert sel.full_bbox[idim][0] == sel.selected_bbox[idim][0]
            assert sel.full_shape[idim] == sel.selected_shape[idim]


@pytest.mark.parametrize("coord", ("latitude", "longitude", "depth"))
def test_selection_isel(ds_xr, coord):

    fields = list(ds_xr.data_vars)

    sel_dict = {coord: slice(1, len(ds_xr.coords[coord]))}
    sel_dict_type = "isel"
    sel = Selection(ds_xr, fields, sel_dict=sel_dict, sel_dict_type=sel_dict_type)
    _isel_tester(ds_xr, sel, fields, coord, 1)

    sel_dict = {coord: [1, 2]}
    sel = Selection(ds_xr, fields, sel_dict=sel_dict, sel_dict_type=sel_dict_type)
    _isel_tester(ds_xr, sel, fields, coord, 1)
    dim_id = ds_xr.data_vars[fields[0]].dims.index(coord)
    assert sel.selected_shape[dim_id] == 2

    sel_dict = {coord: np.array([1, 2])}
    sel = Selection(ds_xr, fields, sel_dict=sel_dict, sel_dict_type=sel_dict_type)
    _isel_tester(ds_xr, sel, fields, coord, 1)
    assert sel.selected_shape[dim_id] == 2

    # check that selecting a single value reduces the dimensionality
    sel_dict = {coord: 1}
    sel = Selection(ds_xr, fields, sel_dict=sel_dict, sel_dict_type=sel_dict_type)
    assert len(sel.selected_shape) == 2
    assert coord not in sel.selected_coords


@pytest.mark.parametrize("coord", ("latitude", "longitude", "depth"))
def test_selection_sel(ds_xr, coord):
    fields = list(ds_xr.data_vars)

    search_for = ds_xr.coords[coord].values[1]

    sel_dict = {coord: slice(search_for, ds_xr.coords[coord].values.max())}
    sel_dict_type = "sel"
    sel = Selection(ds_xr, fields, sel_dict=sel_dict, sel_dict_type=sel_dict_type)
    _isel_tester(ds_xr, sel, fields, coord, 1)

    search_for = ds_xr.coords[coord].isel({coord: [1, 2]}).values
    sel_dict = {coord: search_for}
    sel = Selection(ds_xr, fields, sel_dict=sel_dict, sel_dict_type=sel_dict_type)
    _isel_tester(ds_xr, sel, fields, coord, 1)
    dim_id = ds_xr.data_vars[fields[0]].dims.index(coord)
    assert sel.selected_shape[dim_id] == 2

    search_for = search_for.tolist()
    sel_dict = {coord: search_for}
    sel = Selection(ds_xr, fields, sel_dict=sel_dict, sel_dict_type=sel_dict_type)
    _isel_tester(ds_xr, sel, fields, coord, 1)
    assert sel.selected_shape[dim_id] == 2

    # check that selecting a single value reduces the dimensionality
    sel_dict = {coord: ds_xr.coords[coord].values[1]}
    sel = Selection(ds_xr, fields, sel_dict=sel_dict, sel_dict_type=sel_dict_type)
    assert len(sel.selected_shape) == 2
    assert coord not in sel.selected_coords


@pytest.mark.parametrize("coord_set", range(5))
def test_time_reduction(coord_set):
    ds = construct_ds_with_time(coord_set)

    with pytest.raises(ValueError, match=r".* reduce dimensionality .*"):
        _ = Selection(ds, list(ds.data_vars))

    sel = Selection(ds, list(ds.data_vars), sel_dict={"time": 0})
    assert len(sel.selected_shape) == 3

    timetoselect = ds.time.isel({"time": 1}).values
    sel = Selection(
        ds, list(ds.data_vars), sel_dict={"time": timetoselect}, sel_dict_type="sel"
    )
    assert len(sel.selected_shape) == 3


def test_coord_aliasing():
    clist = ("c1", "c2", "c3")

    coords = {c: _get_test_coord(c, 4) for c in clist}
    var_shape = tuple([len(c) for c in coords.values()])
    vals = np.random.random(var_shape)
    da = xr.DataArray(vals, coords=coords, dims=clist)
    fld = "test_field"
    ds = xr.Dataset(data_vars={fld: da})

    known_coord_aliases["c1"] = "x"
    known_coord_aliases["c2"] = "y"
    known_coord_aliases["c3"] = "z"

    sel = Selection(
        ds,
        [
            fld,
        ],
    )
    for c in clist:
        assert c not in sel.yt_coord_names
    for c in "xyz":
        assert c in sel.yt_coord_names
