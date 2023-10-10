import numpy as np
import pytest
import xarray as xr
import yt

import yt_xarray.accessor._xr_to_yt as xr2yt
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


c_m_ds_kwargs = {"latitude": "y_name", "longitude": "x_name", "altitude": "z_name"}


@pytest.mark.parametrize("coord", ("latitude", "longitude", "altitude"))
def test_selection_aliases(coord):

    for othername in xr2yt._coord_aliases[coord]:

        kwargs = {c_m_ds_kwargs[coord]: othername}
        ds = construct_minimal_ds(**kwargs)
        fields = list(ds.data_vars)
        sel = xr2yt.Selection(ds, fields)
        assert np.all(sel.starting_indices == np.array((0, 0, 0)))

        n_edges = ds.data_vars[fields[0]].shape

        assert np.all(sel.selected_shape == n_edges)

        if othername not in ("latitude", "longitude", "altitude"):
            # only check the non-yt names
            assert othername not in sel.yt_coord_names
            assert xr2yt.known_coord_aliases[othername] in sel.yt_coord_names


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
    sel = xr2yt.Selection(ds_xr, fields, sel_dict=sel_dict, sel_dict_type=sel_dict_type)
    _isel_tester(ds_xr, sel, fields, coord, 1)

    sel_dict = {coord: [1, 2]}
    sel = xr2yt.Selection(ds_xr, fields, sel_dict=sel_dict, sel_dict_type=sel_dict_type)
    _isel_tester(ds_xr, sel, fields, coord, 1)
    dim_id = ds_xr.data_vars[fields[0]].dims.index(coord)
    assert sel.selected_shape[dim_id] == 2

    sel_dict = {coord: np.array([1, 2])}
    sel = xr2yt.Selection(ds_xr, fields, sel_dict=sel_dict, sel_dict_type=sel_dict_type)
    _isel_tester(ds_xr, sel, fields, coord, 1)
    assert sel.selected_shape[dim_id] == 2

    # check that selecting a single value reduces the dimensionality
    sel_dict = {coord: 1}
    sel = xr2yt.Selection(ds_xr, fields, sel_dict=sel_dict, sel_dict_type=sel_dict_type)
    assert len(sel.selected_shape) == 2
    assert coord not in sel.selected_coords

    with pytest.raises(RuntimeError, match="sel_dict_type must be"):
        _ = xr2yt.Selection(ds_xr, fields, sel_dict=sel_dict, sel_dict_type="bad")


def test_selection_units():
    x = np.linspace(0, 1, 5)
    y = np.linspace(0, 1, 5)
    z = np.linspace(0, 1, 5)

    shp = (x.size, y.size, z.size)
    dvals = np.random.rand(*shp)
    cdict = {"x": x, "y": y, "z": z}
    dim = ("x", "y", "z")
    data = {
        "temp": xr.DataArray(dvals, coords=cdict, dims=dim, attrs={"units": "K"}),
        "flips": xr.DataArray(dvals, coords=cdict, dims=dim, attrs={"units": "smoots"}),
        "P": xr.DataArray(dvals, coords=cdict, dims=dim, attrs={"units": "MPa"}),
    }

    ds = xr.Dataset(data)
    ds_yt = ds.yt.load_grid(geometry="cartesian", length_unit="m")
    _ = ds_yt.field_list
    for fld in ("temp", "P"):
        finfo = ds_yt.field_info[("stream", fld)]
        assert finfo.units == ds.data_vars[fld].units

    assert ds_yt.field_info[("stream", "flips")].units == ""


def test_selection_errors(ds_xr):

    coord = "latitude"
    sel_dict = {coord: slice(1, len(ds_xr.coords[coord]))}
    sel_dict_type = "isel"
    with pytest.raises(ValueError, match="Please provide a list of fields"):
        _ = xr2yt.Selection(ds_xr, None, sel_dict=sel_dict, sel_dict_type=sel_dict_type)

    x = np.linspace(0, 1, 5)
    y = np.linspace(0, 1, 6)
    yy = np.linspace(0, 1, 6)
    z = np.linspace(0, 1, 4)

    shp = (x.size, y.size, z.size)

    T1 = xr.DataArray(
        np.random.rand(*shp), coords={"x": x, "y": y, "z": z}, dims=("x", "y", "z")
    )
    T2 = xr.DataArray(
        np.random.rand(*shp[:-1]), coords={"x": x, "y": y}, dims=("x", "y")
    )
    T3 = xr.DataArray(
        np.random.rand(*shp[:-1]), coords={"x": x, "yy": yy}, dims=("x", "yy")
    )
    data = {
        "T1": T1,
        "T2": T2,
        "T3": T3,
    }
    ds = xr.Dataset(data)
    flds = ["T1", "T2"]
    with pytest.raises(RuntimeError, match="does not match"):
        _ = xr2yt.Selection(ds, flds)

    flds = ["T2", "T3"]
    with pytest.raises(RuntimeError, match="coordinates : "):
        _ = xr2yt.Selection(ds, flds)


@pytest.mark.parametrize("coord", ("latitude", "longitude", "depth"))
def test_selection_sel(ds_xr, coord):
    fields = list(ds_xr.data_vars)

    search_for = ds_xr.coords[coord].values[1]

    sel_dict = {coord: slice(search_for, ds_xr.coords[coord].values.max())}
    sel_dict_type = "sel"
    sel = xr2yt.Selection(ds_xr, fields, sel_dict=sel_dict, sel_dict_type=sel_dict_type)
    _isel_tester(ds_xr, sel, fields, coord, 1)

    search_for = ds_xr.coords[coord].isel({coord: [1, 2]}).values
    sel_dict = {coord: search_for}
    sel = xr2yt.Selection(ds_xr, fields, sel_dict=sel_dict, sel_dict_type=sel_dict_type)
    _isel_tester(ds_xr, sel, fields, coord, 1)
    dim_id = ds_xr.data_vars[fields[0]].dims.index(coord)
    assert sel.selected_shape[dim_id] == 2

    search_for = search_for.tolist()
    sel_dict = {coord: search_for}
    sel = xr2yt.Selection(ds_xr, fields, sel_dict=sel_dict, sel_dict_type=sel_dict_type)
    _isel_tester(ds_xr, sel, fields, coord, 1)
    assert sel.selected_shape[dim_id] == 2

    # check that selecting a single value reduces the dimensionality
    sel_dict = {coord: ds_xr.coords[coord].values[1]}
    sel = xr2yt.Selection(ds_xr, fields, sel_dict=sel_dict, sel_dict_type=sel_dict_type)
    assert len(sel.selected_shape) == 2
    assert coord not in sel.selected_coords


@pytest.mark.parametrize("coord_set", range(5))
def test_time_reduction(coord_set):
    ds = construct_ds_with_extra_dim(coord_set)

    with pytest.raises(
        NotImplementedError,
        match="Loading data with time as a dimension is not currently",
    ):
        _ = xr2yt.Selection(ds, list(ds.data_vars))

    sel = xr2yt.Selection(ds, list(ds.data_vars), sel_dict={"time": 0})
    assert len(sel.selected_shape) == 3

    timetoselect = ds.time.isel({"time": 1}).values
    sel = xr2yt.Selection(
        ds, list(ds.data_vars), sel_dict={"time": timetoselect}, sel_dict_type="sel"
    )
    assert len(sel.selected_shape) == 3


def test_dimension_reduction():
    ds = construct_ds_with_extra_dim(0, dim_name="eta")
    with pytest.raises(ValueError, match="ndim is 4"):
        _ = xr2yt.Selection(ds, list(ds.data_vars))

    sel = xr2yt.Selection(ds, list(ds.data_vars), sel_dict={"eta": 0})
    assert len(sel.selected_shape) == 3


def test_coord_aliasing():
    clist = ("c1", "c2", "c3")

    coords = {c: _get_test_coord(c, 4) for c in clist}
    var_shape = tuple([len(c) for c in coords.values()])
    vals = np.random.random(var_shape)
    da = xr.DataArray(vals, coords=coords, dims=clist)
    fld = "test_field"
    ds = xr.Dataset(data_vars={fld: da})

    with pytest.raises(ValueError, match="not a known coordinate"):
        _ = xr2yt.Selection(
            ds,
            [
                fld,
            ],
        )

    xr2yt.known_coord_aliases["c1"] = "x"
    xr2yt.known_coord_aliases["c2"] = "y"
    xr2yt.known_coord_aliases["c3"] = "z"

    sel = xr2yt.Selection(
        ds,
        [
            fld,
        ],
    )
    for c in clist:
        assert c not in sel.yt_coord_names
    for c in "xyz":
        assert c in sel.yt_coord_names


@pytest.mark.parametrize("use_callable", [True, False])
def test_two_dimensional(use_callable):
    x = np.linspace(0, 1, 16)
    y = np.linspace(0, 1, 16)
    z = np.linspace(0, 1, 16)
    time = np.linspace(0, 1, 5)

    shp = (x.size, y.size, z.size)
    n_cells_xy = x.size * y.size

    data = {
        "temp": xr.DataArray(
            np.random.rand(*shp), coords={"x": x, "y": y, "z": z}, dims=("x", "y", "z")
        ),
        "precip": xr.DataArray(
            np.random.rand(*shp[:-1]), coords={"x": x, "y": y}, dims=("x", "y")
        ),
        "precip_t": xr.DataArray(
            np.random.rand(time.size, x.size, y.size),
            coords={"time": time, "x": x, "y": y},
            dims=("time", "x", "y"),
        ),
    }

    ds = xr.Dataset(data)
    yt_2d = ds.yt.load_grid(
        fields=[
            "precip",
        ],
        length_unit=1,
        geometry="cartesian",
        use_callable=use_callable,
    )

    ad = yt_2d.all_data()
    assert ad[("stream", "precip")].size == n_cells_xy

    slc = yt.SlicePlot(yt_2d, "z", ("stream", "precip"))
    slc.render()

    with pytest.raises(
        NotImplementedError,
        match="Loading data with time as a dimension is not currently",
    ):
        _ = ds.yt.load_grid(
            fields=[
                "precip_t",
            ],
            length_unit=1,
            geometry="cartesian",
            use_callable=use_callable,
        )

    yt_2d = ds.yt.load_grid(
        fields=[
            "precip_t",
        ],
        length_unit=1,
        geometry="cartesian",
        sel_dict={"time": 0},
        use_callable=use_callable,
    )
    ad = yt_2d.all_data()
    assert ad[("stream", "precip_t")].size == n_cells_xy

    yt_2d = ds.yt.load_grid(
        fields=[
            "precip_t",
        ],
        length_unit=1,
        geometry="cartesian",
        sel_dict={"time": time[1]},
        sel_dict_type="sel",
        use_callable=use_callable,
    )
    ad = yt_2d.all_data()
    assert ad[("stream", "precip_t")].size == n_cells_xy


_expected_geoms = {
    "cartesian": ("x", "y", "z"),
    "spherical": ("r", "theta", "phi"),
    "geographic": ("altitude", "latitude", "longitude"),
    "internal_geographic": ("depth", "latitude", "longitude"),
}


@pytest.mark.parametrize(
    "geometry", ["cartesian", "spherical", "geographic", "internal_geographic"]
)
def test_finding_3rd_dim(geometry):

    expected = _expected_geoms[geometry]
    # select any 2, make sure we add the 3rd back. repeat for every permutation
    choices = list(expected)
    for axis_pairs in ((0, 1), (0, 2), (1, 2)):
        axes = [choices[axis_pairs[0]], choices[axis_pairs[1]]]
        new_axes = xr2yt._add_3rd_axis_name(geometry, axes)
        assert len(set(new_axes).difference(set(expected))) == 0


@pytest.mark.parametrize(
    "geometry",
    ["cartesian", "spherical", "geographic", "internal_geographic", "geodetic"],
)
def test_geometry(geometry):
    assert xr2yt._validate_geometry(geometry) == geometry


def test_bad_gemoetry():
    with pytest.raises(ValueError, match="is not a valid geometry"):
        _ = xr2yt._validate_geometry("this_is_not_a_geom")


@pytest.mark.parametrize(
    "geometry, coord_list, expected",
    [
        ("cartesian", ["x", "y", "z"], "cartesian"),
        ("spherical", ["r", "theta", "phi"], "spherical"),
        ("geographic", ["r", "theta", "phi"], "geographic"),
        ("internal_geographic", ["r", "theta", "phi"], "internal_geographic"),
        ("geodetic", ["latitude", "longitude", "altitude"], "geographic"),
        ("geodetic", ["latitude", "longitude", "depth"], "internal_geographic"),
        ("blah", ["x", "y", "z"], "error"),
    ],
)
def test_determine_yt_geomtype(geometry, coord_list, expected):
    if expected == "error":
        with pytest.raises(ValueError, match="Unsupported geometry type"):
            _ = xr2yt._determine_yt_geomtype(geometry, coord_list)
    else:
        assert xr2yt._determine_yt_geomtype(geometry, coord_list) == expected


@pytest.mark.parametrize(
    "input_dim, expected_type",
    (
        (np.arange(0, 10), xr2yt._GridType.UNIFORM),
        (np.linspace(0, 10, 9), xr2yt._GridType.UNIFORM),
        (np.array([1.0, 2.0, 2.3, 3.5]), xr2yt._GridType.STRETCHED),
        (np.linspace(0, 10, 3) + np.array([0.0, 0.0, 1e-15]), xr2yt._GridType.UNIFORM),
        (np.linspace(0, 10, 3) + np.array([0.0, 0.0, 1e-7]), xr2yt._GridType.STRETCHED),
    ),
)
def test_grid_type(input_dim, expected_type):
    assert xr2yt._check_grid_stretchiness(input_dim) == expected_type


@pytest.mark.parametrize(
    "dim_name, dim_vals, expected",
    (
        ("x", np.array([1]), False),
        ("x", np.datetime64("2001-01-02").astype("datetime64[ns]"), True),
        ("TiMe", np.array([1]), True),
    ),
)
def test_time_check(dim_name, dim_vals, expected):
    assert xr2yt._check_for_time(dim_name, dim_vals) is expected


@pytest.mark.parametrize(
    "geometry, stretched, interp_required",
    [
        ("cartesian", False, False),
        ("cartesian", True, True),
        ("geographic", False, True),
        ("internal_geographic", False, True),
        ("not_a_geometry", False, True),
    ],
)
def test_selection_interp_validation(geometry, stretched, interp_required):

    if geometry == "cartesian":
        dim_names = ("x", "y", "z")
    elif geometry == "geographic":
        dim_names = ("longitude", "latitude", "altitude")
    elif geometry == "internal_geographic":
        dim_names = ("longitude", "latitude", "depth")
    else:
        dim_names = ("x", "y", "z")

    ds = construct_minimal_ds(
        x_stretched=stretched,
        x_name=dim_names[0],
        y_stretched=False,
        y_name=dim_names[1],
        z_stretched=False,
        z_name=dim_names[2],
    )

    fields = list(ds.data_vars)

    sel_info = xr2yt.Selection(
        ds,
        fields=fields,
    )

    interp_reqd_actual, shp, bbox = sel_info.interp_validation(geometry)

    assert interp_reqd_actual == interp_required


@pytest.mark.parametrize(
    "yt_geom", ("cartesian", "spherical", "geographic", "internal_geographic")
)
def test_add_3rd_axis_name(yt_geom):

    # get full list, remove on and make sure we get it back
    expected = list(xr2yt._expected_yt_axes[yt_geom])
    actual = xr2yt._add_3rd_axis_name(yt_geom, expected[:-1])
    for dim in expected:
        assert dim in actual

    with pytest.raises(RuntimeError, match="This function should only"):
        _ = xr2yt._add_3rd_axis_name(yt_geom, expected)

    with pytest.raises(ValueError, match="Unsupported geometry type"):
        _ = xr2yt._add_3rd_axis_name("bad_geometry", expected[:-1])


def _get_pixelized_slice(yt_ds):
    slc = yt_ds.slice(
        yt_ds.coordinates.axis_id["depth"],
        yt_ds.domain_center[yt_ds.coordinates.axis_id["depth"]],
        center=yt_ds.domain_center,
    )
    vals = yt_ds.coordinates.pixelize(
        0,
        slc,
        ("stream", "test_field"),
        yt_ds.arr([1, 359, -89, 89], "code_length"),
        (400, 400),
    )
    return slc, vals


def _get_ds_for_reverse_tests(stretched, use_callable, chunksizes):
    ds = construct_minimal_ds(
        min_x=1,
        max_x=359,
        min_z=50,
        max_z=650,
        min_y=89,
        max_y=-89,
        n_x=50,
        n_y=100,
        n_z=30,
        z_stretched=stretched,
        npseed=True,
    )
    yt_ds = ds.yt.load_grid(use_callable=use_callable, chunksizes=chunksizes)
    return yt_ds


@pytest.mark.parametrize(
    "stretched,use_callable,chunksizes",
    [
        (True, False, None),
        (False, False, None),
        (False, True, None),
        (False, True, 20),
        (False, False, 20),
    ],
)
def test_reversed_axis(stretched, use_callable, chunksizes):
    # tests for when the incoming data is not positive-monotonic

    yt_ds = _get_ds_for_reverse_tests(stretched, use_callable, chunksizes)

    if stretched:
        grid_obj = yt_ds.index.grids[0]
        ax_id = yt_ds.coordinates.axis_id["latitude"]
        assert np.all(grid_obj.cell_widths[ax_id] > 0)

    slc, vals = _get_pixelized_slice(yt_ds)

    pdy_lats = slc._generate_container_field("pdy")
    assert np.all(pdy_lats > 0)
    assert np.all(np.isfinite(vals))
