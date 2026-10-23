from typing import Any

import numpy as np
import pytest
import xarray as xr
from numpy import typing as npt

import yt_xarray  # noqa: F401
from yt_xarray.utilities._grid_decomposition import (
    ChunkInfo,
    GridSpec,
    plan_stretched_grid_refinement,
    summarize_grid_plan,
)
from yt_xarray.utilities._utilities import construct_ds_with_extra_dim

pytest.importorskip("scipy")

_coefs = (2.0, 3.0, -1.0)


def _linear_field(*coords: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
    mesh = np.meshgrid(*coords, indexing="ij")
    field = np.zeros(mesh[0].shape, dtype=np.float64)
    for c, m in zip(_coefs, mesh):
        field += c * m
    return field


@pytest.fixture()
def ds_stretched():
    # a stretched cartesian grid: fine in the middle of x, fine at the end
    # of y, uniform in z. the field is linear so linear interpolation is exact.
    x = np.concatenate(
        [np.linspace(0, 1, 5), np.linspace(1.1, 2, 10), np.linspace(2.25, 4, 8)]
    )
    y = np.concatenate([np.linspace(0, 1, 3), np.linspace(1.125, 2, 8)])
    z = np.linspace(-1, 1, 5)
    f = _linear_field(x, y, z)
    da = xr.DataArray(f, coords={"x": x, "y": y, "z": z}, dims=("x", "y", "z"))
    return xr.Dataset({"f": da, "g": 2 * da})


_coef_by_name = dict(zip(("x", "y", "z"), _coefs))


def _check_linear(
    ds_yt: Any,
    field: str = "f",
    rtol: float = 1e-10,
    axis_names: tuple[str, ...] = ("x", "y", "z"),
) -> npt.NDArray[np.float64]:
    ad = ds_yt.all_data()
    fc = ad.fcoords.d
    expected = sum(_coef_by_name[name] * fc[:, i] for i, name in enumerate(axis_names))
    vals: npt.NDArray[np.float64] = ad[("stream", field)].d
    np.testing.assert_allclose(vals, expected, rtol=rtol, atol=1e-12)
    return vals


@pytest.mark.parametrize("use_callable", (True, False))
def test_single_grid_default_spacing(ds_stretched, use_callable):
    ds_yt = ds_stretched.yt.load_as_uniform_grid(
        length_unit="m", use_callable=use_callable
    )
    assert len(ds_yt.index.grids) == 1
    # default spacing is the minimum source spacing per dimension
    dds = ds_yt.index.grids[0].dds.d
    np.testing.assert_allclose(dds, [0.1, 0.125, 0.5])
    np.testing.assert_allclose(ds_yt.domain_left_edge.d, [0.0, 0.0, -1.0])
    np.testing.assert_allclose(ds_yt.domain_right_edge.d, [4.0, 2.0, 1.0])
    _check_linear(ds_yt)
    vals = _check_linear(ds_yt, field="f")
    g = ds_yt.all_data()[("stream", "g")].d
    np.testing.assert_allclose(g, 2 * vals)


def test_bbox_and_spacing_forms(ds_stretched):
    # dict forms with xarray names
    ds_yt = ds_stretched.yt.load_as_uniform_grid(
        length_unit="m",
        bbox={"x": (0.5, 3.5)},
        cell_spacing={"x": 0.3, "y": 0.2, "z": 0.5},
    )
    np.testing.assert_allclose(ds_yt.domain_left_edge.d, [0.5, 0.0, -1.0])
    np.testing.assert_allclose(ds_yt.domain_right_edge.d, [3.5, 2.0, 1.0])
    assert tuple(ds_yt.domain_dimensions) == (10, 10, 4)
    _check_linear(ds_yt)

    # array forms and a scalar spacing
    ds_yt = ds_stretched.yt.load_as_uniform_grid(
        length_unit="m",
        bbox=np.array([[0.0, 4.0], [0.0, 2.0], [-1.0, 1.0]]),
        cell_spacing=0.25,
    )
    assert tuple(ds_yt.domain_dimensions) == (16, 8, 8)
    _check_linear(ds_yt)

    # sequence spacing
    ds_yt = ds_stretched.yt.load_as_uniform_grid(
        length_unit="m", cell_spacing=(0.5, 0.5, 0.5)
    )
    assert tuple(ds_yt.domain_dimensions) == (8, 4, 4)


def test_bbox_rounds_up_to_spacing(ds_stretched):
    # 4 / 0.3 is not an integer: with pin_bbox=False the max edge is extended
    ds_yt = ds_stretched.yt.load_as_uniform_grid(
        length_unit="m", cell_spacing={"x": 0.3, "y": 0.5, "z": 0.5}, pin_bbox=False
    )
    assert ds_yt.domain_dimensions[0] == 14
    np.testing.assert_allclose(ds_yt.domain_right_edge.d[0], 14 * 0.3)
    np.testing.assert_allclose(ds_yt.index.grids[0].dds.d[0], 0.3)


def test_pin_bbox_decreases_spacing(ds_stretched):
    # 4 / 0.3 is not an integer: by default the bbox is kept and dx shrinks to 4 / 14
    ds_yt = ds_stretched.yt.load_as_uniform_grid(
        length_unit="m", cell_spacing={"x": 0.3, "y": 0.5, "z": 0.5}
    )
    assert tuple(ds_yt.domain_dimensions) == (14, 4, 4)
    np.testing.assert_allclose(ds_yt.domain_right_edge.d, [4.0, 2.0, 1.0])
    np.testing.assert_allclose(ds_yt.index.grids[0].dds.d, [4.0 / 14, 0.5, 0.5])

    # an explicit bbox is also pinned exactly
    bbox = np.array([[0.5, 3.5], [0.0, 2.0], [-1.0, 1.0]])
    ds_yt = ds_stretched.yt.load_as_uniform_grid(
        length_unit="m", bbox=bbox, cell_spacing=0.7
    )
    np.testing.assert_allclose(ds_yt.domain_left_edge.d, bbox[:, 0])
    np.testing.assert_allclose(ds_yt.domain_right_edge.d, bbox[:, 1])
    assert tuple(ds_yt.domain_dimensions) == (5, 3, 3)

    # when the width is already an integer multiple, the spacing is unchanged
    ds_yt = ds_stretched.yt.load_as_uniform_grid(length_unit="m", cell_spacing=0.5)
    assert tuple(ds_yt.domain_dimensions) == (8, 4, 4)
    np.testing.assert_allclose(ds_yt.index.grids[0].dds.d, 0.5)
    _check_linear(ds_yt)


@pytest.mark.parametrize("extra_kwargs", [{"chunksizes": 3}, {"refine_grid": True}])
def test_pin_bbox_multi_grid(ds_stretched, extra_kwargs: dict[str, Any]):
    ds_yt = ds_stretched.yt.load_as_uniform_grid(
        length_unit="m", cell_spacing=0.3, **extra_kwargs
    )
    assert len(ds_yt.index.grids) > 1
    np.testing.assert_allclose(ds_yt.domain_left_edge.d, [0.0, 0.0, -1.0])
    np.testing.assert_allclose(ds_yt.domain_right_edge.d, [4.0, 2.0, 1.0])
    assert tuple(ds_yt.domain_dimensions) == (14, 7, 7)
    expected_dds = np.array([4.0 / 14, 2.0 / 7, 2.0 / 7])
    for grid in ds_yt.index.grids:
        np.testing.assert_allclose(
            grid.dds.d, expected_dds / ds_yt.refine_by**grid.Level
        )
    np.testing.assert_allclose(
        ds_yt.index.grid_right_edge.d.max(axis=0), [4.0, 2.0, 1.0]
    )
    _check_linear(ds_yt)


def test_fill_value_outside_source(ds_stretched):
    bbox = np.array([[-1.0, 5.0], [0.0, 2.0], [-1.0, 1.0]])
    ds_yt = ds_stretched.yt.load_as_uniform_grid(
        length_unit="m", bbox=bbox, cell_spacing=0.5, fill_value=-999.0
    )
    ad = ds_yt.all_data()
    vals = ad[("stream", "f")].d
    x = ad.fcoords.d[:, 0]
    outside = (x < 0.0) | (x > 4.0)
    assert np.all(vals[outside] == -999.0)
    assert not np.any(vals[~outside] == -999.0)

    # default fill is nan
    ds_yt = ds_stretched.yt.load_as_uniform_grid(
        length_unit="m", bbox=bbox, cell_spacing=0.5
    )
    vals = ds_yt.all_data()[("stream", "f")].d
    assert np.all(np.isnan(vals[outside]))


@pytest.mark.parametrize("use_callable", (True, False))
@pytest.mark.parametrize("chunksizes", (5, (4, 4, 4), (100, 100, 100)))
def test_chunked(ds_stretched, use_callable, chunksizes):
    ds_yt = ds_stretched.yt.load_as_uniform_grid(
        length_unit="m", chunksizes=chunksizes, use_callable=use_callable
    )
    dims = np.array([40, 16, 4])
    chnk = np.broadcast_to(np.asarray(chunksizes), (3,))
    expected = ChunkInfo(tuple(dims), chnk).n_tots
    assert len(ds_yt.index.grids) == expected
    vals = _check_linear(ds_yt)
    assert vals.size == np.prod(dims)


def test_chunked_matches_single(ds_stretched):
    ds_single = ds_stretched.yt.load_as_uniform_grid(length_unit="m")
    ds_chunk = ds_stretched.yt.load_as_uniform_grid(length_unit="m", chunksizes=7)
    vals = []
    for dsyt in (ds_single, ds_chunk):
        cg = dsyt.covering_grid(0, dsyt.domain_left_edge, dsyt.domain_dimensions)
        vals.append(cg[("stream", "f")].d)
    np.testing.assert_allclose(vals[0], vals[1])


def test_nearest(ds_stretched):
    ds_yt = ds_stretched.yt.load_as_uniform_grid(
        length_unit="m", interp_method="nearest", cell_spacing=0.5
    )
    ad = ds_yt.all_data()
    vals = ad[("stream", "f")].d
    fc = ad.fcoords.d
    # every value must be an actual source node value
    src = ds_stretched["f"].values.ravel()
    assert np.all(np.isin(np.round(vals, 10), np.round(src, 10)))
    # and close to linear within the largest source spacing
    expected = sum(c * fc[:, i] for i, c in enumerate(_coefs))
    assert np.abs(vals - expected).max() < 3.0


@pytest.mark.parametrize("use_callable", (True, False))
def test_refined(ds_stretched, use_callable):
    ds_yt = ds_stretched.yt.load_as_uniform_grid(
        length_unit="m", refine_grid=True, use_callable=use_callable
    )
    assert ds_yt.index.max_level == 2
    levels = [g.Level for g in ds_yt.index.grids]
    assert 0 in levels and 1 in levels and 2 in levels
    _check_linear(ds_yt)

    # y sets the max level (spacing ratio of 4) and is resolved exactly at
    # the finest level, x (ratio 2.5) is over-resolved there
    finest = [g for g in ds_yt.index.grids if g.Level == 2]
    np.testing.assert_allclose(finest[0].dds.d[1], 0.125)
    assert finest[0].dds.d[0] <= 0.1
    np.testing.assert_allclose(ds_yt.index.grids[0].dds.d, [0.2, 0.5, 0.5])

    # all data covers the full domain exactly once
    vol = ds_yt.all_data()[("index", "cell_volume")].d.sum()
    np.testing.assert_allclose(vol, 4.0 * 2.0 * 2.0)


def test_refined_max_level(ds_stretched):
    ds_yt = ds_stretched.yt.load_as_uniform_grid(
        length_unit="m", refine_grid=True, max_level=1
    )
    assert ds_yt.index.max_level == 1
    _check_linear(ds_yt)

    ds_yt = ds_stretched.yt.load_as_uniform_grid(
        length_unit="m", refine_grid=True, max_level=0
    )
    assert len(ds_yt.index.grids) == 1
    _check_linear(ds_yt)


@pytest.mark.parametrize(
    "refine_by",
    ((2, 2, 1), np.array([4, 2, 1]), [2, 1, 1], {"x": 2, "y": 2, "z": 1}),
)
def test_refined_anisotropic(ds_stretched, refine_by):
    ds_yt = ds_stretched.yt.load_as_uniform_grid(
        length_unit="m", refine_grid=True, refine_by=refine_by
    )
    if isinstance(refine_by, dict):
        rb = np.array([refine_by[c] for c in ("x", "y", "z")])
    else:
        rb = np.asarray(refine_by)
    for g in ds_yt.index.grids:
        if g.Parent is not None:
            np.testing.assert_allclose(g.dds.d * rb, g.Parent.dds.d)
    _check_linear(ds_yt)
    vol = ds_yt.all_data()[("index", "cell_volume")].d.sum()
    np.testing.assert_allclose(vol, 4.0 * 2.0 * 2.0)

    # a covering grid at the finest level works
    rb_max = rb**ds_yt.index.max_level
    cg = ds_yt.covering_grid(
        ds_yt.index.max_level,
        ds_yt.domain_left_edge,
        ds_yt.domain_dimensions * rb_max,
    )
    vals = cg[("stream", "f")].d
    assert vals.shape == tuple(ds_yt.domain_dimensions * rb_max)


def test_refine_by_dict_ignores_dim_order(ds_stretched):
    # the dict form resolves by name, so it must match the equivalent tuple
    # regardless of the order the dict is given in, and on a dataset whose
    # dimension order differs
    ds_t = ds_stretched.transpose("z", "y", "x")
    rb_dict = {"y": 2, "z": 1, "x": 2}
    ds_a = ds_stretched.yt.load_as_uniform_grid(
        length_unit="m", refine_grid=True, refine_by=rb_dict
    )
    ds_b = ds_stretched.yt.load_as_uniform_grid(
        length_unit="m", refine_grid=True, refine_by=(2, 2, 1)
    )
    ds_c = ds_t.yt.load_as_uniform_grid(
        length_unit="m", refine_grid=True, refine_by=rb_dict
    )
    ds_d = ds_t.yt.load_as_uniform_grid(
        length_unit="m", refine_grid=True, refine_by=(1, 2, 2)
    )
    for dsyt in (ds_a, ds_b):
        _check_linear(dsyt)
    for dsyt in (ds_c, ds_d):
        _check_linear(dsyt, axis_names=("z", "y", "x"))
    assert np.all(ds_a.refine_by == ds_b.refine_by)
    assert np.all(ds_c.refine_by == ds_d.refine_by)
    n_cells = [
        sum(g.ActiveDimensions.prod() for g in d.index.grids)
        for d in (ds_a, ds_b, ds_c, ds_d)
    ]
    assert len(set(n_cells)) == 1

    with pytest.raises(ValueError, match="refine_by must be provided for every"):
        ds_stretched.yt.load_as_uniform_grid(
            length_unit="m", refine_grid=True, refine_by={"x": 2}
        )
    with pytest.raises(KeyError, match="not a selected coordinate"):
        ds_stretched.yt.load_as_uniform_grid(
            length_unit="m", refine_grid=True, refine_by={"x": 2, "y": 1, "q": 1}
        )


def test_refined_anisotropic_cell_savings(ds_stretched):
    # never refining z should use fewer cells than isotropic refinement
    ds_iso = ds_stretched.yt.load_as_uniform_grid(length_unit="m", refine_grid=True)
    ds_ani = ds_stretched.yt.load_as_uniform_grid(
        length_unit="m", refine_grid=True, refine_by=(2, 2, 1)
    )
    n_iso = sum(g.ActiveDimensions.prod() for g in ds_iso.index.grids)
    n_ani = sum(g.ActiveDimensions.prod() for g in ds_ani.index.grids)
    assert n_ani < n_iso


def test_geographic():
    # geographic dataset, stretched in depth, with reversed latitude
    depth = np.array([0.0, 10.0, 20.0, 50.0, 100.0, 200.0, 400.0])
    lat = np.linspace(40.0, 30.0, 6)
    lon = np.linspace(100.0, 120.0, 9)
    f = _linear_field(depth, lat, lon)
    da = xr.DataArray(
        f,
        coords={"depth": depth, "latitude": lat, "longitude": lon},
        dims=("depth", "latitude", "longitude"),
    )
    ds = xr.Dataset({"f": da}, attrs={"geospatial_vertical_units": "km"})

    ds_yt = ds.yt.load_as_uniform_grid(
        cell_spacing={"depth": 10.0, "lat": 2.0, "longitude": 2.5}
    )
    assert ds_yt.coordinates.name == "internal_geographic"
    assert tuple(ds_yt.domain_dimensions) == (40, 5, 8)
    np.testing.assert_allclose(ds_yt.domain_left_edge.d, [0.0, 30.0, 100.0])
    _check_linear(ds_yt)

    ds_yt = ds.yt.load_as_uniform_grid(refine_grid=True)
    assert ds_yt.index.max_level > 0
    _check_linear(ds_yt)


@pytest.mark.parametrize("use_callable", (True, False))
def test_time_selection(use_callable):
    ds = construct_ds_with_extra_dim(1, nd_space=3)
    fld = list(ds.data_vars)[0]
    ds_yt = ds.yt.load_as_uniform_grid(
        fld,
        length_unit="m",
        sel_dict={"time": 2},
        cell_spacing=0.1,
        use_callable=use_callable,
    )
    assert tuple(ds_yt.domain_dimensions) == (10, 10, 10)
    vals = ds_yt.all_data()[("stream", fld)].d
    assert vals.size == 1000
    assert np.all(np.isfinite(vals))
    # the source grid is uniform with 5 nodes on [0, 1] so interpolating at
    # the node positions recovers the node values exactly
    ds_yt = ds.yt.load_as_uniform_grid(
        fld,
        length_unit="m",
        sel_dict={"time": 2},
        bbox=np.array([[-0.125, 1.125]] * 3),
        cell_spacing=0.25,
        use_callable=use_callable,
    )
    cg = ds_yt.covering_grid(0, ds_yt.domain_left_edge, ds_yt.domain_dimensions)
    np.testing.assert_allclose(cg[("stream", fld)].d, ds[fld].isel(time=2).values)


def test_errors(ds_stretched):
    with pytest.raises(ValueError, match="interp_method must be"):
        ds_stretched.yt.load_as_uniform_grid(length_unit="m", interp_method="cubic")
    with pytest.raises(ValueError, match="chunksizes cannot be combined"):
        ds_stretched.yt.load_as_uniform_grid(
            length_unit="m", chunksizes=4, refine_grid=True
        )
    with pytest.raises(ValueError, match="The number of elements in chunksizes"):
        ds_stretched.yt.load_as_uniform_grid(length_unit="m", chunksizes=(4, 4))
    with pytest.raises(KeyError, match="not a selected coordinate"):
        ds_stretched.yt.load_as_uniform_grid(length_unit="m", bbox={"q": (0, 1)})
    with pytest.raises(ValueError, match="cell_spacing must be provided for every"):
        ds_stretched.yt.load_as_uniform_grid(length_unit="m", cell_spacing={"x": 0.1})
    with pytest.raises(ValueError, match="cell_spacing must have 3 elements"):
        ds_stretched.yt.load_as_uniform_grid(length_unit="m", cell_spacing=(0.1, 0.1))
    with pytest.raises(ValueError, match="cell_spacing values must be positive"):
        ds_stretched.yt.load_as_uniform_grid(length_unit="m", cell_spacing=-0.1)
    with pytest.raises(ValueError, match="bbox must have shape"):
        ds_stretched.yt.load_as_uniform_grid(length_unit="m", bbox=np.zeros((2, 2)))
    with pytest.raises(ValueError, match="bbox max must exceed"):
        ds_stretched.yt.load_as_uniform_grid(length_unit="m", bbox={"x": (1.0, 0.5)})
    with pytest.raises(ValueError, match="refine_by must be an integer or"):
        ds_stretched.yt.load_as_uniform_grid(
            length_unit="m", refine_grid=True, refine_by=(2, 2)
        )
    with pytest.raises(ValueError, match="refine_by values must be"):
        ds_stretched.yt.load_as_uniform_grid(
            length_unit="m", refine_grid=True, refine_by=0
        )


def test_not_periodic_by_default(ds_stretched):
    ds_yt = ds_stretched.yt.load_as_uniform_grid(length_unit="m")
    assert tuple(ds_yt.periodicity) == (False, False, False)
    ds_yt = ds_stretched.yt.load_as_uniform_grid(length_unit="m", refine_grid=True)
    assert tuple(ds_yt.periodicity) == (False, False, False)
    ds_yt = ds_stretched.yt.load_as_uniform_grid(
        length_unit="m", periodicity=(True, False, False)
    )
    assert tuple(ds_yt.periodicity) == (True, False, False)


def test_2d_not_implemented():
    x = np.linspace(0, 1, 5)
    da = xr.DataArray(np.ones((5, 5)), coords={"x": x, "y": x}, dims=("x", "y"))
    ds = xr.Dataset({"f": da})
    with pytest.raises(NotImplementedError, match="only implemented for 3D"):
        ds.yt.load_as_uniform_grid(length_unit="m")


@pytest.mark.parametrize("use_callable", (True, False))
@pytest.mark.parametrize(
    "kwargs",
    (
        {},
        {"chunksizes": 7},
        {"refine_grid": True},
        {"refine_grid": True, "refine_by": {"x": 2, "y": 2, "z": 1}},
    ),
)
def test_dask_backed(ds_stretched, use_callable, kwargs):
    # a dask-backed copy of the stretched dataset, with dask chunks that do
    # not line up with any of the yt grids built on top of it.
    da = pytest.importorskip("dask.array")
    ds_dask = ds_stretched.chunk({"x": 5, "y": 4, "z": 3})
    assert isinstance(ds_dask["f"].data, da.Array)

    ds_yt = ds_dask.yt.load_as_uniform_grid(
        length_unit="m", use_callable=use_callable, **kwargs
    )
    ds_ref = ds_stretched.yt.load_as_uniform_grid(
        length_unit="m", use_callable=use_callable, **kwargs
    )

    assert len(ds_yt.index.grids) == len(ds_ref.index.grids)
    _check_linear(ds_yt)
    for g_dask, g_ref in zip(ds_yt.index.grids, ds_ref.index.grids):
        np.testing.assert_allclose(g_dask[("stream", "f")].d, g_ref[("stream", "f")].d)

    # the xarray dataset itself is still lazy
    assert isinstance(ds_dask["f"].data, da.Array)


# the planner on its own


def _check_plan(grids: list[GridSpec], refine_by: Any) -> None:
    rb = np.broadcast_to(np.asarray(refine_by), (3,))
    by_level: dict[int, list[GridSpec]] = {}
    for g in grids:
        by_level.setdefault(g.level, []).append(g)
    assert len(by_level[0]) == 1
    for level in sorted(by_level):
        if level == 0:
            continue
        for g in by_level[level]:
            # each grid has exactly one parent that contains it
            parents = [
                p
                for p in by_level[level - 1]
                if np.all(p.left_edge <= g.left_edge + 1e-12)
                and np.all(p.right_edge >= g.right_edge - 1e-12)
            ]
            assert len(parents) == 1
            dds = (g.right_edge - g.left_edge) / g.dimensions
            p = parents[0]
            pdds = (p.right_edge - p.left_edge) / p.dimensions
            np.testing.assert_allclose(dds * rb, pdds)
        # grids on a level do not overlap
        gl = by_level[level]
        for i in range(len(gl)):
            for j in range(i + 1, len(gl)):
                lo = np.maximum(gl[i].left_edge, gl[j].left_edge)
                hi = np.minimum(gl[i].right_edge, gl[j].right_edge)
                assert not np.all(hi - lo > 1e-12)


_plan_coords = [
    np.concatenate(
        [np.linspace(0, 1, 5), np.linspace(1.1, 2, 10), np.linspace(2.25, 4, 8)]
    ),
    np.concatenate([np.linspace(0, 1, 3), np.linspace(1.125, 2, 8)]),
    np.linspace(0, 1, 5),
]
_plan_bbox = np.array([[0.0, 4.0], [0.0, 2.0], [0.0, 1.0]])


@pytest.mark.parametrize("refine_by", (2, 3, (2, 2, 1), (4, 2, 1), (1, 1, 1)))
@pytest.mark.parametrize("merge_grids", (True, False))
def test_plan_nesting(refine_by, merge_grids):
    grids = plan_stretched_grid_refinement(
        _plan_coords,
        _plan_bbox,
        np.array([0.25, 0.5, 0.25]),
        refine_by=refine_by,
        merge_grids=merge_grids,
    )
    _check_plan(grids, refine_by)
    summary = summarize_grid_plan(grids)
    assert summary[0]["n_grids"] == 1
    assert summary[0]["n_cells"] == 16 * 4 * 4
    if np.all(np.asarray(refine_by) == 1):
        assert len(grids) == 1


def test_plan_merging_reduces_grids():
    merged = plan_stretched_grid_refinement(
        _plan_coords, _plan_bbox, np.array([0.25, 0.5, 0.25]), refine_by=2
    )
    unmerged = plan_stretched_grid_refinement(
        _plan_coords,
        _plan_bbox,
        np.array([0.25, 0.5, 0.25]),
        refine_by=2,
        merge_grids=False,
    )
    assert len(merged) <= len(unmerged)
    s_m = summarize_grid_plan(merged)
    s_u = summarize_grid_plan(unmerged)
    for level in s_m:
        assert s_m[level]["n_cells"] == s_u[level]["n_cells"]


def test_plan_refined_region_location():
    # refinement in x should only cover the finely spaced band [1, 2]
    grids = plan_stretched_grid_refinement(
        [
            _plan_coords[0],
            np.linspace(0.0, 2.0, 5, dtype=np.float64),
            np.linspace(0.0, 1.0, 5, dtype=np.float64),
        ],
        _plan_bbox,
        np.array([0.25, 0.5, 0.25]),
        refine_by=2,
    )
    finest = max(g.level for g in grids)
    assert finest == 2
    for g in grids:
        if g.level > 0:
            assert g.left_edge[0] >= 1.0 - 1e-12
            assert g.right_edge[0] <= 2.0 + 1e-12


def test_plan_errors():
    with pytest.raises(ValueError, match="integer multiple of cell_spacing"):
        plan_stretched_grid_refinement(
            _plan_coords, _plan_bbox, np.array([0.3, 0.5, 0.25])
        )
    with pytest.raises(ValueError, match="max_level must be"):
        plan_stretched_grid_refinement(
            _plan_coords, _plan_bbox, np.array([0.25, 0.5, 0.25]), max_level=-1
        )
