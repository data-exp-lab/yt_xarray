import numpy as np
import pytest
import xarray as xr
import yt
from dask import array as da

import yt_xarray  # noqa: F401
from yt_xarray import sample_data
from yt_xarray.utilities._utilities import construct_minimal_ds


def test_partial_chunks_uni_cartesian():

    # uniform grid, cartesian
    fields = {
        "field0": ("x", "y", "z"),
        "field1": ("x", "y", "z"),
    }
    dims = {"x": (0, 1, 30), "y": (0, 2, 62), "z": (-1, 0.5, 25)}
    ds = sample_data.load_random_xr_data(fields, dims)
    ds_yt = ds.yt.load_grid(length_unit="km", chunksizes=10)
    ad = ds_yt.all_data()
    fld_vals = ad[("stream", "field0")]

    # uniform grid cartesian will wrap it, so number of cells should match
    # the number of grid points initially
    expected_size = np.prod([dim[2] for dim in dims.values()])
    assert fld_vals.size == expected_size


def test_cartesian():

    # builds yt grids with and without chunking (and callables), compares the
    # frb ImageArrays for equivalent SlicePlots
    ds = construct_minimal_ds(x_name="x", y_name="y", z_name="z", n_fields=2)

    ds_yt_nochunk = ds.yt.load_grid(length_unit="km")
    ds_ch_no_call = ds.yt.load_grid(length_unit="km", chunksizes=2, use_callable=False)
    ds_ch_call = ds.yt.load_grid(length_unit="km", chunksizes=2)

    slices = []
    for dsyt in [ds_yt_nochunk, ds_ch_no_call, ds_ch_call]:
        slc = yt.SlicePlot(dsyt, "z", ("stream", "test_field_0"))
        slc.render()
        slices.append(slc.frb[("stream", "test_field_0")])

    assert np.all(slices[0] == slices[1])
    assert np.all(slices[0] == slices[2])


def test_geographic_interp():

    # this case will require interpolation
    ds = construct_minimal_ds(n_fields=2)
    ds_yt_nochunk = ds.yt.load_grid()
    ds_ch_no_call = ds.yt.load_grid(chunksizes=2, use_callable=False)
    ds_ch_call = ds.yt.load_grid(chunksizes=2)

    initial_shape = np.asarray(ds.data_vars["test_field_0"].shape)
    expected_size = np.prod(initial_shape - 1)

    for dsyt in [ds_yt_nochunk, ds_ch_no_call, ds_ch_call]:
        ad = dsyt.all_data()
        fld = ad[("stream", "test_field_0")]
        assert fld.size == expected_size


def test_geographic_no_interp():

    # this case will not require interpolation
    ds = construct_minimal_ds(
        n_fields=2, min_x=30.0, max_x=35.0, min_y=40.0, max_y=42.0
    )
    ds_yt_nochunk = ds.yt.load_grid()
    ds_ch_no_call = ds.yt.load_grid(chunksizes=2, use_callable=False)
    ds_ch_call = ds.yt.load_grid(chunksizes=2)
    ds_tuple = ds.yt.load_grid(chunksizes=(2, 2, 2))

    # number of cells == number of initial node points
    expected_size = ds.data_vars["test_field_0"].size

    for dsyt in [ds_yt_nochunk, ds_ch_no_call, ds_ch_call, ds_tuple]:
        ad = dsyt.all_data()
        fld = ad[("stream", "test_field_0")]
        assert fld.size == expected_size


def test_2d_fields_not_implemented():
    fields = {
        "field0": ("x", "y"),
        "field1": ("x", "y"),
    }
    dims = {"x": (0, 1, 30), "y": (0, 2, 15)}
    ds = sample_data.load_random_xr_data(fields, dims)

    with pytest.raises(
        NotImplementedError, match="Can only load a chunked grid with 3D"
    ):
        _ = ds.yt.load_grid(length_unit="km", chunksizes=10)


def test_chunks_stretched_not_implemented():
    ds = construct_minimal_ds(x_name="x", y_name="y", z_name="z", x_stretched=True)
    with pytest.raises(NotImplementedError, match="Stretched grids cannot set"):
        _ = ds.yt.load_grid(length_unit="km", chunksizes=10, use_callable=False)


def test_dask_array():

    # checks that we can handle a field that is a dask array
    shp = (10, 12, 5)
    f1 = da.random.random(shp, chunks=5)
    coords = {
        "x": np.linspace(0, 1, shp[0]),
        "y": np.linspace(0, 1, shp[1]),
        "z": np.linspace(0, 1, shp[2]),
    }

    data = {"test_field": xr.DataArray(f1, coords=coords, dims=("x", "y", "z"))}
    ds = xr.Dataset(data)

    ds_yt = ds.yt.load_grid(chunksizes=5, length_unit="m")

    n_grids = len(ds_yt.index.grids)
    expected = np.prod(np.ceil(np.array(shp) / 5))
    assert n_grids == expected

    vals = ds_yt.all_data()[("stream", "test_field")]
    assert vals.size == np.prod(shp)


def test_chunk_tuple():
    # uniform grid, cartesian
    fields = {
        "field0": ("x", "y", "z"),
        "field1": ("x", "y", "z"),
    }
    dims = {"x": (0, 1, 30), "y": (0, 2, 40), "z": (-1, 0.5, 20)}
    ds = sample_data.load_random_xr_data(fields, dims)
    ds_yt = ds.yt.load_grid(length_unit="km", chunksizes=(30, 40, 20))
    assert len(ds_yt.index.grids) == 1


def test_chunk_bad_length():
    fields = {
        "field0": ("x", "y", "z"),
        "field1": ("x", "y", "z"),
    }
    dims = {"x": (0, 1, 30), "y": (0, 2, 40), "z": (-1, 0.5, 20)}
    ds = sample_data.load_random_xr_data(fields, dims)

    with pytest.raises(ValueError, match="The number of elements in "):
        _ = ds.yt.load_grid(length_unit="km", chunksizes=(30, 40, 20, 5))
