import numpy as np
import pytest
import yt

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

    # number of cells == number of initial node points
    expected_size = ds.data_vars["test_field_0"].size

    for dsyt in [ds_yt_nochunk, ds_ch_no_call, ds_ch_call]:
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
