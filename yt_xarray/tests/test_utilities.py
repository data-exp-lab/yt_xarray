import numpy as np
import xarray as xr
import yt

import yt_xarray
from yt_xarray.accessor import _xr_to_yt as xr2yt
from yt_xarray.utilities._utilities import _find_file, construct_minimal_ds


def test_construct_minimal_ds():
    ds = construct_minimal_ds()
    assert isinstance(ds, xr.Dataset)

    ds = construct_minimal_ds(n_fields=2)
    assert len(ds.variables) - len(ds.coords) == 2

    ds = construct_minimal_ds(x_stretched=True, x_name="x")

    assert xr2yt._check_grid_stretchiness(ds.x.values) == xr2yt._GridType.STRETCHED

    ds = construct_minimal_ds(
        x_stretched=False,
        x_name="x",
        y_stretched=True,
        y_name="y",
        z_stretched=True,
        z_name="z",
    )

    assert xr2yt._check_grid_stretchiness(ds.x.values) == xr2yt._GridType.UNIFORM
    assert xr2yt._check_grid_stretchiness(ds.y.values) == xr2yt._GridType.STRETCHED
    assert xr2yt._check_grid_stretchiness(ds.z.values) == xr2yt._GridType.STRETCHED


def test_float32_ds():
    # float32 must be upcast for grid positions. This only manifested when
    # using a stretched grid (as the cell_widths were in float32).
    ds = construct_minimal_ds(dtype="float32", x_stretched=True)
    assert ds.test_field.dtype == np.float32


def test_file_validation(tmp_path):

    tdir = tmp_path / "test_data"
    tdir.mkdir()
    b = tdir / "test_file.txt"
    b.write_text("test test test")

    yt.config.ytcfg.set("yt", "test_data_dir", str(tdir))

    _ = _find_file("test_file.txt")


def test_open_dataset(tmp_path):
    tdir = tmp_path / "test_nc_data"
    tdir.mkdir()

    ds = construct_minimal_ds()
    ds.to_netcdf(tdir / "test_nc.nc")
    del ds

    yt.config.ytcfg.set("yt", "test_data_dir", str(tdir))

    ds = yt_xarray.open_dataset(tdir / "test_nc.nc")
    assert isinstance(ds, xr.Dataset)
