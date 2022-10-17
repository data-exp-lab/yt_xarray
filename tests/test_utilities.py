import xarray as xr
import yt

import yt_xarray
from yt_xarray._utilities import _find_file, construct_minimal_ds


def test_construct_minimal_ds():
    ds = construct_minimal_ds()
    assert isinstance(ds, xr.Dataset)

    ds = construct_minimal_ds(n_fields=2)
    assert len(ds.variables) - len(ds.coords) == 2


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
