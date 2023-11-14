import os
import os.path

import pytest

from yt_xarray.sample_data import load_random_xr_data


@pytest.fixture
def xr_ds():
    fields = {"temperature": ("x", "y", "z"), "pressure": ("x", "y", "z")}
    dims = {"x": (0, 1, 15), "y": (0, 1, 10), "z": (0, 1, 15)}
    ds = load_random_xr_data(fields, dims, length_unit="m")
    return ds


@pytest.mark.parametrize("viz_method", ["SlicePlot", "ProjectionPlot"])
def test_2d_volume_plots(tmp_path, xr_ds, viz_method):
    func = getattr(xr_ds.yt.vis, viz_method)
    slc = func("x", "temperature")

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    fname = str(output_dir / f"yt_xarray_{viz_method}.png")
    slc.save(fname)

    assert os.path.isfile(fname)


def test_phase_plot(tmp_path, xr_ds):
    slc = xr_ds.yt.vis.PhasePlot("pressure", "temperature", "temperature")

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    fname = str(output_dir / "yt_xarray_PhasePlot.png")
    slc.save(fname)

    assert os.path.isfile(fname)
