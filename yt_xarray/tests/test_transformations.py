import numpy as np
import pytest

from yt_xarray import transformations
from yt_xarray.sample_data import load_random_xr_data


def test_linear():
    scale = {"whatever": 2.0, "x": 0.5}
    n_c = ("x", "y", "whatever")
    lsc = transformations.LinearScale(n_c, scale=scale)
    x, y, whatever = lsc.to_transformed(x=1.0, y=1.0, whatever=1.0)
    assert (x, y, whatever) == (0.5, 1.0, 2.0)
    round_trip = lsc.to_native(x_sc=x, y_sc=y, whatever_sc=whatever)
    assert tuple(round_trip) == (1.0, 1.0, 1.0)

    lsc = transformations.LinearScale(n_c)
    for ky in n_c:
        assert lsc.scale[ky] == 1.0


def test_geocentric():
    gc = transformations.GeocentricCartesian(r_o=6371.0)

    r_in = 3000.0
    lat_in = 42.0
    lon_in = 10.0
    x, y, z = gc.to_transformed(radius=r_in, latitude=lat_in, longitude=lon_in)
    r_out, lat_out, lon_out = gc.to_native(x=x, y=y, z=z)

    out_ = np.asarray((r_out, lat_out, lon_out))
    in_ = np.asarray((r_in, lat_in, lon_in))

    assert np.allclose(out_, in_)

    with pytest.raises(ValueError, match="radial_type must be one of"):
        _ = transformations.GeocentricCartesian(r_o=6371.0, radial_type="bad_type")

    gc = transformations.GeocentricCartesian(r_o=6000.0, radial_type="depth")
    depth_in = 300.0
    x, y, z = gc.to_transformed(depth=depth_in, latitude=lat_in, longitude=lon_in)
    depth_out, lat_out, lon_out = gc.to_native(x=x, y=y, z=z)
    assert np.allclose(depth_out, depth_in)

    gc = transformations.GeocentricCartesian(radial_type="altitude")
    altitude_in = 300.0
    x, y, z = gc.to_transformed(altitude=altitude_in, latitude=lat_in, longitude=lon_in)
    altitude_out, lat_out, lon_out = gc.to_native(x=x, y=y, z=z)
    assert np.allclose(altitude_out, altitude_in)


def test_geocentric_embedded():
    fields = {
        "field0": ("radius", "latitude", "longitude"),
    }
    dims = {
        "radius": (2000, 5000, 32),
        "latitude": (10, 50, 32),
        "longitude": (10, 50, 22),
    }
    ds = load_random_xr_data(fields, dims)
    ds_yt = transformations.build_interpolated_cartesian_ds(
        ds,
        [
            "field0",
        ],
    )
    ad = ds_yt.all_data()
    mn = np.nanmin(ad[("stream", "field0")])
    mx = np.nanmax(ad[("stream", "field0")])
    assert np.isfinite(mn)
    assert np.isfinite(mx)
    assert mn > 0
    assert mx > mn


def test_bad_coord_names():
    scale = {"whatever": 2.0, "x": 0.5}
    n_c = ("x", "y", "whatever")
    lsc = transformations.LinearScale(n_c, scale=scale)
    with pytest.raises(RuntimeError, match="bad_name is not a valid coordinate name"):
        _ = lsc.to_transformed(x=1.0, y=1.0, bad_name=1.0)

    x, y, whatever = lsc.to_transformed(x=1.0, y=1.0, whatever=1.0)
    assert (x, y, whatever) == (0.5, 1.0, 2.0)
    with pytest.raises(RuntimeError, match="bad_name is not a valid coordinate name"):
        _ = lsc.to_native(x_sc=x, y_sc=y, bad_name=whatever)
