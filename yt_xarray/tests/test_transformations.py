import numpy as np
import pytest
import xarray as xr

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

    bbox_in = np.array([[0.0, 2.0], [0.0, 2.0], [0.0, 2.0]])
    bbox_dict_in = {"x": bbox_in[0, :], "y": bbox_in[1, :], "whatever": bbox_in[2, :]}
    bbox_out = lsc.calculate_transformed_bbox(bbox_dict_in)
    assert np.all(bbox_in == bbox_out)


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


def test_gc_cartesian_bounding_box():
    gc = transformations.GeocentricCartesian(radial_type="altitude", r_o=1000.0)
    bbox_gc = {
        "altitude": [0.0, 1000.0],
        "latitude": [-90.0, 90.0],
        "longitude": [0.0, 360.0],
    }
    bbox_cart = gc.calculate_transformed_bbox(bbox_dict=bbox_gc)
    assert bbox_cart.min() == -2000.0
    assert bbox_cart.max() == 2000.0


def test_gc_negative_lons():
    gc = transformations.GeocentricCartesian()
    gc_nl = transformations.GeocentricCartesian(use_neg_lons=True)
    r_in = 3000.0
    lat_in = 42.0
    lon_in = 220.0
    expected_lon = lon_in - 360.0
    x, y, z = gc.to_transformed(radius=r_in, latitude=lat_in, longitude=lon_in)

    r_out, lat_out, lon_out = gc.to_native(x=x, y=y, z=z)
    r_out_nl, lat_out_nl, lon_out_nl = gc_nl.to_native(x=x, y=y, z=z)
    assert lon_out_nl == expected_lon
    assert lon_out_nl == expected_lon
    assert r_out == r_out_nl
    assert lat_out == lat_out_nl


def test_geocentric_embedded(xr_ds_for_tform):
    fields = {
        "field0": ("altitude", "latitude", "longitude"),
    }
    dims = {
        "altitude": (2000, 5000, 32),
        "latitude": (10, 50, 32),
        "longitude": (10, 50, 22),
    }
    ds = load_random_xr_data(fields, dims)
    gc = transformations.GeocentricCartesian(radial_type="altitude", r_o=6371.0)
    ds_yt = transformations.build_interpolated_cartesian_ds(ds, gc, fields=("field0",))
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
    with pytest.raises(ValueError, match="Coordinate name bad_name not found"):
        _ = lsc.to_transformed(x=1.0, y=1.0, bad_name=1.0)

    x, y, whatever = lsc.to_transformed(x=1.0, y=1.0, whatever=1.0)
    assert (x, y, whatever) == (0.5, 1.0, 2.0)
    with pytest.raises(ValueError, match="Coordinate name bad_name not found"):
        _ = lsc.to_native(x_sc=x, y_sc=y, bad_name=whatever)


def test_missing_coord_names():
    scale = {"whatever": 2.0, "x": 0.5}
    n_c = ("x", "y", "whatever")
    lsc = transformations.LinearScale(n_c, scale=scale)
    with pytest.raises(RuntimeError, match="The native coordinate whatever"):
        _ = lsc.to_transformed(x=1.0, y=1.0)

    with pytest.raises(RuntimeError, match="The transformed coordinate y_sc"):
        _ = lsc.to_native(x_sc=1.0, whatever_sc=1.0)


def test_coord_aliasing():
    coord_aliases = {"what": "radius"}
    gc = transformations.GeocentricCartesian(r_o=6371.0, coord_aliases=coord_aliases)

    r_in = 3000.0
    lat_in = 42.0
    lon_in = 10.0
    xyz0 = gc.to_transformed(radius=r_in, latitude=lat_in, longitude=lon_in)

    xyz = gc.to_transformed(what=r_in, latitude=lat_in, longitude=lon_in)

    assert xyz0 == xyz

    coord_aliases = {"what": "bad"}
    with pytest.raises(ValueError, match="Coordinate alias what must point"):
        gc = transformations.GeocentricCartesian(coord_aliases=coord_aliases)


@pytest.fixture(scope="module")
def xr_ds_for_tform():
    fields = {
        "field0": ("altitude", "latitude", "longitude"),
    }
    dims = {
        "altitude": (2000, 5000, 16),
        "latitude": (10, 50, 16),
        "longitude": (10, 50, 16),
    }
    ds = load_random_xr_data(fields, dims)
    return ds


@pytest.mark.parametrize("refine_method", ("division", "signature_filter"))
def test_geocentric_embedded_decomposed(refine_method, xr_ds_for_tform):
    ds = xr_ds_for_tform
    gc = transformations.GeocentricCartesian(radial_type="altitude", r_o=6371.0)
    ds_yt = transformations.build_interpolated_cartesian_ds(
        ds,
        gc,
        refine_grid=True,
        refinement_method=refine_method,
        refine_max_iters=10,
    )

    assert len(ds_yt.index.grids) > 1


def test_gc_build_interpolated_cartesian_ds_options(xr_ds_for_tform):
    ds = xr_ds_for_tform
    gc = transformations.GeocentricCartesian(radial_type="altitude", r_o=6371.0)
    _ = transformations.build_interpolated_cartesian_ds(
        ds,
        gc,
        fields="field0",
    )

    with pytest.raises(ValueError, match="refinement_method must be "):
        _ = transformations.build_interpolated_cartesian_ds(
            ds,
            gc,
            refine_grid=True,
            refinement_method="not_a_method",
        )

    with pytest.raises(ValueError, match="interp_method must be one of"):
        _ = transformations.build_interpolated_cartesian_ds(
            ds,
            gc,
            interp_method="not_a_method",
        )


def test_embedded_interp_method(xr_ds_for_tform):
    ds = xr_ds_for_tform
    gc = transformations.GeocentricCartesian(radial_type="altitude", r_o=6371.0)
    ds_yt = transformations.build_interpolated_cartesian_ds(
        ds, gc, fields=("field0",), interp_method="interpolate"
    )
    ad = ds_yt.all_data()
    mn = np.nanmin(ad[("stream", "field0")])
    mx = np.nanmax(ad[("stream", "field0")])
    assert np.isfinite(mn)
    assert np.isfinite(mx)
    assert mn > 0
    assert mx > mn

    def over_ride_ones(data=None, coords=None):
        assert len(coords) == 3
        assert isinstance(data, xr.DataArray)
        data_size = coords[0].shape
        return np.ones(data_size)

    ds_yt = transformations.build_interpolated_cartesian_ds(
        ds, gc, fields=("field0",), interp_func=over_ride_ones
    )

    ad = ds_yt.all_data()
    vals = ad["stream", "field0"]
    vals = vals[np.isfinite(vals)]
    assert np.all(vals == 1.0)
