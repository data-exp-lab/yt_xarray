import matplotlib.pyplot as plt
import numpy as np
import pytest

from yt_xarray.transformations import GeocentricCartesian
from yt_xarray.utilities import _grid_decomposition as _gd


def test_create_image_mask():
    bbox_geo = np.array([[4000.0, 6371], [-90.0, 90.0], [0.0, 360.0]])

    gc = GeocentricCartesian(radial_type="radius", radial_axis="r", r_o=6371.0)

    bbox_cart = np.array([[-6600.0, 6600], [-6600.0, 6600], [-6600.0, 6600]])

    res = (32,) * 3

    im = _gd._create_image_mask(bbox_cart, bbox_geo, res, gc, chunks=16)

    assert np.min(im) == 0
    assert np.max(im) == 1
    assert im.shape == res


def test_signature_array():
    phi = np.ones((5, 3))
    sig = _gd.signature_array(phi, 0)
    assert np.all(sig == 3)
    sig = _gd.signature_array(phi, 1)
    assert np.all(sig == 5)

    phi = np.ones((5, 3, 2))
    sig = _gd.signature_array(phi, 2)
    assert np.all(sig == 15)

    with pytest.raises(NotImplementedError, match="Signature array only"):
        phi = np.ones((2,) * 4)
        _ = _gd.signature_array(phi, 2)


def test_grid_bounds_plot():
    le = (0, 0)
    re = (10, 10)
    gb = _gd.GridBounds(le, re)

    f, ax = plt.subplots(1)
    gb.plot(ax)

    le = (0, 0, 0)
    re = (10, 10, 10)
    gb = _gd.GridBounds(le, re)
    with pytest.raises(NotImplementedError, match="Grid patch plotting"):
        gb.plot(ax)

    with pytest.raises(RuntimeError, match="le is undefined."):
        _gd.GridBounds((0, np.nan), (1, 1))

    with pytest.raises(RuntimeError, match="re is undefined."):
        _gd.GridBounds((0, 0), (1, np.nan))


@pytest.mark.parametrize("decomp_method", ("bisect", "signature"))
def test_decompose_image(decomp_method):
    image_mask = np.ones((100, 100), dtype=bool)
    image_mask[0:50, 0:50] = False
    max_iters = 1
    min_grid_size = 5

    if decomp_method == "bisect":
        decomp_func = _gd.decompose_image_mask_bisect
    else:
        decomp_func = _gd.decompose_image_mask

    grids, n_iters = decomp_func(
        image_mask, max_iters=max_iters, min_grid_size=min_grid_size
    )
    assert len(grids) == 3
