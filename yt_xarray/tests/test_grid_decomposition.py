import numpy as np

from yt_xarray.transformations import GeocentricCartesian
from yt_xarray.utilities._grid_decomposition import _create_image_mask


def test_create_image_mask():
    bbox_geo = np.array([[4000.0, 6371], [-90.0, 90.0], [0.0, 360.0]])

    gc = GeocentricCartesian(radial_type="radius", radial_axis="r", r_o=6371.0)

    bbox_cart = np.array([[-6600.0, 6600], [-6600.0, 6600], [-6600.0, 6600]])

    res = (32,) * 3

    im = _create_image_mask(bbox_cart, bbox_geo, res, gc, chunks=16)

    assert np.min(im) == 0
    assert np.max(im) == 1
    assert im.shape == res
