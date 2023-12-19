import abc
from typing import List, Optional, Tuple

import numpy as np
import xarray as xr
import yt
from unyt import earth_radius as _earth_radius

EARTH_RADIUS = _earth_radius * 1.0


class Transformer(abc.ABC):
    def __init__(self, native_coords: Tuple[str], transformed_coords: Tuple[str]):
        self.native_coords = native_coords
        self.transformed_coords = transformed_coords

    @abc.abstractmethod
    def calculate_native(self, **coords):
        pass

    @abc.abstractmethod
    def calculate_transformed(self, **coords):
        pass

    def to_native(self, **coords):
        for dim in coords.keys():
            if dim not in self.transformed_coords:
                raise RuntimeError()

        return self.calculate_native(**coords)

    def to_transformed(self, **coords):
        for dim in coords.keys():
            if dim not in self.native_coords:
                raise RuntimeError()

        return self.calculate_transformed(**coords)


class LinearScale(Transformer):
    def __init__(self, native_coords: Tuple[str], scale: Optional[dict] = None):
        if scale is None:
            scale = {}

        for nc in native_coords:
            if nc not in scale:
                scale[nc] = 1.0
        self.scale = scale
        transformed_coords = tuple([nc + "_sc" for nc in native_coords])
        super().__init__(native_coords, transformed_coords)

    def calculate_transformed(self, **coords):
        transformed = []
        for nc_sc in self.transformed_coords:
            nc = nc_sc[:-3]
            transformed.append(np.asarray(coords[nc]) * self.scale[nc])
        return transformed

    def calculate_native(self, **coords):
        native = []
        for nc in self.native_coords:
            native.append(np.asarray(coords[nc + "_sc"]) / self.scale[nc])
        return native


class GeocentricCartesian(Transformer):
    def __init__(
        self,
        radial_axis: str = "radius",
        radial_type: str = "radius",
        r_o=None,
    ):
        try:
            import aglio  # NOQA
        except ImportError:
            msg = (
                "This operation requires the aglio package. Install "
                "with `pip install aglio` and try again."
            )
            raise ImportError(msg)

        native_coords = (radial_axis, "latitude", "longitude")
        transformed_cords = ("x", "y", "z")
        self.radial_type = radial_type
        if radial_type not in ("radius", "depth", "altitude"):
            raise ValueError()

        if r_o is None:
            self._r_o = EARTH_RADIUS

        self.radial_axis = radial_axis

        super().__init__(native_coords, transformed_cords)

    def calculate_transformed(self, **coords):
        from aglio.coordinate_transformations import geosphere2cart

        if self.radial_type == "depth":
            r_val = self._r_o - coords[self.radial_axis]
        elif self.radial_type == "altitude":
            r_val = self._r_o + coords[self.radial_axis]
        else:
            r_val = coords[self.radial_axis]

        x, y, z = geosphere2cart(coords["latitude"], coords["longitude"], r_val)
        return x, y, z

    def calculate_native(self, **coords):
        from aglio.coordinate_transformations import cart2sphere

        R, lat, lon = cart2sphere(
            coords["x"], coords["y"], coords["z"], geo=True, deg=True
        )

        if self.radial_type == "altitude":
            R = R - self._r_o
        elif self.radial_type == "depth":
            R = self._r_o - R
        return R, lat, lon


def build_interpolated_cartesian_ds(
    xr_ds: xr.Dataset,
    fields: Tuple[str],
    bbox_dict: Optional[dict] = None,
    grid_resolution: Optional[List[int]] = None,
):
    dims = xr_ds.data_vars[fields[0]].dims
    assert "latitude" in dims
    assert "longitude" in dims
    assert "radius" in dims
    coords = ("radius", "latitude", "longitude")

    gc = GeocentricCartesian()

    if bbox_dict is None:
        bbox_dict = {}
        for coord in coords:
            bbox_dict[coord] = [
                np.min(xr_ds.coords[coord].values),
                np.max(xr_ds.coords[coord].values),
            ]

    # get 3d cartesian bounding box
    rads = [
        bbox_dict["radius"][0],
    ] * 4 + [bbox_dict["radius"][1]] * 4
    la = "latitude"
    lo = "longitude"
    lats = [bbox_dict[la][0], bbox_dict[la][0], bbox_dict[la][1], bbox_dict[la][1]] * 2
    lons = [bbox_dict[lo][0], bbox_dict[lo][1], bbox_dict[lo][1], bbox_dict[lo][0]] * 2

    rads = np.array(rads)
    lats = np.array(lats)
    lons = np.array(lons)
    x_y_z = gc.to_transformed(radius=rads, latitude=lats, longitude=lons)

    bbox_cart = []
    for idim in range(3):
        bbox_cart.append([np.min(x_y_z[idim]), np.max(x_y_z[idim])])

    bbox_cart = np.array(bbox_cart)

    def _read_data(grid, field_name):
        # xyz = grid.fcoords
        xyz = []
        for idim in range(3):
            xyz.append(
                np.linspace(
                    grid.LeftEdge[idim], grid.RightEdge[idim], grid.shape[idim]
                ).d
            )

        r, lat, lon = gc.to_native(x=xyz[0], y=xyz[1], z=xyz[2])

        min_r = np.min(np.where(r >= bbox_dict["radius"][0]))
        max_r = np.max(np.where(r <= bbox_dict["radius"][1])) + 1
        min_la = np.min(np.where(lat >= bbox_dict["latitude"][0]))
        max_la = np.max(np.where(lat <= bbox_dict["latitude"][1])) + 1
        min_lo = np.min(np.where(lon >= bbox_dict["longitude"][0]))
        max_lo = np.max(np.where(lon <= bbox_dict["longitude"][1])) + 1

        # find closest points within valid ranges
        output_vals = np.full(grid.shape, np.nan, dtype="float64")
        data = xr_ds.data_vars[field_name[1]]
        vals = data.sel(
            {
                "radius": r[min_r:max_r],
                "latitude": lat[min_la:max_la],
                "longitude": lon[min_lo:max_lo],
            },
            method="nearest",
        )
        # vals = vals.to_numpy().reshape(grid.shape)
        output_vals[min_r:max_r, min_la:max_la, min_lo:max_lo] = vals
        return output_vals

    data_dict = {}
    for field in fields:
        data_dict[field] = _read_data

    if grid_resolution is None:
        grid_resolution = (64, 64, 64)

    ds = yt.load_uniform_grid(
        data_dict,
        grid_resolution,
        geometry="cartesian",
        bbox=bbox_cart,
        length_unit="km",
        axis_order="xyz",
    )

    return ds
