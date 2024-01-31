import abc
from typing import List, Optional, Tuple

import numpy as np
import xarray as xr
import yt
from unyt import earth_radius as _earth_radius

from yt_xarray.utilities._utilities import _import_optional_dep

EARTH_RADIUS = _earth_radius * 1.0


class Transformer(abc.ABC):
    """ "
    The transformer base class, meant to be subclassed, do not use directly.

    Parameters
    ----------
    native_coords: Tuple[str]
        the names of the native coordinates, e.g., ('x0', 'y0', 'z0'), on
        which data is defined.
    transformed_coords: Tuple[str]
        the names of the transformed coordinates, e.g., ('x1', 'y1', 'z1')

    the names of the coordinates will be expected as keyword arguments in the
    `to_native` and `to_transformed` methods.

    """

    def __init__(self, native_coords: Tuple[str], transformed_coords: Tuple[str]):
        self.native_coords = native_coords
        self._native_coords_set = set(native_coords)
        self.transformed_coords = transformed_coords
        self._transformed_coords_set = set(transformed_coords)

    @abc.abstractmethod
    def _calculate_native(self, **coords):
        """
        function to convert from transformed to native coordinates. Must be
        implemented by each child class.
        """

    @abc.abstractmethod
    def _calculate_transformed(self, **coords):
        """
        function to convert from native to transformed coordinates. Must be
        implemented by each child class.
        """

    def to_native(self, **coords):
        """
        Calculate the native coordinates from transformed coordinates.

        Parameters
        ----------
        **coords:
            coordinate values in transformed coordinate system, provided as
            individual keyword arguments.

        Returns
        -------
        tuple
            coordinate values in the native coordinate system, in order
            of .native_coords attribute.

        """

        # Generally, no need to override this method, the actual
        # coordinate transformation implementation should happen by
        # overriding `_calculate_native`
        dim_set = set()
        for dim in coords.keys():
            # check for validity of each dim
            if dim not in self.transformed_coords:
                msg = (
                    f"{dim} is not a valid coordinate name. "
                    f"Expected one of {self.transformed_coords}."
                )
                raise RuntimeError(msg)
            dim_set.add(dim)

        # check that all required coordinates are present
        if dim_set != self._transformed_coords_set:
            # find the missing one and raise an error
            for dim in self.transformed_coords:
                if dim not in coords:
                    msg = (
                        f"The transformed coordinate {dim} was not specified."
                        "Please provide it as an additional keyword argument."
                    )
                    raise RuntimeError(msg)

        return self._calculate_native(**coords)

    def to_transformed(self, **coords):
        """
        Calculate the transformed coordinates from native coordinates.

        Parameters
        ----------
        **coords:
            coordinate values in native coordinate system, provided as
            individual keyword arguments.

        Returns
        -------
        tuple
            coordinate values in the transformed coordinate system, in order
            of .transformed_coords attribute.

        """

        # Generally, no need to override this method, the actual
        # coordinate transformation implementation should happen by
        # overriding `_calculate_transformed`
        dim_set = set()
        for dim in coords.keys():
            if dim not in self.native_coords:
                msg = (
                    f"{dim} is not a valid coordinate name. "
                    f"Expected one of {self.native_coords}."
                )
                raise RuntimeError(msg)
            dim_set.add(dim)

        # check that all required coordinates are present
        if dim_set != self._native_coords_set:
            # find the missing one and raise an error
            for dim in self.native_coords:
                if dim not in coords:
                    msg = (
                        f"The native coordinate {dim} was not specified."
                        "Please provide it as an additional keyword argument."
                    )
                    raise RuntimeError(msg)

        return self._calculate_transformed(**coords)


class LinearScale(Transformer):
    """
    A transformer that linearly scales between coordinate systems.

    This transformer is mostly useful for demonstration purposes and simply
    applies a constant scaling factor for each dimension:

        (x_sc, y_sc, z_sc) = (x_scale, y_scale, z_scale) * (x, y, z)

    Parameters
    ----------
    native_coords: Tuple[str]
        the names of the native coordinates, e.g., ('x', 'y', 'z'), on
        which data is defined.
    scale: dict
        a dictionary containing the scale factor for each dimension. keys
        should match the native_coords names and missing keys default to a
        value of 1.0

    The scaled coordinate names are given by appending `'_sc'` to each native
    coordinate name. e.g., if `native_coords=('x', 'y', 'z')`, then the
    transformed coordinate names are ('x_sc', 'y_sc', 'z_sc').

    Examples
    --------

    >>> from yt_xarray.transformations import LinearScale
    >>> native_coords = ('x', 'y', 'z')
    >>> scale_factors = {'x': 2., 'y':3., 'z':1.5}
    >>> lin_scale = LinearScale(native_coords, scale_factors)
    >>> print(lin_scale.to_transformed(x=1, y=1, z=1))
    [2., 3., 1.5]
    >>> print(lin_scale.to_native(x_sc=2., y_sc=3., z_sc=1.5))
    [1., 1., 1.]

    """

    def __init__(self, native_coords: Tuple[str], scale: Optional[dict] = None):
        if scale is None:
            scale = {}

        for nc in native_coords:
            if nc not in scale:
                scale[nc] = 1.0
        self.scale = scale
        transformed_coords = tuple([nc + "_sc" for nc in native_coords])
        super().__init__(native_coords, transformed_coords)

    def _calculate_transformed(self, **coords):
        transformed = []
        for nc_sc in self.transformed_coords:
            nc = nc_sc[:-3]  # native coord name. e.g., go from "x_sc" to just "x"
            transformed.append(np.asarray(coords[nc]) * self.scale[nc])
        return transformed

    def _calculate_native(self, **coords):
        native = []
        for nc in self.native_coords:
            native.append(np.asarray(coords[nc + "_sc"]) / self.scale[nc])
        return native


_default_radial_axes = dict(
    zip(("radius", "depth", "altitude"), ("radius", "depth", "altitude"))
)


class GeocentricCartesian(Transformer):
    """
    A transformer to convert between Geodetic coordinates and cartesian,
    geocentric coordinates.

    Parameters
    ----------
    radial_type: str
        one of ("radius", "depth", "altitude") to indicate the type of
        radial axis.
    radial_axis: str
        Optional string to use as the name for the radial axis, defaults to
        whatever you provide for radial_type.
    r_o: float like
        The reference radius, default is the radius of the Earth.

    transformed_coords names are ("x", "y", "z") and
    native_coords names are (radial_axis, "latitude", "longitude"). Supply
    latitude and longitude vlaues in degrees.

    Examples
    --------

    >>> from yt_xarray.transformations import GeocentricCartesian
    >>> gc = GeocentricCartesian("depth")
    >>> x, y, z = gc.to_transformed(depth=100., latitude=42., longitude=220.)
    >>> print((x, y, z))
    # (-3626843.0297669284, -3043282.6486153184, 4262969.546178633)
    >>> print(gc.to_native(x=x,y=y,z=z))
    # (100.00000000093132, 42.0, 220.0)

    """

    def __init__(
        self,
        radial_type: str = "radius",
        radial_axis: Optional[str] = None,
        r_o=None,
    ):
        # catch missing dependencies early
        emsg = (
            "This functionality requires the aglio package, "
            "install it with `pip install aglio`"
        )
        _ = _import_optional_dep("aglio", custom_message=emsg)

        transformed_coords = ("x", "y", "z")

        valid_radial_types = ("radius", "depth", "altitude")
        if radial_type not in valid_radial_types:
            msg = (
                f"radial_type must be one of {valid_radial_types}, "
                f"found {radial_type}."
            )
            raise ValueError(msg)
        self.radial_type = radial_type

        if r_o is None:
            r_o = EARTH_RADIUS.to("m").d
        self._r_o = r_o

        if radial_axis is None:
            radial_axis = _default_radial_axes[radial_type]
        self.radial_axis = radial_axis
        native_coords = (radial_axis, "latitude", "longitude")

        super().__init__(native_coords, transformed_coords)

    def _calculate_transformed(self, **coords):
        ct = _import_optional_dep("aglio.coordinate_transformations")

        if self.radial_type == "depth":
            r_val = self._r_o - coords[self.radial_axis]
        elif self.radial_type == "altitude":
            r_val = self._r_o + coords[self.radial_axis]
        else:
            r_val = coords[self.radial_axis]

        x, y, z = ct.geosphere2cart(coords["latitude"], coords["longitude"], r_val)
        return x, y, z

    def _calculate_native(self, **coords):
        ct = _import_optional_dep("aglio.coordinate_transformations")

        R, lat, lon = ct.cart2sphere(
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
