import abc
from typing import List, Optional, Tuple

import numpy as np
import xarray as xr
import yt
from unyt import earth_radius as _earth_radius

from yt_xarray.utilities.logging import ytxr_log

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


def _sphere_to_cart(r, theta, phi):
    # r : radius
    # theta: colatitude
    # phi: azimuth
    # returns x, y, z
    z = r * np.cos(theta)
    xy = r * np.sin(theta)
    x = xy * np.cos(phi)
    y = xy * np.sin(phi)
    return x, y, z


def _cart_to_sphere(x, y, z):
    xy = x**2 + y**2
    r = np.sqrt(xy + z**2)
    theta = np.arccos(z / (r + 1e-12))
    phi = np.arctan2(y, x)
    return r, theta, phi


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
        if self.radial_type == "depth":
            r_val = self._r_o - coords[self.radial_axis]
        elif self.radial_type == "altitude":
            r_val = self._r_o + coords[self.radial_axis]
        else:
            r_val = coords[self.radial_axis]

        lat, lon = coords["latitude"], coords["longitude"]
        theta = (90.0 - lat) * np.pi / 180.0  # colatitude in radians
        phi = lon * np.pi / 180.0  # azimuth in radians
        x, y, z = _sphere_to_cart(r_val, theta, phi)
        return x, y, z

    def _calculate_native(self, **coords):
        r, theta, phi = _cart_to_sphere(coords["x"], coords["y"], coords["z"])
        lat = 90.0 - theta * 180.0 / np.pi
        lon = phi * 180.0 / np.pi
        if isinstance(lon, float):
            if lon < 0:
                lon = lon + 360.0
        else:
            lon = np.mod(lon, 360)
        if self.radial_type == "altitude":
            r = r - self._r_o
        elif self.radial_type == "depth":
            r = self._r_o - r
        return r, lat, lon


def build_interpolated_cartesian_ds(
    xr_ds: xr.Dataset,
    fields: Tuple[str],
    radial_type: str,
    bbox_dict: Optional[dict] = None,
    grid_resolution: Optional[List[int]] = None,
    radial_axis: Optional[str] = None,
    r_o: Optional[float] = None,
    fill_value: Optional[float] = None,
    length_unit: Optional[str] = "km",
    refine_grid: Optional[bool] = False,
    refine_by: Optional[int] = 2,
    refine_max_iters: Optional[int] = 200,
    refine_min_grid_size: Optional[int] = 10,
):
    """
    Build a yt cartesian dataset containing fields interpolated on demand
    from data defined on a 3D Geodetic grid to a uniform, cartesian grid

    Parameters
    ----------
    xr_ds: xr.Dataset
        the xarray dataset
    fields: tuple
        the fields to include
    radial_type: str
        one of 'radius', 'depth', 'altitude' to indicate type of radial axis.
    bbox_dict: dict
        optional bounding box to limit the data by
    grid_resolution:
        the interpolated grid resolution, defaults to (64, 64, 64)
    radial_axis: str
        the name of the radial axis, will try to infer if not provided
    r_o: float
        the reference radius, defaults to the radius of the Earth
    fill_value: float
        Optional value to use for filling grid values that fall outside
        the original data. Defaults to np.nan, but for volume rendering
        you may want to adjust this.
    length_unit: str
        the length unit to use, defaults to 'km'
    refine_grid: bool
        if True (default False), will decompose the interpolated grid one level.
    refine_max_iters: int
        if refine_grid is True, max iterations for grid refinement (default 200)
    refine_min_grid_size:
        if refine_grid is True, minimum number of elements in refined grid (default 10)


    Returns
    -------
    yt.Dataset
        a yt dataset: cartesian, uniform grid with references to the
        provided xarray dataset. Interpolation from geodetic to geocentric
        cartesian happens on demand on data reads.

    """

    if fill_value is None:
        fill_value = np.nan

    dims = xr_ds.data_vars[fields[0]].dims

    # lat/lon disambiguation should happen here
    if "latitude" not in dims:
        msg = f"expected latitude as one dimensions, found {dims}"
        raise ValueError(msg)
    if "longitude" not in dims:
        msg = f"expected longitude as one dimensions, found {dims}"
        raise ValueError(msg)

    # todo: disambiguate lat, lon
    latname = "latitude"
    lonname = "longitude"

    if radial_axis is not None:
        if radial_axis not in dims:
            msg = (
                f"The supplied radial_axis, {radial_axis} is one of the "
                f"dimensions: {dims}"
            )
            raise ValueError(msg)
    else:
        # try to infer it
        for dim in dims:
            if dim not in (latname, lonname):
                radial_axis = dim
        if radial_axis is None:
            msg = (
                f"could not determine radial axis from {dims}, please specify"
                "with the radial_axis keyword argument."
            )
            raise RuntimeError(msg)

    coords = (radial_axis, latname, lonname)

    if r_o is None:
        r_o = EARTH_RADIUS.to(length_unit).d

    gc = GeocentricCartesian(radial_type=radial_type, radial_axis=radial_axis, r_o=r_o)

    if bbox_dict is None:
        bbox_dict = {}
        for coord in coords:
            bbox_dict[coord] = [
                np.min(xr_ds.coords[coord].values),
                np.max(xr_ds.coords[coord].values),
            ]

    bbox_cart = _get_cart_bbox_for_geocentric(
        xr_ds, latname, lonname, bbox_dict, radial_axis, gc
    )

    # round ?
    bbox_cart[:, 0] = np.floor(bbox_cart[:, 0])
    bbox_cart[:, 1] = np.ceil(bbox_cart[:, 1])

    has_neg_lons = any(xr_ds.coords[lonname].values < 0)

    def _read_data(grid, field_name):
        # xyz = grid.fcoords
        # xyz1d = []
        # for idim in range(3):
        #     xyz1d.append(
        #         np.linspace(
        #             grid.LeftEdge[idim], grid.RightEdge[idim], grid.shape[idim]
        #         ).d
        #     )
        # xyz = np.meshgrid(*xyz1d)
        xyz = grid.fcoords.d
        rlatlon = gc.to_native(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2])

        r, lat, lon = rlatlon
        if has_neg_lons:
            # the native data uses -180, 180, lon above will be 0, 360
            lon_mask = lon > 180.0
            lon[lon_mask] = lon[lon_mask] - 360.0

        # mask out points outside the native domain
        r_mask = np.logical_and(
            r >= bbox_dict[radial_axis][0], r <= bbox_dict[radial_axis][1]
        )
        la_mask = np.logical_and(
            lat >= bbox_dict[latname][0], lat <= bbox_dict[latname][1]
        )
        lo_mask = np.logical_and(
            lon >= bbox_dict[lonname][0], lon <= bbox_dict[lonname][1]
        )
        mask = np.logical_and(r_mask, la_mask, lo_mask).ravel()

        # find closest points within valid ranges
        output_vals = np.full(mask.shape, fill_value, dtype="float64")

        if np.any(mask):
            data = xr_ds.data_vars[field_name[1]]

            # interp_dict = {
            #     radial_axis: xr.DataArray(r.ravel()[mask], dims="points"),
            #     latname: xr.DataArray(lat.ravel()[mask], dims="points"),
            #     lonname: xr.DataArray(lon.ravel()[mask], dims="points"),
            #     "method": "nearest",
            # }
            # vals = data.sel(**interp_dict)
            interp_dict = {
                radial_axis: xr.DataArray(r.ravel()[mask], dims="points"),
                latname: xr.DataArray(lat.ravel()[mask], dims="points"),
                lonname: xr.DataArray(lon.ravel()[mask], dims="points"),
            }
            vals = data.interp(kwargs=dict(fill_value=np.nan), **interp_dict)
            output_vals[mask] = vals.to_numpy()
        output_vals = output_vals.reshape(grid.shape)

        return output_vals

    data_dict = {}
    for field in fields:
        data_dict[field] = _read_data

    if grid_resolution is None:
        grid_resolution = (64, 64, 64)

    if refine_grid:
        from yt_xarray.utilities._grid_decomposition import (
            _create_image_mask,
            _get_yt_ds,
        )

        # create an image mask within bbox
        ytxr_log.info("Creating image mask for grid decomposition.")

        bbox_geo = []
        for ax in gc.native_coords:
            bbox_geo.append(bbox_dict[ax])
        bbox_geo = np.array(bbox_geo)
        image_mask = _create_image_mask(
            bbox_cart, bbox_geo, grid_resolution, gc, chunks=50
        )
        ytxr_log.info("Decomposing image mask and building yt dataset.")

        return _get_yt_ds(
            image_mask,
            data_dict,
            bbox_cart,
            max_iters=refine_max_iters,
            min_grid_size=refine_min_grid_size,
            refine_by=refine_by,
            length_unit=length_unit,
        )

    ds = yt.load_uniform_grid(
        data_dict,
        grid_resolution,
        geometry="cartesian",
        bbox=bbox_cart,
        length_unit=length_unit,
        axis_order="xyz",
        nprocs=1,  # placeholder, should relax this when possible.
    )

    return ds


def _get_cart_bbox_for_geocentric(
    xr_ds, latname, lonname, bbox_dict, radial_axis, gc: GeocentricCartesian
):
    # get 3d cartesian bounding box
    # for global data, cant just use min/max as it will miss +/- x and y
    # when latitude varies. When the lat/lon grid is small, just build
    # a full grid to evaluate. When it's large, discretrize with some
    # probe points.
    la = latname
    lo = lonname
    if len(xr_ds.coords[la]) <= 100:
        test_lats = xr_ds.coords[la].values
    else:
        test_lats = np.linspace(bbox_dict[la][0], bbox_dict[la][1], 100)

    if len(xr_ds.coords[lo]) <= 100:
        test_lons = xr_ds.coords[lo].values
    else:
        test_lons = np.linspace(bbox_dict[lo][0], bbox_dict[lo][1], 100)

    test_lats, test_lons = np.meshgrid(test_lats, test_lons)
    test_lats = np.ravel(test_lats)
    test_lons = np.ravel(test_lons)

    rmin = bbox_dict[radial_axis][0]
    cs_to_transform = {radial_axis: rmin, latname: test_lats, lonname: test_lons}
    x_y_z = gc.to_transformed(**cs_to_transform)
    bbox_cart = []
    for idim in range(3):
        bbox_cart.append([np.min(x_y_z[idim]), np.max(x_y_z[idim])])

    rmax = bbox_dict[radial_axis][1]
    cs_to_transform = {radial_axis: rmax, latname: test_lats, lonname: test_lons}
    x_y_z = gc.to_transformed(**cs_to_transform)
    for idim in range(3):
        bbox_cart[idim][0] = np.min((np.min(x_y_z[idim]), bbox_cart[idim][0]))
        bbox_cart[idim][1] = np.max((np.max(x_y_z[idim]), bbox_cart[idim][1]))
    bbox_cart = np.array(bbox_cart)

    # round ?
    bbox_cart[:, 0] = np.floor(bbox_cart[:, 0])
    bbox_cart[:, 1] = np.ceil(bbox_cart[:, 1])
    return bbox_cart
