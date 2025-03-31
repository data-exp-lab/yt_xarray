import importlib
import os.path
from os import PathLike

import numpy as np
import numpy.typing as npt
import xarray as xr
import yt.config

from yt_xarray.accessor._xr_to_yt import known_coord_aliases


class _DimConstructor:
    def __init__(
        self,
        dim_min: float,
        dim_max: float,
        dim_n: int,
        dim_name: str,
        stretched: bool = False,
    ):
        self.dim_min = dim_min
        self.dim_max = dim_max
        self.dim_n = dim_n
        self.dim_name = dim_name
        self.stretched = stretched

    @property
    def dim_n_2(self) -> int:
        return self.dim_n // 2

    @property
    def values(self) -> npt.NDArray:
        dim_vals = np.linspace(self.dim_min, self.dim_max, self.dim_n)
        if self.stretched:
            dx = dim_vals[1:] - dim_vals[:-1]
            dim_vals[self.dim_n_2 :] = dim_vals[self.dim_n_2 :] + dx[0] * 2.0
        return dim_vals


class _DsConstructor:
    def __init__(
        self,
        min_x: float,
        max_x: float,
        n_x: int,
        x_name: str,
        min_y: float,
        max_y: float,
        n_y: int,
        y_name: str,
        min_z: float,
        max_z: float,
        n_z: int,
        z_name: str,
        z_units: str,
        field_name: str,
        coord_order: tuple[str, str, str],
        is_stretched: dict[str, bool],
    ):
        self.x = _DimConstructor(
            min_x, max_x, n_x, x_name, stretched=is_stretched[x_name]
        )
        self.y = _DimConstructor(
            min_y, max_y, n_y, y_name, stretched=is_stretched[y_name]
        )
        self.z = _DimConstructor(
            min_z, max_z, n_z, z_name, stretched=is_stretched[z_name]
        )
        self.z_units = z_units
        self.field_name = field_name
        self.coord_order = coord_order
        self.is_stretched = is_stretched


def construct_minimal_ds(
    min_x: float = 0.0,
    max_x: float = 360.0,
    n_x: int = 4,
    x_name: str = "longitude",
    x_stretched: bool = False,
    min_y: float = -90.0,
    max_y: float = 90.0,
    n_y: int = 5,
    y_name: str = "latitude",
    y_stretched: bool = False,
    min_z: float = 0.0,
    max_z: float = 500.0,
    n_z: int = 6,
    z_name: str = "depth",
    z_units: str = "km",
    z_stretched: bool = False,
    field_name: str = "test_field",
    n_fields: int = 1,
    coord_order: tuple[str, str, str] | list[str] | None = None,
    dtype: str = "float64",
    npseed: bool = False,
) -> xr.Dataset:

    valid_coord_order: tuple[str, str, str] = ("z", "y", "x")
    if isinstance(coord_order, list):
        if len(coord_order) == 3:
            valid_coord_order = tuple(coord_order)  # type: ignore[assignment]
        else:
            raise RuntimeError("coord_order length must be 3")
    elif isinstance(coord_order, tuple):
        valid_coord_order = coord_order

    dtype_to_use = getattr(np, dtype)

    # contruct and return a minimal xarray dataset to use in tests as needed

    # current known z types = ["height", "depth", "radius", "altitude"]
    is_stretched = {x_name: x_stretched, y_name: y_stretched, z_name: z_stretched}
    cdict = _DsConstructor(
        min_x=min_x,
        max_x=max_x,
        n_x=n_x,
        x_name=x_name,
        min_y=min_y,
        max_y=max_y,
        n_y=n_y,
        y_name=y_name,
        min_z=min_z,
        max_z=max_z,
        n_z=n_z,
        z_name=z_name,
        z_units=z_units,
        field_name=field_name,
        coord_order=valid_coord_order,
        is_stretched=is_stretched,
    )

    coords = {}
    var_shape: tuple[int, ...] = ()
    coord_order_rn: tuple[str, ...] = ()
    coord_arrays = []
    for c in valid_coord_order:
        current: _DimConstructor = getattr(cdict, c)
        cvals = current.values
        coords[current.dim_name] = cvals.astype(dtype_to_use)
        coord_arrays.append(coords[current.dim_name])
        coord_order_rn += (current.dim_name,)
        var_shape += (current.dim_n,)

    if npseed:
        np.random.seed(0)

    vals = np.random.random(var_shape).astype(dtype_to_use)
    if n_fields > 1:
        data_vars = {}
        for i in range(n_fields):
            fname = cdict.field_name + f"_{i}"
            da = xr.DataArray(vals, coords=coord_arrays, dims=coord_order_rn)
            data_vars[fname] = da
    else:
        data_vars = {
            cdict.field_name: xr.DataArray(
                vals, coords=coord_arrays, dims=coord_order_rn
            )
        }

    other_atts = {"geospatial_vertical_units": cdict.z_units}

    return xr.Dataset(data_vars=data_vars, attrs=other_atts)


def _test_time_coord(nt: int = 5) -> npt.NDArray:
    t0 = np.datetime64("2001-01-02").astype("datetime64[ns]")
    dt = np.timedelta64(1, "D")
    tvals = [t0]
    for _ in range(nt):
        tvals.append(tvals[-1] + dt)
    return np.array(tvals)


def _get_test_coord(
    cname, n, minv: float | None = None, maxv: float | None = None
) -> npt.NDArray[np.floating]:
    if cname in known_coord_aliases:
        cname = known_coord_aliases[cname]

    if cname == "time":
        return _test_time_coord(nt=n)

    if cname == "latitude":
        if minv is None:
            minv = -90.0
        if maxv is None:
            maxv = 90.0
        return np.linspace(minv, maxv, n)

    if cname == "longitude":
        if minv is None:
            minv = -180.0
        if maxv is None:
            maxv = 180.0
        return np.linspace(minv, maxv, n)

    if minv is None:
        minv = 0.0
    if maxv is None:
        maxv = 1.0

    return np.linspace(minv, maxv, n)


def construct_ds_with_extra_dim(
    icoord: int,
    dim_name: str = "time",
    ncoords: int | None = None,
    nd_space: int = 3,
    reverse_indices: list[int] | None = None,
) -> xr.Dataset:
    coord_configs = {
        0: (dim_name, "x", "y", "z"),
        1: (dim_name, "z", "y", "x"),
        2: (dim_name, "lon", "lat", "lev"),
        3: (dim_name, "longitude", "latitude", "altitude"),
        4: (dim_name, "depth", "longitude", "lat"),
    }

    data_vars = {}
    if ncoords is None:
        ncoords = icoord + 4

    full_dims = coord_configs[icoord]
    dim_order = [full_dims[idim] for idim in range(nd_space + 1)]
    coords = {c: _get_test_coord(c, ncoords) for c in dim_order}

    if reverse_indices is not None:
        for indx in reverse_indices:
            dim = dim_order[indx]
            coords[dim] = coords[dim][::-1]

    var_shape = tuple([len(c) for c in coords.values()])
    vals = np.random.random(var_shape)
    fname = f"test_case_{icoord}"
    da = xr.DataArray(vals, coords=coords, dims=dim_order)
    data_vars[fname] = da

    return xr.Dataset(data_vars=data_vars)


def _find_file(file: PathLike[str] | str) -> PathLike[str] | str:
    if os.path.isfile(file):
        return file

    ddir = yt.config.ytcfg.get("yt", "test_data_dir")
    default_val = yt.config.ytcfg_defaults["yt"]["test_data_dir"]
    if ddir != default_val:
        possible_file = os.path.join(ddir, file)
        if os.path.isfile(possible_file):
            return str(possible_file)

    raise OSError(
        f"Could not find {file} in current directory or in the yt search path ({ddir})"
    )


def _validate_file(function):
    def validate_then_call(file, *args, **kwargs):
        goodfile = _find_file(file)
        return function(goodfile, *args, **kwargs)

    return validate_then_call


def _import_optional_dep(
    name: str, package: str | None = None, custom_message: str | None = None
):
    # wrapper of importlib.import_module
    # name, package get sent to importlib.import_module
    # custom_message will overwrite the ImportError message if the package is not
    # found.
    try:
        return importlib.import_module(name, package=package)
    except ImportError:
        msg = custom_message
        if msg is None:
            msg = f"This functionality requires {name}. Install it and try again."
        raise ImportError(msg)
