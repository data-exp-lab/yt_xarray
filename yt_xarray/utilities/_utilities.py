import os.path
from typing import Optional, Tuple

import numpy as np
import xarray as xr
import yt.config

from yt_xarray.accessor._xr_to_yt import known_coord_aliases


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
    coord_order: Optional[Tuple[str, str, str]] = None,
    dtype: str = "float64",
    npseed: bool = False,
) -> xr.Dataset:

    if coord_order is None:
        coord_order = ("z", "y", "x")

    dtype_to_use = getattr(np, dtype)

    # contruct and return a minimal xarray dataset to use in tests as needed

    # current known z types = ["height", "depth", "radius", "altitude"]

    cdict = dict(
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
        coord_order=coord_order,
    )

    is_stretched = {x_name: x_stretched, y_name: y_stretched, z_name: z_stretched}

    coords = {}
    var_shape = ()
    coord_order_rn = ()
    coord_arrays = []
    for c in coord_order:
        mn = cdict["min_" + c]
        mx = cdict["max_" + c]
        n = cdict["n_" + c]
        cname = c + "_name"
        cvals = np.linspace(mn, mx, n)
        if is_stretched[cdict[cname]]:
            dx = cvals[1:] - cvals[:-1]
            cvals[int(n / 2) :] = cvals[int(n / 2) :] + dx[0] * 2.0
        coords[cdict[cname]] = cvals.astype(dtype_to_use)
        coord_arrays.append(coords[cdict[cname]])
        coord_order_rn += (cdict[cname],)
        var_shape += (n,)

    if npseed:
        np.random.seed(0)

    vals = np.random.random(var_shape).astype(dtype_to_use)
    if n_fields > 1:
        data_vars = {}
        for i in range(n_fields):
            fname = cdict["field_name"] + f"_{i}"
            da = xr.DataArray(vals, coords=coord_arrays, dims=coord_order_rn)
            data_vars[fname] = da
    else:
        data_vars = {
            cdict["field_name"]: xr.DataArray(
                vals, coords=coord_arrays, dims=coord_order_rn
            )
        }

    other_atts = {"geospatial_vertical_units": cdict["z_units"]}

    return xr.Dataset(data_vars=data_vars, attrs=other_atts)


def _test_time_coord(nt=5):
    t0 = np.datetime64("2001-01-02").astype("datetime64[ns]")
    dt = np.timedelta64(1, "D")
    tvals = [t0]
    for _ in range(nt):
        tvals.append(tvals[-1] + dt)
    return np.array(tvals)


def _get_test_coord(
    cname, n, minv: Optional[float] = None, maxv: Optional[float] = None
):

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


def construct_ds_with_extra_dim(icoord: int, dim_name: str = "time"):

    coord_configs = {
        0: (dim_name, "x", "y", "z"),
        1: (dim_name, "z", "y", "x"),
        2: (dim_name, "lon", "lat", "lev"),
        3: (dim_name, "longitude", "latitude", "altitude"),
        4: (dim_name, "depth", "longitude", "lat"),
    }

    data_vars = {}
    coords = {c: _get_test_coord(c, icoord + 4) for c in coord_configs[icoord]}
    var_shape = tuple([len(c) for c in coords.values()])
    vals = np.random.random(var_shape)
    fname = f"test_case_{icoord}"
    da = xr.DataArray(vals, coords=coords, dims=coord_configs[icoord])
    data_vars[fname] = da

    return xr.Dataset(data_vars=data_vars)


def _find_file(file):
    if os.path.isfile(file):
        return file

    ddir = yt.config.ytcfg.get("yt", "test_data_dir")
    default_val = yt.config.ytcfg_defaults["yt"]["test_data_dir"]
    if ddir != default_val:
        possible_file = os.path.join(ddir, file)
        if os.path.isfile(possible_file):
            return possible_file

    raise OSError(
        f"Could not find {file} in current directory or in the yt search path ({ddir})"
    )


def _validate_file(function):
    def validate_then_call(file, *args, **kwargs):
        goodfile = _find_file(file)
        return function(goodfile, *args, **kwargs)

    return validate_then_call
