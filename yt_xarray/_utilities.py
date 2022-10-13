from typing import Optional, Tuple

import numpy as np
import xarray as xr


def construct_minimal_ds(
    min_x: float = 0.0,
    max_x: float = 360.0,
    n_x: int = 4,
    x_name: str = "longitude",
    min_y: float = -90.0,
    max_y: float = 90.0,
    n_y: int = 5,
    y_name: str = "latitude",
    min_z: float = 0.0,
    max_z: float = 500.0,
    n_z: int = 6,
    z_name: str = "depth",
    z_units: str = "km",
    field_name: str = "test_field",
    n_fields: int = 1,
    coord_order: Optional[Tuple[str, str, str]] = None,
) -> xr.Dataset:

    if coord_order is None:
        coord_order = ("z", "y", "x")

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

    coords = {}
    var_shape = ()
    coord_order_rn = ()
    coord_arrays = []
    for c in coord_order:
        mn = cdict["min_" + c]
        mx = cdict["max_" + c]
        n = cdict["n_" + c]
        cname = c + "_name"
        coords[cdict[cname]] = np.linspace(mn, mx, n)
        coord_arrays.append(coords[cdict[cname]])
        coord_order_rn += (cdict[cname],)
        var_shape += (n,)

    vals = np.random.random(var_shape)
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
