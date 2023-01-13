from typing import Dict, Optional

import numpy as np
import xarray as xr


def load_random_xr_data(
    fields: Dict[str, tuple], dims: Dict[str, tuple], length_unit: Optional[str] = None
) -> xr.Dataset:
    """

    Parameters
    ----------
    fields : Dict[str, tuple]
        A dictionary specifying fields and their dimensions
        {'field1': ('x', 'y', 'z'), 'field2': ('x', 'y')}
    dims
        a dictionary mapping any dimensions to their start, stop and size.
        Any dimensions specified in fields must exist here.
        {'x': (0, 1, 10), 'y': (0, 2, 15)} would create an x and y dimension
        that goes from 0 to 1 with 10 elements for x, 0 to 2 with 15 elements
        for y.

    Returns
    -------
    xr.Dataset
        an xarray Dataset with fields of random values with the supplied names
        and dimensions.

    """
    available_coords = {}
    for dim_name, dim_range in dims.items():
        available_coords[dim_name] = np.linspace(*dim_range)

    data = {}
    for field, field_dims in fields.items():

        coords = {}
        sz = []
        for dim_name in field_dims:
            if dim_name not in available_coords:
                raise KeyError(
                    f"{dim_name} is specified as a dimension for "
                    f"{field} but does not exist in the dims argument!"
                )

            sz.append(dims[dim_name][2])
            coords[dim_name] = available_coords[dim_name]

        data[field] = xr.DataArray(np.random.rand(*sz), coords=coords, dims=field_dims)

    attrs = {}
    if length_unit is not None:
        attrs["geospatial_vertical_units"] = length_unit
    return xr.Dataset(data, attrs=attrs)
