import collections.abc
from typing import List, Optional, Tuple

import numpy as np

# collection of helpful classes and functions that ease the communication between
# xarray and yt datasets


class Selection:
    """
    a helper-container that parses an xarray field (or fields) and records the
    information necessary to load into yt:
    * reduction to 3D
    * coordinate aliasing
    * bounding box info

    Does not handle stretched grids, use somethinge else for that....

    Important: this class does not store references to xarray or yt datasets!
    """

    def __init__(
        self,
        xr_ds,
        fields: List[str] = None,
        sel_dict: Optional[dict] = None,
        sel_dict_type: Optional[str] = "isel",
    ):

        self.fields = self._validate_fields(xr_ds, fields)
        self.units = self._find_units(xr_ds)
        self.full_shape = xr_ds.data_vars[self.fields[0]].shape

        self.sel_dict = sel_dict or {}
        if sel_dict_type not in ["sel", "isel"]:
            raise RuntimeError(
                f"sel_dict_type must be 'sel' or 'isel', got {sel_dict_type}"
            )
        self.sel_dict_type = sel_dict_type

        # all these attributes are set in _process_selection
        self.selected_shape: Tuple[int] = None
        self.full_bbox = None
        self.selected_bbox = None
        self.full_coords: Tuple[str] = None
        self.selected_coords: Tuple[str] = None
        self.starting_indices: np.ndarray = None

        self._process_selection(xr_ds)

        self.yt_coord_names = _convert_to_yt_internal_coords(self.selected_coords)

    def _find_units(self, xr_ds) -> dict:
        units = {getattr(xr_ds[field], "units", None) or "" for field in self.fields}
        # could validate with unyt to see if it exists...
        return units

    def _find_starting_index(self, coordname, coord_da, coord_select) -> int:

        si = 0
        selector = coord_select[coordname]
        if self.sel_dict_type == "isel":
            if isinstance(selector, slice):
                si = selector.start
            elif isinstance(selector, int):
                si = selector
            elif isinstance(selector, (collections.abc.Sequence, np.ndarray)):
                # assuming that the list, array, whatever is sequential.
                # could add a validation here. but we wont ever be able to handle
                # a case like [2,3,4, 8, 12, 13] in a yt dataset, doesnt make
                # sense.
                si = selector[0]
            else:
                raise RuntimeError(f"Unexpected sel_dict value: {coordname}:{selector}")
        elif self.sel_dict_type == "sel":
            # find the coordinate index corresponding to the selection
            if isinstance(selector, slice):
                search_for = selector.start
            elif isinstance(selector, float):
                search_for = selector
            elif isinstance(selector, (collections.abc.Sequence, np.ndarray)):
                search_for = selector[0]
            else:
                raise RuntimeError(f"Unexpected sel_dict value: {coordname}:{selector}")
            the_index = np.where(coord_da == search_for)[0]
            if len(the_index) == 0:
                raise RuntimeError(
                    f"Could not find selected value: {coordname}:{search_for}"
                )
            si = the_index[0]

        return si

    def _process_selection(self, xr_ds):

        # the full list of coordinates (in order)
        full_coords = list(xr_ds.data_vars[self.fields[0]].coords)

        # for each coordinate, apply any selector and then store min/max of the
        # coordinate. If the selector results in <= 1 in one of the dimensions,
        # that dimensions is dropped from the selection
        shape = []  # the shape after selctions
        dimranges = []  # the min, max after selections
        full_dimranges = []  # the global min, max
        coord_list = []  # the coord list after selection
        starting_indices = []  # global starting index
        for c in full_coords:
            coord_da = getattr(xr_ds, c)  # the full coordinate data array

            # store the global
            global_min = float(coord_da.min().values)
            global_max = float(coord_da.max().values)
            full_dimranges.append([global_min, global_max])

            si = 0  # starting index
            coord_select = {}  # the selection dictionary just for this coordinate
            if c in self.sel_dict:
                coord_select[c] = self.sel_dict[c]
                si = self._find_starting_index(coord_da, coord_select)

            sel_or_isel = getattr(coord_da, self.sel_dict_type)
            coord_vals = sel_or_isel(coord_select).values
            if coord_vals.size > 1:
                dimranges.append([coord_vals.min(), coord_vals.max()])
                shape.append(coord_vals.size)
                coord_list.append(c)
                starting_indices.append(si)

        self.selected_shape = tuple(shape)
        self.full_bbox = np.array(full_dimranges)
        self.selected_bbox = np.array(dimranges)
        self.full_coords = tuple(full_coords)
        self.selected_coords: tuple(coord_list)
        self.starting_indices = np.array(starting_indices)

        self.grid_dict = {
            "left_edge": self.selected_bbox[:, 0],
            "right_edge": self.selected_bbox[:, 1],
            "dimensions": self.selected_shape,
            "level": 0,
        }

    def _validate_fields(self, xr_ds, fields: List[str]) -> List[str]:

        if fields is None:
            raise ValueError("Please provide a list of fields")

        # ensure that all fields have the same coordinates and the same shape
        shape = xr_ds.data_vars[fields[0]].shape
        coords = tuple(xr_ds.data_vars[fields[0]].coords)

        if len(fields) > 1:
            msg = "Provided fields must have the same "

            for f in fields[1:]:

                if xr_ds.data_vars[f].shape != shape:
                    rmsg = msg + f"shape : {f} does not match {fields[0]}"
                    raise RuntimeError(rmsg)

                if tuple(xr_ds.data_vars[f].coords) != coords:
                    rmsg = msg + f"coordinates : {f} does not match {fields[0]}"
                    raise RuntimeError(rmsg)

        return fields

    def select_from_xr(self, xr_ds, field):
        if self.sel_dict_type == "isel":
            return xr_ds[field].isel(self.sel_dict)
        else:
            return xr_ds[field].sel(self.sel_dict)


_coord_aliases = {
    "altitude": ["altitude", "height", "level", "lev"],
    "latitude": ["latitude", "lat"],
    "longitude": ["longitude", "lon"],
}

_coord_aliases_rev = {}
for ky, vals in _coord_aliases.items():
    for val in vals:
        _coord_aliases_rev[val] = ky


def _convert_to_yt_internal_coords(coord_list):
    yt_coords = []
    for c in coord_list:
        if c.lower() in _coord_aliases_rev:
            yt_coords.append(_coord_aliases_rev[c.lower()])
        else:
            yt_coords.append(c)

    return yt_coords
