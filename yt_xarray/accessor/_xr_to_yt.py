import collections.abc
import enum
from typing import List, Optional, Tuple

import numpy as np
import unyt
import xarray as xr

from yt_xarray.utilities.logging import ytxr_log

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
        self.units: dict = self._find_units(xr_ds)
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
        self.selected_time = None
        self.ndims: int = None
        self.grid_type = None  # one of _GridType members
        self.cell_widths: list = None
        self.global_dims: list = None
        self._process_selection(xr_ds)

        self.yt_coord_names = _convert_to_yt_internal_coords(self.selected_coords)

    def _find_units(self, xr_ds) -> dict:

        units = {}
        for field in self.fields:
            unit = getattr(xr_ds[field], "units", "")
            if unit != "" and hasattr(unyt.unit_symbols, unit) is False:
                unit = ""
            units[field] = unit
        return units

    def _find_starting_index(self, coordname, coord_da, coord_select) -> int:

        si = 0
        selector = coord_select[coordname]
        if self.sel_dict_type == "isel":
            if isinstance(selector, slice):
                si = selector.start
            elif isinstance(selector, int):
                si = selector
            elif isinstance(
                selector, (collections.abc.Sequence, np.ndarray, xr.DataArray)
            ):
                # assuming that the list, array, whatever is sequential.
                # could add a validation here. but we wont ever be able to handle
                # a case like [2,3,4, 8, 12, 13] in a yt dataset, doesnt make
                # sense.
                if _size_of_array_like(selector) > 1:
                    si = selector[0]
                else:
                    si = selector
            else:
                raise RuntimeError(f"Unexpected sel_dict value: {coordname}:{selector}")
        elif self.sel_dict_type == "sel":
            # find the coordinate index corresponding to the selection
            if isinstance(selector, slice):
                search_for = selector.start
            elif isinstance(selector, (float, np.datetime64, int)):
                search_for = selector
            elif isinstance(
                selector, (collections.abc.Sequence, np.ndarray, xr.DataArray)
            ):
                if _size_of_array_like(selector) > 1:
                    search_for = selector[0]
                else:
                    search_for = selector
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
        full_coords = list(xr_ds.data_vars[self.fields[0]].dims)
        time = 0.0
        # for each coordinate, apply any selector and then store min/max of the
        # coordinate. If the selector results in <= 1 in one of the dimensions,
        # that dimensions is dropped from the selection
        n_edges = []  # number of edges after selection
        n_cells = []  # number of cells after selection
        dimranges = []  # the min, max after selections
        full_dimranges = []  # the global min, max
        coord_list = []  # the coord list after selection
        starting_indices = []  # global starting index
        cell_widths = []  # cell widths after selection
        grid_type = _GridType.UNIFORM  # start with uniform assumption
        reverse_axis = []  # axes must be positive-monitonic for yt
        reverse_axis_names = []
        global_dims = []  # the global shape
        for c in full_coords:
            coord_da = getattr(xr_ds, c)  # the full coordinate data array

            # check if coordinate values are increasing
            if coord_da.size > 1:
                rev_ax = coord_da[1] <= coord_da[0]
                reverse_axis.append(bool(rev_ax.values))
                if rev_ax:
                    reverse_axis_names.append(c)

            # store the global ranges
            global_dims.append(coord_da.size)
            global_min = float(coord_da.min().values)
            global_max = float(coord_da.max().values)
            full_dimranges.append([global_min, global_max])

            si = 0  # starting xarray-index for any pre-selections from user
            coord_select = {}  # the selection dictionary just for this coordinate
            if c in self.sel_dict:
                coord_select[c] = self.sel_dict[c]
                si = self._find_starting_index(c, coord_da, coord_select)

            # apply any selections and extract coordinates
            sel_or_isel = getattr(coord_da, self.sel_dict_type)
            coord_vals = sel_or_isel(coord_select).values.astype(np.float64)
            is_time_dim = _check_for_time(c, coord_vals)

            if coord_vals.size > 1:

                # not positive-monotonic? reverse it for cell width calculations
                # changes to indexing are accounted for when extracting data.
                if reverse_axis[-1]:
                    coord_vals = coord_vals[::-1]

                cell_widths.append(coord_vals[1:] - coord_vals[:-1])
                dimranges.append([coord_vals.min(), coord_vals.max()])
                n_edges.append(coord_vals.size)
                n_cells.append(coord_vals.size - 1)
                coord_list.append(c)
                starting_indices.append(si)

                if is_time_dim:
                    raise NotImplementedError(
                        "Loading data with time as a dimension is not currently"
                        " supported. Please provide a selection dictionary to "
                        "select a single time to load."
                    )
                else:
                    # check if this dimension is a stretched grid. If it is, the
                    # grid will be treated as stretched.
                    if _check_grid_stretchiness(coord_vals) == _GridType.STRETCHED:
                        grid_type = _GridType.STRETCHED

            elif coord_vals.size == 1 and is_time_dim:
                time = coord_vals

        if len(n_edges) > 3:
            raise ValueError(
                f"ndim is {len(n_edges)}, please provide a sel_dict to"
                f" reduce dimensionality to 3."
            )
        self.ndims = len(n_edges)
        self.selected_shape = tuple(n_edges)
        self.select_shape_cells = tuple(n_cells)
        self.full_bbox = np.array(full_dimranges).astype(np.float64)
        self.selected_bbox = np.array(dimranges).astype(np.float64)
        self.full_coords = tuple(full_coords)
        self.selected_coords = tuple(coord_list)
        self.starting_indices = np.array(starting_indices)
        self.selected_time = time
        self.grid_type = grid_type
        self.cell_widths = cell_widths
        self.reverse_axis = reverse_axis
        self.reverse_axis_names = reverse_axis_names
        self.global_dims = np.array(global_dims)
        # self.coord_selected_arrays = coord_selected_arrays

        # set the yt grid dictionary
        self.grid_dict = {
            "left_edge": self.selected_bbox[:, 0],
            "right_edge": self.selected_bbox[:, 1],
            "dimensions": self.select_shape_cells,
            "level": 0,
        }

    def _validate_fields(self, xr_ds, fields: List[str]) -> List[str]:

        if fields is None:
            raise ValueError("Please provide a list of fields")

        # ensure that all fields have the same coordinates and the same shape
        shape = xr_ds.data_vars[fields[0]].shape
        coords = xr_ds.data_vars[fields[0]].dims

        if len(fields) > 1:
            msg = "Provided fields must have the same "

            for f in fields[1:]:

                if xr_ds.data_vars[f].shape != shape:
                    rmsg = msg + f"shape : {f} does not match {fields[0]}"
                    raise RuntimeError(rmsg)

                if tuple(xr_ds.data_vars[f].dims) != coords:
                    rmsg = msg + f"coordinates : {f} does not match {fields[0]}"
                    raise RuntimeError(rmsg)

        # check that we have vertex-centered data (coordinate dimensions should
        # match the field dimensions)
        dim_list = list(xr_ds.data_vars[fields[0]].dims)
        coord_shape = np.array([getattr(xr_ds, dim).size for dim in dim_list])
        field_shape = np.array(shape)
        if np.all(coord_shape == field_shape) is False:
            raise RuntimeError(
                "coordinate dimensions do not match field " "dimensions."
            )
        return fields

    def select_from_xr(self, xr_ds, field):
        if self.sel_dict_type == "isel":
            vars = xr_ds[field].isel(self.sel_dict)
        else:
            vars = xr_ds[field].sel(self.sel_dict)

        for axname in self.reverse_axis_names:
            dimvals = getattr(vars, axname)
            vars = vars.sel({axname: dimvals[::-1]})

        return vars

    def interp_validation(self, geometry):
        # checks if yt will need to interpolate to cell center
        # returns a tuple of (bool, shape, bbox). If the bool is True then
        # interpolation is required.
        ytxr_log.info(
            "Attempting to detect if yt_xarray will require field interpolation:"
        )

        if self.grid_type == _GridType.STRETCHED:
            ytxr_log.info("    stretched grid detected: yt_xarray will interpolate.")
            return True, self.select_shape_cells, self.selected_bbox
        elif self.grid_type == _GridType.UNIFORM:
            dxyz = np.array([cell_wids[0] for cell_wids in self.cell_widths])
            bbox = self.selected_bbox.copy()
            bbox_wid = bbox[:, 1] - bbox[:, 0]
            bbox[:, 0] = bbox[:, 0] - dxyz / 2
            bbox[:, 1] = bbox[:, 0] + bbox_wid + dxyz
            if geometry == "cartesian":
                # OK to wrap in a pseudo-grid, offset bounds by +/- 1/2 cell
                # spacing, return number of nodes as the number of cells for
                # the pseudo-grid and do not require interpolation.
                ytxr_log.info(
                    "    Cartesian geometry on uniform grid: yt_xarray will not interpolate."
                )
                return False, self.selected_shape, bbox

            elif geometry in ("geographic", "internal_geographic"):
                msg = (
                    "    Geodetic geometry bounds exceeded: yt_xarray will interpolate."
                )
                # check if still within bounds, if not, require interpolation
                for idim, dim in enumerate(self.selected_coords):
                    if dim == "latitude":
                        if bbox[idim, 1] > 90.0 or bbox[idim, 0] < -90.0:
                            ytxr_log.info(msg)
                            return True, self.select_shape_cells, self.selected_bbox
                    elif dim == "longitude":
                        if bbox[idim, 1] > 180.0 or bbox[idim, 0] < -180.0:
                            ytxr_log.info(msg)
                            return True, self.select_shape_cells, self.selected_bbox

                # should be OK to pad cells
                ytxr_log.info(
                    "    Geodetic geometry on uniform grid within geodetic "
                    "bounds: yt_xarray will not interpolate."
                )
                return False, self.selected_shape, bbox

            else:
                # the others would require similar bounds checks, so just
                # require interpolation for now at least.
                return True, self.select_shape_cells, self.selected_bbox
        else:
            raise RuntimeError(f"Unexptected grid type: {self.grid_type}")


_coord_aliases = {
    "altitude": ["altitude", "height", "level", "lev"],
    "latitude": ["latitude", "lat"],
    "longitude": ["longitude", "lon"],
}


known_coord_aliases = {}
for ky, vals in _coord_aliases.items():
    for val in vals:
        known_coord_aliases[val] = ky

_expected_yt_axes = {
    "cartesian": set(["x", "y", "z"]),
    "spherical": set(["r", "phi", "theta"]),
    "geographic": set(["altitude", "latitude", "longitude"]),
    "internal_geographic": set(["depth", "latitude", "longitude"]),
}

valid_geometries = tuple(_expected_yt_axes.keys()) + ("geodetic",)

_yt_coord_names = []
for vals in _expected_yt_axes.values():
    _yt_coord_names += list(vals)


def _convert_to_yt_internal_coords(coord_list):
    yt_coords = []
    for c in coord_list:
        cname = c.lower()
        if cname in known_coord_aliases:
            yt_coords.append(known_coord_aliases[cname])
        elif cname in _yt_coord_names:
            yt_coords.append(cname)
        else:
            raise ValueError(
                f"{c} is not a known coordinate. To load in yt, you "
                f"must supply an alias via the yt_xarray.known_coord_aliases"
                f" dictionary."
            )

    return yt_coords


def _determine_yt_geomtype(coord_type: str, coord_list: List[str]) -> Optional[str]:
    # mainly for expanding geodetic into internal_geographic or geographic as used
    # by yt
    if coord_type == "geodetic":
        # is it internal or external
        possible_alts = _coord_aliases["altitude"]
        if "depth" in coord_list:
            return "internal_geographic"
        elif any([i in coord_list for i in possible_alts]):
            return "geographic"
    elif coord_type in _expected_yt_axes.keys():
        return coord_type
    else:
        raise ValueError(f"Unsupported geometry type: {coord_type}")


def _add_3rd_axis_name(yt_geometry: str, axis_order: list) -> list:
    if len(axis_order) != 2:
        raise RuntimeError("This function should only be called for 2d data.")

    axis_set = set(axis_order)
    if yt_geometry in _expected_yt_axes.keys():
        yt_axes = _expected_yt_axes[yt_geometry]
    else:
        raise ValueError(f"Unsupported geometry type: {yt_geometry}")

    missing = list(yt_axes.difference(axis_set))
    if len(missing) == 1:
        return axis_order + [
            missing[0],
        ]

    raise RuntimeError("Could not determine missing coordinate.")


def _size_of_array_like(v):

    if isinstance(v, (np.ndarray, xr.DataArray)):
        return v.size

    return len(v)


def _validate_geometry(possible_geom: str) -> str:
    if possible_geom in valid_geometries:
        return possible_geom
    raise ValueError(f"{possible_geom} is not a valid geometry")


class _GridType(enum.Flag):
    UNIFORM = enum.auto()
    STRETCHED = enum.auto()
    UNKNOWN = enum.auto()


def _check_grid_stretchiness(x):
    dx = np.unique(x[1:] - x[:-1])
    grid_tol = 1e-10  # could be a user setting
    if np.allclose(dx, [dx[0]], grid_tol):
        return _GridType.UNIFORM
    else:
        return _GridType.STRETCHED


def _check_for_time(dim_name, dim_vals: np.ndarray):
    return "time" in dim_name.lower() or type(dim_vals) is np.datetime64


def _interpolate_to_cell_centers(data: xr.DataArray):
    # linear interpolation from nodes to cell centers across all dimensions of
    # a DataArray
    interp_dict = {}
    for dim in data.dims:
        dimvals = data.coords[dim].values
        interp_dict[dim] = (dimvals[1:] + dimvals[:-1]) / 2.0
    return data.interp(interp_dict)


def _load_full_field_from_xr(
    ds_xr, field: str, sel_info: Selection, interp_required: bool = False
):
    vals = sel_info.select_from_xr(ds_xr, field).load()

    if interp_required:
        vals = _interpolate_to_cell_centers(vals)

    vals = vals.values.astype(np.float64)
    if sel_info.ndims == 2:
        vals = np.expand_dims(vals, axis=-1)
    return vals
