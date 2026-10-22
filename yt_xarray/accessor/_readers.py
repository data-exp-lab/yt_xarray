from typing import Literal, Optional

import numpy as np
from numpy import typing as npt

from yt_xarray.accessor import _xr_to_yt


def _get_xarray_reader(
    handle, sel_info: _xr_to_yt.Selection, interp_required: Optional[bool] = True
):
    # the callable generator for an open xarray handle.
    # handle: an open xarray handle
    # sel_info: a Selection object for the handle
    # reader_type: specifies what type of reader (only "node_to_cell" exists now)

    def _reader(grid, field_name):
        # grid: a yt grid object
        _, fname = field_name

        # first get the internal yt index ranges
        si = grid.get_global_startindex()
        ei = si + grid.ActiveDimensions

        # convert to the xarray indexing in 3 steps
        # 1. account for dimension direction
        # 2. account for any prior xarray subselections
        # 3. account for interpolation requirements

        # step 1 (if axis has been reversed, node ordering is also reversed)
        for idim in range(sel_info.ndims):
            if sel_info.reverse_axis[idim]:
                # note that the si, ei are exchanged!
                si0 = si.copy()
                ei0 = ei.copy()
                si[idim] = sel_info.global_dims_no_time[idim] - ei0[idim]
                ei[idim] = sel_info.global_dims_no_time[idim] - si0[idim]

        # step 2: this global start index accounts for indexing after any
        # subselections on the xarray DataArray are made. might
        # be a way to properly set this with yt grid objects.
        gsi = sel_info.starting_indices
        if sel_info.ndims == 2:
            gsi = np.append(gsi, 0)
        si = si + gsi
        ei = ei + gsi

        # step 3: if we are interpolating to cell centers, grab some extra nodes
        if interp_required:
            for idim in range(sel_info.ndims):
                if sel_info.reverse_axis[idim]:
                    si[idim] = si[idim] - 1
                else:
                    ei[idim] = ei[idim] + 1

        # now we can select the data (still accounting for any dimension reversal)

        # build the index-selector for each dimension
        c_list = sel_info.selected_coords  # the xarray coord names
        i_select_dict = {}
        for idim in range(sel_info.ndims):
            i_select_dict[c_list[idim]] = slice(si[idim], ei[idim])

        # set any of the initial selections that will reduce the
        # dimensionality or size of the full DataArray
        first_selection = {}
        for ky, val in sel_info.sel_dict.items():
            if ky not in i_select_dict:
                if sel_info.sel_dict_type == "sel":
                    first_selection[ky] = val
                else:
                    # just add it to the i_select_dict
                    i_select_dict[ky] = val

        var = getattr(handle, fname)  # the xarray DataArray
        if len(first_selection) > 0:
            datavals = var.sel(first_selection).isel(i_select_dict)
        else:
            datavals = var.isel(i_select_dict)

        # load into memory (if its not) as xr DataArray
        datavals = datavals.load()

        # reverse axis ordering if needed
        for axname in sel_info.reverse_axis_names:
            dimvals = getattr(datavals, axname)
            datavals = datavals.sel({axname: dimvals[::-1]})

        if interp_required:
            # interpolate from nodes to cell centers across all remaining dims
            datavals = _xr_to_yt._interpolate_to_cell_centers(datavals)

        # return the plain values
        vals = datavals.values.astype(np.float64)
        if sel_info.ndims == 2:
            vals = np.expand_dims(vals, axis=-1)
        return vals

    return _reader


def _interpolate_to_uniform_grid(
    handle,
    sel_info: _xr_to_yt.Selection,
    field: str,
    left_edge: npt.NDArray[np.float64],
    dds: npt.NDArray[np.float64],
    dimensions: npt.NDArray[np.int64],
    interp_method: Literal["linear", "nearest"] = "linear",
    fill_value: float = np.nan,
) -> npt.NDArray[np.float64]:
    # interpolate a field from an open xarray handle onto the cell centers of
    # a uniform grid patch defined by its left edge, cell spacing and
    # dimensions (all in the native coordinates of the xarray dataset,
    # ordered by sel_info.selected_coords).

    data = sel_info.select_from_xr(handle, field)

    interp_dict = {}
    isel_dict = {}
    for idim, c in enumerate(sel_info.selected_coords):
        centers = left_edge[idim] + (np.arange(dimensions[idim]) + 0.5) * dds[idim]
        interp_dict[c] = centers

        # only pull the source nodes that bracket this grid (plus one node of
        # padding on each side) so that lazily loaded data stays lazy.
        cvals = data.coords[c].values
        i0 = np.searchsorted(cvals, centers[0], side="right") - 1
        i1 = np.searchsorted(cvals, centers[-1], side="left") + 1
        i0 = int(np.clip(i0, 0, cvals.size - 2))
        i1 = int(np.clip(max(i1, i0 + 2), 0, cvals.size))
        isel_dict[c] = slice(i0, i1)

    data = data.isel(isel_dict).load()
    interpolated = data.interp(
        interp_dict, method=interp_method, kwargs={"fill_value": fill_value}
    )
    vals = interpolated.transpose(*sel_info.selected_coords, ...).values
    result: npt.NDArray[np.float64] = np.reshape(vals, tuple(dimensions)).astype(
        np.float64
    )
    return result


def _get_interpolating_reader(
    handle,
    sel_info: _xr_to_yt.Selection,
    interp_method: Literal["linear", "nearest"] = "linear",
    fill_value: float = np.nan,
):
    # the callable generator for interpolating from an open xarray handle
    # onto any uniform yt grid patch (at any refinement level).

    def _reader(grid, field_name):
        _, fname = field_name
        left_edge = grid.LeftEdge.to("code_length").d
        dds = grid.dds.to("code_length").d
        return _interpolate_to_uniform_grid(
            handle,
            sel_info,
            fname,
            left_edge,
            dds,
            grid.ActiveDimensions,
            interp_method=interp_method,
            fill_value=fill_value,
        )

    return _reader
