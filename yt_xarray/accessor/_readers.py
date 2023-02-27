from typing import Optional

import numpy as np

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

        # this global start index accounts for indexing after any
        # subselections on the xarray DataArray are made. might
        # be a way to properly set this with yt grid objects.
        gsi = sel_info.starting_indices
        if sel_info.ndims == 2:
            gsi = np.append(gsi, 0)

        si = grid.get_global_startindex() + gsi
        ei = si + grid.ActiveDimensions
        global_dims = sel_info.global_dims.copy()
        if interp_required:
            # if interpolating, si and ei must be node indices so
            # we offset by an additional element
            cell_to_node_offset = np.ones((3,), dtype=int)
            ei = ei + cell_to_node_offset

        # build the index-selector for our yt grid object
        c_list = sel_info.selected_coords  # the xarray coord names
        i_select_dict = {}
        for idim in range(sel_info.ndims):
            if sel_info.reverse_axis[idim]:
                # the xarray axis is in negative ordering. a yt index of
                # 0 should point to the maximum index in xarray
                si_idim = global_dims[idim] - si[idim]
                ei_idim = global_dims[idim] - ei[idim]
                # when reverse slicing, the final index will not be included...
                # if the end index is within bounds, just bump it by one more,
                # otherwise if end index is already 0, just pass in None to
                # slice
                if ei_idim > 0:
                    ei_idim = ei_idim - 1
                elif ei_idim == 0:
                    ei_idim = None
                i_select_dict[c_list[idim]] = slice(si_idim, ei_idim, -1)
            else:
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

        if interp_required:
            # interpolate from nodes to cell centers across all remaining dims
            datavals = _xr_to_yt._interpolate_to_cell_centers(datavals)

        # return the plain values
        vals = datavals.values.astype(np.float64)
        if sel_info.ndims == 2:
            vals = np.expand_dims(vals, axis=-1)
        return vals

    return _reader
