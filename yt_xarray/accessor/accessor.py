from collections import defaultdict
from typing import List, Optional

import numpy as np
import xarray as xr
import yt
from unyt import unyt_quantity

from yt_xarray.accessor import _xr_to_yt
from yt_xarray.accessor._readers import _get_xarray_reader
from yt_xarray.accessor._xr_to_yt import _load_full_field_from_xr
from yt_xarray.utilities.logging import ytxr_log


@xr.register_dataset_accessor("yt")
class YtAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._bbox_cart = {}
        self._bbox = {}
        self._field_grids = defaultdict(lambda: None)

    def load_grid(
        self,
        fields: Optional[List[str]] = None,
        geometry: str = None,
        use_callable: bool = True,
        sel_dict: Optional[dict] = None,
        sel_dict_type: Optional[str] = "isel",
        chunksizes: Optional[int] = None,
        **kwargs,
    ):
        """
        Initializes a yt gridded dataset for the supplied fields.

        Parameters
        ----------
        fields : list[str]
            list of fields to include. If None, will try to use all fields

        geometry : str
            the geometry to pass to yt.load_uniform grid. If not provided,
            will attempt to infer.

        use_callable : bool
            if True (default), then the yt dataset will utilize links to the open
            xarray Dataset handle to avoid copying memory.

        sel_dict: dict
            an optional selection dictionary to apply to the fields before yt dataset
            initialization

        sel_dict_type: str
            either "isel" (default) or "sel" to indicate index or value selection for
            sel_dict.

        kwargs :
            any additional keyword arguments to pass to yt.load_uniform_grid

        Returns
        -------
        yt StreamDataset

        """
        if fields is None:
            # might as well try!
            fields = list(self._obj.data_vars)

        sel_info = _xr_to_yt.Selection(
            self._obj,
            fields=fields,
            sel_dict=sel_dict,
            sel_dict_type=sel_dict_type,
        )
        if sel_info.grid_type == _xr_to_yt._GridType.STRETCHED and use_callable:
            # why not? this should work now, shouldnt it?
            raise NotImplementedError(
                "Detected a stretched grid, which is not yet supported for callables, "
                "set use_callable=False."
            )

        if geometry is None:
            geometry = self.geometry
        geometry = _xr_to_yt._determine_yt_geomtype(geometry, sel_info.selected_coords)
        if geometry is None:
            raise ValueError(
                "Cannot determine yt geometry type, please provide"
                "geometry = 'geographic', 'internal_geographic' or 'cartesian'"
            )

        if "length_unit" in kwargs:
            length_unit = kwargs.pop("length_unit")
        else:
            length_unit = self._infer_length_unit()
            if length_unit is None:
                raise ValueError(
                    "cannot determine length_unit, please provide as"
                    " a keyword argument."
                )

        axis_order = sel_info.yt_coord_names

        geom = (geometry, axis_order)

        simtime = sel_info.selected_time
        if isinstance(sel_info.selected_time, np.datetime64):
            # float(simtime) will be nanoseconds before/after 1970-01-01
            # would be nice to have yt ingest a np datetime, especially cause
            # this will be converted to a float, so the roundtrip will not
            # match exactly.
            simtime = unyt_quantity(int(simtime), "ns")
        kwargs.update({"sim_time": simtime})

        if chunksizes is None:
            return _load_single_grid(
                self._obj, sel_info, geom, use_callable, fields, length_unit, **kwargs
            )
        elif sel_info.grid_type == _xr_to_yt._GridType.STRETCHED:
            raise NotImplementedError(
                "Stretched grids cannot set the chunksizes argument."
            )
        else:
            return _load_chunked_grid(
                self._obj,
                sel_info,
                geom,
                use_callable,
                fields,
                length_unit,
                chunksizes,
                **kwargs,
            )

    def _infer_length_unit(self):
        if self.geometry == "geodetic":
            return 1.0
        elif hasattr(self._obj, "geospatial_vertical_units"):
            # some netcdf conventions have this!
            return self._obj.geospatial_vertical_units
        return None

    _geometry = None

    def set_geometry(self, geometry: str):
        """
        Set the geometry for the dataset.

        Parameters
        ----------

        geometry: str
        the dataset's coordinate type. See yt_xarray.valid_geometries
        for possible values.

        """

        self._geometry = _xr_to_yt._validate_geometry(geometry)

    @property
    def geometry(self) -> str:
        """the dataset geometry"""
        if self._geometry is None:
            self._geometry = self._infer_geometry()
        return self._geometry

    def _infer_geometry(self) -> str:
        # try to infer if we have a geodetic dataset. the differentiation
        # between internal and not used by yt (internal_geographic vs geographic)
        # is not applied here, but is when converting to a yt dataset (see
        # _xr_to_yt._determine_yt_geomtype). Default is to assume cartesian.
        geodetic_names = (
            _xr_to_yt._coord_aliases["latitude"] + _xr_to_yt._coord_aliases["longitude"]
        )
        ctype = "cartesian"
        for coord in list(self._obj.coords):
            if coord.lower() in geodetic_names:
                ctype = "geodetic"

        ytxr_log.info(
            f"Inferred geometry type is {ctype}. To override, use ds.yt.set_geometry"
        )
        return ctype

    @property
    def _coord_list(self):
        # a list of all dataset coordinates. Note that dataset fields
        # may use a different ordering!!!
        return list(self._obj.coords.keys())

    def get_bbox(
        self,
        field: str,
        sel_dict: Optional[dict] = None,
        sel_dict_type: Optional[str] = "isel",
    ) -> np.ndarray:
        """
        return the bounding box array for a field, with possible selections

        Parameters
        ----------
        field
            the field to check the bounding box for

        Returns
        -------

        np.ndarray

        an array with shape (3, 2) with the min, max values for each dimension
        of the coordinates of a field.

        """

        sel_info = _xr_to_yt.Selection(
            self._obj, fields=[field], sel_dict=sel_dict, sel_dict_type=sel_dict_type
        )
        return sel_info.selected_bbox


def _load_single_grid(
    ds_xr, sel_info, geom, use_callable, fields, length_unit, **kwargs
):

    geometry = geom[0]

    interp_required, data_shp, bbox = sel_info.interp_validation(geometry)
    g_dict = sel_info.grid_dict.copy()
    g_dict["dimensions"] = data_shp
    g_dict["left_edge"] = bbox[:, 0]
    g_dict["right_edge"] = bbox[:, 1]

    if sel_info.ndims == 2:
        axis_order = geom[1]
        axis_order = _xr_to_yt._add_3rd_axis_name(geom[0], axis_order)
        geom = (geom[0], axis_order)
        data_shp = data_shp + (1,)
        bbox = np.vstack([bbox, [-0.5, 0.5]])

    data = {}
    if use_callable:
        reader = _get_xarray_reader(ds_xr, sel_info, interp_required=interp_required)

    for field in fields:
        units = sel_info.units[field]
        if use_callable:
            data[field] = (reader, units)
        else:
            vals = _load_full_field_from_xr(
                ds_xr, field, sel_info, interp_required=interp_required
            )
            data[field] = (vals, units)

    if sel_info.ndims == 2:
        g_dict["left_edge"] = np.append(g_dict["left_edge"], -0.5)
        g_dict["right_edge"] = np.append(g_dict["right_edge"], 0.5)
        g_dict["dimensions"] += (1,)

    if sel_info.grid_type == _xr_to_yt._GridType.STRETCHED:
        return yt.load_uniform_grid(
            data,
            data_shp,
            geometry=geom[0],
            bbox=bbox,
            length_unit=length_unit,
            cell_widths=sel_info.cell_widths,
            axis_order=geom[1],
            **kwargs,
        )
    else:
        data.update(g_dict)
        grid_data = [
            data,
        ]
        return yt.load_amr_grids(
            grid_data,
            data_shp,
            geometry=geom[0],
            bbox=bbox,
            length_unit=length_unit,
            axis_order=geom[1],
            **kwargs,
        )


def _load_chunked_grid(
    ds_xr, sel_info, geom, use_callable, fields, length_unit, chunksizes, **kwargs
):

    if isinstance(chunksizes, int):
        chunksizes = np.array((chunksizes,) * sel_info.ndims)
    elif len(chunksizes) != sel_info.ndims:
        raise ValueError(
            f"The number of elements in chunksizes {len(chunksizes)} "
            f"must match the dimensionality {sel_info.ndims}"
        )
    else:
        chunksizes = np.asarray(chunksizes, dtype=int)

    if sel_info.ndims != 3:
        raise NotImplementedError(
            "Can only load a chunked grid with 3D fields at present."
        )

    geometry = geom[0]

    # get the global shape and bounding box
    interp_required, data_shp, bbox = sel_info.interp_validation(geometry)

    # note: if interp_required, data_shp is number of cells
    # otherwise it is number of nodes (which are treated as new cell centers).
    # the bbox will already account for this as well.

    # do some grid/chunk counting
    n_chnk = np.asarray(data_shp) / chunksizes  # may not be int
    n_whl_chnk = np.floor(n_chnk).astype(int)  # whole chunks in each dim
    n_part_chnk = np.ceil(n_chnk - n_whl_chnk).astype(int)  # partial chunks

    n_tots = np.prod(n_part_chnk + n_whl_chnk)
    ytxr_log.info(f"Constructing a yt chunked grid with {n_tots} chunks.")

    # initialize the global starting index
    si = np.array([0, 0, 0], dtype=int)
    si = sel_info.starting_indices + si

    # select field for grabbing coordinate arrays -- fields should all be
    # verified by now
    fld = fields[0]
    cnames = sel_info.selected_coords

    if interp_required is False:
        dxyz = np.array([cell_wids[0] for cell_wids in sel_info.cell_widths])

    # build arrays of the left_edges, right_edges and dimensions. these will
    # be organized by dimension first (all of the left edges in x, all
    # the left edges in y, etc.)
    left_edges = []
    right_edges = []
    subgrid_sizes = []
    subgrid_start = []
    subgrid_end = []
    for idim in range(sel_info.ndims):

        si_0 = si[idim] + chunksizes[idim] * np.arange(n_whl_chnk[idim])
        ei_0 = si_0 + chunksizes[idim]

        if n_part_chnk[idim] == 1:
            si_0_partial = ei_0[-1]
            ei_0_partial = data_shp[idim] - si_0_partial
            si_0 = np.concatenate(
                [
                    si_0,
                    [
                        si_0_partial,
                    ],
                ]
            )
            ei_0 = np.concatenate(
                [
                    ei_0,
                    [
                        ei_0[-1] + ei_0_partial,
                    ],
                ]
            )

        c = cnames[idim]
        rev_ax = sel_info.reverse_axis[idim]
        if rev_ax is False:
            le_0 = ds_xr[fld].coords[c].isel({c: si_0}).values
            if interp_required is False:
                # move the edges so the node is now a cell center
                le_0 = le_0 - dxyz[idim] / 2.0

            # bbox value below already accounts for interp_required
            max_val = bbox[idim, 1]
            re_0 = np.concatenate([le_0[1:], [max_val]])

        else:
            re_0 = ds_xr[fld].coords[c].isel({c: si_0[::-1]}).values
            if interp_required is False:
                # move the edges so the node is now a cell center
                re_0 = re_0 - dxyz[idim] / 2.0
            min_val = bbox[idim, 0]
            le_0 = np.concatenate([[min_val], re_0[:-1]])

        # sizes also already account for interp_required
        subgrid_size = ei_0 - si_0

        left_edges.append(le_0)
        right_edges.append(re_0)
        subgrid_sizes.append(subgrid_size)
        subgrid_start.append(si_0)
        subgrid_end.append(ei_0)

    # these arrays are ordered by dimension. e.g., left_edges[0] will be the
    # all first dimension left edges
    left_edges = np.meshgrid(*left_edges, indexing="ij")
    right_edges = np.meshgrid(*right_edges, indexing="ij")
    subgrid_sizes = np.meshgrid(*subgrid_sizes, indexing="ij")
    subgrid_start = np.meshgrid(*subgrid_start, indexing="ij")
    subgrid_end = np.meshgrid(*subgrid_end, indexing="ij")

    # re-organize by grid number so that, e.g., the left_edges are the usual
    # left_edges (left_edges[0] is the min x, y, z of grid 0)
    left_edges = np.column_stack([le.ravel() for le in left_edges])
    right_edges = np.column_stack([re.ravel() for re in right_edges])
    dimensions = np.column_stack([sz.ravel() for sz in subgrid_sizes])
    subgrid_start = np.column_stack([sz.ravel() for sz in subgrid_start])
    subgrid_end = np.column_stack([sz.ravel() for sz in subgrid_end])

    # now ready to build the list of grids
    if use_callable:
        reader = _get_xarray_reader(ds_xr, sel_info, interp_required=interp_required)

    grid_data = []
    n_grids = len(left_edges)

    if use_callable is False:
        full_field_vals = {}
        for field in fields:
            vals = _load_full_field_from_xr(
                ds_xr, field, sel_info, interp_required=interp_required
            )
            full_field_vals[field] = vals

    for igrid in range(n_grids):
        gdict = {
            "left_edge": left_edges[igrid],
            "right_edge": right_edges[igrid],
            "dimensions": dimensions[igrid],
            "level": 0,
        }
        for field in fields:
            units = sel_info.units[field]
            if use_callable:
                gdict[field] = (reader, units)
            else:
                si = subgrid_start[igrid]
                ei = subgrid_end[igrid]
                gridvals = full_field_vals[field][
                    si[0] : ei[0], si[1] : ei[1], si[2] : ei[2]
                ]
                gdict[field] = (gridvals, units)
        grid_data.append(gdict)

    return yt.load_amr_grids(
        grid_data,
        data_shp,
        geometry=geom[0],
        bbox=bbox,
        length_unit=length_unit,
        axis_order=geom[1],
        **kwargs,
    )
