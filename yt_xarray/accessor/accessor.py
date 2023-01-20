from collections import defaultdict
from typing import List, Optional

import numpy as np
import xarray as xr
import yt
from unyt import unyt_quantity

from yt_xarray.accessor import _xr_to_yt
from yt_xarray.logging import ytxr_log


def _get_xarray_reader(handle, sel_info: _xr_to_yt.Selection):
    # the callable generator for an open xarray handle.
    # handle: an open xarray handle
    # sel_info: a Selection object for the handle
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

        # build the index-selector for our yt grid object
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
        vals = datavals.values
        if sel_info.ndims == 2:
            vals = np.expand_dims(vals, axis=-1)
        return vals

    return _reader


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
        if sel_info.grid_type == _xr_to_yt._GridType.STRETCHED:
            raise NotImplementedError(
                "Detected a stretched grid, which is not yet supported."
            )

        if geometry is None:
            geometry = self.geometry
        geometry = _xr_to_yt._determine_yt_geomtype(geometry, sel_info.selected_coords)
        if geometry is None:
            raise ValueError(
                "Cannot determine yt geometry type, please provide"
                "geometry = 'geographic', 'internal_geopgraphic' or 'cartesian'"
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

        data_shp = sel_info.selected_shape
        bbox = sel_info.selected_bbox
        if sel_info.ndims == 2:
            axis_order = geom[1]
            axis_order = _xr_to_yt._add_3rd_axis_name(geom[0], axis_order)
            geom = (geom[0], axis_order)
            data_shp = data_shp + (1,)
            bbox = np.vstack([bbox, [-0.5, 0.5]])

        data = {}
        if use_callable:
            reader = _get_xarray_reader(self._obj, sel_info)

        for field in fields:
            units = sel_info.units[field]
            if use_callable:
                data[field] = (reader, units)
            else:
                vals = sel_info.select_from_xr(self._obj, field).values
                if sel_info.ndims == 2:
                    vals = np.expand_dims(vals, axis=-1)
                data[field] = (vals, units)

        g_dict = sel_info.grid_dict.copy()
        if sel_info.ndims == 2:
            g_dict["left_edge"] = np.append(g_dict["left_edge"], -0.5)
            g_dict["right_edge"] = np.append(g_dict["right_edge"], 0.5)
            g_dict["dimensions"] += (1,)

        data.update(g_dict)
        grid_data = [
            data,
        ]
        return yt.load_amr_grids(
            grid_data,
            data_shp,
            geometry=geom,
            bbox=bbox,
            length_unit=length_unit,
            **kwargs,
        )

    def _infer_length_unit(self):
        if self.geometry == "geodetic":
            return 1
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
