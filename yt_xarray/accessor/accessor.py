from collections import defaultdict
from typing import List, Optional

import numpy as np
import xarray as xr
import yt
from unyt import unyt_quantity

from yt_xarray.accessor import _xr_to_yt
from yt_xarray.accessor._xr_to_yt import _add_3rd_axis_name, _determine_yt_geomtype


@xr.register_dataset_accessor("yt")
class YtAccessor:
    def __init__(self, xarray_obj):
        # __init__ can ONLY have xarray_obj here
        self._obj = xarray_obj
        self._bbox_cart = {}
        self._bbox = {}
        self._field_grids = defaultdict(lambda: None)

    def _load_uniform_grid(
        self,
        fields: Optional[List[str]] = None,
        geometry=None,
        use_callable=True,
        sel_dict: Optional[dict] = None,
        sel_dict_type: Optional[str] = "isel",
        allow_time_as_dim: Optional[bool] = False,
        **kwargs,
    ):

        if fields is None:
            # might as well try!
            fields = list(self._obj.data_vars)

        sel_info = _xr_to_yt.Selection(
            self._obj,
            fields=fields,
            sel_dict=sel_dict,
            sel_dict_type=sel_dict_type,
            allow_time_as_dim=allow_time_as_dim,
        )

        if geometry is None:
            geometry = _determine_yt_geomtype(self.coord_type, sel_info.selected_coords)
            if geometry is None:
                raise ValueError(
                    "Cannot determine yt geometry type, please provide"
                    "geometry = 'geographic', 'internal_geopgraphic' or 'cartesian'"
                )

        # need to possibly account for stretched grid here... or at
        # least check for it and raise an error...

        if "length_unit" in kwargs:
            length_unit = kwargs.pop("length_unit")
        else:
            length_unit = self._infer_length_unit()
            if length_unit is None:
                raise ValueError(
                    "cannot determine length_unit, please provide as"
                    "a keyword argument."
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
            axis_order = _add_3rd_axis_name(geom[0], axis_order)
            geom = (geom[0], axis_order)
            data_shp = data_shp + (1,)
            bbox = np.vstack([bbox, [0, 1]])

        if use_callable:

            def _read_data(handle, sel_info):
                def _reader(grid, field_name):
                    ftype, fname = field_name

                    gsi = (
                        sel_info.starting_indices
                    )  # should just set it for the grid....
                    if sel_info.ndims == 2:
                        gsi = np.concatenate(
                            [
                                gsi,
                                [
                                    0,
                                ],
                            ]
                        )
                    si = grid.get_global_startindex() + gsi
                    ei = si + grid.ActiveDimensions

                    # build the index-selector for our grid
                    c_list = sel_info.selected_coords  # the xarray coord names
                    i_select_dict = {}
                    for idim in range(sel_info.ndims):  # 2d/3d issue here
                        i_select_dict[c_list[idim]] = slice(si[idim], ei[idim])

                    # apply any initial selections, only along dimensions
                    # not covered in the grid here, accounting for sel vs isel.
                    # these selections **should** reduce the dimensionality
                    # of the array.
                    first_selection = {}
                    for ky, val in sel_info.sel_dict.items():
                        if ky not in i_select_dict:
                            if sel_info.sel_dict_type == "sel":
                                first_selection[ky] = val
                            else:
                                # just add it to the i_select_dict
                                i_select_dict[ky] = val

                    var = getattr(handle, fname)
                    if first_selection:
                        data = var.sel(first_selection).isel(i_select_dict)
                    else:
                        data = var.isel(i_select_dict)
                    vals = data.values
                    if sel_info.ndims == 2:
                        vals = np.expand_dims(vals, axis=-1)
                    return vals

                return _reader

            reader = _read_data(self._obj, sel_info)

            data = {}
            for field in fields:
                units = sel_info.units[field]
                data[field] = (reader, units)

            g_dict = sel_info.grid_dict.copy()
            if sel_info.ndims == 2:
                g_dict["left_edge"] = np.concatenate(
                    [
                        g_dict["left_edge"],
                        [
                            -0.5,
                        ],
                    ]
                )
                g_dict["right_edge"] = np.concatenate(
                    [
                        g_dict["right_edge"],
                        [
                            0.5,
                        ],
                    ]
                )
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

        else:
            # should account for stretched grid here?
            data = {}

            for field in fields:
                vals = sel_info.select_from_xr(self._obj, field).values
                if sel_info.ndims == 2:
                    vals = np.expand_dims(vals, axis=-1)
                units = sel_info.units[field]
                data[field] = (vals, units)

            return yt.load_uniform_grid(
                data,
                data_shp,
                length_unit=length_unit,
                bbox=bbox,
                geometry=geom,
                **kwargs,
            )

    def load_grid_from_callable(
        self,
        fields: Optional[List[str]] = None,
        geometry=None,
        sel_dict: Optional[dict] = None,
        sel_dict_type: Optional[str] = "isel",
        allow_time_as_dim: Optional[bool] = False,
        **kwargs,
    ):
        """
        returns a uniform grid yt dataset linked to the open xarray handle.

        Parameters
        ----------
        fields : list of fields to include. If None, will try to use all fields
        geometry : the geometry to pass to yt.load_uniform grid. If not provided,
                   will attempt to infer.
        kwargs : any additional keyword arguments to pass to yt.load_uniform_grid

        Returns
        -------
        yt StreamDataset

        Notes
        -----

        This function relies on the stream callable functionality in yt>=4.1.0
        in order to read directly from an open xarray handle without creating
        additional in-memory copies of the data.
        """
        return self._load_uniform_grid(
            fields=fields,
            geometry=geometry,
            sel_dict=sel_dict,
            sel_dict_type=sel_dict_type,
            allow_time_as_dim=allow_time_as_dim,
            **kwargs,
        )

    def load_uniform_grid(
        self,
        fields: Optional[List[str]] = None,
        geometry: Optional[str] = None,
        sel_dict: Optional[dict] = None,
        sel_dict_type: Optional[str] = "isel",
        allow_time_as_dim: Optional[bool] = False,
        **kwargs,
    ):
        """
        return an in-memory uniform grid yt dataset

        Parameters
        ----------
        fields : list of fields to include. If None, will try to use all fields
        geometry : the geometry to pass to yt.load_uniform grid. If not provided,
                   will attempt to infer.
        kwargs : any additional keyword arguments to pass to yt.load_uniform_grid

        Returns
        -------
        yt StreamDataset
        """
        return self._load_uniform_grid(
            fields=fields,
            geometry=geometry,
            use_callable=False,
            sel_dict=sel_dict,
            sel_dict_type=sel_dict_type,
            allow_time_as_dim=allow_time_as_dim,
            **kwargs,
        )

    def _infer_length_unit(self):
        if self.coord_type == "geodetic":
            return 1
        elif hasattr(self._obj, "geospatial_vertical_units"):
            # some netcdf conventions have this!
            return self._obj.geospatial_vertical_units
        return None

    _coord_type = None

    def set_coordinate_type(self, coordinate_type: str):
        self._coord_type = coordinate_type

    @property
    def coord_type(self) -> str:
        if self._coord_type is None:
            self._coord_type = self._infer_coordinate_type()
        return self._coord_type

    def _infer_coordinate_type(self) -> str:
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
        print(
            f"Inferred coordinate type is {ctype} -- to override, use ds.yt.set_coordinate_type"
        )
        return ctype

    def ds(self):
        """try to return a yt dataset with all data fields"""
        return self.load_grid_from_callable()

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
