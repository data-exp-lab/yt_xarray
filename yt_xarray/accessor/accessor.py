from collections import defaultdict
from typing import List, Optional, Tuple

import numpy as np
import xarray as xr
import yt


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
        **kwargs,
    ):

        if geometry is None:
            geomtype = _determine_yt_geomtype(self.coord_type, self._coord_list)
            if geomtype is None:
                raise ValueError(
                    "Cannot determine yt geometry type, please provide"
                    "geometry = 'geographic' or 'internal_geopgraphic'"
                )

        did_not_provide_fields = fields is None
        if did_not_provide_fields:
            fields = list(self._obj.data_vars)

        shape = self._obj.data_vars[fields[0]].shape
        coords = tuple(self._obj.data_vars[fields[0]].coords)
        fld_msg = " Please provide a subset of fields to load."
        for f in fields:
            if self._obj.data_vars[f].shape != shape:
                msg = "Provided fields must have the same shape."
                if did_not_provide_fields:
                    msg += fld_msg
                raise RuntimeError(msg)
            elif tuple(self._obj.data_vars[f].coords) != coords:
                msg = "Provided fields must have the same coordinates."
                if did_not_provide_fields:
                    msg += fld_msg
                raise RuntimeError(msg)

            if self._obj.data_vars[f].ndim != 3:
                msg = (
                    f"Fields must be 3D when loaded into yt and "
                    f"{f}.ndim={self._obj.data_vars[f].ndim}."
                )
                if sel_dict is None:
                    msg = msg + " Provide an sel_dict to reduce the dimensionality."
                    raise RuntimeError(msg)
                elif self._obj.data_vars[f].ndim - len(sel_dict) != 3:
                    ndim1 = self._obj.data_vars[f].ndim - len(sel_dict)
                    msg = (
                        msg
                        + f" Provided sel_dict will result in a dimensionality of {ndim1}"
                    )
                    raise RuntimeError(msg)

        # need to possibly account for stretched grid here... or at
        # least check for it and raise an error...

        # single grid, use whole domain for L/R edges
        # also returns shape, corrected for selection dict
        bbox_vals, shape, coord_list, starting_indices = self.get_single_bbox(
            fields, sel_dict=sel_dict, sel_dict_type=sel_dict_type
        )

        if "length_unit" in kwargs:
            length_unit = kwargs.pop("length_unit")
        else:
            length_unit = self._infer_length_unit()
            if length_unit is None:
                raise ValueError(
                    "cannot determine length_unit, please provide as"
                    "a keyword argument."
                )

        yt_coord_list = _convert_to_yt_internal_coords(coord_list)
        geom = (geomtype, yt_coord_list)
        print(geom)

        if use_callable:

            def _read_data(handle, sel_dict=None, sel_dict_type="isel"):

                if sel_dict is None:
                    sel_dict = {}

                def _reader(grid, field_name):
                    ftype, fname = field_name
                    si = (
                        grid.get_global_startindex() + starting_indices
                    )  # should just set it for the grid....
                    ei = si + grid.ActiveDimensions
                    var = getattr(handle, fname)
                    i_select_dict = {}
                    # e.g.,
                    # co2 = ds.co23D.isel({"time": 0, "lat": slice(0, 3), "lon": slice(0, 3), "lev": slice(0, 5)})
                    for idim in range(3):
                        i_select_dict[coord_list[idim]] = slice(si[idim], ei[idim])

                    # apply any initial selections, only along dimensions
                    # not covered in the grid here, accounting for sel vs isel.
                    first_selection = {}
                    for ky, val in sel_dict.items():
                        if ky not in i_select_dict:
                            if sel_dict_type == "sel":
                                first_selection[ky] = val
                            else:
                                # just add it to the i_select_dict
                                i_select_dict[ky] = val

                    if first_selection:
                        data = var.sel(first_selection).isel(i_select_dict)
                    else:
                        data = var.isel(i_select_dict)

                    # data = var[si[0] : ei[0], si[1] : ei[1], si[2] : ei[2]]
                    return data.values

                return _reader

            reader = _read_data(
                self._obj, sel_dict=sel_dict, sel_dict_type=sel_dict_type
            )

            data = {}
            for field in fields:
                units = getattr(self._obj.data_vars[field], "units", None) or ""
                data[field] = (reader, units)

            data.update(
                {
                    "left_edge": bbox_vals[:, 0],
                    "right_edge": bbox_vals[:, 1],
                    "dimensions": shape,
                    "level": 0,
                }
            )

            grid_data = [
                data,
            ]

            return yt.load_amr_grids(
                grid_data,
                shape,
                geometry=geom,
                bbox=bbox_vals,
                length_unit=length_unit,
                **kwargs,
            )

        else:
            # should account for stretched grid here?

            # should account for sel_dict_type here too.
            data = {}
            for field in fields:
                if sel_dict is not None:
                    if sel_dict_type == "isel":
                        vals = self._obj[field].isel(sel_dict).values
                    elif sel_dict_type == "sel":
                        vals = self._obj[field].sel(sel_dict).values
                    else:
                        raise ValueError(
                            f"Unexpected value for sel_dict_type. Expected 'isel' or 'sel', found {sel_dict_type}"
                        )
                else:
                    vals = self._obj[field].values
                # could add on units here
                data[field] = vals

            return yt.load_uniform_grid(
                data,
                shape,
                length_unit=length_unit,
                bbox=bbox_vals,
                geometry=geom,
                **kwargs,
            )

    def load_grid_from_callable(
        self,
        fields: Optional[List[str]] = None,
        geometry=None,
        sel_dict: Optional[dict] = None,
        sel_dict_type: Optional[str] = "isel",
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
            **kwargs,
        )

    def load_uniform_grid(
        self,
        fields: Optional[List[str]] = None,
        geometry: Optional[str] = None,
        sel_dict: Optional[dict] = None,
        sel_dict_type: Optional[str] = "isel",
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
            **kwargs,
        )

    def _infer_length_unit(self):
        if self.coord_type == "geodetic":
            return 1
        elif hasattr(self._obj, "geospatial_vertical_units"):
            # some netcdf conventions have this!
            return self._obj.geospatial_vertical_units
        return None

    def _get_yt_coordlist(self) -> Tuple[str]:
        # yt expects certain coordinate names, need a way to handle those.
        # in some cases, simple aliasing may be enough. Leaving this as a
        # placeholder. see yt/geometry/coordinates/geographic_coordinates.py
        # e.g., geographic expects 'altitude'.
        return self._coord_list

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
        # _determine_yt_geomtype). Default is to assume cartesian.
        geodetic_names = ["latitude", "longitude", "lat", "lon"]
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

    @property
    def field_list(self):
        # a list of variables that are not coordinates
        return [i for i in self._obj.variables if i not in self._obj.coords]

    def _get_field_coord_tuple(
        self, field: str, isel: Optional[dict] = None
    ) -> Tuple[str]:
        # return the ordered coordinate names for a field
        c = list(self._obj[field].coords)  # e.g., (time, lev, lat, lon)
        if isel is not None:
            c = [_ for _ in c if _ not in isel]
        return tuple(c)

    def get_field_coords(self, field: str, isel: Optional[dict] = None):
        field_coords = self._get_field_coord_tuple(field, isel=isel)
        c = [self._obj[f].values for f in field_coords]
        return c

    def get_bbox(
        self,
        field: str,
        sel_dict: Optional[dict] = None,
        sel_dict_type: Optional[str] = "isel",
    ) -> np.ndarray:
        """
        return the bounding box array for a field

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

        # import xarray as xr
        # import yt_xarray
        # ds = xr.open_dataset(
        #     "/home/chavlin/hdd/data/yt_data/yt_sample_sets/cmip6/co23D_Emon_GISS-E2-1-G_piControl_r101i1p1f1_gn_185001-190012.nc")
        # print(ds.yt.get_bbox('co23D'))
        # print("\nwith time selection:")
        # print(ds.yt.get_bbox('co23D', sel_dict=dict(time=0)))

        if sel_dict_type not in ["isel", "sel"]:
            raise ValueError(
                f"sel_dict_type must be one of isel or sel, found {sel_dict_type}"
            )

        # the full list of coordinates (in order)
        full_coords = list(getattr(self._obj, field).coords)

        # find bounding box, accounting for isel
        if sel_dict is None:
            sel_dict = {}

        # for each coordinate, apply any selector and then store min/max of the
        # coordinate. If the selector results in <= 1 in one of the dimensions,
        # that dimensions is dropped.
        shape = []
        dimranges = []
        coord_list = []
        starting_indices = []
        for c in full_coords:
            coord_select = {}
            si = 0
            if c in sel_dict:
                coord_select[c] = sel_dict[c]
                if isinstance(sel_dict[c], slice):
                    si = sel_dict[c].start
            coord_da = getattr(self._obj, c)
            selector = getattr(coord_da, sel_dict_type)  # ds.variable.sel or .isel
            coord_vals = selector(coord_select).values
            if coord_vals.size > 1:
                dimranges.append([coord_vals.min(), coord_vals.max()])
                shape.append(coord_vals.size)
                coord_list.append(c)
                starting_indices.append(si)
        shape = tuple(shape)
        bbox = np.array(dimranges)
        starting_indices = np.array(starting_indices)

        return bbox, shape, coord_list, starting_indices

    def get_single_bbox(
        self,
        fields: List[str],
        sel_dict: Optional[dict] = None,
        sel_dict_type: Optional[str] = "isel",
    ) -> np.ndarray:
        # return the bounding box for a set of fields. Will return a single
        # bounding box if all fields share a bounding box, otherwise will
        # error.

        # assemble the set of coordinates just to check that all the fields have
        # the same coordinates
        coord_sets = set()
        for field in fields:
            coord_sets.add(self._get_field_coord_tuple(field))

        if len(coord_sets) > 1:
            msg = (
                "multiple bounding boxes found for given field list during an "
                "operation that requires a single bounding box."
            )
            raise RuntimeError(msg)

        return self.get_bbox(fields[0], sel_dict=sel_dict, sel_dict_type=sel_dict_type)


def _determine_yt_geomtype(coord_type: str, coord_list: List[str]) -> Optional[str]:
    if coord_type == "geodetic":
        # is it internal or external
        possible_alts = ["altitude", "height", "level", "lev"]
        if "depth" in coord_list:
            return "internal_geographic"
        elif any([i in coord_list for i in possible_alts]):
            return "geographic"
        return None
    elif coord_type == "cartesian":
        return "cartesian"


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
