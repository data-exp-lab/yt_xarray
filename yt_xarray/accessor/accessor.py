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
        *args,
        fields: Optional[List[str]] = None,
        geometry=None,
        use_callable=True,
        **kwargs,
    ):

        if geometry is None:
            geomtype = _determine_yt_geomtype(self.coord_type, self._coord_list)
            if geomtype is None:
                raise ValueError(
                    "Cannot determine yt geometry type, please provide"
                    "geometry = 'geographic' or 'internal_geopgraphic'"
                )

        if fields is None:
            fields = list(self._obj.data_vars)

        shape = self._obj.data_vars[fields[0]].shape
        for f in fields:
            if self._obj.data_vars[f].shape != shape:
                msg = "Provided fields must have the same shape."
                raise RuntimeError(msg)
            if self._obj.data_vars[f].ndim != 3:
                msg = (
                    f"Only 3D fields are supported at present and "
                    f"{f}.ndim={self._obj.data_vars[f].ndim}."
                )
                raise NotImplementedError(msg)

        # need to possibly account for stretched grid here... or at
        # least check for it and raise an error...

        # single grid, use whole domain for L/R edges
        bbox_vals = self.get_single_bbox(fields)
        l_e = bbox_vals[:, 0]
        r_e = bbox_vals[:, 1]

        if "length_unit" in kwargs:
            length_unit = kwargs.pop("length_unit")
        else:
            length_unit = self._infer_length_unit()
            if length_unit is None:
                raise ValueError(
                    "cannot determine length_unit, please provide as"
                    "a keyword argument."
                )

        coord_list = self._get_yt_coordlist()
        geom = (geomtype, coord_list)
        if use_callable:

            def _read_data(handle):
                def _reader(grid, field_name):
                    ftype, fname = field_name
                    si = grid.get_global_startindex()
                    ei = si + grid.ActiveDimensions
                    var = getattr(handle, fname)
                    data = var[si[0] : ei[0], si[1] : ei[1], si[2] : ei[2]]
                    return data.values

                return _reader

            reader = _read_data(self._obj)

            data = {}
            for field in fields:
                units = getattr(self._obj.data_vars[field], "units", None) or ""
                data[field] = (reader, units)

            data.update(
                {
                    "left_edge": l_e,
                    "right_edge": r_e,
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
            # should account for stretched grid here!
            data = {field: self._obj[field].values for field in fields}
            return yt.load_uniform_grid(
                data,
                shape,
                length_unit=length_unit,
                bbox=bbox_vals,
                geometry=geom,
                **kwargs,
            )

    def load_grid_from_callable(
        self, fields: Optional[List[str]] = None, geometry=None, **kwargs
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
        return self._load_uniform_grid(fields=fields, geometry=geometry, **kwargs)

    def load_uniform_grid(
        self,
        fields: Optional[List[str]] = None,
        geometry: Optional[str] = None,
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
            fields=fields, geometry=geometry, use_callable=False, **kwargs
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

    def _get_field_coord_tuple(self, field: str) -> Tuple[str]:
        # return the ordered coordinate names for a field
        return tuple(self._obj[field].coords)

    def get_field_coords(self, field: str):
        field_coords = self._get_field_coord_tuple(field)
        c0 = self._obj[field_coords[0]].values
        c1 = self._obj[field_coords[1]].values
        c2 = self._obj[field_coords[2]].values
        return c0, c1, c2

    def get_bbox(self, field: str) -> np.ndarray:
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

        # note that netcdf fields can use different coordinates, so the
        # bounding box is potentially dependent on field!
        if field in self._bbox:
            return self._bbox[field]

        c0, c1, c2 = self.get_field_coords(field)

        dimranges = []
        for dvals in (c0, c1, c2):
            dimranges.append([dvals.min(), dvals.max()])
        bbox = np.array(dimranges)

        self._bbox[field] = bbox
        return bbox

    def get_single_bbox(self, fields: List[str]) -> np.ndarray:
        # return the bounding box for a set of fields. Will return a single
        # bounding box if all fields share a bounding box, otherwise will
        # error.

        # assemble the set of coordinates
        coord_sets = set()
        for field in fields:
            coord_sets.add(self._get_field_coord_tuple(field))

        if len(coord_sets) > 1:
            msg = (
                "multiple bounding boxes found for given field list during an "
                "operation that requires a single bounding box."
            )
            raise RuntimeError(msg)

        return self.get_bbox(fields[0])


def _determine_yt_geomtype(coord_type: str, coord_list: List[str]) -> Optional[str]:
    if coord_type == "geodetic":
        # is it internal or external
        possible_alts = ["altitude", "height", "level"]
        if "depth" in coord_list:
            return "internal_geographic"
        elif any([i in coord_list for i in possible_alts]):
            return "geographic"
        return None
    elif coord_type == "cartesian":
        return "cartesian"
