from collections import defaultdict
from typing import List, Optional, Union

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
        self._active_yt_ds = None

    @property
    def _yt_ds(self):
        # a property for storing the last-loaded yt dataset.
        return self._active_yt_ds

    @_yt_ds.setter
    def _yt_ds(self, yt_ds):
        self._active_yt_ds = yt_ds

    def load_grid(
        self,
        fields: Optional[Union[str, List[str]]] = None,
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
        fields : str, list[str]
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

        if isinstance(fields, str):
            fields = [
                fields,
            ]

        sel_info = _xr_to_yt.Selection(
            self._obj,
            fields=fields,
            sel_dict=sel_dict,
            sel_dict_type=sel_dict_type,
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
            ds_yt = _load_single_grid(
                self._obj, sel_info, geom, use_callable, fields, length_unit, **kwargs
            )
        elif sel_info.grid_type == _xr_to_yt._GridType.STRETCHED:
            raise NotImplementedError(
                "Stretched grids cannot set the chunksizes argument."
            )
        else:
            ds_yt = _load_chunked_grid(
                self._obj,
                sel_info,
                geom,
                use_callable,
                fields,
                length_unit,
                chunksizes,
                **kwargs,
            )
        self._yt_ds = ds_yt
        return ds_yt

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

    def SlicePlot(self, normal, field, **im_kwargs):
        """
        Wrapper of `yt.SlicePlot`. For off-axis plots, first construct a
        yt dataset object with `ds.yt.load_grid` and then use `yt.SlicePlot`

        Parameters
        ----------
        normal: str or int
            The normal to the slice.
        field: str
            The field to plot
        **im_kwargs
            any of the keyword arguments accepted by yt.SlicePlot

        Returns
        -------
        yt PlotWindow
        """
        return _yt_2D_plot(yt.SlicePlot, self._obj, normal, field, **im_kwargs)

    def ProjectionPlot(self, normal, field, **im_kwargs):
        """
        Wrapper of `yt.ProjectionPlot`. For off-axis plots, first construct a
        yt dataset object with `ds.yt.load_grid` and then use `yt.ProjectionPlot`

        Parameters
        ----------
        normal: str or int or 3-element tuple
            The normal to the slice.
        field: str
            The field to plot
        **im_kwargs
            any of the keyword arguments accepted by yt.ProjectionPlot

        Returns
        -------
        yt PlotWindow
        """
        return _yt_2D_plot(yt.ProjectionPlot, self._obj, normal, field, **im_kwargs)

    def PhasePlot(
        self,
        x_field: str,
        y_field: str,
        z_fields: Union[str, List[str]],
        weight_field: Optional[str] = None,
        x_bins: Optional[int] = 128,
        y_bins: Optional[int] = 128,
        accumulation: Optional[bool] = False,
        fractional: Optional[bool] = False,
        fontsize: Optional[int] = 18,
        figure_size: Optional[int] = 8.0,
        shading: Optional[str] = "nearest",
    ):
        """
        Construct a `yt.PhasePlot`.

        Parameters
        ----------
        x_field : str
            The x binning field for the profile.
        y_field : str
            The y binning field for the profile.
        z_fields : str or list
            The field or fields to be profiled.
        weight_field : str
            The weight field for calculating weighted averages.  If None,
            the profile values are the sum of the field values within the bin.
            Otherwise, the values are a weighted average.
            Default : ("gas", "mass")
        x_bins : int
            The number of bins in x field for the profile.
            Default: 128.
        y_bins : int
            The number of bins in y field for the profile.
            Default: 128.
        accumulation : bool or list of bools
            If True, the profile values for a bin n are the cumulative sum of
            all the values from bin 0 to n.  If -True, the sum is reversed so
            that the value for bin n is the cumulative sum from bin N (total bins)
            to n.  A list of values can be given to control the summation in each
            dimension independently.
            Default: False.
        fractional : If True the profile values are divided by the sum of all
            the profile data such that the profile represents a probability
            distribution function.
        fontsize : int
            Font size for all text in the plot.
            Default: 18.
        figure_size : int
            Size in inches of the image.
            Default: 8 (8x8)
        shading : str
            This argument is directly passed down to matplotlib.axes.Axes.pcolormesh
            see
            https://matplotlib.org/3.3.1/gallery/images_contours_and_fields/pcolormesh_grids.html#sphx-glr-gallery-images-contours-and-fields-pcolormesh-grids-py  # noqa
            Default: 'nearest'

        """
        if isinstance(z_fields, str):
            z_fields = [
                z_fields,
            ]

        fields_needed = list(set([x_field, y_field] + z_fields))
        ds = _get_default_ds(self._obj, fields_needed)

        x_field = ("stream", x_field)
        y_field = ("stream", y_field)
        z_fields = [("stream", z_f) for z_f in z_fields]

        return yt.PhasePlot(
            ds,
            x_field,
            y_field,
            z_fields,
            weight_field=weight_field,
            x_bins=x_bins,
            y_bins=y_bins,
            accumulation=accumulation,
            fractional=fractional,
            fontsize=fontsize,
            figure_size=figure_size,
            shading=shading,
        )

    def ProfilePlot(
        self,
        x_field,
        y_fields,
        weight_field=None,
        n_bins=64,
        accumulation=False,
        fractional=False,
        label=None,
        plot_spec=None,
        x_log=True,
        y_log=True,
    ):
        """
        Construct a `yt.ProfilePlot`.

        Parameters
        ----------
        x_field : str
            The binning field for the profile.
        y_fields : str or list
            The field or fields to be profiled.
        weight_field : str
            The weight field for calculating weighted averages. If None,
            the profile values are the sum of the field values within the bin.
            Otherwise, the values are a weighted average.
            Default : None
        n_bins : int
            The number of bins in the profile.
            Default: 64.
        accumulation : bool
            If True, the profile values for a bin N are the cumulative sum of
            all the values from bin 0 to N.
            Default: False.
        fractional : If True the profile values are divided by the sum of all
            the profile data such that the profile represents a probability
            distribution function.
        label : str or list of strings
            If a string, the label to be put on the line plotted.  If a list,
            this should be a list of labels for each profile to be overplotted.
            Default: None.
        plot_spec : dict or list of dicts
            A dictionary or list of dictionaries containing plot keyword
            arguments.  For example, dict(color="red", linestyle=":").
            Default: None.
        x_log : bool
            Whether the x_axis should be plotted with a logarithmic
            scaling (True), or linear scaling (False).
            Default: True.
        y_log : dict or bool
            A dictionary containing field:boolean pairs, setting the logarithmic
            property for that field. May be overridden after instantiation using
            set_log
            A single boolean can be passed to signify all fields should use
            logarithmic (True) or linear scaling (False).
            Default: True.


        Returns
        -------

        """
        fields_needed = list(set([x_field, y_fields]))

        if weight_field is not None and weight_field not in fields_needed:
            fields_needed.append(weight_field)

        ds = _get_default_ds(self._obj, fields_needed)
        ad = ds.all_data()

        return yt.ProfilePlot(
            ad,
            x_field,
            y_fields,
            weight_field=weight_field,
            n_bins=n_bins,
            accumulation=accumulation,
            fractional=fractional,
            label=label,
            plot_spec=plot_spec,
            x_log=x_log,
            y_log=y_log,
        )


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


def _get_default_ds(ds_xr: xr.Dataset, field):
    geom = ds_xr.yt.geometry  # will trigger inference here

    # other load_grid options
    # use_callable: bool = True,
    # sel_dict: Optional[dict] = None,  dont need this one
    # sel_dict_type: Optional[str] = "isel",  dont need this one
    # chunksizes: Optional[int] = None,

    # if the grid were cached this might be easier...
    ds = ds_xr.yt.load_grid(fields=field, geometry=geom)
    return ds


def _yt_2D_plot(yt_function, ds_xr: xr.Dataset, normal, field, **im_kwargs):
    ds = _get_default_ds(ds_xr, field)
    # normal = validate_normal(normal)
    return yt_function(ds, normal, ("stream", field), **im_kwargs)
