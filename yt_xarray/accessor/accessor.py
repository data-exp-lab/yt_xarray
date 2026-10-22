from typing import Any, List, Literal, Mapping, Union

import numpy as np
import xarray as xr
import yt
from numpy import typing as npt
from numpy.typing import ArrayLike, NDArray
from yt.data_objects.static_output import Dataset as ytDataset

from yt_xarray.accessor import _xr_to_yt
from yt_xarray.accessor._readers import (
    _get_interpolating_reader,
    _get_xarray_reader,
    _interpolate_to_uniform_grid,
)
from yt_xarray.accessor._xr_to_yt import _load_full_field_from_xr
from yt_xarray.utilities._grid_decomposition import (
    ChunkInfo,
    GridSpec,
    _uniform_chunk_grids,
    _validate_refine_by,
    plan_stretched_grid_refinement,
    summarize_grid_plan,
)
from yt_xarray.utilities.logging import ytxr_log


@xr.register_dataset_accessor("yt")  # type: ignore[no-untyped-call]
class YtAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._bbox_cart = {}
        self._bbox = {}
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
        fields: Union[str, List[str]] | None = None,
        geometry: str | None = None,
        use_callable: bool = True,
        sel_dict: dict[str, Any] | None = None,
        sel_dict_type: str = "isel",
        chunksizes: Union[int, ArrayLike] | None = None,
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
            xarray Dataset handle to avoid copying data.

        sel_dict: dict
            an optional selection dictionary to apply to the fields before yt dataset
            initialization

        sel_dict_type: str
            either "isel" (default) or "sel" to indicate index or value selection for
            sel_dict.

        chunksizes: int or ArrayLike
            if set, will decompose the grid into multiple grids with grid dimensions
            of chunksizes. Can be a single integer (same chunksize in each dimensions)
            or an ArrayLike object of the same length as the number of dimensions.

        kwargs :
            any additional keyword arguments to pass to yt.load_uniform_grid

        Returns
        -------
        yt StreamDataset

        """
        fields, sel_info, geom, length_unit, kwargs = self._prepare_load(
            fields, geometry, sel_dict, sel_dict_type, kwargs
        )

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

    def _prepare_load(
        self,
        fields: Union[str, List[str]] | None,
        geometry: str | None,
        sel_dict: dict[str, Any] | None,
        sel_dict_type: str,
        kwargs: dict[str, Any],
    ) -> tuple[
        List[str],
        _xr_to_yt.Selection,
        tuple[str, list[str]],
        str | float,
        dict[str, Any],
    ]:
        # the shared validation for the load_* methods: field list, selection
        # info, yt geometry and axis order, length unit and simulation time.
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

        length_unit: str | float
        if "length_unit" in kwargs:
            length_unit = kwargs.pop("length_unit")
        else:
            inferred_unit = self._infer_length_unit()
            if inferred_unit is None:
                raise ValueError(
                    "cannot determine length_unit, please provide as"
                    " a keyword argument."
                )
            length_unit = inferred_unit

        axis_order: list[str] = list(sel_info.yt_coord_names)

        geom = (geometry, axis_order)

        kwargs.update({"sim_time": sel_info.selected_time})

        return fields, sel_info, geom, length_unit, kwargs

    def load_as_uniform_grid(
        self,
        fields: Union[str, List[str]] | None = None,
        geometry: str | None = None,
        bbox: Mapping[str, ArrayLike] | ArrayLike | None = None,
        cell_spacing: Mapping[str, float] | ArrayLike | float | None = None,
        pin_bbox: bool = False,
        use_callable: bool = True,
        sel_dict: dict[str, Any] | None = None,
        sel_dict_type: str = "isel",
        chunksizes: Union[int, ArrayLike] | None = None,
        interp_method: Literal["linear", "nearest"] = "linear",
        fill_value: float = np.nan,
        refine_grid: bool = False,
        refine_by: Mapping[str, int] | Union[int, ArrayLike] = 2,
        max_level: int | None = None,
        **kwargs,
    ):
        """
        Initializes a yt dataset of uniform grids, interpolating the supplied
        fields on demand from the (possibly stretched) xarray grid in the
        native coordinates of the xarray dataset.

        This is useful for consumers of yt datasets that require a constant
        cell spacing within each grid, e.g., yt_idv, when the xarray dataset
        is defined on a stretched grid: the interpolation happens in native
        coordinates, which is cheaper and more faithful than a transformation
        to cartesian coordinates.

        Parameters
        ----------
        fields : str, list[str]
            list of fields to include. If None, will try to use all fields

        geometry : str
            the geometry to pass to yt. If not provided, will attempt to infer.

        bbox : dict or ArrayLike
            the bounding box of the new grid in native coordinates. Either a
            dictionary mapping coordinate names to (min, max) pairs or an
            array of shape (ndim, 2) in the dimension order of the fields.
            Defaults to the extent of the (selected) xarray data. If the
            width in a dimension is not an integer multiple of the cell spacing,
            the maximum is extended so that it is, unless pin_bbox is True.

        cell_spacing : dict, ArrayLike or float
            the cell spacing of the new grid in native coordinates: a dictionary
            mapping coordinate names to spacing, an array of length ndim in the
            dimension order of the fields or a single value for all dimensions.
            If refine_grid is False, defaults to the smallest spacing of the
            xarray grid in each dimension. If refine_grid is True, this is the
            level 0 spacing and defaults to the largest value not exceeding the
            largest spacing of the xarray grid for which the finest level
            exactly resolves the smallest spacing in each dimension.

        pin_bbox : bool
            controls how a bbox width that is not an integer multiple of the
            cell spacing is reconciled. If False (default), the bbox maximum is
            extended to the next cell boundary and the cell spacing is kept.
            If True, the bbox is kept exactly and the cell spacing is decreased
            in that dimension to the largest value that divides the width into
            an integer number of cells. With refine_grid, the adjusted spacing
            is the level 0 spacing.

        use_callable : bool
            if True (default), then the yt dataset will interpolate on demand
            from the open xarray Dataset handle. If False, interpolation is
            done up front for every grid.

        sel_dict: dict
            an optional selection dictionary to apply to the fields before yt dataset
            initialization

        sel_dict_type: str
            either "isel" (default) or "sel" to indicate index or value selection for
            sel_dict.

        chunksizes: int or ArrayLike
            if set, will decompose the uniform grid into multiple grids with
            at most chunksizes cells in each dimension. Cannot be combined
            with refine_grid.

        interp_method: str
            ``'linear'`` (default) or ``'nearest'``, passed to xarray's interp.

        fill_value: float
            the value for cells of the new grid that fall outside the extent
            of the xarray data (default np.nan).

        refine_grid: bool
            if True (default False), build a hierarchy of uniform grids of
            increasing refinement wherever the xarray grid spacing is finer
            than the current level's spacing. The hierarchy is planned from
            the xarray coordinate values alone, no field values are read.

        refine_by: int, ArrayLike or dict
            the refinement factor between levels when refine_grid is True.
            Either a single integer (default 2), one integer per dimension in
            the dimension order of the fields, or a dictionary mapping
            coordinate names to integers, e.g., ``{'x': 2, 'y': 1, 'z': 2}``.
            A dimension with a refinement factor of 1 is never refined.

        max_level: int
            the maximum refinement level when refine_grid is True. Defaults
            to the level required to resolve the smallest spacing of the
            xarray grid.

        kwargs :
            any additional keyword arguments to pass to yt.load_uniform_grid
            or yt.load_amr_grids. Note that the domain is not periodic by
            default: pass ``periodicity`` to override.

        Returns
        -------
        yt StreamDataset

        """

        fields, sel_info, geom, length_unit, kwargs = self._prepare_load(
            fields, geometry, sel_dict, sel_dict_type, kwargs
        )

        if sel_info.ndims != 3:
            raise NotImplementedError(
                "load_as_uniform_grid is only implemented for 3D fields at present."
            )

        if interp_method not in ("linear", "nearest"):
            raise ValueError(
                f"interp_method must be 'linear' or 'nearest', found {interp_method}"
            )

        if refine_grid and chunksizes is not None:
            raise ValueError("chunksizes cannot be combined with refine_grid=True.")

        # the interpolated grid is not periodic unless the user says so
        kwargs.setdefault("periodicity", (False, False, False))

        if isinstance(refine_by, Mapping):
            refine_by = _refine_by_dict_to_array(sel_info, refine_by)
        rb = _validate_refine_by(refine_by, sel_info.ndims)
        dxyz = _validate_cell_spacing(sel_info, cell_spacing, refine_grid, rb)
        bbox_arr, dxyz = _validate_uniform_bbox(sel_info, bbox, dxyz, pin_bbox)
        dims = np.round((bbox_arr[:, 1] - bbox_arr[:, 0]) / dxyz).astype(int)

        if refine_grid:
            coord_arrays = _selected_coord_arrays(sel_info)
            grids = plan_stretched_grid_refinement(
                coord_arrays, bbox_arr, dxyz, refine_by=rb, max_level=max_level
            )
            _log_grid_plan(grids, coord_arrays, bbox_arr)
        elif chunksizes is not None:
            chunksizes = _validate_chunksizes(chunksizes, sel_info.ndims)
            grids = _uniform_chunk_grids(bbox_arr, dxyz, chunksizes)
            ytxr_log.info(f"Constructing a yt chunked grid with {len(grids)} chunks.")
        else:
            grids = [
                GridSpec(
                    left_edge=bbox_arr[:, 0],
                    right_edge=bbox_arr[:, 1],
                    dimensions=dims,
                    level=0,
                )
            ]

        if use_callable:
            reader = _get_interpolating_reader(
                self._obj, sel_info, interp_method=interp_method, fill_value=fill_value
            )

        def _field_data(grid: GridSpec) -> dict[str, Any]:
            field_data = {}
            for field in fields:
                units = sel_info.units[field]
                if use_callable:
                    field_data[field] = (reader, units)
                else:
                    dds = (grid.right_edge - grid.left_edge) / grid.dimensions
                    vals = _interpolate_to_uniform_grid(
                        self._obj,
                        sel_info,
                        field,
                        grid.left_edge,
                        dds,
                        grid.dimensions,
                        interp_method=interp_method,
                        fill_value=fill_value,
                    )
                    field_data[field] = (vals, units)
            return field_data

        if len(grids) == 1 and not refine_grid:
            ds_yt = yt.load_uniform_grid(
                _field_data(grids[0]),
                dims,
                geometry=geom[0],
                bbox=bbox_arr,
                length_unit=length_unit,
                axis_order=_as_axis_order(geom[1]),
                nprocs=1,
                **kwargs,
            )
        else:
            grid_data = []
            for grid in grids:
                gdict = grid.to_yt_dict()
                gdict.update(_field_data(grid))
                grid_data.append(gdict)
            if refine_grid:
                kwargs["refine_by"] = rb
            ds_yt = yt.load_amr_grids(
                grid_data,
                dims,
                geometry=geom[0],
                bbox=bbox_arr,
                length_unit=length_unit,
                axis_order=_as_axis_order(geom[1]),
                **kwargs,
            )

        self._yt_ds = ds_yt
        return ds_yt

    def _infer_length_unit(self) -> str | float | None:
        if self.geometry == "geodetic":
            return 1.0
        elif hasattr(self._obj, "geospatial_vertical_units"):
            # some netcdf conventions have this!
            return str(self._obj.geospatial_vertical_units)
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
        sel_dict: dict[str, Any] | None = None,
        sel_dict_type: str = "isel",
    ) -> NDArray[Any]:
        """
        return the bounding box array for a field, with possible selections

        Parameters
        ----------
        field
            the field to check the bounding box for

        sel_dict: dict
            an optional selection dictionary to apply to the fields before yt dataset
            initialization

        sel_dict_type: str
            either "isel" (default) or "sel" to indicate index or value selection for
            sel_dict.

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
        Wrapper of ``yt.SlicePlot``. For off-axis plots, first construct a
        yt dataset object with ``ds.yt.load_grid`` and then use ``yt.SlicePlot``

        Parameters
        ----------
        normal: str or int
            The normal to the slice.
        field: str
            The field to plot
        im_kwargs
            any additional keyword arguments accepted by ``yt.SlicePlot``

        Returns
        -------
        yt PlotWindow
        """
        return _yt_2D_plot(yt.SlicePlot, self._obj, normal, field, **im_kwargs)

    def ProjectionPlot(self, normal, field, **im_kwargs):
        """
        Wrapper of ``yt.ProjectionPlot``. For off-axis plots, first construct a
        yt dataset object with ``ds.yt.load_grid`` and then use ``yt.ProjectionPlot``

        Parameters
        ----------
        normal: str or int or 3-element tuple
            The normal to the slice.
        field: str
            The field to plot
        im_kwargs
            any of the keyword arguments accepted by ``yt.ProjectionPlot``

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
        weight_field: str | None = None,
        x_bins: int = 128,
        y_bins: int = 128,
        accumulation: bool = False,
        fractional: bool = False,
        fontsize: int | float = 18,
        figure_size: int | float = 8.0,
        shading: str = "nearest",
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

        pp_x_field = ("stream", x_field)
        pp_y_field = ("stream", y_field)
        pp_zfields = [("stream", z_f) for z_f in z_fields]

        return yt.PhasePlot(  # type: ignore[no-untyped-call]
            ds,
            pp_x_field,
            pp_y_field,
            pp_zfields,
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

        return yt.ProfilePlot(  # type: ignore[no-untyped-call]
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


def _as_axis_order(axis_order: list[str]) -> tuple[str, str, str]:
    return (axis_order[0], axis_order[1], axis_order[2])


def _validate_chunksizes(
    chunksizes: Union[int, ArrayLike], ndims: int
) -> npt.NDArray[np.int64]:
    if np.ndim(chunksizes) == 0:
        return np.full(ndims, chunksizes, dtype=np.int64)
    chunksizes_arr = np.asarray(chunksizes, dtype=np.int64)
    if chunksizes_arr.size != ndims:
        raise ValueError(
            f"The number of elements in chunksizes {chunksizes_arr.size} "
            f"must match the dimensionality {ndims}"
        )
    return chunksizes_arr


def _coord_name_to_index(sel_info: _xr_to_yt.Selection, name: str) -> int:
    # map an xarray coordinate name (or its yt name, or a known alias) to the
    # dimension index
    if name in sel_info.selected_coords:
        return int(sel_info.selected_coords.index(name))
    if name in sel_info.yt_coord_names:
        return int(sel_info.yt_coord_names.index(name))
    alias = _xr_to_yt.known_coord_aliases.get(name.lower())
    if alias is not None and alias in sel_info.yt_coord_names:
        return int(sel_info.yt_coord_names.index(alias))
    raise KeyError(
        f"{name} is not a selected coordinate. Valid names are "
        f"{sel_info.selected_coords} (or {tuple(sel_info.yt_coord_names)})."
    )


def _refine_by_dict_to_array(
    sel_info: _xr_to_yt.Selection, refine_by: Mapping[str, int]
) -> npt.NDArray[np.int64]:
    rb = np.zeros(sel_info.ndims, dtype=np.int64)
    for name, val in refine_by.items():
        rb[_coord_name_to_index(sel_info, name)] = val
    if np.any(rb == 0):
        raise ValueError(
            "refine_by must be provided for every dimension: "
            f"{sel_info.selected_coords}"
        )
    return rb


def _selected_coord_arrays(
    sel_info: _xr_to_yt.Selection,
) -> list[npt.NDArray[np.float64]]:
    # reconstruct the (increasing) coordinate values of the selected dimensions
    coord_arrays = []
    for idim in range(sel_info.ndims):
        c0 = sel_info.selected_bbox[idim, 0]
        widths = np.asarray(sel_info.cell_widths[idim], dtype=np.float64)
        coord_arrays.append(np.concatenate([[c0], c0 + np.cumsum(widths)]))
    return coord_arrays


def _validate_cell_spacing(
    sel_info: _xr_to_yt.Selection,
    cell_spacing: Mapping[str, float] | ArrayLike | float | None,
    refine_grid: bool,
    refine_by: npt.NDArray[np.int64],
) -> npt.NDArray[np.float64]:
    ndims = sel_info.ndims
    min_widths = np.array([np.min(w) for w in sel_info.cell_widths], dtype=np.float64)
    max_widths = np.array([np.max(w) for w in sel_info.cell_widths], dtype=np.float64)

    dxyz: npt.NDArray[np.float64]
    if cell_spacing is None:
        if refine_grid:
            # choose the largest level 0 spacing that does not exceed the
            # largest source spacing and for which the finest level lands
            # exactly on the smallest source spacing in each dimension
            dxyz = min_widths.copy()
            for idim in range(ndims):
                if refine_by[idim] > 1:
                    ratio = max_widths[idim] / min_widths[idim]
                    n_lev = np.floor(np.log(ratio) / np.log(refine_by[idim]) + 1e-8)
                    dxyz[idim] = min_widths[idim] * refine_by[idim] ** n_lev
        else:
            dxyz = min_widths
        ytxr_log.info(f"Using a cell_spacing of {dxyz}.")
        return dxyz

    if isinstance(cell_spacing, Mapping):
        dxyz = np.full(ndims, np.nan, dtype=np.float64)
        for name, val in cell_spacing.items():
            dxyz[_coord_name_to_index(sel_info, name)] = val
        if np.any(np.isnan(dxyz)):
            raise ValueError(
                "cell_spacing must be provided for every dimension: "
                f"{sel_info.selected_coords}"
            )
    elif np.ndim(cell_spacing) == 0:
        dxyz = np.full(ndims, cell_spacing, dtype=np.float64)
    else:
        dxyz = np.asarray(cell_spacing, dtype=np.float64)
        if dxyz.shape != (ndims,):
            raise ValueError(
                f"cell_spacing must have {ndims} elements, found {cell_spacing}"
            )

    if np.any(dxyz <= 0):
        raise ValueError(f"cell_spacing values must be positive, found {dxyz}")
    return dxyz


def _validate_uniform_bbox(
    sel_info: _xr_to_yt.Selection,
    bbox: Mapping[str, ArrayLike] | ArrayLike | None,
    dxyz: npt.NDArray[np.float64],
    pin_bbox: bool = False,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    ndims = sel_info.ndims
    bbox_arr: npt.NDArray[np.float64]
    if bbox is None:
        bbox_arr = sel_info.selected_bbox.copy()
    elif isinstance(bbox, Mapping):
        bbox_arr = sel_info.selected_bbox.copy()
        for name, val in bbox.items():
            bbox_arr[_coord_name_to_index(sel_info, name), :] = np.asarray(val)
    else:
        bbox_arr = np.asarray(bbox, dtype=np.float64)
        if bbox_arr.shape != (ndims, 2):
            raise ValueError(
                f"bbox must have shape ({ndims}, 2), found {bbox_arr.shape}"
            )
    bbox_arr = bbox_arr.astype(np.float64)

    if np.any(bbox_arr[:, 1] <= bbox_arr[:, 0]):
        raise ValueError(f"bbox max must exceed bbox min, found {bbox_arr}")

    widths = bbox_arr[:, 1] - bbox_arr[:, 0]
    dims = np.ceil(widths / dxyz - 1e-8).astype(int)
    if pin_bbox:
        new_dxyz = widths / dims
        if not np.allclose(new_dxyz, dxyz):
            ytxr_log.info(
                "bbox width is not an integer multiple of cell_spacing, decreasing "
                f"cell_spacing from {dxyz} to {new_dxyz} to match the bbox."
            )
        return bbox_arr, new_dxyz

    new_max = bbox_arr[:, 0] + dims * dxyz
    if not np.allclose(new_max, bbox_arr[:, 1]):
        ytxr_log.info(
            "bbox width is not an integer multiple of cell_spacing, extending "
            f"bbox max from {bbox_arr[:, 1]} to {new_max}."
        )
    bbox_arr[:, 1] = new_max
    return bbox_arr, dxyz


def _log_grid_plan(
    grids: list[GridSpec],
    coord_arrays: list[npt.NDArray[np.float64]],
    bbox: npt.NDArray[np.float64],
) -> None:
    summary = summarize_grid_plan(grids)
    total = 0
    for level, entry in sorted(summary.items()):
        total += entry["n_cells"]
        ytxr_log.info(
            f"    level {level}: {entry['n_grids']} grids, {entry['n_cells']} cells"
        )
    min_widths = np.array([np.min(c[1:] - c[:-1]) for c in coord_arrays])
    n_uniform = int(np.prod(np.ceil((bbox[:, 1] - bbox[:, 0]) / min_widths)))
    ytxr_log.info(
        f"Refined grid plan has {total} cells in {len(grids)} grids. A uniform "
        f"grid at the finest source spacing would have {n_uniform} cells."
    )


def _load_single_grid(
    ds_xr, sel_info, geom, use_callable, fields, length_unit, **kwargs
) -> ytDataset:
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

    ds: ytDataset
    if sel_info.grid_type == _xr_to_yt._GridType.STRETCHED:
        ds = yt.load_uniform_grid(
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
        ds = yt.load_amr_grids(
            grid_data,
            data_shp,
            geometry=geom[0],
            bbox=bbox,
            length_unit=length_unit,
            axis_order=geom[1],
            **kwargs,
        )
    return ds


def _load_chunked_grid(
    ds_xr, sel_info, geom, use_callable, fields, length_unit, chunksizes, **kwargs
) -> ytDataset:
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

    # initialize the global starting index
    si = np.array([0, 0, 0], dtype=int)
    si = sel_info.starting_indices + si

    # do some grid/chunk counting
    chnkinfo = ChunkInfo(data_shp, chunksizes, starting_index_offset=si)
    ytxr_log.info(f"Constructing a yt chunked grid with {chnkinfo.n_tots} chunks.")

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
        si_0 = chnkinfo.si[idim]
        ei_0 = chnkinfo.ei[idim]
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
        subgrid_size = chnkinfo.sizes[idim]

        left_edges.append(le_0)
        right_edges.append(re_0)
        subgrid_sizes.append(subgrid_size)
        subgrid_start.append(si_0)
        subgrid_end.append(ei_0)

    # these arrays are ordered by dimension. e.g., left_edges[0] will be the
    # all first dimension left edges
    le_mesh = np.meshgrid(*left_edges, indexing="ij")
    re_mesh = np.meshgrid(*right_edges, indexing="ij")
    sz_mesh = np.meshgrid(*subgrid_sizes, indexing="ij")
    st_mesh = np.meshgrid(*subgrid_start, indexing="ij")
    en_mesh = np.meshgrid(*subgrid_end, indexing="ij")

    # re-organize by grid number so that, e.g., the left_edges are the usual
    # left_edges (left_edges[0] is the min x, y, z of grid 0)
    grid_le = np.column_stack([le.ravel() for le in le_mesh])
    grid_re = np.column_stack([re.ravel() for re in re_mesh])
    grid_dims = np.column_stack([sz.ravel() for sz in sz_mesh])
    grid_start = np.column_stack([sz.ravel() for sz in st_mesh])
    grid_end = np.column_stack([sz.ravel() for sz in en_mesh])

    # now ready to build the list of grids
    if use_callable:
        reader = _get_xarray_reader(ds_xr, sel_info, interp_required=interp_required)

    grid_data = []
    n_grids = len(grid_le)

    if use_callable is False:
        full_field_vals = {}
        for field in fields:
            vals = _load_full_field_from_xr(
                ds_xr, field, sel_info, interp_required=interp_required
            )
            full_field_vals[field] = vals

    for igrid in range(n_grids):
        gdict = {
            "left_edge": grid_le[igrid],
            "right_edge": grid_re[igrid],
            "dimensions": grid_dims[igrid],
            "level": 0,
        }
        for field in fields:
            units = sel_info.units[field]
            if use_callable:
                gdict[field] = (reader, units)
            else:
                si = grid_start[igrid]
                ei = grid_end[igrid]
                gridvals = full_field_vals[field][
                    si[0] : ei[0], si[1] : ei[1], si[2] : ei[2]
                ]
                gdict[field] = (gridvals, units)
        grid_data.append(gdict)

    ds: ytDataset = yt.load_amr_grids(
        grid_data,
        data_shp,
        geometry=geom[0],
        bbox=bbox,
        length_unit=length_unit,
        axis_order=geom[1],
        **kwargs,
    )
    return ds


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
