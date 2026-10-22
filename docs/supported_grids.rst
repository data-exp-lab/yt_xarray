Supported Grids
===============

``yt_xarray`` supports loading in regularly gridded data: this includes uniform
grids where the spacing in each dimension is constant (but variable between
dimensions) and stretched grids where the spacing in each dimension varies
regularly.

The grid objects in ``yt`` are primarily built to work with cell-centered values
as opposed to the node-centered values common in many netcdf datasets. While some
limited support for nodal values exists in ``yt``, ``yt_xarray`` will attempt to
load nodal values as cell-centered values in a number of ways:

Wrapped Grids
#############

If the input data is a uniform grid then a new grid is built around the nodal
values so that they are cell-centered values of the new grid. For geographic
data, however, this only works if the new grid does not extend beyond the
allowed coordinate extremes (e.g., latitude must remain between -90, 90).
In te event that the new grid extends beyond allowable bounds for the coordinate
system, an interpolated grid is built.

Interpolated Grids
##################

For stretched grids, or for uniform grids that cannot be wrapped (see above), a
new grid is built by using the nodal coordinates as cell boundaries and
interpolating to cell centers. This is done on the fly as data is loaded (so
there is no need to interpolate beforehand) and it actually relies on xarray's
interpolation procedures (which in turn rely on scipy). Interpolation is limited
to linear interpolation at present.

Chunked Grids
#############

For uniform grids, yt_xarray can also load data in chunks. In this case, each
chunk will correspond to a yt grid object, and yt functions will process chunks
sequentially to avoid having to load data fully in memory. This means that if
the xarray dataset relies on dask arrays, then yt will only have to load in a
subset of the whole dataset. At present the chunking is not a one to one match,
meaning that multiple dask chunks may be contained within a yt grid object, see the
examples for a more detailed look.

Resampled Uniform Grids
#######################

Some consumers of yt datasets, notably ``yt_idv``, require every grid to have a
constant cell spacing in each dimension and so cannot work with stretched grids
directly. For these cases, ``ds.yt.load_as_uniform_grid`` builds a new uniform
grid with a user-specified bounding box and cell spacing and interpolates the
xarray fields onto it on demand, as yt reads data. The interpolation happens in
the native coordinates of the xarray dataset (e.g., depth, latitude, longitude),
which is both cheaper and more faithful than transforming to cartesian coordinates
first. The resampled grid may be chunked into multiple yt grids with the
``chunksizes`` argument, exactly as for ``ds.yt.load_grid``.

By default the cell spacing is the smallest spacing of the stretched grid in each
dimension, so that every feature of the original grid is resolved. For grids where
the fine spacing is confined to a small region, this can be wasteful, and
``refine_grid=True`` instead builds a hierarchy of uniform grids: a coarse level 0
grid covering the whole domain with finer grids only where the stretched grid
spacing is finer than the current level's spacing. The hierarchy is planned from
the coordinate values of the stretched grid alone, so no field data is read until
yt requests it.

Note that yt refines all dimensions together by ``refine_by``, so a region that
is finely spaced in one dimension will also be refined in the others. For a grid
that is stretched in only some dimensions, a per-dimension ``refine_by`` (e.g.,
``refine_by=(2, 1, 1)`` to only ever refine the first dimension, or
``refine_by={'x': 2, 'y': 1, 'z': 1}`` to specify it by coordinate name) avoids this. The
loader logs the cell count of the planned hierarchy next to that of a uniform grid
at the finest spacing, so it is easy to check whether refinement is worthwhile for
a given dataset.
