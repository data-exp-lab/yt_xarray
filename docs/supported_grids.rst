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
