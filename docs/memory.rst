Memory Management in yt_xarray
==============================



yt_xarray does its best to avoid creating new copies of data. But there are
some instances where this may fail and yt_xarray must resort to fully copying
data.



yt_xarray: loading via callables
********************************

To load data, yt_xarray relies on yt's S
Rather than making copies of arrays to store in the yt dataset, references to
the open xarray dataset handle are stored and then accessed only when needed
In some cases, this
may cause problems, and so it is also possible to create a yt dataset from copies
of the xarray values.

yt grids
********

yt grids are built as cell-centered.
