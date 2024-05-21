yt\_xarray.YtAccessor
=====================

To use the ``YtAccessor`` methods, simply ``import yt_xarray`` before
loading your dataset. Accessor methods will then be available at ``ds.yt``.
For example::


    import xarray as xr
    import yt_xarray

    ds = xr.open_datset(...)
    ds.yt.load_grid(...)

The full method definitions are as follows:

.. autoclass:: yt_xarray.YtAccessor
   :members:
   :show-inheritance:
