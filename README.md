# yt_xarray


[![PyPI version](https://badge.fury.io/py/yt_xarray.svg)](https://badge.fury.io/py/yt_xarray)


interfaces between yt and xarray


## Features

This is an experimental package for improving the exhange of data between yt
and xarray. The primary access point is the `YtAccessor` object for xarray. To
use it, simply import the package and the `.yt` object will be available to
xarray datasets. For example, to convert the xarray dataset into a full-fledged
yt dataset:

```python
import xarray as xr
import yt_xarray

ds = xr.open_dataset(...)
yt_ds = ds.yt.ds()
```
In this example, `ds.yt.ds()` returns a yt dataset built using the new (as of
yt 4.1.0) callable functionality to reference the open xarray ds handle. See
the [example notebooks](https://github.com/data-exp-lab/yt_xarray/tree/main/examples)
for more approaches.

## Limitations

There are many known (and likely unknown) limitations, here are the most pressing:
* 3D variables are required for yt functionality
* If you have variables on different grids, you must specify the variables you want to load in yt.
* Geometry: the beta version was built using sample domain data from seismology, so there are likely lingering assumptions...

If you try it out and find any issues, please report via [github](https://github.com/data-exp-lab/yt_xarray/issues).


## Contributions

Contributions are welcome, see [CONTRIBUTING.md](CONTRIBUTING.md) for details.
