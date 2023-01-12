# yt_xarray


[![PyPI version](https://badge.fury.io/py/yt_xarray.svg)](https://badge.fury.io/py/yt_xarray)


An interface between yt and xarray


## Features

This is a package for improving the exchange of data between yt
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
In this example, `ds.yt.ds()` returns a yt dataset, ready to use with any yt command. See
the [example notebooks](https://github.com/data-exp-lab/yt_xarray/tree/main/examples)
for more approaches.

## On Memory Management

yt_xarray strives to avoid creating new copies of data when possible. Rather than
making copies of arrays to store in the yt dataset, references to the open xarray
dataset handle are stored and then accessed only when needed. In some cases, this
may cause problems, and so it is also possible to create a yt dataset from copies
of the xarray values.

## Common problems and their solutions

### xarray datasets with a mix of dimensionality

If you have variables on different grids, you must specify the variables you want to load in yt.

At present, the fields that are loaded into yt must have the same dimensionality.
See lkawejralserkj.

2D, 3D, time

### unknown coordinate names

## Supported geometries and grids

### regular grids

uniform or stretched

### geometries

cartesian, spherical and geographic

### non-spatial coordinate systems

It is common for atmospheric and oceanographic data to use a pressure coordinate
systems.

While these can be loaded into yt, it is important
to recognize that while most of yt's routines will succesfully run, the results may
be incorrect. Plotting slices and selecting data should work as expected, but any
routines that involve integration over the volume (projection plots, volume-averaging)
will treat the pressure coordinate as a length and thus yield incorrect results.

The solution for full functionality is (at present) not great: calculate the
altitude and then interpolate onto a uniform grid.

## Getting Help

If you try it out and find any issues, please report via [github](https://github.com/data-exp-lab/yt_xarray/issues).


## Contributions

Contributions are welcome, see [CONTRIBUTING.md](CONTRIBUTING.md) for details.
