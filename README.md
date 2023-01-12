# yt_xarray


[![PyPI version](https://badge.fury.io/py/yt_xarray.svg)](https://badge.fury.io/py/yt_xarray)


An interface between yt and xarray


## Usage

At present, the primary purpose of yt_xarray is to streamline the creation of a yt
dataset from an existing xarray dataset, allowing instant access to all of yt's
functionality [without fully copying data](#on-memory-management).

The main access point is the `YtAccessor` object for xarray. To
use it, simply import `yt_xarray` and the `.yt` object will be available to
xarray datasets. For example, to convert the xarray dataset into a full-fledged
yt dataset:

```python
import xarray as xr
import yt_xarray

ds = xr.open_dataset(...)
yt_ds = ds.yt.ds()
```
In this example, `ds.yt.ds()` returns a yt dataset using all of the fields in
the xarray dataset, ready to use with any yt command. See
the [example notebooks](#example-notebooks) for more approaches.

## Example Notebooks

The [examples directory](https://github.com/data-exp-lab/yt_xarray/tree/main/examples)
of the repository contains a number of jupyter notebooks demonstrating functionality:

TODO: link to notebooks exactly. Or link to a documentation page.

## On Memory Management

yt_xarray does its best to avoid creating new copies of data. Rather than
making copies of arrays to store in the yt dataset, references to the open xarray
dataset handle are stored and then accessed only when needed. In some cases, this
may cause problems, and so it is also possible to create a yt dataset from copies
of the xarray values.

## Common problems and their solutions

### xarray datasets with a mix of dimensionality

It is common for xarray datasets to have variables that have a mix of dimensionality.
Some variables may be 3D in space (x, y, z), 2D (x, y), 3D in space and time (x, y, t), etc.
But with the present implmentation,  the fields that are loaded into yt must have the
same dimensionality. So keep in mind that for most of the functionality here, you will have
to provide subsets of fields that you want to work with. For variables with time dimensions,
you will generally have to provide a timestep to load.

See the example HERE for a detailed look.

### unknown coordinate names

yt datasets have a fixed expectation for coordinate names. In cartesian, these
coordinate names are `'x'`, `'y'`, `'z'` while for geographic coordinate systems
the coordinate names are `'latitude'`, `'longtiude'` and then either `'altitude'`
or `'depth'`. To work with xarray variables defined with coordinate names that
differ from these, yt_xarray provides some coordinate aliasing, see HERE for an
example.

## Some notes on grids and coordinate systems

### regular grids and geometries

The present implementation works with regularly gridded data. Grids may be uniform
(equal spacing in each dimension) or stretched (variable spacing in each dimension)
and spacing need to be equal between dimensions. But hierarchical or adaptive grids
are not presently supported (though it may be possible to load your data in yt, it
just might take a bit more work).

### non-spatial coordinate systems

yt is designed for volumetric data: 3D (or 2D) data defined in a
coordinate system with simple transformations to length axes. The simplest case is
cartesian coordinates where x, y and z are already in units of length but simple
non-cartesian coordinate systems are also implemented, including spherical or
geographic coordinate systems. Time-dependence is generally handled by subsequent
loading of single time steps.

In xarray, there are no restrictions on coordinate systems: as long as your
dimensions are self-described in the dataset, you can extract your arrays. For
example, it is common for atmospheric and oceanographic data to use a pressure-related
coordinate systems (such as a [sigma pressure coordinate system](https://en.wikipedia.org/wiki/Sigma_coordinate_system))

While these more general coordinate systems can be loaded into yt via yt_xarray
coordinate aliasing, it is important to recognize that while most of yt's routines
will succesfully run, the results may  be unexpected. Plotting slices, selecting
data, finding extrema and more will work as expected, but any routines that involve
integration or averaging over the volume (projection plots, volume-averaging)
will treat the all coordinates as length coordinates and thus may yield incorrect results.

The solution for full functionality is to interpolate your data onto a volumetric
grid (e.g., if using sigma-pressure coordinates, first calculate altitudes then
interpolate your data onto a regular altitude-latitude-longitude grid).

## Getting Help

If you try it out and find any issues, please report via [github](https://github.com/data-exp-lab/yt_xarray/issues).

## Contributions

Contributions are welcome, see [CONTRIBUTING.md](CONTRIBUTING.md) for details.
