Some common problems (and how to solve them)
============================================

yt and xarray have many similarities in how they handle their Datasets, but
there are also many aspects that differ to varying degree. This page describes
some of the difficulties you may encounter while using yt_xarray to communicate
between the two.

xarray datasets with a mix of dimensionality
********************************************

It is common for xarray datasets to have variables that have a mix of dimensionality.
Some variables may be 3D in space ``(x, y, z)``, 2D ``(x, y)``, 3D in space and time ``(x, y, t)``, etc.
But with the present implementation,  the fields that are loaded into yt must have the
same dimensionality. So keep in mind that for most of the functionality here, you will have
to provide subsets of fields that you want to work with. For variables with time dimensions,
you will generally have to provide a timestep to load.

See :doc:`examples/example_001_xr_to_yt` and :doc:`examples/example_003_two_dimensional_fields`
for examples.

unknown coordinate names
************************

yt datasets have a fixed expectation for coordinate names. In cartesian, these
coordinate names are ``'x'``, ``'y'``, ``'z'`` while for geographic coordinate systems
the coordinate names are ``'latitude'``, ``'longtiude'`` and then either ``'altitude'``
or ``'depth'``. To work with xarray variables defined with coordinate names that
differ from these, yt_xarray provides some coordinate aliasing.

See :doc:`examples/example_002_coord_aliases` for an example.

unknown coordinate systems
**************************

yt is designed for volumetric data: 3D (or 2D) data defined in a
coordinate system with simple transformations to length axes. The simplest case is
cartesian coordinates where ``'x'``, ``'y'`` and ``'z'`` are already in units of
length but simple non-cartesian coordinate systems are also implemented, including
spherical or geographic coordinate systems. Time-dependence is generally handled
by subsequent loading of single time steps.

In xarray, there are no restrictions on coordinate systems: as long as your
dimensions are self-described in the dataset, you can extract your arrays. For
example, it is common for atmospheric and oceanographic data to use a pressure-related
coordinate systems (such as a `sigma coordinate system <https://en.wikipedia.org/wiki/Sigma_coordinate_system>`_).

While these more general coordinate systems can be loaded into yt via yt_xarray
coordinate aliasing (see `above <#unknown-coordinate-names>`_), it is important to recognize that while most of yt's routines
will successfully run, the results may  be unexpected. Many of yt's analysis routines
involve projecting or volume-averaging, which would yield incorrect results when
treating, for example, a pressure-coordinate as a length. Generally, the functionality
that will work is that related to data selection such as plotting slices (``yt.SlicePlot()``),
subselecting data or finding extrema. Some methods, like ``yt.PhasePlot()`` may work
if care is taken to set the averaging keyword arguments appropriately.

The solution for full functionality is to interpolate your data onto a volumetric
grid (e.g., if using sigma-pressure coordinates, first calculate altitudes then
interpolate your data onto a regular altitude-latitude-longitude grid).

Irregular Grids
***************

The present implementation works with regularly gridded data. Grids may be uniform
(equal spacing in each dimension) or stretched (variable spacing in each dimension)
and spacing need to be equal between dimensions. But hierarchical or adaptive grids
are not presently supported: though it may be possible to load your data in yt, it
just might take a bit more manual work with yt's
`generic data loaders <https://yt-project.org/doc/examining/generic_array_data.html>`_
(NOTE: if you **do** go this route, consider submitting a PR to yt_xarray to expand
the dataset types it can handle!).
