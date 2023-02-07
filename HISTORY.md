# History

## 0.1.1 (2022-02-07)

This release builds out the loading functions introduced in v0.1.0 and includes
improvements to documentation

### New

`ds.yt.load_grid` now has:
* initial support for loading 3D uniform grid data with chunks
* support for 2D fields for non-chunked data
* support for stretched (non-chunked) and uniform grids

CI improvements, including:
* test coverage checks on PR
* rtd docs build, including builds on PR

### Changes
* replaced the multiple load functions with a single function, `ds.yt.load_grid`

### Fixes
* correctly handle nodes as cell centers, interpolating if necessary

## 0.1.0 (2022-03-17)
* First release on PyPI.
