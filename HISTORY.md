# History

## 0.2.0 (2023-10-10)

Maintenance release. Minimum python version is now 3.9.

### Changes
* improve error messaging for load_grid
* fix deprecation warning from yt.load_amr_grids
* consolidate to pyproject.toml
* expand testing: all platforms, new weekly run
* switch to pre-commit.ci

## 0.1.4 (2023-03-21)

Bug fix release.

### Fixes:
* fix coordinate flipping bug (PR [46](https://github.com/data-exp-lab/yt_xarray/pull/46))
## 0.1.3 (2023-03-10)

Bug fix release.

### Fixes:
* handle the case where data coordinate lengths are 1 (PR [41](https://github.com/data-exp-lab/yt_xarray/pull/41))

## 0.1.2 (2023-02-27)

Bug fix release.

### Fixes:
* fixes handling of xarray variables with dimensions in decreasing
  order (e.g., latitude from 90 to -90) without
having to re-index the whole variable (PR [39](https://github.com/data-exp-lab/yt_xarray/pull/39)).

## 0.1.1 (2023-02-07)

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
