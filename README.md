# yt_xarray


[![PyPI version](https://badge.fury.io/py/yt_xarray.svg)](https://badge.fury.io/py/yt_xarray)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/data-exp-lab/yt_xarray/main.svg)](https://results.pre-commit.ci/latest/github/data-exp-lab/yt_xarray/main)

An interface between yt and xarray


## Overview

yt_xarray streamlines communication between
[xarray](https://docs.xarray.dev/en/stable/#) and [yt](https://yt-project.org),
making it easier use yt's visualization and analysis methods with data loaded
via xarray.

Presently, yt_xarray primarily adds methods to generate a yt dataset from a
subset of xarray dataset fields, without making copies of data when possible.

For a more detailed description, check out the full documentation at
[https://yt-xarray.readthedocs.io](https://yt-xarray.readthedocs.io/en/latest/).

## Quick Start

### Installation

The latest stable version can be installed using `pip` with:

```commandline
$ pip install yt_xarray
```

This will install both xarray and yt if you are missing one or the other.

### Usage

The main access point is the `YtAccessor` object for xarray. To
use it, simply import `yt_xarray` and the `.yt` object will be available to
xarray datasets. For example, to convert the xarray dataset into a full-fledged
yt dataset:

```python
import xarray as xr
import yt_xarray

ds = xr.open_dataset(...)
yt_ds = ds.yt.load_grid()
```
In this example, `ds.yt.grid()` returns a yt dataset using all of the fields in
the xarray dataset, ready to use with any yt command. This will, however, only
work if all of your data is defined on the same grid using coordinate names that
yt understands. So for more complex cases, see the [example notebooks](https://yt-xarray.readthedocs.io/en/latest/examples.html)
and the [FAQ](https://yt-xarray.readthedocs.io/en/latest/examples.html) on how
to handle those cases.

## Examples

Check out the [example notebooks](https://yt-xarray.readthedocs.io/en/latest/examples.html)
and the [FAQ](https://yt-xarray.readthedocs.io/en/latest/examples.html) for examples and
descriptions of common issues.

## Getting Help

Bug reports and questions are welcome via [github issues](https://github.com/data-exp-lab/yt_xarray/issues).
You can also reach out via the yt slack channel
([see here for how to join](https://yt-project.org/community.html)) by messaging
Chris Havlin directly or posting in help (though you should tag @Chris Havlin in
your post to get a faster response.)

## Contributions

Contributions are welcome, see [CONTRIBUTING.md](CONTRIBUTING.md) for details.
