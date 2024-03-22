import numpy as np
import yt_idv

import yt_xarray
from yt_xarray import transformations

ds = yt_xarray.open_dataset("IRIS/wUS-SH-2010_percent.nc")
grid_resolution = (32, 32, 32)
ds_yt = transformations.build_interpolated_cartesian_ds(
    ds,
    [
        "dvs",
    ],
    "depth",
    grid_resolution=grid_resolution,
    refine_grid=True,
    refine_max_iters=2000,
    refine_min_grid_size=8,
    refine_by=4,
)


def _slow_vels(field, data):
    # return negative velocities only, 0 all other elements
    dvs = data["dvs"].copy()
    dvs[np.isnan(dvs)] = 0.0
    dvs[dvs > 0] = 0.0
    return np.abs(dvs)


ds_yt.add_field(
    name=("stream", "slow_dvs"),
    function=_slow_vels,
    sampling_type="local",
)


rc = yt_idv.render_context(height=800, width=800, gui=True)
sg = rc.add_scene(ds_yt, "slow_dvs", no_ghost=True)
rc.run()
