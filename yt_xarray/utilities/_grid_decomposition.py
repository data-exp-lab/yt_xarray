from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Tuple, Union

import numpy as np
from numpy import typing as npt
from numpy.typing import ArrayLike
from yt import load_amr_grids

from yt_xarray.transformations import Transformer
from yt_xarray.utilities._utilities import _import_optional_dep
from yt_xarray.utilities.logging import ytxr_log

if TYPE_CHECKING:
    import matplotlib.pyplot as plt


def _dsig2_dpx2(sig1d: npt.NDArray) -> npt.NDArray:
    dsig1_dpx1 = sig1d[1:] - sig1d[:-1]
    dsig2_dpx2 = np.zeros(sig1d.shape)
    dsig2_dpx2[1:-1] = dsig1_dpx1[1:] - dsig1_dpx1[:-1]
    return dsig2_dpx2


def _lowpass_filter(sig_1d: npt.NDArray) -> npt.NDArray:
    # pass a signature array through a lowpass filter. should expose some of these
    # for when a grid is not properly decomposing...
    msg = "This functionality requires scipy. Install it and try again."
    scp_signal = _import_optional_dep("scipy.signal", custom_message=msg)

    order = 2  # keep at 2
    fs = 1.0  # sampling frequency [Hz]
    Wn = 0.1  # critical frequency [Hz]
    b, a = scp_signal.butter(order, Wn=Wn, fs=fs, btype="low", analog=False)
    y = scp_signal.filtfilt(b, a, sig_1d)
    return y


class GridBounds:
    # minimal grid representation. not trying to recreate yt here... just
    # need a convenient container.
    def __init__(self, le: Tuple[int, ...], re: Tuple[int, ...]):
        self.le = le
        self.re = re
        if np.any(~np.isfinite(self.le)):
            raise RuntimeError("le is undefined.")
        if np.any(~np.isfinite(self.re)):
            raise RuntimeError("re is undefined.")

        size = np.asarray(self.re) - np.asarray(self.le)
        self.size = size.astype(int)
        self.nd = len(le)

    def plot(self, ax: "plt.Axes", **kwargs):
        # add a rectangle patch for this grid to a matplotlib axis.
        # all kwargs forwarded to matplotlib.patches.Rectangle
        if self.nd > 2:
            msg = "Grid patch plotting is only implemented for 2d grids."
            raise NotImplementedError(msg)

        from matplotlib.patches import Rectangle

        width = self.size[1]
        height = self.size[0]
        xy = list(self.le)
        xy.reverse()
        rect = Rectangle(xy, width, height, **kwargs)
        ax.add_patch(rect)


def _find_max_change(f_d2_d2: npt.NDArray, f: npt.NDArray) -> int:
    # find an index corresponding to the maximum change across 0
    # f_d2_d2 : second derivative of f
    # f: the function
    signs = np.sign(f_d2_d2)
    sign_changes = np.where(signs[:-2] != signs[1:-1])[0] + 1
    mag_change = f_d2_d2[sign_changes + 1] - f_d2_d2[sign_changes]
    max_coord = sign_changes[np.max(mag_change) == mag_change]
    coord: int = max_coord[0]
    offset = -2 * np.sign(f[coord + 1] - f[coord])
    coord = int(coord + offset)  # offset 1 extra depending on sign
    if coord < 5:
        coord = 4
    elif coord > f_d2_d2.size - 5:
        coord = f_d2_d2.size - 5
    return coord


_sig_axis_sum_order = {0: (1, -1), 1: (0, -1), 2: (0, 0)}


def _signature_array(phi: npt.NDArray, dim: int) -> npt.NDArray:
    # calculate the signature array for an image mask for the
    # specified dimension
    if phi.ndim == 2:
        axid = 1 - dim  # axis to sum over
        sig_array = np.sum(phi, axis=axid)
    elif phi.ndim == 3:
        sig_array = np.sum(phi, axis=_sig_axis_sum_order[dim][0])
        sig_array = np.sum(sig_array, axis=_sig_axis_sum_order[dim][1])
    else:
        raise NotImplementedError("Signature array only implemented in 2d, 3d.")
    assert sig_array.size == phi.shape[dim]
    return sig_array


def _find_grid_division_coord_1d(
    grid: GridBounds, phi: npt.NDArray, dim: int
) -> int | None:
    # find the pixel index at which to split the grid for the present
    # dimension
    if phi.shape[dim] < 10:
        return None

    # get signature array
    sig = _signature_array(phi, dim)
    # smooth it
    filtered = _lowpass_filter(sig)
    # find second derivative
    dx2d2 = _dsig2_dpx2(filtered)
    # identify location of max change across zero
    coord = _find_max_change(dx2d2, filtered)
    # bump to global index
    coord = coord + grid.le[dim]
    return coord


def _find_grid_division_coord(phi: npt.NDArray, grid: GridBounds) -> tuple[int, ...]:
    phi_subset = _grid_subset(grid.le, grid.re, phi)

    grid_coords: list[int | None] = []
    for grid_dim in range(grid.nd):
        grid_coords.append(_find_grid_division_coord_1d(grid, phi_subset, grid_dim))
    # ignoring type checking on following line because:
    # Argument 1 to "tuple" has incompatible type "list[int | None]"; expected "Iterable[int]"
    return tuple(grid_coords)  # type: ignore[arg-type]


def _grid_subset(
    le: Tuple[int, ...], re: Tuple[int, ...], phi: npt.NDArray
) -> npt.NDArray:
    nd = len(le)
    slcs = tuple([slice(le[idim], re[idim]) for idim in range(nd)])
    phivals = phi[slcs]
    return phivals


def _grid_filled_fraction(
    le: Tuple[int, ...], re: Tuple[int, ...], phi: npt.NDArray
) -> float:
    grid_vals = _grid_subset(le, re, phi)
    np_sum: float = np.sum(grid_vals)
    grid_size: int = grid_vals.size
    return np_sum / grid_size


def _grid_contains_points(
    le: Tuple[int, ...], re: Tuple[int, ...], phi: npt.NDArray
) -> bool:
    phi_subset = _grid_subset(le, re, phi)
    contains: bool = np.any(phi_subset > 0)
    return contains


def _split_grid_at_coord(grid: GridBounds, coord, phi: npt.NDArray) -> List[GridBounds]:
    new_grids = []
    le = grid.le
    re = grid.re

    # get list of new left and new right edges
    new_l_edges: list[tuple[float, ...] | float] = []
    new_r_edges: list[tuple[float, ...] | float] = []
    idim_res: tuple[float, ...] | float
    idim_les: tuple[float, ...] | float
    for idim in range(phi.ndim):
        if coord[idim] is not None:
            idim_les = (le[idim], coord[idim])
            idim_res = (coord[idim], re[idim])
        else:
            idim_les = (le[idim],)
            idim_res = re[idim]
        new_l_edges.append(idim_les)
        new_r_edges.append(idim_res)

    # build permutation of new edges
    ixyz_le = np.meshgrid(*new_l_edges, indexing="ij")
    ixyz_re = np.meshgrid(*new_r_edges, indexing="ij")
    ixyz_le = np.column_stack([ixyz_le[idim].ravel() for idim in range(phi.ndim)])
    ixyz_re = np.column_stack([ixyz_re[idim].ravel() for idim in range(phi.ndim)])

    # for each new grid, check if it contains nonzero image values then add
    # the new grid. The loop here isn't too bad, because the size of ixyz_le is
    # 2**nd (so 8 for 3D).
    for igrid in range(ixyz_le.shape[0]):
        le_i = tuple(ixyz_le[igrid])
        re_i = tuple(ixyz_re[igrid])
        if _grid_contains_points(le_i, re_i, phi):
            new_grids.append(GridBounds(le_i, re_i))

    return new_grids


def decompose_image_mask_bisect(
    phi: npt.NDArray,
    max_iters: int = 100,
    ideal_grid_fill: float = 0.95,
    min_grid_size: int = 10,
) -> Tuple[List[GridBounds], int]:
    # this decomposition divides each dimension in half, adding 2**nd new potential
    # grids. If a new grid is empty, it is discarded.

    grids = []
    grids.append(GridBounds((0,) * phi.ndim, phi.shape))

    keepgoing = True
    igrid = 0
    n_iters = 0

    while keepgoing:
        fill_frac = _grid_filled_fraction(grids[igrid].le, grids[igrid].re, phi)
        grid_size_checks = tuple(grids[igrid].size >= min_grid_size)
        if fill_frac < ideal_grid_fill and np.any(grid_size_checks):
            grid = grids[igrid]
            coords = []
            for dim in range(phi.ndim):
                coords.append(int(grid.le[dim] + grid.size[dim] / 2))
            coords_in = tuple(coords)

            new_grids = _split_grid_at_coord(grids[igrid], coords_in, phi)
            if len(new_grids) > 0:
                # remove divided grid
                del grids[igrid]
                # add the new ones to the end
                grids.extend(new_grids)
                # do **not** increment igrid, because we deleted one.
            else:
                igrid += 1
        else:
            igrid += 1

        if igrid >= len(grids) or n_iters >= max_iters:
            keepgoing = False

        n_iters += 1

    return grids, n_iters


def decompose_image_mask(
    phi: npt.NDArray,
    max_iters: int = 100,
    ideal_grid_fill: float = 0.9,
    min_grid_size: int = 10,
) -> Tuple[List[GridBounds], int]:
    # following berger and rigoutsos 1991 (https://doi.org/10.1109/21.120081)
    # to take a binary image mask and construct grids that include the
    # non-zero pixels of the image mask.

    if min_grid_size < 10:
        # must be >=10 due to lowpass filter requirement
        min_grid_size = 10

    grids = []
    grids.append(GridBounds((0,) * phi.ndim, phi.shape))

    keepgoing = True
    igrid = 0
    n_iters = 0

    while keepgoing:
        fill_frac = _grid_filled_fraction(grids[igrid].le, grids[igrid].re, phi)
        grid_size_checks = tuple(grids[igrid].size >= min_grid_size)
        if fill_frac < ideal_grid_fill and np.any(grid_size_checks):
            coords = _find_grid_division_coord(phi, grids[igrid])
            new_grids = _split_grid_at_coord(grids[igrid], coords, phi)
            if len(new_grids) > 0:
                # remove divided grid
                del grids[igrid]
                # add the new ones to the end
                grids.extend(new_grids)
                # do **not** increment igrid, because we deleted one.
            else:
                igrid += 1
        else:
            igrid += 1

        if igrid >= len(grids) or n_iters >= max_iters:
            keepgoing = False

        n_iters += 1

    return grids, n_iters


def _create_image_mask(
    bbox_cart: npt.NDArray,
    bbox_native: npt.NDArray,
    res: Union[tuple[int, ...], npt.NDArray],
    tform: Transformer,
    chunks: Optional[int] = 100,
) -> npt.NDArray:
    emsg = (
        "This functionality requires dask[array], "
        "install it with `pip install dask[array]`"
    )
    da = _import_optional_dep("dask.array", custom_message=emsg)

    # create a geometry image mask
    res = np.asarray(res)
    wid = (bbox_cart[:, 1] - bbox_cart[:, 0]) / res

    # cell centers
    xyz_1d = [
        bbox_cart[i, 0] + wid[i] / 2.0 + wid[i] * da.arange(res[i], chunks=chunks)
        for i in range(3)
    ]
    xyz = da.meshgrid(*xyz_1d, indexing="ij")
    del xyz_1d

    # mark 1 if any cell corner falls in domain.
    corner_masks = []

    def _get_corner_mask(
        bbox_native: npt.NDArray,
        tform: Transformer,
        xyz: Tuple[npt.NDArray, npt.NDArray, npt.NDArray],
        wid: npt.NDArray,
        ix: float,
        iy: float,
        iz: float,
    ) -> npt.NDArray:
        # check if an individual corner of a mesh element falls within the native
        # bounding box.
        #
        # bbox_native: bounding box array in native coordinates, in order expected
        #              by transform
        # tform: transformer instance
        # xyz: tuple of xyz arrays
        # wid: grid spacing in xyz
        # ix, iy, iz are one of (-1, 1)

        x = xyz[0] + ix * wid[0] / 2.0
        y = xyz[1] + iy * wid[1] / 2.0
        z = xyz[2] + iz * wid[2] / 2.0
        coords = tform.to_native(x=x, y=y, z=z)
        dim_masks = []
        for idim in range(3):
            min_val = bbox_native[idim, 0]
            max_val = bbox_native[idim, 1]
            dim_v = coords[idim]
            msk = np.logical_and(dim_v >= min_val, dim_v <= max_val)
            dim_masks.append(msk)
        corner_mask = np.logical_and(dim_masks[0], dim_masks[1])
        corner_mask = np.logical_and(corner_mask, dim_masks[2])
        return corner_mask

    for ix in [-1.0, 1.0]:
        for iy in [-1.0, 1.0]:
            for iz in [-1.0, 1.0]:
                corner_masks.append(
                    _get_corner_mask(bbox_native, tform, xyz, wid, ix, iy, iz)
                )

    image_mask = corner_masks[0]
    for msk in corner_masks[1:]:
        image_mask = np.logical_or(image_mask, msk)
    return image_mask.compute()


def _get_yt_ds(
    image_mask: npt.NDArray,
    data_callables: dict[str, Callable[[Any], Any]],
    bbox: npt.NDArray,
    max_iters=200,
    min_grid_size=10,
    refine_by=2,
    refinement_method="division",
    **load_kwargs,
):
    # first get the grids in pixel dims
    if refinement_method == "signature_filter":
        grids, n_iters = decompose_image_mask(
            image_mask, max_iters=max_iters, min_grid_size=min_grid_size
        )
    elif refinement_method == "division":
        # always divide by 2
        grids, n_iters = decompose_image_mask_bisect(
            image_mask, max_iters=max_iters, min_grid_size=min_grid_size
        )
    else:
        msg = (
            f"refinement_method must be 'signature_filter' or 'division' but "
            f"found {refinement_method}"
        )
        raise ValueError(msg)

    msg = f"Decomposed into {len(grids)} grids after {n_iters} iterations."
    ytxr_log.info(msg)

    # build the grid dict list expected by yt
    dxyx = (bbox[:, 1] - bbox[:, 0]) / image_mask.shape
    grid_data = []

    # first add a single grid covering the whole domain
    gdict = {
        "left_edge": bbox[:, 0],
        "right_edge": bbox[:, 1],
        "dimensions": image_mask.shape,
        "level": 0,
    }
    for ky, val in data_callables.items():
        gdict[ky] = val
    grid_data.append(gdict)

    # all other grids are level 1
    for grid in grids:
        le = bbox[:, 0] + grid.le * dxyx
        re = le + grid.size * dxyx

        lev0_size = grid.size
        lev1_size = tuple(lev0_size * refine_by)

        gdict = {
            "left_edge": le,
            "right_edge": re,
            "dimensions": lev1_size,
            "level": 1,
        }
        for ky, val in data_callables.items():
            gdict[ky] = val
        grid_data.append(gdict)

    return load_amr_grids(
        grid_data,
        image_mask.shape,
        geometry="cartesian",
        bbox=bbox,
        axis_order="xyz",
        refine_by=refine_by,
        **load_kwargs,
    )


class ChunkInfo:
    """
    Class for tracking info related to chunked-decomposition of a domain

    Parameters
    ----------
    data_shp: Tuple[int,]
        the global shape of the data to chunk
    chunksizes: np.ndarray[int]
        the chunksizes in each dimension of data_shp
    starting_index_offset: np.ndarray[int]
        global index offset. start and end indices will be offset
        by this array. Defaults to [0,0,0].
    """

    def __init__(
        self,
        data_shp: Tuple[int, ...],
        chunksizes: npt.NDArray[np.int64],
        starting_index_offset: npt.NDArray[np.int64] | None = None,
    ):

        self.chunksizes = chunksizes
        self.data_shape = np.asarray(data_shp).astype(np.int64)
        self.n_chnk: npt.NDArray[np.int64] = (
            self.data_shape / chunksizes
        )  # may not be int
        self.n_whl_chnk = np.floor(self.n_chnk).astype(int)  # whole chunks in each dim
        self.n_part_chnk = np.ceil(self.n_chnk - self.n_whl_chnk).astype(int)
        self.n_tots = np.prod(self.n_part_chnk + self.n_whl_chnk)

        self.ndim = len(data_shp)
        if starting_index_offset is None:
            starting_index_offset = np.zeros(self.data_shape.shape, dtype=np.int64)
        self.starting_index_offset: npt.NDArray[np.int64] = starting_index_offset

    _si: List[npt.NDArray[np.int64]] | None = None
    _ei: List[npt.NDArray[np.int64]] | None = None
    _sizes: List[npt.NDArray[np.int64]] | None = None

    @property
    def si(self) -> List[npt.NDArray[np.int64]]:
        """
        The starting indices of individual chunks by dimension.
        Includes any global offset.
        """
        if self._si is None:
            si_list = []
            ei_list = []
            size_list = []
            for idim in range(self.ndim):

                # first get the starting and end points of whole chunks
                si0 = self.starting_index_offset[idim]
                si_0 = si0 + self.chunksizes[idim] * np.arange(self.n_whl_chnk[idim])
                ei_0 = si_0 + self.chunksizes[idim]

                # if this dim has a partial chunk at the end, add on a
                # partial chunk.
                if self.n_part_chnk[idim] == 1:
                    si_0_partial = si0 + self.chunksizes[idim] * self.n_whl_chnk[idim]
                    ei_0_partial = si0 + self.data_shape[idim]
                    si_0 = np.concatenate([si_0, [si_0_partial]])
                    ei_0 = np.concatenate([ei_0, [ei_0_partial]])
                si_list.append(si_0)
                ei_list.append(ei_0)
                size_list.append(ei_0 - si_0)
            self._si = si_list
            self._ei = ei_list
            self._sizes = size_list
        return self._si

    @property
    def ei(self) -> List[npt.NDArray[np.int64]]:
        """
        The ending indices of individual chunks by dimension.
        Includes any global offset.
        """
        if self._ei is None:
            _ = self.si
            assert self._ei is not None
        return self._ei

    @property
    def sizes(self) -> List[npt.NDArray[np.int64]]:
        if self._sizes is None:
            _ = self.si
            assert self._sizes is not None
        return self._sizes


_Box = Tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]


@dataclass
class GridSpec:
    """
    Minimal description of a single yt grid patch: physical edges, cell
    dimensions and refinement level.
    """

    left_edge: npt.NDArray[np.float64]
    right_edge: npt.NDArray[np.float64]
    dimensions: npt.NDArray[np.int64]
    level: int = 0

    @property
    def n_cells(self) -> int:
        return int(np.prod(self.dimensions))

    def to_yt_dict(self) -> dict[str, Any]:
        return {
            "left_edge": self.left_edge,
            "right_edge": self.right_edge,
            "dimensions": self.dimensions,
            "level": self.level,
        }


def _validate_refine_by(
    refine_by: Union[int, ArrayLike], ndim: int
) -> npt.NDArray[np.int64]:
    rb: npt.NDArray[np.int64]
    if np.ndim(refine_by) == 0:
        rb = np.full(ndim, int(refine_by), dtype=np.int64)
    else:
        rb = np.asarray(refine_by, dtype=np.int64)
        if rb.shape != (ndim,):
            raise ValueError(
                f"refine_by must be an integer or a sequence of {ndim} integers, "
                f"found {refine_by!r}"
            )
    if np.any(rb < 1):
        raise ValueError(f"refine_by values must be >= 1, found {refine_by!r}")
    return rb


def _uniform_chunk_grids(
    bbox: npt.NDArray[np.float64],
    dxyz: npt.NDArray[np.float64],
    chunksizes: npt.NDArray[np.int64],
) -> List[GridSpec]:
    # decompose a uniform grid spanning bbox with spacing dxyz into chunks
    # of at most chunksizes cells, as a list of level 0 GridSpec
    dims = np.round((bbox[:, 1] - bbox[:, 0]) / dxyz).astype(int)
    chnkinfo = ChunkInfo(tuple(dims), chunksizes)
    ndim = len(dims)
    starts_mesh = np.meshgrid(*chnkinfo.si, indexing="ij")
    ends_mesh = np.meshgrid(*chnkinfo.ei, indexing="ij")
    starts = np.column_stack([s.ravel() for s in starts_mesh])
    ends = np.column_stack([e.ravel() for e in ends_mesh])

    grids = []
    for igrid in range(starts.shape[0]):
        si = starts[igrid]
        ei = ends[igrid]
        grids.append(
            GridSpec(
                left_edge=bbox[:, 0] + si * dxyz,
                right_edge=bbox[:, 0] + ei * dxyz,
                dimensions=(ei - si).astype(int),
                level=0,
            )
        )
    assert len(grids) == chnkinfo.n_tots
    assert ndim == grids[0].left_edge.size
    return grids


def _required_levels_1d(
    widths: npt.NDArray[np.float64], dx0: float, refine_by: int, max_level: int
) -> npt.NDArray[np.int64]:
    # the refinement level required to resolve each source cell width, given
    # a level 0 spacing of dx0: the smallest L with dx0 / refine_by**L <= width
    tol = 1e-8
    if refine_by == 1:
        return np.zeros(widths.shape, dtype=np.int64)
    ratio = dx0 / widths
    levels = np.ceil(np.log(ratio) / np.log(refine_by) - tol)
    levels = np.clip(levels, 0, max_level)
    result: npt.NDArray[np.int64] = levels.astype(np.int64)
    return result


def _merge_intervals(
    intervals: npt.NDArray[np.int64],
) -> npt.NDArray[np.int64]:
    # merge overlapping or adjacent integer intervals. intervals is an (n, 2) array
    if intervals.shape[0] == 0:
        return intervals.reshape((0, 2))
    order = np.argsort(intervals[:, 0])
    intervals = intervals[order]
    merged = [intervals[0].copy()]
    for lo, hi in intervals[1:]:
        if lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append(np.array([lo, hi]))
    return np.array(merged)


def _needed_intervals_1d(
    coord_vals: npt.NDArray[np.float64],
    left_edge: float,
    n0: int,
    dx0: float,
    refine_by: int,
    max_level: int,
) -> List[npt.NDArray[np.int64]]:
    # for a single dimension, find the index intervals (in units of the finest
    # level's cells) that need refinement at each level. Returns a list with
    # an (n, 2) array for each level from 0 to max_level (level 0 is always
    # the full domain).

    widths = coord_vals[1:] - coord_vals[:-1]
    req_levels = _required_levels_1d(widths, dx0, refine_by, max_level)

    n_finest = n0 * refine_by**max_level
    needed = [np.array([[0, n_finest]], dtype=np.int64)]
    eps = 1e-6
    for level in range(1, max_level + 1):
        # source cells needing this level or finer
        msk = req_levels >= level
        if not np.any(msk):
            needed.append(np.zeros((0, 2), dtype=np.int64))
            continue
        starts = coord_vals[:-1][msk]
        ends = coord_vals[1:][msk]

        # snap outward to level-1 cell boundaries, in level-1 index units
        dx_parent = dx0 / refine_by ** (level - 1)
        n_parent = n0 * refine_by ** (level - 1)
        i0 = np.floor((starts - left_edge) / dx_parent + eps)
        i1 = np.ceil((ends - left_edge) / dx_parent - eps)
        i0 = np.clip(i0, 0, n_parent).astype(np.int64)
        i1 = np.clip(i1, 0, n_parent).astype(np.int64)
        keep = i1 > i0
        intervals = np.column_stack([i0[keep], i1[keep]])
        # convert to finest-level units
        intervals = intervals * refine_by ** (max_level - level + 1)
        needed.append(_merge_intervals(intervals))
    return needed


def _partition_1d(
    needed: List[npt.NDArray[np.int64]], level: int
) -> npt.NDArray[np.int64]:
    # breakpoints for the partition of a dimension at a given level: the union
    # of all interval endpoints for levels <= level.
    pts = [needed[0].ravel()]
    for lev in range(1, level + 1):
        pts.append(needed[lev].ravel())
    return np.unique(np.concatenate(pts))


def _interval_is_needed(
    lo: int, hi: int, needed_at_level: npt.NDArray[np.int64]
) -> bool:
    # partition intervals never straddle a needed-interval endpoint, so a
    # midpoint check is exact
    mid = (lo + hi) / 2.0
    return bool(np.any((needed_at_level[:, 0] <= mid) & (mid <= needed_at_level[:, 1])))


def _merge_boxes_along_dim(
    boxes: List[_Box],
    idim: int,
    parent_breakpoints: npt.NDArray[np.int64],
) -> List[_Box]:
    # greedily merge boxes adjacent along idim when they share all other
    # extents and the shared face is not a parent-level breakpoint (so the
    # merged box still sits within a single parent grid).
    parent_set = set(parent_breakpoints.tolist())
    groups: dict[tuple[int, ...], list[_Box]] = {}
    for lo, hi in boxes:
        key = tuple(np.delete(lo, idim).tolist() + np.delete(hi, idim).tolist())
        groups.setdefault(key, []).append((lo, hi))

    merged = []
    for group in groups.values():
        group.sort(key=lambda b: b[0][idim])
        current_lo, current_hi = group[0]
        for lo, hi in group[1:]:
            adjacent = lo[idim] == current_hi[idim]
            if adjacent and int(lo[idim]) not in parent_set:
                current_hi = current_hi.copy()
                current_hi[idim] = hi[idim]
            else:
                merged.append((current_lo, current_hi))
                current_lo, current_hi = lo, hi
        merged.append((current_lo, current_hi))
    return merged


def _auto_max_level(
    coord_arrays: List[npt.NDArray[np.float64]],
    cell_spacing: npt.NDArray[np.float64],
    refine_by: npt.NDArray[np.int64],
) -> int:
    max_level = 0
    for idim, cvals in enumerate(coord_arrays):
        if refine_by[idim] == 1:
            continue
        min_w = np.min(cvals[1:] - cvals[:-1])
        levels = _required_levels_1d(
            np.array([min_w]), cell_spacing[idim], int(refine_by[idim]), 1000
        )
        max_level = max(max_level, int(levels[0]))
    return max_level


def plan_stretched_grid_refinement(
    coord_arrays: List[npt.NDArray[np.float64]],
    bbox: npt.NDArray[np.float64],
    cell_spacing: npt.NDArray[np.float64],
    refine_by: Union[int, ArrayLike] = 2,
    max_level: int | None = None,
    merge_grids: bool = True,
) -> List[GridSpec]:
    """
    Plan an AMR hierarchy of uniform grids that resolves a stretched grid,
    based only on the stretched grid's coordinate spacing.

    Parameters
    ----------
    coord_arrays: list of 1d arrays
        the monotonically increasing coordinate values of the stretched grid,
        one array per dimension.
    bbox: np.ndarray
        the (ndim, 2) bounding box of the level 0 grid. The width in each
        dimension must be an integer multiple of cell_spacing.
    cell_spacing: np.ndarray
        the level 0 cell spacing in each dimension.
    refine_by: int or sequence of int
        the refinement factor per level, either a single integer or one per
        dimension. A dimension with a factor of 1 is never refined.
    max_level: int
        the maximum refinement level. If None, the hierarchy is as deep as
        needed to resolve the smallest spacing of the stretched grid in every
        dimension with a refinement factor greater than 1.
    merge_grids: bool
        if True (default), adjacent grids are merged where possible.

    Returns
    -------
    list of GridSpec
        the grid patches, level 0 first. Every grid above level 0 is fully
        contained within a single grid of the level below.

    """

    ndim = len(coord_arrays)
    bbox = np.asarray(bbox, dtype=np.float64)
    cell_spacing = np.asarray(cell_spacing, dtype=np.float64)
    rb = _validate_refine_by(refine_by, ndim)

    n0 = np.round((bbox[:, 1] - bbox[:, 0]) / cell_spacing).astype(int)
    if not np.allclose(bbox[:, 0] + n0 * cell_spacing, bbox[:, 1]):
        raise ValueError(
            "The bbox width must be an integer multiple of cell_spacing in each "
            "dimension."
        )

    if max_level is None:
        max_level = _auto_max_level(coord_arrays, cell_spacing, rb)
    if max_level < 0:
        raise ValueError(f"max_level must be >= 0, found {max_level}")

    for idim in range(ndim):
        if rb[idim] == 1:
            min_w = np.min(coord_arrays[idim][1:] - coord_arrays[idim][:-1])
            if cell_spacing[idim] > min_w * (1 + 1e-8):
                ytxr_log.warning(
                    f"Dimension {idim} has refine_by of 1 but the cell spacing "
                    f"({cell_spacing[idim]}) is larger than the smallest source "
                    f"spacing ({min_w}): the source grid will be under-resolved."
                )

    # finest-level cell counts and the 1d needed intervals per dimension
    needed = []
    for idim in range(ndim):
        needed.append(
            _needed_intervals_1d(
                coord_arrays[idim],
                bbox[idim, 0],
                int(n0[idim]),
                cell_spacing[idim],
                int(rb[idim]),
                max_level,
            )
        )

    # the size of a cell at each level in finest-level units, by dimension
    def _cells_per_level_cell(level: int) -> npt.NDArray[np.int64]:
        result: npt.NDArray[np.int64] = rb ** (max_level - level)
        return result

    dx_finest = cell_spacing / _cells_per_level_cell(0)

    grids = [
        GridSpec(
            left_edge=bbox[:, 0].copy(),
            right_edge=bbox[:, 1].copy(),
            dimensions=n0.astype(np.int64),
            level=0,
        )
    ]

    for level in range(1, max_level + 1):
        partitions = [_partition_1d(needed[idim], level) for idim in range(ndim)]
        intervals_1d = [
            np.column_stack([p[:-1], p[1:]]) for p in partitions
        ]  # (n_int, 2) per dim
        needed_flags = [
            np.array(
                [
                    _interval_is_needed(lo, hi, needed[idim][level])
                    for lo, hi in intervals_1d[idim]
                ],
                dtype=bool,
            )
            for idim in range(ndim)
        ]

        # cartesian product of the per-dimension intervals
        index_grids = np.meshgrid(
            *[np.arange(len(iv)) for iv in intervals_1d], indexing="ij"
        )
        index_combos = np.column_stack([ig.ravel() for ig in index_grids])

        boxes: List[_Box] = []
        for combo in index_combos:
            is_needed = any(needed_flags[idim][combo[idim]] for idim in range(ndim))
            if not is_needed:
                continue
            lo = np.array([intervals_1d[idim][combo[idim], 0] for idim in range(ndim)])
            hi = np.array([intervals_1d[idim][combo[idim], 1] for idim in range(ndim)])
            boxes.append((lo, hi))

        if merge_grids:
            for idim in range(ndim):
                parent_pts = _partition_1d(needed[idim], level - 1)
                boxes = _merge_boxes_along_dim(boxes, idim, parent_pts)

        cell_size = _cells_per_level_cell(level)
        for lo, hi in boxes:
            dims = (hi - lo) // cell_size
            assert np.all(dims * cell_size == hi - lo)
            grids.append(
                GridSpec(
                    left_edge=bbox[:, 0] + lo * dx_finest,
                    right_edge=bbox[:, 0] + hi * dx_finest,
                    dimensions=dims.astype(np.int64),
                    level=level,
                )
            )

    return grids


def summarize_grid_plan(grids: List[GridSpec]) -> dict[int, dict[str, int]]:
    """
    Count the grids and cells at each level of a grid plan.

    Returns
    -------
    dict
        keyed by level, with "n_grids" and "n_cells" entries.
    """
    summary: dict[int, dict[str, int]] = {}
    for grid in grids:
        entry = summary.setdefault(grid.level, {"n_grids": 0, "n_cells": 0})
        entry["n_grids"] += 1
        entry["n_cells"] += grid.n_cells
    return summary
