import numpy as np
from scipy.signal import butter, filtfilt


def dsig2_dpx2(sig1d):
    dsig1_dpx1 = sig1d[1:] - sig1d[:-1]
    dsig2_dpx2 = np.zeros(sig1d.shape)
    dsig2_dpx2[1:-1] = dsig1_dpx1[1:] - dsig1_dpx1[:-1]
    return dsig2_dpx2


def lowpass_filter(sig_1d):
    # pass a signature array through a lowpass filter. should expose some of these
    # for when a grid is not properly decomposing...
    order = 2  # keep at 2
    fs = 1.0  # sampling frequency [Hz]
    Wn = 0.1  # critical frequency [Hz]
    b, a = butter(order, Wn=Wn, fs=fs, btype="low", analog=False)
    y = filtfilt(b, a, sig_1d)
    return y


class GridBounds:
    # minimal grid representation. not trying to recreate yt here... just
    # need a convenient container.
    def __init__(self, le, re):
        self.le = le
        self.re = re
        if np.any(~np.isfinite(self.le)):
            raise RuntimeError("le is undefined.")
        if np.any(~np.isfinite(self.re)):
            raise RuntimeError("re is undefined.")

        size = np.asarray(self.re) - np.asarray(self.le)
        self.size = size.astype(int)
        self.nd = len(le)

    def plot(self, ax, **kwargs):
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


def find_max_change(f_d2_d2, f):
    # find an index corresponding to the maximum change across 0
    # f_d2_d2 : second derivative of f
    # f: the function
    signs = np.sign(f_d2_d2)
    sign_changes = np.where(signs[:-2] != signs[1:-1])[0] + 1
    mag_change = f_d2_d2[sign_changes + 1] - f_d2_d2[sign_changes]
    max_coord = sign_changes[np.max(mag_change) == mag_change]
    coord = max_coord[0]
    offset = -2 * np.sign(f[coord + 1] - f[coord])
    coord = int(coord + offset)  # offset 1 extra depending on sign
    if coord < 5:
        coord = 4
    elif coord > f_d2_d2.size - 5:
        coord = f_d2_d2.size - 5
    return coord


_sig_axis_sum_order = {0: (1, -1), 1: (0, -1), 2: (0, 0)}


def signature_array(phi, dim):
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


def find_grid_division_coord_1d(grid, phi, dim):
    if phi.shape[dim] < 10:
        return None

    # get signature array
    sig = signature_array(phi, dim)
    # smooth it
    filtered = lowpass_filter(sig)
    # find second derivative
    dx2d2 = dsig2_dpx2(filtered)
    # identify location of max change across zero
    coord = find_max_change(dx2d2, filtered)
    # bump to global index
    coord = coord + grid.le[dim]
    if coord < grid.le[dim] or coord > grid.re[dim]:
        raise RuntimeError()

    return coord


def find_grid_division_coord(phi, grid: GridBounds):
    phi_subset = _grid_subset(grid.le, grid.re, phi)

    grid_coords = []
    for grid_dim in range(grid.nd):
        grid_coords.append(find_grid_division_coord_1d(grid, phi_subset, grid_dim))

    return tuple(grid_coords)


def _grid_subset(le, re, phi):
    nd = len(le)
    slcs = tuple([slice(le[idim], re[idim]) for idim in range(nd)])
    phivals = phi[slcs]
    return phivals


def grid_filled_fraction(le, re, phi):
    grid_vals = _grid_subset(le, re, phi)
    return np.sum(grid_vals) / grid_vals.size


def grid_contains_points(le, re, phi):
    phi_subset = _grid_subset(le, re, phi)
    return np.any(phi_subset > 0)


def split_grid_at_coord(grid, coord, phi):
    new_grids = []
    le = grid.le
    re = grid.re

    # get list of new left and new right edges
    new_l_edges = []
    new_r_edges = []
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
        if grid_contains_points(le_i, re_i, phi):
            new_grids.append(GridBounds(le_i, re_i))

    return new_grids


def decompose_image_mask(
    phi: np.ndarray, max_iters=100, ideal_grid_fill=0.9, min_grid_size=10
):
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
        fill_frac = grid_filled_fraction(grids[igrid].le, grids[igrid].re, phi)
        grid_size_checks = tuple(grids[igrid].size >= min_grid_size)
        if fill_frac < ideal_grid_fill and np.any(grid_size_checks):
            coords = find_grid_division_coord(phi, grids[igrid])
            new_grids = split_grid_at_coord(grids[igrid], coords, phi)
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


def _get_yt_ds(
    image_mask: np.ndarray,
    data_callables: dict,
    bbox: np.ndarray,
    max_iters=200,
    min_grid_size=10,
    refine_by=2,
):
    # first get the grids in pixel dims
    grids, n_iters = decompose_image_mask(
        image_mask, max_iters=max_iters, min_grid_size=min_grid_size
    )

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

    import yt

    return yt.load_amr_grids(
        grid_data,
        image_mask.shape,
        geometry="cartesian",
        bbox=bbox,
        axis_order="xyz",
    )
