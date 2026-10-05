"""Compatible H(curl) transfer between non-nested rectilinear Maxwell grids.

The local-self convergence audit uses the same physical box at several mesh
steps, but ``stretched_axis`` does not make those grids nested.  Interpolating
Cartesian field components at edge centres is therefore not a compatible edge
transfer and can create a very large artificial curl on the refined grid.

For lowest-order Cartesian Nedelec fields the component parallel to an edge is
constant in the longitudinal coordinate inside a coarse cell and bilinear in
the two transverse coordinates.  A fine edge degree of freedom is its line
integral, so the exact geometric prolongation is obtained by integrating that
coarse Nedelec reconstruction along every fine edge, splitting the integral at
coarse-cell boundaries.  This construction preserves constant vector fields
and commutes with the nodal gradient interpolation up to roundoff.
"""
from __future__ import annotations

from array import array

import numpy as np
import scipy.sparse as sp


def _axes_tuple(axes):
    out = tuple(np.asarray(axis, float).reshape(-1) for axis in axes)
    if len(out) != 3:
        raise ValueError("H(curl) transfer requires three Cartesian axes")
    for axis in out:
        if axis.size < 3 or np.any(np.diff(axis) <= 0.0):
            raise ValueError("H(curl) transfer axes must be strictly increasing")
    return out


def _edge_layout(axes):
    x, y, z = _axes_tuple(axes)
    nx, ny, nz = len(x) - 1, len(y) - 1, len(z) - 1
    nx_edges = nx * (ny + 1) * (nz + 1)
    ny_edges = (nx + 1) * ny * (nz + 1)
    nz_edges = (nx + 1) * (ny + 1) * nz
    offsets = (0, nx_edges, nx_edges + ny_edges)
    return (nx, ny, nz), offsets, nx_edges + ny_edges + nz_edges


def _edge_id(axis, i, j, k, shape, offsets):
    nx, ny, nz = shape
    if axis == 0:
        return offsets[0] + (int(i) * (ny + 1) + int(j)) * (nz + 1) + int(k)
    if axis == 1:
        return offsets[1] + (int(i) * ny + int(j)) * (nz + 1) + int(k)
    return offsets[2] + (int(i) * (ny + 1) + int(j)) * nz + int(k)


def _locate_interval(nodes, value):
    value = float(value)
    scale = max(1.0, float(np.max(np.abs(nodes))))
    tol = 32.0 * np.finfo(float).eps * scale
    if value < nodes[0] - tol or value > nodes[-1] + tol:
        raise ValueError("fine-grid point lies outside coarse H(curl) transfer box")
    if value <= nodes[0] + tol:
        return 0, 0.0
    if value >= nodes[-1] - tol:
        return len(nodes) - 2, 1.0
    index = int(np.searchsorted(nodes, value, side="right") - 1)
    index = min(max(index, 0), len(nodes) - 2)
    t = (value - nodes[index]) / (nodes[index + 1] - nodes[index])
    return index, float(np.clip(t, 0.0, 1.0))


def _longitudinal_segments(nodes, start, stop):
    start = float(start)
    stop = float(stop)
    if not stop > start:
        raise ValueError("fine edge has non-positive length")
    scale = max(1.0, float(np.max(np.abs(nodes))))
    tol = 64.0 * np.finfo(float).eps * scale
    if start < nodes[0] - tol or stop > nodes[-1] + tol:
        raise ValueError("fine edge leaves coarse H(curl) transfer box")
    position = max(start, float(nodes[0]))
    stop = min(stop, float(nodes[-1]))
    index = int(np.searchsorted(nodes, position, side="right") - 1)
    index = min(max(index, 0), len(nodes) - 2)
    out = []
    while position < stop - tol:
        right = min(stop, float(nodes[index + 1]))
        length = right - position
        if length > tol:
            out.append((index, length))
        position = right
        if position >= stop - tol:
            break
        index += 1
        if index >= len(nodes) - 1:
            raise ValueError("coarse H(curl) segmentation exhausted before fine edge ended")
    if not out:
        index, _ = _locate_interval(nodes, 0.5 * (start + stop))
        out.append((index, stop - start))
    return out


def build_hcurl_prolongation(coarse_axes, fine_background):
    """Return sparse ``P`` mapping coarse edge line integrals to fine ones.

    C-backed ``array`` buffers avoid the very large temporary Python-object
    overhead that ordinary triplet lists would create for the ~254k-edge
    validation transfer.
    """
    coarse_axes = _axes_tuple(coarse_axes)
    fine_axes = (fine_background.x, fine_background.y, fine_background.z)
    for coarse, fine in zip(coarse_axes, fine_axes):
        scale = max(1.0, abs(float(coarse[0])), abs(float(coarse[-1])))
        tol = 64.0 * np.finfo(float).eps * scale
        if abs(float(coarse[0]) - float(fine[0])) > tol or abs(float(coarse[-1]) - float(fine[-1])) > tol:
            raise ValueError("coarse/fine H(curl) grids must share the same physical box")

    coarse_shape, coarse_offsets, coarse_edges = _edge_layout(coarse_axes)
    rows = array("q")
    cols = array("q")
    data = array("d")

    def append(row, col, value):
        if abs(value) > 0.0:
            rows.append(int(row)); cols.append(int(col)); data.append(float(value))

    for fine_edge, (axis, i, j, k) in enumerate(fine_background.edge_tuples):
        if axis == 0:
            segments = _longitudinal_segments(coarse_axes[0], fine_background.x[i], fine_background.x[i + 1])
            jt, eta = _locate_interval(coarse_axes[1], fine_background.y[j])
            kt, zeta = _locate_interval(coarse_axes[2], fine_background.z[k])
            transverse = ((jt, 1.0 - eta), (jt + 1, eta)), ((kt, 1.0 - zeta), (kt + 1, zeta))
            for ic, overlap in segments:
                longitudinal_weight = overlap / (coarse_axes[0][ic + 1] - coarse_axes[0][ic])
                for jc, wy in transverse[0]:
                    for kc, wz in transverse[1]:
                        append(fine_edge, _edge_id(0, ic, jc, kc, coarse_shape, coarse_offsets), longitudinal_weight * wy * wz)
        elif axis == 1:
            segments = _longitudinal_segments(coarse_axes[1], fine_background.y[j], fine_background.y[j + 1])
            it, xi = _locate_interval(coarse_axes[0], fine_background.x[i])
            kt, zeta = _locate_interval(coarse_axes[2], fine_background.z[k])
            transverse = ((it, 1.0 - xi), (it + 1, xi)), ((kt, 1.0 - zeta), (kt + 1, zeta))
            for jc, overlap in segments:
                longitudinal_weight = overlap / (coarse_axes[1][jc + 1] - coarse_axes[1][jc])
                for ic, wx in transverse[0]:
                    for kc, wz in transverse[1]:
                        append(fine_edge, _edge_id(1, ic, jc, kc, coarse_shape, coarse_offsets), longitudinal_weight * wx * wz)
        else:
            segments = _longitudinal_segments(coarse_axes[2], fine_background.z[k], fine_background.z[k + 1])
            it, xi = _locate_interval(coarse_axes[0], fine_background.x[i])
            jt, eta = _locate_interval(coarse_axes[1], fine_background.y[j])
            transverse = ((it, 1.0 - xi), (it + 1, xi)), ((jt, 1.0 - eta), (jt + 1, eta))
            for kc, overlap in segments:
                longitudinal_weight = overlap / (coarse_axes[2][kc + 1] - coarse_axes[2][kc])
                for ic, wx in transverse[0]:
                    for jc, wy in transverse[1]:
                        append(fine_edge, _edge_id(2, ic, jc, kc, coarse_shape, coarse_offsets), longitudinal_weight * wx * wy)

    row_values = np.frombuffer(rows, dtype=np.int64)
    col_values = np.frombuffer(cols, dtype=np.int64)
    data_values = np.frombuffer(data, dtype=np.float64)
    P = sp.csr_matrix(
        (data_values, (row_values, col_values)),
        shape=(fine_background.n_edges, coarse_edges),
    )
    P.sum_duplicates()
    P.eliminate_zeros()
    row_nnz = np.diff(P.indptr)
    if P.shape[0] == 0 or np.any(row_nnz == 0):
        raise RuntimeError("H(curl) prolongation contains an empty fine-edge row")
    return P


def build_nodal_prolongation(coarse_axes, fine_background):
    """Trilinear nodal interpolation, used to certify the commuting diagram."""
    coarse_axes = _axes_tuple(coarse_axes)
    cx, cy, cz = coarse_axes
    nx, ny, nz = len(cx) - 1, len(cy) - 1, len(cz) - 1
    coarse_nodes = (nx + 1) * (ny + 1) * (nz + 1)

    def node_id(i, j, k):
        return (int(i) * (ny + 1) + int(j)) * (nz + 1) + int(k)

    rows, cols, data = [], [], []
    fine_node = 0
    for x in fine_background.x:
        ic, xi = _locate_interval(cx, x)
        for y in fine_background.y:
            jc, eta = _locate_interval(cy, y)
            for z in fine_background.z:
                kc, zeta = _locate_interval(cz, z)
                for di, wx in ((0, 1.0 - xi), (1, xi)):
                    for dj, wy in ((0, 1.0 - eta), (1, eta)):
                        for dk, wz in ((0, 1.0 - zeta), (1, zeta)):
                            weight = wx * wy * wz
                            if abs(weight) > 0.0:
                                rows.append(fine_node); cols.append(node_id(ic + di, jc + dj, kc + dk)); data.append(weight)
                fine_node += 1
    return sp.csr_matrix(
        (np.asarray(data, float), (np.asarray(rows, int), np.asarray(cols, int))),
        shape=(fine_node, coarse_nodes),
    )


__all__ = ["build_hcurl_prolongation", "build_nodal_prolongation"]
