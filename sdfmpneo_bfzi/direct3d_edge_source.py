"""Consistent 3-D Nedelec source loading and closed-port measurement.

For any oriented Whitney-edge field a, one and the same exact matrix L
satisfies (L@a)_p = sum_strands turns * integral_Gamma_p A_h.dl.
Use L.T for coil forcing and L for port voltage. This is the discrete
Galerkin duality missing from the former volume-imposed source approach.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from .direct3d_aphi import _topology, _EDGES


def closed_edge_load(mesh, source_strands):
    """Exact Nedelec edge loads (n_ports,n_edges) from real closed filaments.

    Chords are clipped against every intersected tetrahedron. On each
    straight subchord, Whitney tangential circulation is constant.
    A shared-face chord is integrated exactly once.
    """
    vertices = mesh.nodes[mesh.tetrahedra]
    centres = vertices.mean(axis=1)
    max_radius = float(np.linalg.norm(vertices-centres[:, None, :], axis=2).max())
    tree = cKDTree(centres)
    edges, local_edges, signs, *_ = _topology(mesh)
    grads = np.zeros((len(vertices), 4, 3))
    grads[:, 1:, :] = np.linalg.inv(
        (vertices[:, 1:]-vertices[:, 0, None, :]).transpose(0, 2, 1))
    grads[:, 0, :] = -grads[:, 1:, :].sum(axis=1)
    load = np.zeros((len(source_strands), len(edges)), dtype=float)
    for port, paths in enumerate(source_strands):
        if not paths:
            raise ValueError("Every port must contain a closed source strand")
        for weight, wire in paths:
            points = np.asarray(wire, float)
            if (points.ndim != 2 or points.shape[1] != 3
                    or len(points) < 4 or not np.isfinite(points).all()
                    or not np.array_equal(points[0], points[-1])
                    or not np.isreal(weight) or not np.isfinite(weight)
                    or np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) <= 0)):
                raise ValueError("Source must be a real-weight, exactly closed, nondegenerate path")
            for start, stop in zip(points[:-1], points[1:]):
                direction = stop-start
                ids = tree.query_ball_point(
                    (start+stop)/2,
                    float(np.linalg.norm(direction))/2+1.00001*max_radius+1e-12)
                ids = np.asarray(sorted(ids), dtype=int)
                if not len(ids):
                    raise ValueError("Source chord lies outside the tetrahedral domain")
                g = grads[ids]
                lambda123 = np.einsum(
                    "tij,tj->ti", g[:, 1:, :], start-vertices[ids, 0])
                lambdas = np.column_stack(
                    (1-lambda123.sum(axis=1), lambda123))
                slopes = np.einsum("tij,j->ti", g, direction)
                lower = np.zeros(len(ids))
                upper = np.ones(len(ids))
                valid = np.ones(len(ids), bool)
                for k in range(4):
                    slope = slopes[:, k]
                    flat = abs(slope) < 1e-13
                    valid &= ~(flat & (lambdas[:, k] < -1e-10))
                    negative = slope < -1e-13
                    positive = slope > 1e-13
                    upper[negative] = np.minimum(
                        upper[negative], -lambdas[negative, k]/slope[negative])
                    lower[positive] = np.maximum(
                        lower[positive], -lambdas[positive, k]/slope[positive])
                included = valid & (upper-lower > 1e-11)
                cells = ids[included]
                low = lower[included]
                high = upper[included]
                if not len(cells):
                    raise ValueError("Source chord misses the reference domain")
                boundaries = np.sort(np.r_[0., 1., low, high])
                for a, b in zip(boundaries[:-1], boundaries[1:]):
                    if b-a <= 1e-11:
                        continue
                    mid = (a+b)/2
                    covering = np.flatnonzero(
                        (low <= mid+1e-11) & (high >= mid-1e-11))
                    if not len(covering):
                        raise ValueError("Source chord escapes the reference domain")
                    cell = int(cells[int(covering[0])])
                    gcell = grads[cell]
                    dlam = gcell@direction
                    local123 = gcell[1:]@(start+mid*direction-vertices[cell, 0])
                    l = np.r_[1-local123.sum(), local123]
                    tangents = np.array([
                        (l[i]*dlam[j]-l[j]*dlam[i])*signs[cell, e]
                        for e, (i, j) in enumerate(_EDGES)])
                    np.add.at(load[port], local_edges[cell],
                              float(weight)*(b-a)*tangents)
    return load


def elemental_current_edge_load(mesh, element_current_density):
    """Exact Whitney load from piecewise-constant impressed J_s [A/m^2].

    Input is [port, tetra, xyz] in *this mesh's element order*.  For each
    constant J_s in a tetra, the P1 Whitney edge integral is analytic:
      L[p,e] += V/4 * J_s[p,t] dot
                (grad(lambda_j)-grad(lambda_i)) * edge_orientation.
    Use with the SAME two separately solved source current densities in an
    independent Elmer reference; current continuity, 1-A normalization and
    mapping of Elmer element IDs to this mesh are separate physical gates.
    This is NOT a conversion of the existing ideal line source into a
    finite-volume coil, nor a numerical equivalence certificate.
    """
    currents=np.asarray(element_current_density,dtype=complex)
    if (currents.ndim!=3 or currents.shape[0]<1 or
            currents.shape[1:]!=(len(mesh.tetrahedra),3) or
            not np.isfinite(currents).all()):
        raise ValueError(
            "Expected finite impressed J [ports,tetrahedra,3] in A/m^2")
    edges,local_edges,signs,*_= _topology(mesh)
    vertices=mesh.nodes[mesh.tetrahedra]
    transforms=(vertices[:,1:]-vertices[:,0,None,:]).transpose(0,2,1)
    determinants=np.linalg.det(transforms)
    if np.any(abs(determinants)<1e-16):
        raise ValueError("Degenerate tetrahedron in volume source mesh")
    volume=abs(determinants)/6
    grads=np.zeros((len(vertices),4,3))
    grads[:,1:]=np.linalg.inv(transforms)
    grads[:,0]=-grads[:,1:].sum(axis=1)
    means=np.stack([
        (grads[:,j]-grads[:,i])/4
        for i,j in _EDGES],axis=1)*signs[:,:,None]
    local=np.einsum('ptc,tec,t->pte',
                    currents,means,volume,optimize=True)
    load=np.zeros((len(currents),len(edges)),dtype=complex)
    for port in range(len(currents)):
        np.add.at(load[port],local_edges.ravel(),local[port].ravel())
    return load
