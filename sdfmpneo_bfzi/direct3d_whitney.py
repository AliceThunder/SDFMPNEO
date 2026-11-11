"""Elementwise original Whitney H(curl) B and exact tetrahedral magnetic energy.

Lowest-order Nedelec B=curl(A_h) is constant in each affine tetrahedron.
No nodal projection, smoothing or VTU point interpolation occurs here.
"""
import numpy as np
from .direct3d_aphi import MU0, _EDGES, _topology


def curl_b(mesh, coeffs):
    """Recover raw curl from globally oriented edge circulations.

    coeffs: (ports, global_edges) or (global_edges,), as returned by
    AphiResult.edge_coefficients. Output is always (ports, tetra, xyz).
    """
    a = np.asarray(coeffs)
    edges, loc, sign, *_ = _topology(mesh)
    if a.ndim == 1:
        a = a[None, :]
    if (a.ndim != 2 or a.shape[1] != len(edges)
            or not np.isfinite(a).all()):
        raise ValueError('Expected finite Whitney edge circulations [port,edge]')
    xyz = mesh.nodes[mesh.tetrahedra]
    mat = (xyz[:, 1:] - xyz[:, 0, None, :]).transpose(0, 2, 1)
    if np.any(abs(np.linalg.det(mat)) < 1e-16):
        raise ValueError('Degenerate tetrahedron')
    grad = np.empty((len(xyz), 4, 3))
    grad[:, 1:] = np.linalg.inv(mat)
    grad[:, 0] = -grad[:, 1:].sum(axis=1)
    rot = 2*np.stack([np.cross(grad[:, i], grad[:, j])
                      for i, j in _EDGES], axis=1)
    rot *= sign[:, :, None]
    return np.einsum('pte,tec->ptc', a[:, loc], rot, optimize=True)


def magnetic_energy(mesh, b, *, phase_rad=0., average=False):
    """Integrate raw per-tetra B magnetic energy (J) for uniform mu0.

    b: complex peak-phasor field (tetra,3) or (ports,tetra,3).
    For average=False, use instantaneous real B at the given phase.
    For average=True, use full-period mean |B_peak|^2/(4*mu0).
    """
    b = np.asarray(b)
    if b.ndim == 2:
        b = b[None, :]
    if b.ndim != 3 or b.shape[1:] != (len(mesh.tetrahedra), 3) or not np.isfinite(b).all():
        raise ValueError('Expected B [port,tetra,xyz]')
    xyz = mesh.nodes[mesh.tetrahedra]
    mat = (xyz[:, 1:] - xyz[:, 0, None, :]).transpose(0, 2, 1)
    vol = abs(np.linalg.det(mat))/6
    if average:
        sq = 0.5*np.sum(abs(b)**2, axis=-1)
    else:
        sq = np.sum(np.real(b*np.exp(1j*phase_rad))**2, axis=-1)
    return np.einsum('t,pt->p', vol, sq)/(2*MU0)
