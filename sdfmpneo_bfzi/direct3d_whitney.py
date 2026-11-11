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


def bdf2_joule(mesh, a_hist, phi, dt):
    """Recover transient E, q and loss at the ORIGINAL Whitney tetra quadrature.

    a_hist: (3,n_edge) global oriented A circulations at n-2,n-1,n.
    phi: scalar electric potential at n on all mesh vertices; only conducting
    tetrahedra contribute. Temporal gauge transforms must use the same BDF2
    derivative for phi. This is a real transient-field operator, not phasor.
    Returns (E[n_tet,4,3], q[n_tet,4], total_watts); insulation entries zero.
    """
    from .direct3d_aphi import _BARY4
    edges, loc, sign, *_ = _topology(mesh)
    if np.iscomplexobj(a_hist) or np.iscomplexobj(phi):
        raise ValueError("BDF2 transient E/J requires real fields")
    a = np.asarray(a_hist, dtype=float)
    p = np.asarray(phi, dtype=float)
    if (a.shape != (3, len(edges)) or p.shape != (len(mesh.nodes),)
            or not np.isfinite(a).all() or not np.isfinite(p).all()
            or not np.isfinite(dt) or dt <= 0):
        raise ValueError("Expected real finite Whitney history, nodal phi and positive dt")
    e = np.zeros((len(mesh.tetrahedra), 4, 3))
    q = np.zeros((len(mesh.tetrahedra), 4))
    wet = np.flatnonzero(mesh.sigma > 0)
    if not len(wet):
        return e, q, 0.
    tet = mesh.tetrahedra[wet]
    xyz = mesh.nodes[tet]
    mat = (xyz[:, 1:] - xyz[:, 0, None, :]).transpose(0, 2, 1)
    det = np.linalg.det(mat)
    if np.any(abs(det) < 1e-16):
        raise ValueError("Degenerate tetrahedron")
    grad = np.empty((len(wet), 4, 3))
    grad[:, 1:] = np.linalg.inv(mat)
    grad[:, 0] = -grad[:, 1:].sum(axis=1)
    bas = np.stack([
        _BARY4[None, :, i, None] * grad[:, None, j]
        - _BARY4[None, :, j, None] * grad[:, None, i]
        for i, j in _EDGES], axis=1)
    bas *= sign[wet, :, None, None]
    da = (3*a[2] - 4*a[1] + a[0])/(2*dt)
    grad_phi = np.einsum("ti,tic->tc", p[tet], grad)
    e[wet] = (-np.einsum("te,teqc->tqc", da[loc[wet]], bas,
                          optimize=True) - grad_phi[:, None, :])
    q[wet] = mesh.sigma[wet, None]*np.sum(e[wet]**2, axis=-1)
    watts = float(np.dot(abs(det)/6, q[wet].mean(axis=1)))
    return e, q, watts
