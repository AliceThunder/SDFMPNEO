"""Continuous UWPT geometry controls on a common material mesh topology."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from .geometry_chart import AffineTetrahedralGeometryChart
from .uwpt_geometry import RigidPose


def uwpt_geometry_chart(tagged, transmitter, receiver, tags, parameters):
    """Prescribe solid/interface motion and harmonically extend into materials.

    Planar scaling changes radius, pitch and conductor width together. Thickness,
    receiver translation, package size and the seawater radius are independent.
    Shape/turn count/connectivity remain fixed. Units of translations/radius: m.
    These are exact controls of this mesh chart, not independent CAD remeshing.
    """
    mesh, X = tagged.mesh, tagged.mesh.vertices
    copper, interfaces, centers, rotations = {}, {}, {}, {}
    water_nodes = np.unique(mesh.tetrahedra[tagged.tetra_mask(tags['seawater'])])
    for side, settings in [('tx', transmitter), ('rx', receiver)]:
        copper[side] = np.unique(mesh.tetrahedra[tagged.tetra_mask(tags[side+'_copper'])])
        package_nodes = np.unique(mesh.tetrahedra[tagged.tetra_mask(tags[side+'_package'])])
        interfaces[side] = np.intersect1d(package_nodes, water_nodes)
        centers[side] = np.asarray(settings['translation'], float)
        rotations[side] = RigidPose(centers[side], *settings['angles']).rotation
        if not len(copper[side]) or not len(interfaces[side]):
            raise ValueError('UWPT family requires tagged copper and package/water interfaces')
    outer = tagged.boundary_nodes(tags['outer_boundary'])
    # Infer the actual meshed sphere, including CAD chord-tolerance padding.
    sphere = X[outer]
    fit = np.linalg.lstsq(np.column_stack([2*sphere, np.ones(len(sphere))]),
                          np.sum(sphere*sphere, axis=1), rcond=None)[0]
    center = fit[:3]
    radius = float(np.sqrt(fit[3]+center@center))
    reference = {f'{s}_{kind}_scale': 1. for s in ('tx', 'rx')
                 for kind in ('planar', 'thickness', 'package')}
    reference.update(rx_offset_x=centers['rx'][0]-centers['tx'][0],
                     rx_offset_y=centers['rx'][1]-centers['tx'][1],
                     rx_gap=centers['rx'][2]-centers['tx'][2], seawater_radius=radius)
    if not parameters or set(parameters)-set(reference):
        raise ValueError(f'geometry parameters must be a nonempty subset of {sorted(reference)}')
    fixed = np.unique(np.concatenate([outer, *copper.values(), *interfaces.values()]))
    free = np.setdiff1d(np.arange(mesh.n_nodes), fixed)
    edge = mesh.edge_vertices
    w = 1./np.linalg.norm(X[edge[:, 1]]-X[edge[:, 0]], axis=1)**2
    i, j = edge.T
    L = sp.coo_matrix((np.concatenate([w,w,-w,-w]),
                      (np.concatenate([i,j,i,j]), np.concatenate([i,j,j,i]))),
                     shape=(mesh.n_nodes, mesh.n_nodes)).tocsr()
    directions = []
    for name in parameters:
        D = np.zeros_like(X)
        if name == 'seawater_radius':
            D[outer] = (X[outer]-center)/radius
        elif name in ('rx_offset_x', 'rx_offset_y', 'rx_gap'):
            axis = ('rx_offset_x', 'rx_offset_y', 'rx_gap').index(name)
            D[np.union1d(copper['rx'], interfaces['rx']), axis] = 1.
        else:
            side, kind, _ = name.split('_')
            nodes = interfaces[side] if kind == 'package' else copper[side]
            local = (X[nodes]-centers[side]) @ rotations[side]
            if kind == 'planar':
                local[:, 2] = 0.
            elif kind == 'thickness':
                local[:, :2] = 0.
            D[nodes] = local @ rotations[side].T
        if len(free):
            D[free] = spsolve(L[free][:,free], -L[free][:,fixed]@D[fixed])
        directions.append(D)
    chart = AffineTetrahedralGeometryChart(X, mesh.tetrahedra, np.array(directions), tuple(parameters))
    return chart, np.array([reference[n] for n in parameters])
