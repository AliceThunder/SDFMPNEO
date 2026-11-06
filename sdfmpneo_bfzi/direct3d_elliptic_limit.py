"""Independent 3D A-phi low-induction E/Joule reference for elliptical cylinders.

The transverse electric scalar potential is NONZERO when the two axes
differ, so this checks the actual charge-conservation part of A-phi.
It is an independent Maxwell limiting solution, not an attempt to fit
any published paper table, and not a TEAM-7 or BFZI geometry.
"""
from __future__ import annotations

import numpy as np
from .direct3d_aphi import MU0, _BARY4, TetraMesh, solve_aphi


def elliptic_prism_mesh(a, b, height, sigma, *, sectors=24, radial=3, axial=3):
    """All-conductor 3D linear-tetra extrusion of an inscribed ellipse."""
    if not all(np.isfinite(x) and x > 0 for x in (a, b, height, sigma)):
        raise ValueError("Positive finite ellipse geometry and sigma required")
    if (any(type(v) is not int for v in (sectors, radial, axial))
            or sectors < 6 or radial < 1 or axial < 1):
        raise ValueError("Integer sectors>=6, radial>=1, axial>=1 required")
    cross = [[0., 0.]]
    for j in range(1, radial+1):
        for k in range(sectors):
            theta = 2*np.pi*k/sectors
            cross.append([a*j/radial*np.cos(theta),
                          b*j/radial*np.sin(theta)])
    xy = np.asarray(cross, float)
    nxy = len(xy)
    z = np.linspace(-height/2, height/2, axial+1)
    nodes = np.column_stack((
        np.tile(xy, (axial+1, 1)), np.repeat(z, nxy)))

    def ring(j, k):
        return 1+(j-1)*sectors+(k % sectors)

    triangles = [(0, ring(1,k), ring(1,k+1))
                 for k in range(sectors)]
    for j in range(1, radial):
        for k in range(sectors):
            aa, bb = ring(j,k), ring(j,k+1)
            cc, dd = ring(j+1,k), ring(j+1,k+1)
            triangles.extend(((aa,cc,dd), (aa,dd,bb)))
    cells = []
    for level in range(axial):
        lo, hi = level*nxy, (level+1)*nxy
        for tri in triangles:
            u, v, w = sorted(tri)
            cells.extend(((lo+u,lo+v,lo+w,hi+w),
                          (lo+u,lo+v,hi+v,hi+w),
                          (lo+u,hi+u,hi+v,hi+w)))
    return TetraMesh(nodes, np.asarray(cells, int),
                     np.full(len(cells), sigma))


def axial_uniform_potential(points, B_peak):
    x = np.asarray(points, float)
    return .5*B_peak*np.column_stack(
        (-x[:,1], x[:,0], np.zeros(len(x))))


def elliptic_low_induction_e(points, frequency, B_peak, a, b):
    """Analytic leading-order E, peak phasors exp(+i omega t)."""
    p = np.asarray(points, float)
    factor = 1j*2*np.pi*frequency*B_peak/(a*a+b*b)
    return factor*np.column_stack((
        a*a*p[:,1], -b*b*p[:,0], np.zeros(len(p))))


def elliptic_low_induction_phi(points, frequency, B_peak, a, b):
    """Nonzero scalar potential when a!=b; an arbitrary constant is free."""
    p = np.asarray(points, float)
    return (-1j*np.pi*frequency*B_peak*(a*a-b*b)/(a*a+b*b)
            *p[:,0]*p[:,1])


def elliptic_low_induction_power(a, b, height, sigma, frequency, B_peak):
    """Continuous ellipse cycle-average loss at omega*mu*sigma*a² << 1."""
    omega = 2*np.pi*frequency
    return (sigma*omega**2*abs(B_peak)**2*np.pi*height*a**3*b**3
            /(8*(a*a+b*b)))


def elliptic_3d_benchmark(*, a=.12, b=.07, height=.1, sigma=4.,
                          frequency=1000., B_peak=.001,
                          sectors=24, radial=3, axial=3):
    """Independently compare real 3D A-phi E/q with an analytic limit.

    Reaction A has zero tangential trace on the cylinder surface (there
    is no external air domain in this inexpensive limiting test).
    Only interpret as physical low-induction limit when
    alpha=omega*mu0*sigma*max(a,b)² is much smaller than 1.
    The polygonal ellipse is a separate geometric discretization error.
    """
    if not all(np.isfinite(v) and v > 0 for v in (frequency, B_peak)):
        raise ValueError("Frequency and B_peak must be finite and positive")
    mesh = elliptic_prism_mesh(
        a, b, height, sigma, sectors=sectors, radial=radial, axial=axial)

    def source(points):
        return axial_uniform_potential(points, B_peak)

    solution = solve_aphi(mesh, frequency, [source])
    corners = mesh.nodes[mesh.tetrahedra]
    quadrature = np.einsum("qi,tij->tqj", _BARY4, corners)
    analytic = elliptic_low_induction_e(
        quadrature.reshape(-1,3), frequency, B_peak, a, b)
    analytic = analytic.reshape(len(corners), 4, 3)
    computed = solution.electric_at_quadrature[0]
    bare_induction = -1j*(2*np.pi*frequency)*axial_uniform_potential(
        quadrature.reshape(-1,3), B_peak).reshape(len(corners),4,3)
    volume = np.abs(np.linalg.det(
        (corners[:,1:]-corners[:,0,None,:]).transpose(0,2,1)))/6
    weight = volume[:,None,None]
    error = np.sqrt(np.sum(weight*np.abs(computed-analytic)**2)/
                    np.sum(weight*np.abs(analytic)**2))
    no_phi_error = np.sqrt(
        np.sum(weight*np.abs(bare_induction-analytic)**2)/
        np.sum(weight*np.abs(analytic)**2))
    predicted = elliptic_low_induction_power(
        a, b, height, sigma, frequency, B_peak)
    actual = solution.volume_power(np.array([1.+0j]))
    polygon_field_integral = (
        .5*sigma*np.sum(volume[:,None,None]*np.abs(analytic)**2)/4)
    return {
        "frequency_hz": frequency,
        "magnetic_induction_parameter":
            float(2*np.pi*frequency*MU0*sigma*max(a,b)**2),
        "n_tetrahedra": len(corners),
        "relative_e_l2_error": float(error),
        "induction_only_no_phi_e_error":float(no_phi_error),
        "fem_joule_W": float(actual),
        "analytic_ellipse_joule_W": float(predicted),
        "analytic_field_integral_on_polygon_W":
            float(polygon_field_integral),
        "relative_joule_error": float(actual/predicted-1),
        "relative_polygon_geometry_integral_error":
            float(polygon_field_integral/predicted-1),
        "scope":"Low-induction noncircular ellipse; D2 field reference still OPEN",
    }
