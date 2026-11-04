"""P0 source-only reference for the exact BFZI continuous port path.

Straight spiral sides and the complete polygonal return use analytic finite
segment Biot--Savart and vector-potential integrals. Only the *curved* BFZI
corners use Gauss quadrature (order is independent of the production source).
There is no seawater R/K/L integration, artificial Green softening, or FEM.
"""
from __future__ import annotations

import math
import numpy as np

from .scene import BFZIAssembly
from .source import LineSource, MU0


def straight_segment_field(start, stop, positions):
    """Exact C=A/mu0 and B from an oriented, finite straight line.

    C(x)=(1/4pi) int_a^b dl/|x-r|; B=mu0 curl C.
    Evaluation must be outside the ideal zero-radius conductor.
    """
    a = np.asarray(start, dtype=float)
    b = np.asarray(stop, dtype=float)
    x = np.asarray(positions, dtype=float)
    if (a.shape != (3,) or b.shape != (3,)
            or x.ndim != 2 or x.shape[1] != 3
            or not np.isfinite(a).all() or not np.isfinite(b).all()
            or not np.isfinite(x).all()):
        raise ValueError("expected finite segment endpoints and (n,3) queries")
    delta = b-a
    length = float(np.linalg.norm(delta))
    if length <= 0.0:
        raise ValueError("straight source segment has zero length")
    first = x-a
    last = x-b
    r1 = np.linalg.norm(first, axis=1)
    r2 = np.linalg.norm(last, axis=1)
    distance_sum = r1+r2
    gap = distance_sum-length
    # The line-integrated potential is singular only on the actual segment.
    if np.any(gap <= 1e-13*length):
        raise ValueError("source query lies on or too near ideal wire segment")
    potential = (
        delta[None, :]/length
        * (np.log1p(2.0*length/gap)/(4.0*math.pi))[:, None]
    )
    # Stable finite-wire Biot--Savart form using both endpoint vectors:
    # B = mu0/(4pi) (r1 x r2) (|r1|+|r2|)
    #     / (|r1||r2| (|r1||r2| + r1.r2)).
    # Unlike subtracting two nearly equal cos(theta), this remains stable
    # for points on extensions of the segment.
    dot = np.einsum("nc,nc->n", first, last)
    denominator = r1*r2*(r1*r2+dot)
    if np.any(denominator <= 0.0):
        raise ValueError("source segment kernel is singular at query")
    magnetic = (
        MU0/(4.0*math.pi)
        * np.cross(first, last)
        * (distance_sum/denominator)[:, None]
    )
    return potential, magnetic


def continuous_port_field(
    assembly: BFZIAssembly, positions, *, corner_order: int = 20,
):
    """Independent straight-exact/curved-Gauss source field on the full port.

    The complete path is spiral OUTER->INNER then return INNER->OUTER.
    This is a higher-accuracy *source-only* comparison, not an exact
    full-wave finite-wire or coupled copper/seawater solution.
    """
    if (isinstance(corner_order, bool)
            or not isinstance(corner_order, int) or corner_order < 2):
        raise ValueError("corner_order must be an integer >= 2")
    x = np.asarray(positions, dtype=float)
    if x.ndim != 2 or x.shape[1] != 3 or not np.isfinite(x).all():
        raise ValueError("source reference queries must be finite (n,3)")
    c = np.zeros_like(x)
    b = np.zeros_like(x)
    coil = assembly.coil
    pose = assembly.pose
    q, w = np.polynomial.legendre.leggauss(corner_order)
    u = (math.pi/4.0)*(q+1.0)
    wu = (math.pi/4.0)*w
    corners = []
    directed = []
    for turn in range(coil.turns):
        for side in range(4):
            start, _ = coil.segment(turn, "side", side, -1.0, pose)
            stop, _ = coil.segment(turn, "side", side, 1.0, pose)
            # The circular BFZI limit has chi=0: its four "sides"
            # vanish geometrically, leaving only four curved arcs.
            if np.linalg.norm(stop-start) > 1e-14:
                cp, bp = straight_segment_field(start, stop, x)
                c += cp
                b += bp
            points, tangents = coil.segment(
                turn, "corner", side, u, pose,
            )
            corners.append(points)
            directed.append(tangents*wu[:, None])
    line = LineSource(np.vstack(corners), np.vstack(directed))
    c += line.potential_kernel(x)
    b += np.real(line.magnetic_field(x))
    outer, inner = coil.endpoints()
    route = pose.position(np.vstack((
        inner, np.asarray(assembly.return_waypoints), outer,
    )))
    for start, stop in zip(route[:-1], route[1:]):
        cp, bp = straight_segment_field(start, stop, x)
        c += cp
        b += bp
    return c, b


def polygonal_port_vertices(
    assembly: BFZIAssembly, *, panels_per_corner: int = 32,
):
    """Vertices of one EXACTLY CLOSED finite-segment current path.

    Every rounded BFZI corner is replaced by chords between points on
    its true analytic centerline. All genuine straight sides and the
    explicitly routed return are kept exact, preserving their direction.
    This is a *separate* geometric approximation, not Gauss point currents.
    """
    if (isinstance(panels_per_corner, bool)
            or not isinstance(panels_per_corner, int)
            or panels_per_corner < 2):
        raise ValueError("panels_per_corner must be an integer >= 2")
    coil, pose = assembly.coil, assembly.pose
    vertices = []
    knots = np.linspace(0.0, math.pi / 2.0, panels_per_corner + 1)[1:]
    for turn in range(coil.turns):
        for side in range(4):
            start, _ = coil.segment(turn, "side", side, -1.0, pose)
            stop, _ = coil.segment(turn, "side", side, 1.0, pose)
            if not vertices:
                vertices.append(start)
            elif np.linalg.norm(vertices[-1]-start) > 1e-11:
                raise ValueError("BFZI side/corner polyline is discontinuous")
            if np.linalg.norm(stop-vertices[-1]) > 1e-14:
                vertices.append(stop)
            corner_points, _ = coil.segment(
                turn, "corner", side, knots, pose,
            )
            vertices.extend(corner_points)
    outer, inner = coil.endpoints()
    route = pose.position(np.vstack((
        inner, np.asarray(assembly.return_waypoints), outer,
    )))
    if np.linalg.norm(vertices[-1]-route[0]) > 1e-11:
        raise ValueError("BFZI inner terminal does not join return")
    vertices.extend(route[1:])
    path = np.asarray(vertices, dtype=float)
    if np.linalg.norm(path[-1]-path[0]) > 1e-11:
        raise ValueError("BFZI port source polyline is not closed")
    if np.any(np.linalg.norm(np.diff(path,axis=0),axis=1) <= 1e-14):
        raise ValueError("zero-length physical port polyline segment")
    # Force bitwise closure, so exact segment-divergence terms telescope.
    path[-1] = path[0]
    return path


def polygonal_port_field(
    assembly: BFZIAssembly, positions, *, panels_per_corner: int = 32,
):
    """Closed, distributionally solenoidal polygonal current C/B/divC.

    The polygon approximates curved BFZI geometry with progressively
    shorter *physical chords*. Each chord is integrated analytically;
    unlike a Gauss point-line source, the whole polygon is an actually
    closed current distribution, with div C=0 off the source to rounding.
    Compare panel refinement against the independent continuous curve,
    NOT against another Gauss discretization of the same corner.
    """
    x = np.asarray(positions, dtype=float)
    if x.ndim != 2 or x.shape[1] != 3 or not np.isfinite(x).all():
        raise ValueError("polygonal source queries must be finite (n,3)")
    vertices = polygonal_port_vertices(
        assembly, panels_per_corner=panels_per_corner,
    )
    c = np.zeros_like(x)
    b = np.zeros_like(x)
    div_c = np.zeros(len(x))
    for start, stop in zip(vertices[:-1], vertices[1:]):
        cc, bb = straight_segment_field(start, stop, x)
        c += cc
        b += bb
        div_c += (
            1.0/np.linalg.norm(x-start, axis=1)
            -1.0/np.linalg.norm(x-stop, axis=1)
        )/(4.0*math.pi)
    return c, b, div_c
