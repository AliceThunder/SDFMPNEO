"""Mesh-independent finite-support terminal contacts for open spiral sources.

The physical source is a uniform stranded current over the conductor's finite
rectangular cross section, with a smooth finite-length feed/return contact.  The
support dimensions are geometry parameters and never depend on the Maxwell
mesh.  Numerical quadrature, however, must resolve that fixed support as the
mesh is refined: a fixed 3x3 set of points would asymptotically become nine
filaments and would therefore re-introduce a mesh-dependent self singularity.

This module consequently uses composite three-point Gauss quadrature.  The
number of panels may increase with numerical resolution, while the integrated
physical rectangle, total ampere-turns, terminal contact length and path moment
remain unchanged.
"""
from __future__ import annotations

import math
import numpy as np


_CONTACT_WIDTH_MULTIPLIER = 3.0
_PROFILE = "cubic_smoothstep_distributed_terminal_contact"
_SOURCE_MODEL = "stranded_rectangular_cross_section_composite_gauss3_terminal_contact"
_TERMINAL_MODEL = "distributed_terminal_contact_with_charge_balance"
# Each composite panel is no wider than about half the finest local Cartesian
# cell.  This controls only integration accuracy; it does not alter support.
_QUADRATURE_PANEL_TO_MESH = 0.5


def _smoothstep01(value):
    u = np.clip(np.asarray(value, float), 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def terminal_contact_length(coil, total_length):
    """Physical feed/return support length, independent of Maxwell mesh size."""
    width = float(coil.conductor_width)
    thickness = float(coil.conductor_thickness)
    if width <= 0.0 or thickness <= 0.0:
        raise ValueError("terminal contact requires positive conductor dimensions")
    length = _CONTACT_WIDTH_MULTIPLIER * max(width, thickness)
    total = float(total_length)
    if total <= 0.0:
        raise ValueError("terminal contact requires a positive source path length")
    return float(min(length, 0.25 * total))


def terminal_segment_weights(coil, points):
    """Return midpoint current amplitudes and physical terminal support length."""
    p = np.asarray(points, float)
    if p.ndim != 2 or p.shape[1] != 3 or p.shape[0] < 2:
        raise ValueError("terminal contact needs a polyline with at least two points")
    delta = np.diff(p, axis=0)
    lengths = np.linalg.norm(delta, axis=1)
    if np.any(lengths <= np.finfo(float).tiny):
        raise ValueError("terminal contact polyline contains a zero-length segment")
    total = float(np.sum(lengths))
    contact = terminal_contact_length(coil, total)
    starts = np.concatenate(([0.0], np.cumsum(lengths[:-1])))
    mid = starts + 0.5 * lengths
    feed = _smoothstep01(mid / contact)
    ret = _smoothstep01((total - mid) / contact)
    weights = np.asarray(feed * ret, float)
    if np.any(weights < 0.0) or np.any(weights > 1.0 + 1e-14):
        raise FloatingPointError("terminal contact current profile left [0,1]")
    if float(np.max(weights)) < 1.0 - 1e-12:
        raise ValueError("terminal contact supports overlap; source has no unit-current interior")
    return weights, contact


def regularized_path_vector(coil, points):
    p = np.asarray(points, float)
    weights, contact = terminal_segment_weights(coil, p)
    delta = np.diff(p, axis=0)
    return np.asarray(np.sum(weights[:, None] * delta, axis=0), float), float(contact)


def _composite_gauss_1d(span, panel_width):
    """Normalized quadrature for a fixed physical interval of length ``span``."""
    length = float(span)
    target = float(panel_width)
    if length <= 0.0 or target <= 0.0:
        raise ValueError("composite source quadrature requires positive lengths")
    panels = max(1, int(math.ceil(length / target)))
    nodes, weights = np.polynomial.legendre.leggauss(3)
    offsets = []
    normalized = []
    width = length / panels
    for panel in range(panels):
        center = -0.5 * length + (panel + 0.5) * width
        for node, weight in zip(nodes, weights):
            offsets.append(center + 0.5 * width * float(node))
            normalized.append(0.5 * width * float(weight) / length)
    offsets = np.asarray(offsets, float)
    normalized = np.asarray(normalized, float)
    normalized /= np.sum(normalized)
    return offsets, normalized, panels


def _cross_section_quadrature(background, coil):
    resolution = float(min(np.min(background.dx), np.min(background.dy), np.min(background.dz)))
    target = max(_QUADRATURE_PANEL_TO_MESH * resolution, np.finfo(float).tiny)
    u, wu, up = _composite_gauss_1d(float(coil.conductor_width), target)
    v, wv, vp = _composite_gauss_1d(float(coil.conductor_thickness), target)
    return u, wu, v, wv, int(up), int(vp), resolution


def install(background_cls):
    if bool(getattr(background_cls, "_distributed_terminal_contact_installed", False)):
        return background_cls

    original_spatial_context = background_cls._spatial_context

    def deposit_stranded_coil(self, coil, points):
        """Deposit the fixed physical rectangular source with resolved quadrature."""
        p = np.asarray(points, float)
        self._require_inside(p, "coil centerline")
        segment_weights, _contact = terminal_segment_weights(coil, p)
        u_offsets, u_weights, v_offsets, v_weights, _up, _vp, _resolution = (
            _cross_section_quadrature(self, coil)
        )
        source = np.zeros(self.n_edges, float)
        heat = np.zeros(self.n_cells, float)

        for segment, (p0, p1) in enumerate(zip(p[:-1], p[1:])):
            d = np.asarray(p1 - p0, float)
            length = float(np.linalg.norm(d))
            if length <= np.finfo(float).tiny:
                continue
            current_weight = float(segment_weights[segment])
            tangent = d / length
            width_axis, thickness_axis = self._cross_section_frame(coil, tangent)
            center = 0.5 * (p0 + p1)
            for u, wu in zip(u_offsets, u_weights):
                for v, wv in zip(v_offsets, v_weights):
                    qweight = float(wu * wv)
                    point = center + float(u) * width_axis + float(v) * thickness_axis
                    # Material occupancy is the complete wire, whereas the source
                    # amplitude additionally carries the terminal-contact profile.
                    for cell, weight in self._cell_stencil(point):
                        heat[cell] += length * qweight * weight
                    for axis, component in enumerate(d):
                        if abs(component) <= np.finfo(float).tiny:
                            continue
                        stencil = self._edge_stencil(axis, point)
                        if not stencil:
                            continue
                        for edge, weight in stencil:
                            source[edge] += (
                                current_weight
                                * qweight
                                * component
                                * weight
                                / self.edge_lengths[edge]
                            )
        if heat.sum() <= 0.0 or np.linalg.norm(source) <= np.finfo(float).tiny:
            raise ValueError("finite-support terminal source produced a zero physical source")
        heat /= heat.sum()
        return source, heat

    def spatial_context(self, geometry):
        context = original_spatial_context(self, geometry)
        rows = list(getattr(context, "source_regularization", ()))
        updated = []
        for port, (coil, row) in enumerate(zip(context.geometry.coils, rows)):
            points = np.asarray(self._physical_centerline(coil), float)
            expected_vector, contact = regularized_path_vector(coil, points)
            deposited_vector = self._deposited_path_integral(context.source_shape[:, port])
            path_length = float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
            endpoint_vector = np.asarray(points[-1] - points[0], float)
            error = float(
                np.linalg.norm(deposited_vector - expected_vector)
                / max(path_length, np.finfo(float).tiny)
            )
            _u, _wu, _v, _wv, up, vp, resolution = _cross_section_quadrature(self, coil)
            item = dict(row)
            item.update(
                model=self.source_model,
                terminal_model=self.terminal_model,
                terminal_profile=_PROFILE,
                terminal_contact_length=float(contact),
                terminal_contact_width_multiplier=float(_CONTACT_WIDTH_MULTIPLIER),
                regularized_source_vector=expected_vector.tolist(),
                terminal_path_integral_relative_error=error,
                terminal_regularization_mesh_independent=True,
                cross_section_support_mesh_independent=True,
                cross_section_quadrature="composite_gauss3",
                cross_section_width_panels=int(up),
                cross_section_thickness_panels=int(vp),
                cross_section_quadrature_points=int(9 * up * vp),
                source_quadrature_resolution=float(resolution),
                endpoint_vector_change_relative=float(
                    np.linalg.norm(expected_vector - endpoint_vector)
                    / max(path_length, np.finfo(float).tiny)
                ),
            )
            updated.append(item)
        context.source_regularization = tuple(updated)
        return context

    background_cls._deposit_stranded_coil = deposit_stranded_coil
    background_cls._spatial_context = spatial_context
    background_cls.source_model = _SOURCE_MODEL
    background_cls.terminal_model = _TERMINAL_MODEL
    background_cls.terminal_contact_profile = _PROFILE
    background_cls.terminal_contact_width_multiplier = _CONTACT_WIDTH_MULTIPLIER
    background_cls.source_cross_section_quadrature = "composite_gauss3"
    background_cls._distributed_terminal_contact_installed = True
    return background_cls


__all__ = [
    "install",
    "regularized_path_vector",
    "terminal_contact_length",
    "terminal_segment_weights",
]
