"""Mesh-independent finite-support terminal contacts for open spiral sources.

The stranded source already regularizes the conductor cross section with physical
width/thickness quadrature.  An open path also needs a physical longitudinal
terminal model: abruptly truncating a unit path current leaves the terminal
charge support controlled by the Maxwell mesh.  This module replaces that hard
truncation by a distributed feed/contact region tied only to conductor geometry.

The impressed current amplitude rises smoothly from zero to one over the feed
contact, remains one through the interior spiral, and falls smoothly to zero over
the return contact.  Its divergence therefore integrates to the same balanced
unit feed/return current while the terminal charge is distributed over a fixed
physical length rather than a grid-dependent endpoint.
"""
from __future__ import annotations

import numpy as np


_CONTACT_WIDTH_MULTIPLIER = 3.0
_PROFILE = "cubic_smoothstep_distributed_terminal_contact"
_SOURCE_MODEL = "stranded_rectangular_cross_section_gauss3_terminal_contact"
_TERMINAL_MODEL = "distributed_terminal_contact_with_charge_balance"


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
    # Keep a non-contact interior even for unusually short paths.  This clipping
    # depends only on physical geometry, never on the discretization.
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


def install(background_cls):
    if bool(getattr(background_cls, "_distributed_terminal_contact_installed", False)):
        return background_cls

    original_spatial_context = background_cls._spatial_context

    def deposit_stranded_coil(self, coil, points):
        """Deposit a unit interior current with finite physical terminal contacts."""
        p = np.asarray(points, float)
        self._require_inside(p, "coil centerline")
        segment_weights, _contact = terminal_segment_weights(coil, p)
        nodes, weights = np.polynomial.legendre.leggauss(3)
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
            for u, wu in zip(nodes, weights):
                for v, wv in zip(nodes, weights):
                    qweight = float(wu * wv / 4.0)
                    point = (
                        center
                        + 0.5 * float(coil.conductor_width) * float(u) * width_axis
                        + 0.5 * float(coil.conductor_thickness) * float(v) * thickness_axis
                    )
                    # Thermal/material occupancy still represents the entire wire,
                    # not only the impressed terminal-current profile.
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
    background_cls._distributed_terminal_contact_installed = True
    return background_cls


__all__ = [
    "install",
    "regularized_path_vector",
    "terminal_contact_length",
    "terminal_segment_weights",
]
