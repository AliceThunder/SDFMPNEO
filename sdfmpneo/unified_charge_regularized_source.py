"""Compatible finite-volume regularization of open-terminal charge continuity.

The stranded edge source already carries the intended physical path current. Its
terminal divergence, however, must not be left to the incidental cloud-in-cell
edge stencil: as the mesh is refined that stencil can represent the same finite
contact by a sharper and sharper nodal charge cloud, which makes the
pure-longitudinal self energy mesh dependent.

This module therefore declares the terminal charge independently from the cubic
contact profile. ``q_target`` is the nodal Galerkin load of ``-dI/ds`` over the
same fixed rectangular conductor cross section and the same fixed physical
contact length. The edge source is changed by the minimum Euclidean compatible
gradient lift

    S <- S + G psi,
    G.T G psi = q_target - G.T S.

Because ``C G = 0`` exactly, the correction has identically zero discrete curl.
This is the precise invariant certified here; in heterogeneous material it is
not claimed that the resulting Maxwell field's transverse component is pointwise
unchanged. The lift is not a divergence-free projection: the final source keeps
the non-zero open-terminal divergence ``q_target`` by construction.
"""
from __future__ import annotations

import time

import numpy as np
import scipy.sparse.linalg as spla

from .unified_gradient_block_maxwell import (
    gradient_operator,
    node_coordinates,
    source_terminal_divergence,
)
from .unified_terminal_contact_source import (
    _cross_section_quadrature,
    _smoothstep01,
    terminal_contact_length,
)


_CHARGE_MODEL = "volume_integrated_cubic_terminal_charge"
_LIFT_MODEL = "compatible_curl_free_gradient_charge_lift"
_SOURCE_MODEL = (
    "stranded_rectangular_cross_section_composite_gauss3_terminal_contact_"
    "compatible_charge_lift"
)
_TERMINAL_MODEL = "distributed_terminal_contact_with_compatible_charge_balance"
_LINE_GAUSS_ORDER = 3
_MAX_TARGET_ERROR = 5e-11
_MAX_LIFT_CURL = 1e-12


def _smoothstep_prime01(value):
    u = float(value)
    if not 0.0 < u < 1.0:
        return 0.0
    return 6.0 * u * (1.0 - u)


def _node_id(background, i, j, k):
    return (int(i) * (background.ny + 1) + int(j)) * (background.nz + 1) + int(k)


def _node_stencil(background, point):
    stencils = [
        background._linear_stencil(background.x, point[0]),
        background._linear_stencil(background.y, point[1]),
        background._linear_stencil(background.z, point[2]),
    ]
    out = []
    for (i, wi) in stencils[0]:
        for (j, wj) in stencils[1]:
            for (k, wk) in stencils[2]:
                out.append((_node_id(background, i, j, k), float(wi * wj * wk)))
    return out


def terminal_charge_target(background, coil):
    """Integrate the physical terminal ``-dI/ds`` onto nodal basis functions."""
    points = np.asarray(background._physical_centerline(coil), float)
    background._require_inside(points, "terminal charge centerline")
    delta = np.diff(points, axis=0)
    lengths = np.linalg.norm(delta, axis=1)
    if np.any(lengths <= np.finfo(float).tiny):
        raise ValueError("terminal charge path contains a zero-length segment")
    total_length = float(np.sum(lengths))
    contact = float(terminal_contact_length(coil, total_length))
    starts = np.concatenate(([0.0], np.cumsum(lengths[:-1])))

    u_offsets, u_weights, v_offsets, v_weights, up, vp, resolution = (
        _cross_section_quadrature(background, coil)
    )
    line_nodes, line_weights = np.polynomial.legendre.leggauss(_LINE_GAUSS_ORDER)
    n_nodes = (background.nx + 1) * (background.ny + 1) * (background.nz + 1)
    q = np.zeros(n_nodes, float)

    for segment, (p0, p1) in enumerate(zip(points[:-1], points[1:])):
        d = np.asarray(p1 - p0, float)
        length = float(lengths[segment])
        tangent = d / length
        width_axis, thickness_axis = background._cross_section_frame(coil, tangent)
        s0 = float(starts[segment])
        for line_node, line_weight in zip(line_nodes, line_weights):
            tau = 0.5 * (float(line_node) + 1.0)
            s = s0 + tau * length
            feed = float(_smoothstep01(s / contact))
            ret = float(_smoothstep01((total_length - s) / contact))
            dfeed = _smoothstep_prime01(s / contact) / contact
            dret_ds = -_smoothstep_prime01((total_length - s) / contact) / contact
            current_derivative = dfeed * ret + feed * dret_ds
            charge_density_1d = -current_derivative
            center = p0 + tau * d
            ds_weight = 0.5 * length * float(line_weight)
            if abs(charge_density_1d * ds_weight) <= np.finfo(float).tiny:
                continue
            for u, wu in zip(u_offsets, u_weights):
                for v, wv in zip(v_offsets, v_weights):
                    point = center + float(u) * width_axis + float(v) * thickness_axis
                    contribution = (
                        charge_density_1d * ds_weight * float(wu) * float(wv)
                    )
                    for node, weight in _node_stencil(background, point):
                        q[node] += contribution * weight

    negative = float(-np.sum(q[q < 0.0]))
    positive = float(np.sum(q[q > 0.0]))
    if negative <= np.finfo(float).tiny or positive <= np.finfo(float).tiny:
        raise RuntimeError("terminal charge target did not contain balanced feed/return support")
    # Enforce exactly one unit of distributed feed and return current while
    # retaining the quadrature-resolved shape of each terminal cloud.
    q[q < 0.0] /= negative
    q[q > 0.0] /= positive
    net = float(np.sum(q))
    if abs(net) > 5e-14:
        raise FloatingPointError(f"terminal charge target is not balanced: sum={net:.3e}")

    coords = node_coordinates(background)
    moment = np.asarray(coords.T @ q, float).reshape(3)
    absq = np.abs(q)
    total_abs = float(np.sum(absq))
    centroid = np.asarray((coords.T @ absq) / max(total_abs, np.finfo(float).tiny), float)
    centered = coords - centroid[None, :]
    second = np.asarray(
        (centered.T * absq) @ centered / max(total_abs, np.finfo(float).tiny),
        float,
    )
    return q, {
        "terminal_charge_model": _CHARGE_MODEL,
        "terminal_charge_contact_length": contact,
        "terminal_charge_line_gauss_order": int(_LINE_GAUSS_ORDER),
        "terminal_charge_cross_section_width_panels": int(up),
        "terminal_charge_cross_section_thickness_panels": int(vp),
        "terminal_charge_quadrature_resolution": float(resolution),
        "terminal_charge_negative_total": float(-np.sum(q[q < 0.0])),
        "terminal_charge_positive_total": float(np.sum(q[q > 0.0])),
        "terminal_charge_vector": moment.tolist(),
        "terminal_charge_absolute_centroid": centroid.tolist(),
        "terminal_charge_absolute_second_moment": second.tolist(),
        "terminal_charge_support_nodes": int(np.count_nonzero(absq > 1e-15)),
    }


def _graph_gradient_factor(background):
    cached = getattr(background, "_sdfmpneo_charge_lift_factor", None)
    if cached is not None:
        return cached
    started = time.perf_counter()
    G = gradient_operator(background, gauge_fixed=True)
    laplacian = (G.T @ G).tocsc()
    laplacian.sum_duplicates()
    laplacian.eliminate_zeros()
    try:
        factor = spla.splu(
            laplacian,
            permc_spec="MMD_AT_PLUS_A",
            diag_pivot_thresh=0.0,
            options={"Equil": True},
        )
    except (RuntimeError, ValueError):
        factor = spla.splu(laplacian)
    cached = (G, factor, float(time.perf_counter() - started))
    background._sdfmpneo_charge_lift_factor = cached
    return cached


def compatible_charge_lift(background, source, target):
    """Replace source divergence by the target with an exactly curl-free lift."""
    raw = np.asarray(source, float).reshape(-1)
    q_target = np.asarray(target, float).reshape(-1)
    Gfull = gradient_operator(background, gauge_fixed=False)
    if raw.shape != (background.n_edges,) or q_target.shape != (Gfull.shape[1],):
        raise ValueError("charge-lift source/target dimension mismatch")
    q_raw = np.asarray(Gfull.T @ raw, float).reshape(-1)
    delta_q = q_target - q_raw
    target_scale = max(float(np.linalg.norm(q_target)), np.finfo(float).tiny)
    raw_target_error = float(np.linalg.norm(delta_q) / target_scale)
    if abs(float(np.sum(delta_q))) > 1e-10 * max(float(np.linalg.norm(delta_q, 1)), 1.0):
        raise FloatingPointError("charge-lift divergence correction is not globally balanced")

    G, factor, build_seconds = _graph_gradient_factor(background)
    psi = np.asarray(factor.solve(np.asarray(delta_q[1:], float)), float).reshape(-1)
    correction = np.asarray(G @ psi, float).reshape(-1)
    lifted = raw + correction
    q_new = np.asarray(Gfull.T @ lifted, float).reshape(-1)
    target_error = float(np.linalg.norm(q_new - q_target) / target_scale)
    curl = np.asarray(background.curl @ correction, float).reshape(-1)
    curl_error = float(
        np.linalg.norm(curl)
        / max(float(np.linalg.norm(correction)), np.finfo(float).tiny)
    )
    if target_error > _MAX_TARGET_ERROR:
        raise RuntimeError(
            "compatible terminal charge lift failed target divergence: "
            f"error={target_error:.3e}"
        )
    if curl_error > _MAX_LIFT_CURL:
        raise RuntimeError(
            "compatible terminal charge lift changed source curl: "
            f"relative={curl_error:.3e}"
        )
    return lifted, {
        "terminal_charge_lift_model": _LIFT_MODEL,
        "terminal_charge_raw_target_relative_error": raw_target_error,
        "terminal_charge_target_relative_error": target_error,
        "terminal_charge_lift_relative_curl": curl_error,
        "terminal_charge_lift_relative_norm": float(
            np.linalg.norm(correction)
            / max(float(np.linalg.norm(raw)), np.finfo(float).tiny)
        ),
        "terminal_charge_lift_factor_seconds": float(build_seconds),
    }


def install(background_cls):
    if bool(getattr(background_cls, "_compatible_charge_regularization_installed", False)):
        return background_cls

    original_spatial_context = background_cls._spatial_context

    def spatial_context(self, geometry):
        context = original_spatial_context(self, geometry)
        source = np.asarray(context.source_shape, float).copy()
        metadata = [dict(row) for row in getattr(context, "source_regularization", ())]
        if len(metadata) != source.shape[1]:
            metadata = [dict() for _ in range(source.shape[1])]
        for port, coil in enumerate(context.geometry.coils):
            q_target, target_meta = terminal_charge_target(self, coil)
            lifted, lift_meta = compatible_charge_lift(self, source[:, port], q_target)
            source[:, port] = lifted
            q, net_error, moment = source_terminal_divergence(self, lifted)
            declared = np.asarray(target_meta["terminal_charge_vector"], float)
            path_length = float(
                np.sum(np.linalg.norm(np.diff(self._physical_centerline(coil), axis=0), axis=1))
            )
            moment_error = float(
                np.linalg.norm(moment - declared)
                / max(path_length, np.finfo(float).tiny)
            )
            deposited_vector = self._deposited_path_integral(lifted)
            path_error = float(
                np.linalg.norm(deposited_vector - declared)
                / max(path_length, np.finfo(float).tiny)
            )
            item = metadata[port]
            item.update(target_meta)
            item.update(lift_meta)
            item.update(
                model=self.source_model,
                terminal_model=self.terminal_model,
                source_norm=float(np.linalg.norm(lifted)),
                regularized_source_vector=declared.tolist(),
                terminal_path_integral_relative_error=path_error,
                terminal_charge_moment_relative_error=moment_error,
                terminal_charge_net_balance_relative_error=float(net_error),
                terminal_charge_divergence_relative_norm=float(
                    np.linalg.norm(q)
                    / max(float(np.linalg.norm(lifted)), np.finfo(float).tiny)
                ),
                terminal_charge_support_mesh_independent=True,
                terminal_charge_curl_preserved=True,
            )
            metadata[port] = item
        context.source_shape = source
        context.source_regularization = tuple(metadata)
        return context

    background_cls._spatial_context = spatial_context
    background_cls.source_model = _SOURCE_MODEL
    background_cls.terminal_model = _TERMINAL_MODEL
    background_cls.terminal_charge_model = _CHARGE_MODEL
    background_cls.terminal_charge_lift_model = _LIFT_MODEL
    background_cls._compatible_charge_regularization_installed = True
    return background_cls


__all__ = [
    "compatible_charge_lift",
    "install",
    "terminal_charge_target",
]
