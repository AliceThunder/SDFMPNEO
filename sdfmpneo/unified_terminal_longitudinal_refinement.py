"""Terminal-scale nested refinement for the global longitudinal scalar defect.

The global coarse scalar solution owns the nonlocal return path and supplies the
Dirichlet trace on the artificial patch boundary. Earlier near-field patches
refined the whole package at 3 mm / 2.25 mm. That is still much coarser than the
physical terminal charge support (sub-mm conductor thickness), so the scalar
self energy can remain strongly mesh dependent even when the patch reproduces
the parent equations to machine precision.

This adapter keeps the certified full-geometry patch and replaces only its
refined axes. Every coarse patch node is retained. Extra nodes are inserted only
inside small intervals containing the feed/return contact volumes plus a
physical conductor-scale halo. Per-axis resolution is derived from the physical
contact tangent, conductor width direction and conductor thickness direction;
therefore a rotated thin conductor cannot hide its sub-mm scale behind a much
larger axis-aligned contact bounding box. The requested 3 mm reference remains
an upper bound. Validation multiplies the local steps by the same requested
fine/validation ratio (normally 0.75).
"""
from __future__ import annotations

import math
import numpy as np

from . import unified_longitudinal_patch_consistency as _consistency
from .unified_terminal_contact_source import terminal_contact_length


_MODEL = "global_boundary_conditioned_longitudinal_terminal_refinement_v3"


def _as_geometry(module, geometry):
    cls = module.UnifiedUWPTGeometry
    return geometry if isinstance(geometry, cls) else cls.from_mapping(geometry)


def _contact_segments(background, coil):
    points = np.asarray(background._physical_centerline(coil), float)
    delta = np.diff(points, axis=0)
    lengths = np.linalg.norm(delta, axis=1)
    if points.ndim != 2 or points.shape[1] != 3 or np.any(lengths <= np.finfo(float).tiny):
        raise ValueError("terminal refinement requires a valid physical centerline")
    total = float(np.sum(lengths))
    contact = float(terminal_contact_length(coil, total))
    starts = np.concatenate(([0.0], np.cumsum(lengths[:-1])))
    terminal_intervals = ((0.0, contact), (total - contact, total))
    return points, delta, lengths, starts, terminal_intervals, contact


def _contact_boxes(module, background, geometry, port):
    """Return physical feed/return contact AABBs for one port."""
    g = _as_geometry(module, geometry)
    coil = g.coils[int(port)]
    points, delta, lengths, starts, terminal_intervals, contact = _contact_segments(
        background, coil
    )
    boxes = []
    for lower_s, upper_s in terminal_intervals:
        lo = np.full(3, np.inf)
        hi = np.full(3, -np.inf)
        for index, (p0, p1) in enumerate(zip(points[:-1], points[1:])):
            seg_lo = float(starts[index])
            seg_hi = seg_lo + float(lengths[index])
            a = max(seg_lo, float(lower_s))
            b = min(seg_hi, float(upper_s))
            if b <= a:
                continue
            d = np.asarray(delta[index], float)
            length = float(lengths[index])
            tangent = d / length
            width_axis, thickness_axis = background._cross_section_frame(coil, tangent)
            extent = 0.5 * (
                np.abs(np.asarray(width_axis, float)) * float(coil.conductor_width)
                + np.abs(np.asarray(thickness_axis, float)) * float(coil.conductor_thickness)
            )
            pa = np.asarray(p0, float) + ((a - seg_lo) / length) * d
            pb = np.asarray(p0, float) + ((b - seg_lo) / length) * d
            lo = np.minimum(lo, np.minimum(pa, pb) - extent)
            hi = np.maximum(hi, np.maximum(pa, pb) + extent)
        if np.any(~np.isfinite(lo)) or np.any(~np.isfinite(hi)) or np.any(hi <= lo):
            raise RuntimeError("failed to construct terminal contact support box")
        boxes.append((lo, hi))
    return tuple(boxes), contact, coil


def _terminal_options(background):
    root = dict(getattr(background, "background_config", {}) or {})
    own = dict(root.get("global_longitudinal_correction", {}) or {})
    cells = float(own.get("terminal_cells_per_support", 4.0))
    padding_factor = float(own.get("terminal_core_padding_factor", 1.5))
    max_cells = int(own.get("terminal_patch_max_cells", 575000))
    if not np.isfinite(cells) or cells < 1.25:
        raise ValueError("terminal_cells_per_support must be >= 1.25")
    if not np.isfinite(padding_factor) or padding_factor < 0.0:
        raise ValueError("terminal_core_padding_factor must be non-negative")
    if max_cells < 1000:
        raise ValueError("terminal_patch_max_cells is unreasonably small")
    return cells, padding_factor, max_cells


def _directional_axis_steps(background, coil, cells, requested_fine):
    """Resolve tangent/width/thickness directions rather than only their AABB."""
    points, delta, lengths, starts, terminal_intervals, contact = _contact_segments(
        background, coil
    )
    limits = np.full(3, float(requested_fine))
    dimensions = (
        ("tangent", float(contact)),
        ("width", float(coil.conductor_width)),
        ("thickness", float(coil.conductor_thickness)),
    )
    for lower_s, upper_s in terminal_intervals:
        for index, _pair in enumerate(zip(points[:-1], points[1:])):
            seg_lo = float(starts[index])
            seg_hi = seg_lo + float(lengths[index])
            if min(seg_hi, float(upper_s)) <= max(seg_lo, float(lower_s)):
                continue
            tangent = np.asarray(delta[index], float) / float(lengths[index])
            width_axis, thickness_axis = background._cross_section_frame(coil, tangent)
            directions = {
                "tangent": tangent,
                "width": np.asarray(width_axis, float),
                "thickness": np.asarray(thickness_axis, float),
            }
            for name, dimension in dimensions:
                direction = directions[name]
                for axis in range(3):
                    component = abs(float(direction[axis]))
                    if component <= 1e-10:
                        continue
                    candidate = dimension / (float(cells) * component)
                    limits[axis] = min(limits[axis], candidate)
    if np.any(~np.isfinite(limits)) or np.any(limits <= 0.0):
        raise FloatingPointError("terminal directional refinement produced invalid steps")
    return limits


def _base_axis_steps(module, background, geometry, port, requested_fine):
    boxes, contact, coil = _contact_boxes(module, background, geometry, port)
    cells, padding_factor, _max_cells = _terminal_options(background)
    steps = _directional_axis_steps(background, coil, cells, requested_fine)
    halo = padding_factor * max(float(coil.conductor_width), float(coil.conductor_thickness))
    intervals = []
    for axis in range(3):
        intervals.append(
            tuple((float(lo[axis] - halo), float(hi[axis] + halo)) for lo, hi in boxes)
        )
    return np.asarray(steps, float), tuple(intervals), boxes, float(contact)


def _merge_intervals(intervals, lo, hi):
    clipped = []
    for a, b in intervals:
        a = max(float(lo), float(a))
        b = min(float(hi), float(b))
        if b > a:
            clipped.append((a, b))
    clipped.sort()
    merged = []
    for a, b in clipped:
        if not merged or a > merged[-1][1] + 1e-14 * max(abs(a), abs(b), 1.0):
            merged.append([a, b])
        else:
            merged[-1][1] = max(merged[-1][1], b)
    return tuple((float(a), float(b)) for a, b in merged)


def _subdivide_axis(coarse_axis, intervals, target_step):
    """Retain all coarse nodes and add fine nodes only inside terminal intervals."""
    coarse = np.asarray(coarse_axis, float)
    step = float(target_step)
    if coarse.ndim != 1 or coarse.size < 3 or np.any(np.diff(coarse) <= 0.0):
        raise ValueError("terminal refinement requires a monotone coarse axis")
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("terminal refinement step must be positive")
    active = _merge_intervals(intervals, coarse[0], coarse[-1])
    if not active:
        raise RuntimeError("terminal refinement support lies outside the scalar patch")
    out = [float(coarse[0])]
    eps = 64.0 * np.finfo(float).eps * max(float(np.max(np.abs(coarse))), 1.0)
    for a, b in zip(coarse[:-1], coarse[1:]):
        pieces = []
        for u, v in active:
            left = max(float(a), u)
            right = min(float(b), v)
            if right > left + eps:
                pieces.append((left, right))
        cursor = float(a)
        for left, right in pieces:
            if left > cursor + eps:
                if left > out[-1] + eps:
                    out.append(float(left))
            elif abs(left - cursor) > eps and left > out[-1] + eps:
                out.append(float(left))
            width = float(right - left)
            count = max(1, int(math.ceil(width / step)))
            local = np.linspace(left, right, count + 1)
            for value in local[1:]:
                if value > out[-1] + eps:
                    out.append(float(value))
            cursor = right
        if float(b) > out[-1] + eps:
            out.append(float(b))
        elif abs(float(b) - out[-1]) <= eps:
            out[-1] = float(b)
    refined = np.asarray(out, float)
    for value in coarse:
        if np.min(np.abs(refined - value)) > 128.0 * np.finfo(float).eps * max(abs(value), 1.0):
            raise AssertionError("terminal-refined axis lost a coarse parent node")
    if np.any(np.diff(refined) <= 0.0):
        raise FloatingPointError("terminal-refined axis is not strictly monotone")
    return refined


def _refined_axes_for_request(
    module,
    background,
    geometry,
    port,
    coarse_axes,
    requested_step,
    base_requested_step,
):
    base_steps, intervals, boxes, contact = _base_axis_steps(
        module, background, geometry, port, base_requested_step
    )
    ratio = float(requested_step) / max(float(base_requested_step), np.finfo(float).tiny)
    if not (0.0 < ratio <= 1.0 + 1e-12):
        raise ValueError("terminal refinement requested-step ratio must lie in (0,1]")
    axis_steps = base_steps * ratio
    axes = tuple(
        _subdivide_axis(coarse_axes[k], intervals[k], axis_steps[k])
        for k in range(3)
    )
    return axes, axis_steps, boxes, contact


def install(module):
    if bool(getattr(module, "_terminal_longitudinal_refinement_installed", False)):
        return module
    if not bool(getattr(module, "_patch_consistency_installed", False)):
        raise RuntimeError("terminal longitudinal refinement requires patch consistency first")

    def port_states(
        background,
        geometry,
        port,
        global_potential,
        target_steps,
        *,
        phi=None,
    ):
        base_cfg = module._config(background)
        tolerance = _consistency._consistency_tolerance(background)
        attempts = []
        for padding in _consistency._padding_candidates(module, background):
            cfg = dict(base_cfg)
            cfg["boundary_padding"] = float(padding)
            coarse_axes, _center, _aabb_half, full_geometry = (
                _consistency._full_geometry_patch_axes(
                    module, background, geometry, port, cfg
                )
            )
            coarse = _consistency._make_full_patch_background(
                module,
                background,
                coarse_axes,
                fine_step=module._background_step(background),
            )
            coarse_state = _consistency._full_patch_state(
                module,
                background,
                coarse,
                full_geometry,
                global_potential,
                source_port=int(port),
                phi=phi,
                fine_step=module._background_step(background),
                certify_parent=True,
            )
            consistency_value = float(
                coarse_state["parent_restriction_relative_residual"]
            )
            attempts.append(
                {
                    "boundary_padding": float(padding),
                    "parent_restriction_relative_residual": consistency_value,
                }
            )
            print(
                "boundary-conditioned longitudinal patch consistency: "
                f"port={int(port) + 1}, padding={padding:.6g}m, "
                f"residual={consistency_value:.3e}, "
                f"accepted={'yes' if consistency_value <= tolerance else 'no'}",
                flush=True,
            )
            if consistency_value > tolerance:
                continue

            coarse_state = dict(coarse_state)
            coarse_state["selected_boundary_padding"] = float(padding)
            coarse_state["boundary_selection_attempts"] = attempts.copy()
            coarse_state["terminal_nested_refinement"] = False
            states = {"coarse": coarse_state}
            requested_base = float(base_cfg["fine_step"])
            resolution_cells, padding_factor, max_cells = _terminal_options(background)
            for requested in target_steps:
                axes, axis_steps, boxes, contact = _refined_axes_for_request(
                    module,
                    background,
                    full_geometry,
                    int(port),
                    coarse_axes,
                    float(requested),
                    requested_base,
                )
                n_cells = int(np.prod([len(axis) - 1 for axis in axes], dtype=np.int64))
                print(
                    "boundary-conditioned terminal refinement: "
                    f"port={int(port) + 1}, requested={float(requested):.6g}m, "
                    f"axis_steps={[float(v) for v in axis_steps]}, cells={n_cells}, "
                    f"budget={max_cells}",
                    flush=True,
                )
                if n_cells > max_cells:
                    raise RuntimeError(
                        "terminal-scale longitudinal patch exceeds the certified cell budget: "
                        f"port={int(port) + 1}, cells={n_cells}, limit={max_cells}, "
                        f"axis_steps={axis_steps.tolist()}"
                    )
                patch = _consistency._make_full_patch_background(
                    module,
                    background,
                    axes,
                    fine_step=float(np.min(axis_steps)),
                )
                state = dict(
                    _consistency._full_patch_state(
                        module,
                        background,
                        patch,
                        full_geometry,
                        global_potential,
                        source_port=int(port),
                        phi=phi,
                        fine_step=float(np.min(axis_steps)),
                        certify_parent=False,
                    )
                )
                state["selected_boundary_padding"] = float(padding)
                state["terminal_nested_refinement"] = True
                state["requested_reference_step"] = float(requested)
                state["terminal_refinement_axis_steps"] = axis_steps.tolist()
                state["terminal_resolution_cells_per_support"] = float(resolution_cells)
                state["terminal_core_padding_factor"] = float(padding_factor)
                state["terminal_contact_length"] = float(contact)
                state["terminal_contact_boxes"] = [
                    {"lo": np.asarray(lo, float).tolist(), "hi": np.asarray(hi, float).tolist()}
                    for lo, hi in boxes
                ]
                state["terminal_patch_cell_budget"] = int(max_cells)
                states[float(requested)] = state
            return states

        raise RuntimeError(
            "no full-geometry boundary-conditioned longitudinal patch reproduces "
            "the parent scalar restriction; attempts=" + repr(attempts)
        )

    module._port_states = port_states
    module._MODEL = _MODEL

    original_resolve = module._resolve_settings

    def resolve_settings(settings, background):
        original_resolve(settings, background)
        own = settings["BACKGROUND"].setdefault("global_longitudinal_correction", {})
        own.setdefault("terminal_cells_per_support", 4.0)
        own.setdefault("terminal_core_padding_factor", 1.5)
        own.setdefault("terminal_patch_max_cells", 575000)
        if isinstance(getattr(background, "background_config", None), dict):
            target = background.background_config.setdefault(
                "global_longitudinal_correction", {}
            )
            target["terminal_cells_per_support"] = float(
                own["terminal_cells_per_support"]
            )
            target["terminal_core_padding_factor"] = float(
                own["terminal_core_padding_factor"]
            )
            target["terminal_patch_max_cells"] = int(
                own["terminal_patch_max_cells"]
            )

    module._resolve_settings = resolve_settings
    module._terminal_longitudinal_refinement_installed = True
    return module


__all__ = ["install"]
