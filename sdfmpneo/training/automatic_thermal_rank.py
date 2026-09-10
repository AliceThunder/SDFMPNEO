"""Automatic thermal-rank selection and restart-state box construction."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import scipy.sparse.linalg as spla

_CACHE_FORMAT_VERSION = 1
_AUTO_KEYS = {
    "mode",
    "relative_tolerance",
    "absolute_tolerance",
    "initial_coordinate_bound",
    "probe_start_rank",
    "source_bound_safety_factor",
    "restart_state_safety_factor",
    "boundary_fraction",
    "temperature_probe_axes",
    "cache",
    "cache_file",
}


def _unique_rows(rows):
    result, seen = [], set()
    for row in rows:
        value = np.asarray(row, dtype=float).reshape(-1)
        key = tuple(float(v) for v in value)
        if key not in seen:
            seen.add(key)
            result.append(value)
    return tuple(result)


def _operating_anchors(lower, upper):
    lo = np.asarray(lower, dtype=float).reshape(-1)
    hi = np.asarray(upper, dtype=float).reshape(-1)
    if lo.shape != hi.shape or np.any(~np.isfinite(lo + hi)) or np.any(lo > hi):
        raise ValueError("invalid operating box for automatic thermal-rank probing")
    center = 0.5 * (lo + hi)
    rows = [center, lo, hi]
    for i in range(lo.size):
        left, right = center.copy(), center.copy()
        left[i], right[i] = lo[i], hi[i]
        rows.extend((left, right))
    return _unique_rows(rows)


def _thermal_state_anchors(rank, coordinate_bound, max_axes=4):
    rank = int(rank)
    bound = float(coordinate_bound)
    if rank < 1 or not np.isfinite(bound) or bound < 0.0:
        raise ValueError("invalid automatic thermal-rank state probe settings")
    rows = [np.zeros(rank, dtype=float)]
    for i in range(min(rank, int(max_axes))):
        plus, minus = np.zeros(rank), np.zeros(rank)
        plus[i], minus[i] = bound, -bound
        rows.extend((plus, minus))
    return tuple(rows)


def select_rank_from_modal_response_envelope(
    steady_response_envelope,
    *,
    relative_tolerance,
    absolute_tolerance=0.0,
    safety_factor=1.0,
    boundary_fraction=0.25,
    unresolved=True,
):
    """Return the smallest resolved modal prefix satisfying the response-tail target."""
    envelope = np.asarray(steady_response_envelope, dtype=float).reshape(-1)
    if envelope.size < 1 or np.any(~np.isfinite(envelope)) or np.any(envelope < 0.0):
        raise ValueError("modal response envelope must be finite and non-negative")
    rtol, atol = float(relative_tolerance), float(absolute_tolerance)
    safety, fraction = float(safety_factor), float(boundary_fraction)
    if (
        not np.isfinite(rtol + atol + safety + fraction)
        or rtol < 0.0
        or atol < 0.0
        or safety < 1.0
        or not 0.0 < fraction <= 1.0
    ):
        raise ValueError("invalid automatic thermal-rank tolerances")
    total = float(np.linalg.norm(envelope))
    limit = atol + rtol * max(total, np.finfo(float).tiny)
    boundary_width = min(2, envelope.size)
    for rank in range(1, envelope.size + 1):
        tail = float(np.linalg.norm(envelope[rank:]))
        if safety * tail > limit:
            continue
        boundary = 0.0
        if unresolved and rank < envelope.size:
            start = max(rank, envelope.size - boundary_width)
            boundary = float(np.linalg.norm(envelope[start:]))
            if safety * boundary > fraction * max(limit, np.finfo(float).tiny):
                continue
        return rank, {
            "total_response_norm": total,
            "resolved_tail_norm": tail,
            "selection_limit": limit,
            "boundary_tail_norm": boundary,
        }
    return envelope.size, {
        "total_response_norm": total,
        "resolved_tail_norm": 0.0,
        "selection_limit": limit,
        "boundary_tail_norm": 0.0,
    }


def _parse_materials(tagged, config):
    from sdfmpneo.em import (
        ConductivityRegion,
        ConstantConductivity,
        ReciprocalLinearResistivity,
    )
    mesh = tagged.mesh
    nu = np.zeros(mesh.n_tetrahedra)
    capacity = np.zeros(mesh.n_tetrahedra)
    conductivity = np.zeros(mesh.n_tetrahedra)
    regions = []
    ambient = float(config.get("ambient_temperature", 293.15))
    covered = np.zeros(mesh.n_tetrahedra, bool)
    for tag, material in config["materials"].items():
        mask = tagged.tetra_mask(int(tag))
        covered |= mask
        sigma = float(material["electrical_conductivity"])
        alpha = float(material.get("resistivity_temperature_coefficient", 0.0))
        law = (
            ReciprocalLinearResistivity(
                sigma,
                alpha,
                float(material.get("reference_temperature", ambient)),
            )
            if alpha
            else ConstantConductivity(sigma)
        )
        regions.append(ConductivityRegion(material["name"], mask, law))
        nu[mask] = 1.0 / (
            4e-7 * np.pi * float(material.get("relative_permeability", 1.0))
        )
        capacity[mask] = float(material["volumetric_heat_capacity"])
        conductivity[mask] = float(material["thermal_conductivity"])
    if not np.all(covered):
        raise ValueError("material definitions must cover all tetrahedral physical tags")
    return ambient, nu, capacity, conductivity, tuple(regions)


def _rhs_map_for_ports(ports, config):
    from sdfmpneo.training import AffineOperatingRHSMap
    p = ports.n_ports
    offset = (
        np.zeros(p, complex)
        if config.get("current_offset") is None
        else np.asarray(config["current_offset"], complex)
    )
    matrix = (
        np.eye(p, dtype=complex)
        if config.get("current_matrix") is None
        else np.asarray(config["current_matrix"], complex)
    )
    if offset.shape != (p,) or matrix.ndim != 2 or matrix.shape[0] != p:
        raise ValueError("current offset/matrix must match port count")
    return AffineOperatingRHSMap(
        ports.coordinate_rhs @ offset,
        ports.coordinate_rhs @ matrix,
    )


def _probe_modal_response_envelope(
    tagged,
    config,
    *,
    rank,
    coordinate_bound,
    max_state_axes,
    monitor=None,
):
    from sdfmpneo.em import SolidTerminalPortSet
    from sdfmpneo.tetra_core import TetrahedralElectroThermalCore

    mesh = tagged.mesh
    ambient, nu, capacity, conductivity, regions = _parse_materials(tagged, config)
    core = TetrahedralElectroThermalCore.build_nonlinear(
        mesh,
        omega=2 * np.pi * float(config["frequency_hz"]),
        reluctivity_tetra=nu,
        conductivity_regions=regions,
        temperature_reference_nodal=np.full(mesh.n_nodes, ambient),
        constitutive_relative_error_budget=float(
            config.get("constitutive_relative_error", 1e-8)
        ),
        rho_cp_tetra=capacity,
        thermal_conductivity_tetra=conductivity,
        source_current=np.zeros(mesh.n_edges),
        thermal_rank=int(rank),
    )
    problem = core.electromagnetic_problem
    ports = SolidTerminalPortSet.build(
        tagged,
        problem,
        config["terminal_pairs"],
        names=config.get("port_names"),
    )
    rhs_map = _rhs_map_for_ports(ports, config)
    training = config["training"]
    operating = _operating_anchors(
        training["operating_lower"], training["operating_upper"]
    )
    if any(row.shape != (rhs_map.n_operating,) for row in operating):
        raise ValueError(
            "training operating bounds do not match current-map input dimension"
        )
    states = _thermal_state_anchors(
        core.thermal_model.rank, coordinate_bound, max_state_axes
    )
    envelope = np.zeros(core.thermal_model.rank)
    zero = states[0]
    if monitor is not None:
        monitor.phase("thermal_rank_selection")
    lambdas = np.asarray(core.thermal_model.lambdas, dtype=float)
    lu = spla.splu(problem.operator_sparse(zero).tocsc())
    for u in operating:
        if monitor is not None:
            monitor.checkpoint()
        x = lu.solve(rhs_map.evaluate(u))
        q = np.asarray(
            [
                np.real(
                    np.vdot(
                        x,
                        problem.loss_operator_sparse(j, zero) @ x,
                    )
                )
                for j in range(core.thermal_model.rank)
            ],
            dtype=float,
        )
        envelope = np.maximum(envelope, np.abs(q) / lambdas)

    hot_u = max(operating, key=lambda value: float(np.linalg.norm(value)))
    hot_rhs = rhs_map.evaluate(hot_u)
    for state in states[1:]:
        if monitor is not None:
            monitor.checkpoint()
        x = spla.spsolve(problem.operator_sparse(state).tocsc(), hot_rhs)
        q = np.asarray(
            [
                np.real(
                    np.vdot(
                        x,
                        problem.loss_operator_sparse(j, state) @ x,
                    )
                )
                for j in range(core.thermal_model.rank)
            ],
            dtype=float,
        )
        envelope = np.maximum(envelope, np.abs(q) / lambdas)
    return envelope, len(operating), len(states)


def _restart_bounds(envelope, rank, initial_bound, restart_safety):
    response = np.asarray(envelope[:rank], dtype=float)
    bounds = float(initial_bound) + float(restart_safety) * response
    return np.maximum(bounds, float(initial_bound))


def _mesh_file(config_path, config):
    mesh_path = Path(config["mesh"]).expanduser()
    if not mesh_path.is_absolute():
        mesh_path = Path(config_path).parent / mesh_path
    return mesh_path.resolve(strict=False)


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _thermal_rank_cache_path(config_path, truncation):
    configured = truncation.get("cache_file")
    if configured:
        result = Path(configured).expanduser()
        if not result.is_absolute():
            result = Path(config_path).parent / result
        return result
    path = Path(config_path)
    return path.with_name(path.stem + ".thermal_rank.cache.json")


def _thermal_rank_cache_key(config_path, config):
    """Hash exactly the inputs that affect the automatic rank/restart-box probe."""
    truncation = {
        key: value
        for key, value in dict(config.get("thermal_truncation") or {}).items()
        if key not in {"cache", "cache_file"}
    }
    training = dict(config["training"])
    payload = {
        "cache_format_version": _CACHE_FORMAT_VERSION,
        "mesh_sha256": _file_sha256(_mesh_file(config_path, config)),
        "frequency_hz": config["frequency_hz"],
        "ambient_temperature": config.get("ambient_temperature", 293.15),
        "constitutive_relative_error": config.get("constitutive_relative_error", 1e-8),
        "materials": config["materials"],
        "terminal_pairs": config["terminal_pairs"],
        "port_names": config.get("port_names"),
        "current_offset": config.get("current_offset"),
        "current_matrix": config.get("current_matrix"),
        "thermal_truncation": truncation,
        "operating_lower": training["operating_lower"],
        "operating_upper": training["operating_upper"],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_thermal_rank_cache(path, cache_key):
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if (
            payload.get("format_version") != _CACHE_FORMAT_VERSION
            or payload.get("cache_key") != cache_key
        ):
            return None
        rank = int(payload["selected_rank"])
        report = dict(payload["report"])
        if rank < 1 or int(report.get("selected_rank", rank)) != rank:
            return None
    except (OSError, ValueError, TypeError, KeyError):
        return None
    report.update(
        cache_hit=True,
        cache_key=cache_key,
        cache_path=str(Path(path)),
    )
    return rank, report


def _save_thermal_rank_cache(path, cache_key, rank, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stored_report = dict(report)
    stored_report.update(
        cache_hit=False,
        cache_key=cache_key,
        cache_path=str(path),
    )
    payload = {
        "format_version": _CACHE_FORMAT_VERSION,
        "cache_key": cache_key,
        "selected_rank": int(rank),
        "report": stored_report,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def automatic_thermal_rank(config_path, config, *, monitor=None):
    """Expand the physical probe until a resolved tail is found or the full spectrum is reached."""
    from sdfmpneo.spatial import read_gmsh_v22_ascii

    path = Path(config_path)
    truncation = dict(config.get("thermal_truncation") or {})
    cache_enabled = bool(truncation.get("cache", True))
    cache_path = _thermal_rank_cache_path(path, truncation)
    cache_key = None
    if cache_enabled:
        cache_key = _thermal_rank_cache_key(path, config)
        cached = _load_thermal_rank_cache(cache_path, cache_key)
        if cached is not None:
            if monitor is not None:
                monitor.phase("thermal_rank_cache_hit")
            return cached

    tagged = read_gmsh_v22_ascii(_mesh_file(path, config))
    mesh = tagged.mesh
    _, _, capacity, conductivity, _ = _parse_materials(tagged, config)
    assembly = mesh.assemble_p1_thermal(
        rho_cp_tetra=capacity,
        conductivity_tetra=conductivity,
        homogeneous_dirichlet_boundary=True,
    )
    n_free = int(assembly.M.shape[0])
    if n_free < 1:
        raise ValueError("thermal boundary treatment produced no free degrees of freedom")

    relative = float(truncation.get("relative_tolerance", 1e-3))
    absolute = float(truncation.get("absolute_tolerance", 0.0))
    coordinate_bound = float(truncation.get("initial_coordinate_bound", 0.1))
    restart_safety = float(truncation.get("restart_state_safety_factor", 1.5))
    start = max(2, int(truncation.get("probe_start_rank", 4)))
    safety = float(truncation.get("source_bound_safety_factor", 2.0))
    boundary_fraction = float(truncation.get("boundary_fraction", 0.25))
    state_axes = int(truncation.get("temperature_probe_axes", 4))
    if not np.isfinite(restart_safety) or restart_safety < 1.0:
        raise ValueError("restart_state_safety_factor must be finite and at least one")

    def finish(rank, report):
        result = dict(report)
        result.update(
            cache_hit=False,
            cache_key=cache_key,
            cache_path=str(cache_path) if cache_enabled else None,
        )
        if cache_enabled:
            _save_thermal_rank_cache(cache_path, cache_key, rank, result)
        return int(rank), result

    if n_free == 1:
        bounds = [coordinate_bound]
        return finish(1, {
            "method": "automatic_physics_envelope",
            "selected_rank": 1,
            "full_dimension": 1,
            "probe_rank": 1,
            "certified_continuous_domain": False,
            "recommended_restart_coordinate_bound": bounds,
            "scope": "single free thermal degree of freedom; no truncation",
        })

    probe = min(n_free, start)
    while True:
        envelope, n_operating, n_states = _probe_modal_response_envelope(
            tagged,
            config,
            rank=probe,
            coordinate_bound=coordinate_bound,
            max_state_axes=state_axes,
            monitor=monitor,
        )
        rank, details = select_rank_from_modal_response_envelope(
            envelope,
            relative_tolerance=relative,
            absolute_tolerance=absolute,
            safety_factor=safety,
            boundary_fraction=boundary_fraction,
            unresolved=probe < n_free,
        )
        restart_bounds = _restart_bounds(
            envelope, rank, coordinate_bound, restart_safety
        )
        report = {
            "method": "automatic_physics_envelope",
            "selected_rank": int(rank),
            "full_dimension": n_free,
            "probe_rank": probe,
            "relative_tolerance": relative,
            "absolute_tolerance": absolute,
            "source_bound_safety_factor": safety,
            "restart_state_safety_factor": restart_safety,
            "initial_coordinate_bound": coordinate_bound,
            "operating_anchor_count": n_operating,
            "temperature_anchor_count": n_states,
            "maximum_modal_steady_response": envelope.tolist(),
            "recommended_restart_coordinate_bound": restart_bounds.tolist(),
            "certified_continuous_domain": False,
            "scope": (
                "deterministic full-order EM equilibrium anchors on the reference geometry; "
                "restart bounds include initial-state allowance plus a safety-scaled response envelope"
            ),
            **details,
        }
        if rank < probe:
            return finish(rank, report)
        if probe >= n_free:
            report.update(
                selected_rank=n_free,
                fallback="full_discrete_thermal_space_only_after_full_spectrum_probe",
            )
            report["recommended_restart_coordinate_bound"] = _restart_bounds(
                envelope, n_free, coordinate_bound, restart_safety
            ).tolist()
            return finish(n_free, report)
        probe = min(n_free, max(probe + 1, 2 * probe))


def resolve_training_bounds(training, rank, truncation, rank_report=None):
    """Resolve the state box used both for first-segment inputs and rollout restarts."""
    training = dict(training)
    lower, upper = training.get("initial_lower"), training.get("initial_upper")
    automatic = (
        lower is None
        or upper is None
        or (len(lower) == 0 and len(upper) == 0)
    )
    if automatic:
        bounds = None
        if rank_report is not None:
            values = rank_report.get("recommended_restart_coordinate_bound")
            if values is not None:
                values = np.asarray(values, dtype=float).reshape(-1)
                if values.size >= int(rank):
                    bounds = values[: int(rank)]
        if bounds is None:
            bound = float(truncation.get("initial_coordinate_bound", 0.1))
            if not np.isfinite(bound) or bound < 0.0:
                raise ValueError(
                    "initial_coordinate_bound must be finite and non-negative"
                )
            bounds = np.full(int(rank), bound, dtype=float)
        if np.any(~np.isfinite(bounds)) or np.any(bounds < 0.0):
            raise ValueError("automatic restart-state bounds are invalid")
        training["initial_lower"] = (-bounds).tolist()
        training["initial_upper"] = bounds.tolist()
    elif len(lower) != rank or len(upper) != rank:
        raise ValueError(
            f"thermal rank selected {rank} modes but explicit initial bounds have another dimension"
        )
    return training


def strict_truncation_kwargs(truncation):
    """Return only TetrahedralElectroThermalCore's strict certificate arguments."""
    data = dict(truncation or {})
    for key in _AUTO_KEYS:
        data.pop(key, None)
    return data


__all__ = [
    "automatic_thermal_rank",
    "resolve_training_bounds",
    "select_rank_from_modal_response_envelope",
    "strict_truncation_kwargs",
]
