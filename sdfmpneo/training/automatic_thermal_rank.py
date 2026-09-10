"""Automatic thermal-rank selection with one full diagnostic and layered caches."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import scipy.sparse.linalg as spla

_CACHE_FORMAT_VERSION = 2
_AUTO_KEYS = {
    "mode",
    "relative_tolerance",
    "absolute_tolerance",
    "initial_coordinate_bound",
    "probe_start_rank",  # ignored by the current one-shot diagnostic; stripped from strict kwargs
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
    unresolved=False,
):
    """Return the smallest modal prefix satisfying the response-tail target."""
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


def _json_hash(payload):
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _thermal_rank_cache_path(config_path, truncation):
    configured = truncation.get("cache_file")
    if configured:
        result = Path(configured).expanduser()
        if not result.is_absolute():
            result = Path(config_path).parent / result
        return result
    path = Path(config_path)
    return path.with_name(path.stem + ".thermal_rank.cache.json")


def _thermal_cache_paths(config_path, truncation):
    selection = _thermal_rank_cache_path(config_path, truncation)
    base = selection.with_suffix("") if selection.suffix else selection
    return {
        "selection": selection,
        "spectrum": Path(str(base) + ".spectrum.npz"),
        "envelope": Path(str(base) + ".envelope.npz"),
    }


def _thermal_material_signature(materials):
    return {
        str(tag): {
            "thermal_conductivity": material["thermal_conductivity"],
            "volumetric_heat_capacity": material["volumetric_heat_capacity"],
        }
        for tag, material in materials.items()
    }


def _thermal_spectrum_cache_key(config_path, config):
    return _json_hash(
        {
            "format_version": _CACHE_FORMAT_VERSION,
            "kind": "thermal_spectrum",
            "mesh_sha256": _file_sha256(_mesh_file(config_path, config)),
            "materials": _thermal_material_signature(config["materials"]),
            "homogeneous_dirichlet_boundary": True,
        }
    )


def _thermal_envelope_cache_key(config_path, config, spectrum_key=None):
    if spectrum_key is None:
        spectrum_key = _thermal_spectrum_cache_key(config_path, config)
    truncation = dict(config.get("thermal_truncation") or {})
    training = dict(config["training"])
    return _json_hash(
        {
            "format_version": _CACHE_FORMAT_VERSION,
            "kind": "thermal_response_envelope",
            "spectrum_key": spectrum_key,
            "frequency_hz": config["frequency_hz"],
            "ambient_temperature": config.get("ambient_temperature", 293.15),
            "constitutive_relative_error": config.get(
                "constitutive_relative_error", 1e-8
            ),
            "materials": config["materials"],
            "terminal_pairs": config["terminal_pairs"],
            "port_names": config.get("port_names"),
            "current_offset": config.get("current_offset"),
            "current_matrix": config.get("current_matrix"),
            "operating_lower": training["operating_lower"],
            "operating_upper": training["operating_upper"],
            "initial_coordinate_bound": truncation.get(
                "initial_coordinate_bound", 0.1
            ),
            "temperature_probe_axes": truncation.get("temperature_probe_axes", 4),
        }
    )


def _thermal_selection_cache_key(config_path, config, envelope_key=None):
    if envelope_key is None:
        envelope_key = _thermal_envelope_cache_key(config_path, config)
    truncation = dict(config.get("thermal_truncation") or {})
    return _json_hash(
        {
            "format_version": _CACHE_FORMAT_VERSION,
            "kind": "thermal_rank_selection",
            "envelope_key": envelope_key,
            "relative_tolerance": truncation.get("relative_tolerance", 1e-3),
            "absolute_tolerance": truncation.get("absolute_tolerance", 0.0),
            "source_bound_safety_factor": truncation.get(
                "source_bound_safety_factor", 2.0
            ),
            "restart_state_safety_factor": truncation.get(
                "restart_state_safety_factor", 1.5
            ),
            "boundary_fraction": truncation.get("boundary_fraction", 0.25),
        }
    )


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as output:
        np.savez_compressed(output, **arrays)
    temporary.replace(path)


def _load_selection_cache(path, cache_key):
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
        selection_cache_hit=True,
        cache_key=cache_key,
        cache_path=str(Path(path)),
    )
    return rank, report


def _save_selection_cache(path, cache_key, rank, report):
    stored_report = dict(report)
    stored_report.update(
        cache_hit=False,
        selection_cache_hit=False,
        cache_key=cache_key,
        cache_path=str(Path(path)),
    )
    _atomic_json(
        path,
        {
            "format_version": _CACHE_FORMAT_VERSION,
            "cache_key": cache_key,
            "selected_rank": int(rank),
            "report": stored_report,
        },
    )


def _load_spectrum_cache(path, cache_key, M, K):
    from sdfmpneo.thermal import ThermalSpectralModel

    try:
        with np.load(path, allow_pickle=False) as data:
            version = int(np.asarray(data["format_version"]).reshape(()))
            stored_key = str(np.asarray(data["cache_key"]).reshape(()))
            if version != _CACHE_FORMAT_VERSION or stored_key != cache_key:
                return None
            phi = np.asarray(data["Phi"], dtype=float)
            lambdas = np.asarray(data["lambdas"], dtype=float)
        n = int(M.shape[0])
        if (
            K.shape != (n, n)
            or phi.shape != (n, n)
            or lambdas.shape != (n,)
            or np.any(~np.isfinite(phi))
            or np.any(~np.isfinite(lambdas))
            or np.any(lambdas <= 0.0)
        ):
            return None
    except (OSError, ValueError, TypeError, KeyError):
        return None
    return ThermalSpectralModel(M=M, K=K, Phi=phi, lambdas=lambdas)


def _save_spectrum_cache(path, cache_key, spectrum):
    _atomic_npz(
        path,
        format_version=np.array(_CACHE_FORMAT_VERSION, dtype=np.int64),
        cache_key=np.array(cache_key),
        Phi=np.asarray(spectrum.Phi, dtype=float),
        lambdas=np.asarray(spectrum.lambdas, dtype=float),
    )


def _load_envelope_cache(path, cache_key):
    try:
        with np.load(path, allow_pickle=False) as data:
            version = int(np.asarray(data["format_version"]).reshape(()))
            stored_key = str(np.asarray(data["cache_key"]).reshape(()))
            if version != _CACHE_FORMAT_VERSION or stored_key != cache_key:
                return None
            envelope = np.asarray(data["envelope"], dtype=float)
            n_operating = int(np.asarray(data["operating_anchor_count"]).reshape(()))
            n_states = int(np.asarray(data["temperature_anchor_count"]).reshape(()))
        if (
            envelope.ndim != 1
            or envelope.size < 1
            or np.any(~np.isfinite(envelope))
            or np.any(envelope < 0.0)
            or n_operating < 1
            or n_states < 1
        ):
            return None
    except (OSError, ValueError, TypeError, KeyError):
        return None
    return envelope, n_operating, n_states


def _save_envelope_cache(path, cache_key, envelope, n_operating, n_states):
    _atomic_npz(
        path,
        format_version=np.array(_CACHE_FORMAT_VERSION, dtype=np.int64),
        cache_key=np.array(cache_key),
        envelope=np.asarray(envelope, dtype=float),
        operating_anchor_count=np.array(int(n_operating), dtype=np.int64),
        temperature_anchor_count=np.array(int(n_states), dtype=np.int64),
    )


def _thermal_assembly(tagged, config):
    _, _, capacity, conductivity, _ = _parse_materials(tagged, config)
    assembly = tagged.mesh.assemble_p1_thermal(
        rho_cp_tetra=capacity,
        conductivity_tetra=conductivity,
        homogeneous_dirichlet_boundary=True,
    )
    if int(assembly.M.shape[0]) < 1:
        raise ValueError("thermal boundary treatment produced no free degrees of freedom")
    return assembly


def _full_thermal_spectrum(
    tagged, config, *, cache_path, cache_key, cache_enabled
):
    from sdfmpneo.thermal import ThermalSpectralModel

    assembly = _thermal_assembly(tagged, config)
    if cache_enabled:
        cached = _load_spectrum_cache(
            cache_path, cache_key, assembly.M, assembly.K
        )
        if cached is not None:
            return assembly, cached, True

    spectrum = ThermalSpectralModel.build(assembly.M, assembly.K)
    if cache_enabled:
        _save_spectrum_cache(cache_path, cache_key, spectrum)
    return assembly, spectrum, False


def _local_modes_from_spectrum(tagged, assembly, spectrum):
    if assembly.M.shape[0] != spectrum.full_dimension:
        raise ValueError("cached thermal spectrum dimension does not match current mesh")
    return np.asarray(
        [
            assembly.expand_free(spectrum.Phi[:, k])[tagged.mesh.tetrahedra]
            for k in range(spectrum.rank)
        ],
        dtype=float,
    )


def _probe_full_modal_response_envelope(
    tagged,
    config,
    assembly,
    spectrum,
    *,
    coordinate_bound,
    max_state_axes,
    monitor=None,
):
    """Compute the complete modal response envelope once."""
    from sdfmpneo.em import NonlinearTetrahedralApsiProblem, SolidTerminalPortSet

    mesh = tagged.mesh
    ambient, nu, _, _, regions = _parse_materials(tagged, config)
    local_modes = _local_modes_from_spectrum(tagged, assembly, spectrum)
    problem = NonlinearTetrahedralApsiProblem(
        mesh,
        omega=2 * np.pi * float(config["frequency_hz"]),
        reluctivity_tetra=nu,
        source_current=np.zeros(mesh.n_edges),
        temperature_reference_local=np.full(mesh.n_nodes, ambient)[mesh.tetrahedra],
        thermal_modes_local=local_modes,
        conductivity_regions=regions,
        constitutive_relative_error_budget=float(
            config.get("constitutive_relative_error", 1e-8)
        ),
    )
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
        spectrum.rank, coordinate_bound, max_state_axes
    )
    envelope = np.zeros(spectrum.rank)
    lambdas = np.asarray(spectrum.lambdas, dtype=float)

    if monitor is not None:
        monitor.phase("thermal_rank_selection")

    zero = states[0]
    lu = spla.splu(problem.operator_sparse(zero).tocsc())
    rhs_matrix = np.column_stack([rhs_map.evaluate(u) for u in operating])
    X = np.asarray(lu.solve(rhs_matrix), dtype=complex)

    # At a fixed state H_j does not depend on the operating anchor. The former
    # implementation rebuilt H_j once per anchor; evaluate all zero-state
    # operating anchors with one H_j assembly instead.
    for j in range(spectrum.rank):
        if monitor is not None:
            monitor.checkpoint()
        H = problem.loss_operator_sparse(j, zero)
        HX = H @ X
        q = np.real(np.sum(np.conj(X) * HX, axis=0))
        envelope[j] = float(np.max(np.abs(q))) / lambdas[j]

    hot_u = max(operating, key=lambda value: float(np.linalg.norm(value)))
    hot_rhs = rhs_map.evaluate(hot_u)
    for state in states[1:]:
        if monitor is not None:
            monitor.checkpoint()
        x = spla.spsolve(problem.operator_sparse(state).tocsc(), hot_rhs)
        for j in range(spectrum.rank):
            H = problem.loss_operator_sparse(j, state)
            q = float(np.real(np.vdot(x, H @ x)))
            envelope[j] = max(envelope[j], abs(q) / lambdas[j])

    return envelope, len(operating), len(states)


def _restart_bounds(envelope, rank, initial_bound, restart_safety):
    response = np.asarray(envelope[:rank], dtype=float)
    bounds = float(initial_bound) + float(restart_safety) * response
    return np.maximum(bounds, float(initial_bound))


def automatic_thermal_rank(config_path, config, *, monitor=None, tagged=None):
    """Select rank from one complete physical envelope and reuse layered caches."""
    from sdfmpneo.spatial import read_gmsh_v22_ascii

    path = Path(config_path)
    truncation = dict(config.get("thermal_truncation") or {})
    cache_enabled = bool(truncation.get("cache", True))
    paths = _thermal_cache_paths(path, truncation)

    spectrum_key = _thermal_spectrum_cache_key(path, config)
    envelope_key = _thermal_envelope_cache_key(path, config, spectrum_key)
    selection_key = _thermal_selection_cache_key(path, config, envelope_key)

    if cache_enabled:
        cached = _load_selection_cache(paths["selection"], selection_key)
        if cached is not None:
            if monitor is not None:
                monitor.phase("thermal_rank_cache_hit")
            return cached

    relative = float(truncation.get("relative_tolerance", 1e-3))
    absolute = float(truncation.get("absolute_tolerance", 0.0))
    coordinate_bound = float(truncation.get("initial_coordinate_bound", 0.1))
    restart_safety = float(truncation.get("restart_state_safety_factor", 1.5))
    safety = float(truncation.get("source_bound_safety_factor", 2.0))
    boundary_fraction = float(truncation.get("boundary_fraction", 0.25))
    state_axes = int(truncation.get("temperature_probe_axes", 4))
    if not np.isfinite(restart_safety) or restart_safety < 1.0:
        raise ValueError(
            "restart_state_safety_factor must be finite and at least one"
        )

    envelope_cached = (
        _load_envelope_cache(paths["envelope"], envelope_key)
        if cache_enabled
        else None
    )
    spectrum_cache_hit = False
    if envelope_cached is not None:
        envelope, n_operating, n_states = envelope_cached
        envelope_cache_hit = True
        if monitor is not None:
            monitor.phase("thermal_rank_selection")
    else:
        envelope_cache_hit = False
        if tagged is None:
            tagged = read_gmsh_v22_ascii(_mesh_file(path, config))
        assembly, spectrum, spectrum_cache_hit = _full_thermal_spectrum(
            tagged,
            config,
            cache_path=paths["spectrum"],
            cache_key=spectrum_key,
            cache_enabled=cache_enabled,
        )
        envelope, n_operating, n_states = _probe_full_modal_response_envelope(
            tagged,
            config,
            assembly,
            spectrum,
            coordinate_bound=coordinate_bound,
            max_state_axes=state_axes,
            monitor=monitor,
        )
        if cache_enabled:
            _save_envelope_cache(
                paths["envelope"],
                envelope_key,
                envelope,
                n_operating,
                n_states,
            )

    rank, details = select_rank_from_modal_response_envelope(
        envelope,
        relative_tolerance=relative,
        absolute_tolerance=absolute,
        safety_factor=safety,
        boundary_fraction=boundary_fraction,
        unresolved=False,
    )
    restart_bounds = _restart_bounds(
        envelope, rank, coordinate_bound, restart_safety
    )
    report = {
        "method": "automatic_full_spectrum_physics_envelope",
        "selected_rank": int(rank),
        "full_dimension": int(envelope.size),
        "diagnostic_rank": int(envelope.size),
        "relative_tolerance": relative,
        "absolute_tolerance": absolute,
        "source_bound_safety_factor": safety,
        "restart_state_safety_factor": restart_safety,
        "initial_coordinate_bound": coordinate_bound,
        "operating_anchor_count": int(n_operating),
        "temperature_anchor_count": int(n_states),
        "maximum_modal_steady_response": np.asarray(envelope, float).tolist(),
        "recommended_restart_coordinate_bound": restart_bounds.tolist(),
        "certified_continuous_domain": False,
        "scope": (
            "one complete deterministic full-order EM equilibrium envelope on "
            "the reference geometry; the full thermal spectrum is diagnostic "
            "and only the selected prefix is retained by the surrogate"
        ),
        "cache_hit": False,
        "selection_cache_hit": False,
        "envelope_cache_hit": bool(envelope_cache_hit),
        "spectrum_cache_hit": bool(spectrum_cache_hit),
        "selection_cache_path": str(paths["selection"]) if cache_enabled else None,
        "envelope_cache_path": str(paths["envelope"]) if cache_enabled else None,
        "spectrum_cache_path": str(paths["spectrum"]) if cache_enabled else None,
        **details,
    }
    if rank == envelope.size:
        report["fallback"] = (
            "full_discrete_thermal_space_after_complete_envelope_scan"
        )
    if cache_enabled:
        _save_selection_cache(paths["selection"], selection_key, rank, report)
    return int(rank), report


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
