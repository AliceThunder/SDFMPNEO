from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from tempfile import NamedTemporaryFile
import json

import numpy as np
import scipy.sparse.linalg as spla

_ORIGINAL_MODEL_FROM_CONFIG = None
_AUTO_KEYS = (
    "mode", "relative_tolerance", "absolute_tolerance", "initial_coordinate_bound",
    "probe_start_rank", "max_probe_rank", "source_bound_safety_factor",
    "boundary_fraction", "temperature_probe_axes",
)


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
    steady_response_envelope, *, relative_tolerance,
    absolute_tolerance=0.0, safety_factor=1.0,
    boundary_fraction=0.25, unresolved=True,
):
    """Select the smallest modal prefix with a negligible resolved response tail.

    Entry j estimates max |q_j|/lambda_j at deterministic physical anchors. A
    boundary-band test prevents stopping at a partial probe whose highest modes
    are still energetic. This is a convergence criterion, not a theorem-level
    continuous-domain certificate.
    """
    envelope = np.asarray(steady_response_envelope, dtype=float).reshape(-1)
    if envelope.size < 1 or np.any(~np.isfinite(envelope)) or np.any(envelope < 0.0):
        raise ValueError("modal response envelope must be finite and non-negative")
    rtol, atol = float(relative_tolerance), float(absolute_tolerance)
    safety, fraction = float(safety_factor), float(boundary_fraction)
    if (not np.isfinite(rtol + atol + safety + fraction) or rtol < 0.0
            or atol < 0.0 or safety < 1.0 or not 0.0 < fraction <= 1.0):
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
    from sdfmpneo.em import ConductivityRegion, ConstantConductivity, ReciprocalLinearResistivity

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
        law = (ReciprocalLinearResistivity(
            sigma, alpha, float(material.get("reference_temperature", ambient)))
            if alpha else ConstantConductivity(sigma))
        regions.append(ConductivityRegion(material["name"], mask, law))
        nu[mask] = 1.0 / (4e-7 * np.pi * float(material.get("relative_permeability", 1.0)))
        capacity[mask] = float(material["volumetric_heat_capacity"])
        conductivity[mask] = float(material["thermal_conductivity"])
    if not np.all(covered):
        raise ValueError("material definitions must cover all tetrahedral physical tags")
    return ambient, nu, capacity, conductivity, tuple(regions)


def _rhs_map_for_ports(ports, config):
    from sdfmpneo.training import AffineOperatingRHSMap

    p = ports.n_ports
    offset = config.get("current_offset")
    matrix = config.get("current_matrix")
    offset = np.zeros(p, complex) if offset is None else np.asarray(offset, complex)
    matrix = np.eye(p, dtype=complex) if matrix is None else np.asarray(matrix, complex)
    if offset.shape != (p,) or matrix.ndim != 2 or matrix.shape[0] != p:
        raise ValueError("current offset/matrix must match port count")
    return AffineOperatingRHSMap(ports.coordinate_rhs @ offset, ports.coordinate_rhs @ matrix)


def _probe_modal_response_envelope(
    tagged, config, *, rank, coordinate_bound, max_state_axes, monitor=None,
):
    from sdfmpneo.em import SolidTerminalPortSet
    from sdfmpneo.tetra_core import TetrahedralElectroThermalCore

    mesh = tagged.mesh
    ambient, nu, capacity, conductivity, regions = _parse_materials(tagged, config)
    core = TetrahedralElectroThermalCore.build_nonlinear(
        mesh, omega=2 * np.pi * float(config["frequency_hz"]),
        reluctivity_tetra=nu, conductivity_regions=regions,
        temperature_reference_nodal=np.full(mesh.n_nodes, ambient),
        constitutive_relative_error_budget=float(config.get("constitutive_relative_error", 1e-8)),
        rho_cp_tetra=capacity, thermal_conductivity_tetra=conductivity,
        source_current=np.zeros(mesh.n_edges), thermal_rank=int(rank))
    problem = core.electromagnetic_problem
    ports = SolidTerminalPortSet.build(
        tagged, problem, config["terminal_pairs"], names=config.get("port_names"))
    rhs_map = _rhs_map_for_ports(ports, config)
    training = config["training"]
    operating = _operating_anchors(training["operating_lower"], training["operating_upper"])
    if any(row.shape != (rhs_map.n_operating,) for row in operating):
        raise ValueError("training operating bounds do not match current-map input dimension")
    states = _thermal_state_anchors(core.thermal_model.rank, coordinate_bound, max_state_axes)
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
        q = np.asarray([
            np.real(np.vdot(x, problem.loss_operator_sparse(j, zero) @ x))
            for j in range(core.thermal_model.rank)], dtype=float)
        envelope = np.maximum(envelope, np.abs(q) / lambdas)
    hot_u = max(operating, key=lambda value: float(np.linalg.norm(value)))
    hot_rhs = rhs_map.evaluate(hot_u)
    for state in states[1:]:
        if monitor is not None:
            monitor.checkpoint()
        x = spla.spsolve(problem.operator_sparse(state).tocsc(), hot_rhs)
        q = np.asarray([
            np.real(np.vdot(x, problem.loss_operator_sparse(j, state) @ x))
            for j in range(core.thermal_model.rank)], dtype=float)
        envelope = np.maximum(envelope, np.abs(q) / lambdas)
    return envelope, len(operating), len(states)


def automatic_thermal_rank(config_path, config, *, monitor=None):
    """Progressively enlarge the thermal probe until the response tail resolves.

    If the configured probe cap is reached without a resolved tail, the result is
    the full discrete thermal space. Therefore this automatic convenience path can
    cost more, but it never silently promotes an unresolved partial rank.
    """
    from sdfmpneo.spatial import read_gmsh_v22_ascii

    path = Path(config_path)
    tagged = read_gmsh_v22_ascii(path.parent / config["mesh"])
    mesh = tagged.mesh
    _, _, capacity, conductivity, _ = _parse_materials(tagged, config)
    assembly = mesh.assemble_p1_thermal(
        rho_cp_tetra=capacity, conductivity_tetra=conductivity,
        homogeneous_dirichlet_boundary=True)
    n_free = int(assembly.M.shape[0])
    if n_free < 1:
        raise ValueError("thermal boundary treatment produced no free degrees of freedom")
    if n_free == 1:
        return 1, {"method": "automatic_physics_envelope", "selected_rank": 1,
                   "full_dimension": 1, "probe_rank": 1,
                   "certified_continuous_domain": False,
                   "scope": "single free thermal degree of freedom; no truncation"}
    truncation = dict(config.get("thermal_truncation") or {})
    relative = float(truncation.get("relative_tolerance", 1e-3))
    absolute = float(truncation.get("absolute_tolerance", 0.0))
    coordinate_bound = float(truncation.get("initial_coordinate_bound", 0.1))
    start = max(2, int(truncation.get("probe_start_rank", 4)))
    cap_value = truncation.get("max_probe_rank", 32)
    max_probe = n_free if cap_value is None else min(n_free, int(cap_value))
    safety = float(truncation.get("source_bound_safety_factor", 2.0))
    boundary_fraction = float(truncation.get("boundary_fraction", 0.25))
    state_axes = int(truncation.get("temperature_probe_axes", 4))
    probe = min(n_free, max_probe, start)
    while True:
        envelope, n_operating, n_states = _probe_modal_response_envelope(
            tagged, config, rank=probe, coordinate_bound=coordinate_bound,
            max_state_axes=state_axes, monitor=monitor)
        rank, details = select_rank_from_modal_response_envelope(
            envelope, relative_tolerance=relative, absolute_tolerance=absolute,
            safety_factor=safety, boundary_fraction=boundary_fraction,
            unresolved=probe < n_free)
        report = {
            "method": "automatic_physics_envelope", "selected_rank": int(rank),
            "full_dimension": n_free, "probe_rank": probe,
            "relative_tolerance": relative, "absolute_tolerance": absolute,
            "source_bound_safety_factor": safety,
            "initial_coordinate_bound": coordinate_bound,
            "operating_anchor_count": n_operating, "temperature_anchor_count": n_states,
            "maximum_modal_steady_response": envelope.tolist(),
            "certified_continuous_domain": False,
            "scope": ("deterministic full-order EM equilibrium anchors on the reference geometry; "
                      "exact residual training and geometry checks remain separate"),
            **details,
        }
        if rank < probe:
            return int(rank), report
        if probe >= n_free:
            report["selected_rank"] = n_free
            report["fallback"] = "full_discrete_thermal_space"
            return n_free, report
        if probe >= max_probe:
            report["selected_rank"] = n_free
            report["fallback"] = "probe_cap_reached_use_full_discrete_thermal_space"
            return n_free, report
        probe = min(n_free, max_probe, max(probe + 1, 2 * probe))


def _resolved_training(training, rank, truncation):
    training = dict(training)
    lower, upper = training.get("initial_lower"), training.get("initial_upper")
    automatic = lower is None or upper is None or (len(lower) == 0 and len(upper) == 0)
    if automatic:
        bound = float(truncation.get("initial_coordinate_bound", 0.1))
        if not np.isfinite(bound) or bound < 0.0:
            raise ValueError("initial_coordinate_bound must be finite and non-negative")
        training["initial_lower"] = [-bound] * int(rank)
        training["initial_upper"] = [bound] * int(rank)
    elif len(lower) != rank or len(upper) != rank:
        raise ValueError(
            f"automatic thermal rank selected {rank} modes but explicit initial bounds "
            "have a different dimension; use empty bounds for automatic expansion")
    return training


def _temporary_config(path, payload):
    handle = NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix=path.stem + ".auto.", suffix=".json",
        dir=path.parent, delete=False)
    try:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.close()
        return Path(handle.name)
    except Exception:
        handle.close()
        Path(handle.name).unlink(missing_ok=True)
        raise


def automatic_model_from_config(path, *, monitor=None):
    global _ORIGINAL_MODEL_FROM_CONFIG
    source = Path(path)
    config = json.loads(source.read_text(encoding="utf-8"))
    truncation = dict(config.get("thermal_truncation") or {})
    mode = truncation.get("mode")
    if config.get("thermal_rank") is not None or mode not in (
        "auto", "automatic", "physics_envelope", "automatic_physics_envelope"):
        return _ORIGINAL_MODEL_FROM_CONFIG(path)

    strict_keys = ("initial_temperature_deviation_free", "source_dual_bound", "requested_state_tolerance")
    strict = [truncation.get(key) is not None for key in strict_keys]
    if any(strict):
        if not all(strict):
            raise ValueError("strict thermal certificate inputs must be supplied together")
        compatible = dict(config)
        physical = dict(truncation)
        for key in _AUTO_KEYS:
            physical.pop(key, None)
        compatible["thermal_truncation"] = physical
        temporary = _temporary_config(source, compatible)
        try:
            model, training = _ORIGINAL_MODEL_FROM_CONFIG(temporary)
        finally:
            temporary.unlink(missing_ok=True)
        certificate = model.core.thermal_tail_certificate
        model.thermal_rank_report = {
            "method": "strict_projection_tail_certificate",
            "selected_rank": int(model.core.thermal_model.rank),
            "certificate": None if certificate is None else asdict(certificate),
            "certified_continuous_domain": bool(certificate is not None and certificate.certified),
        }
        return model, training

    rank, report = automatic_thermal_rank(source, config, monitor=monitor)
    compatible = dict(config)
    compatible["thermal_rank"] = rank
    compatible["thermal_truncation"] = {}
    compatible["training"] = _resolved_training(config["training"], rank, truncation)
    compatible.pop("em_candidate_states", None)
    temporary = _temporary_config(source, compatible)
    try:
        model, training = _ORIGINAL_MODEL_FROM_CONFIG(temporary)
    finally:
        temporary.unlink(missing_ok=True)
    model.thermal_rank_report = report
    return model, training


def install_automatic_thermal_rank() -> None:
    global _ORIGINAL_MODEL_FROM_CONFIG
    if _ORIGINAL_MODEL_FROM_CONFIG is not None:
        return
    import sdfmpneo.research as research_module
    _ORIGINAL_MODEL_FROM_CONFIG = research_module.model_from_config
    research_module.model_from_config = automatic_model_from_config
    import sdfmpneo as package
    package.model_from_config = automatic_model_from_config


__all__ = [
    "automatic_model_from_config", "automatic_thermal_rank",
    "install_automatic_thermal_rank", "select_rank_from_modal_response_envelope",
]
