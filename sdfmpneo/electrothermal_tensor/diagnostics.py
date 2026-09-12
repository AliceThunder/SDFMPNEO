"""Optional true-EM diagnostics evaluated on neural-predicted thermal states."""
from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass

import numpy as np

from ..em.tetra_nonlinear_diagnostics import NonlinearTetrahedralRegionLossEvaluator
from .adapters import geometry_research_temperature_reconstructor
from .signatures import fixed_research_physical_signature, geometry_research_physical_signature


@dataclass(frozen=True)
class NeuralStateEMDiagnostics:
    thermal_state: np.ndarray
    geometry: np.ndarray
    operating: np.ndarray
    temperature_field: np.ndarray
    maximum_temperature: float
    neural_heat_source: np.ndarray
    physical_heat_source: np.ndarray
    heat_source_error_norm: float
    heat_source_relative_error: float
    drive_rhs_residual_dual_norm: float
    impedance: object
    region_losses: dict[str, float]


def _port_result(ports, em, state, requested_impedance_error: float):
    if hasattr(em, "residual_certificate_for_rhs"):
        return ports.evaluate_reduced_physical_certified(
            em.problem,
            state,
            em,
            requested_impedance_element_error=float(requested_impedance_error),
        )
    return ports.evaluate(em.problem, state, reduced_basis=em.V)


def _evaluate_context(
    *,
    neural_model,
    em,
    ports,
    rhs_map,
    state,
    geometry,
    operating,
    temperature,
    requested_impedance_error: float,
):
    a = np.asarray(state, dtype=float).reshape(-1)
    g = np.asarray(geometry, dtype=float).reshape(-1)
    u = np.asarray(operating, dtype=float).reshape(-1)
    neural_q = np.asarray(neural_model.field.heat_source(a, g, u), dtype=float)
    rhs = rhs_map.evaluate(u)
    physical_q = np.asarray(em.heat_source_for_rhs(a, rhs), dtype=float)
    error = float(np.linalg.norm(neural_q - physical_q))
    scale = max(float(np.linalg.norm(physical_q)), np.finfo(float).tiny)
    drive_residual = float(em.residual_dual_norm_for_rhs(a, rhs))
    port_result = _port_result(ports, em, a, requested_impedance_error)
    em_state = em.state_for_rhs(a, rhs)
    region_losses = NonlinearTetrahedralRegionLossEvaluator(em.problem).evaluate_state(em_state, a)
    T = np.asarray(temperature, dtype=float)
    return NeuralStateEMDiagnostics(
        thermal_state=a,
        geometry=g,
        operating=u,
        temperature_field=T,
        maximum_temperature=float(np.max(T)),
        neural_heat_source=neural_q,
        physical_heat_source=physical_q,
        heat_source_error_norm=error,
        heat_source_relative_error=error / scale,
        drive_rhs_residual_dual_norm=drive_residual,
        impedance=port_result,
        region_losses={str(key): float(value) for key, value in region_losses.items()},
    )


def diagnose_neural_state(
    neural_model,
    physical_model,
    *,
    state,
    operating,
    geometry=None,
    requested_impedance_error: float = 1e-6,
    allow_extrapolation: bool = False,
) -> NeuralStateEMDiagnostics:
    """Audit one neural thermal state with the matching real reduced EM model.

    ``geometry`` is normalized ``[-1,1]^d`` for a geometry-family model and is
    omitted/empty for a fixed model. The physical compatibility signature and
    the saved neural training domain are checked before any diagnostic EM solve.
    Set ``allow_extrapolation=True`` only for an explicit extrapolation study.
    """
    requested = float(requested_impedance_error)
    if not np.isfinite(requested) or requested <= 0.0:
        raise ValueError("requested_impedance_error must be finite and positive")
    a = np.asarray(state, dtype=float).reshape(-1)
    u = np.asarray(operating, dtype=float).reshape(-1)
    if hasattr(physical_model, "geometry_names"):
        signature = geometry_research_physical_signature(physical_model)
        if neural_model.physical_signature != signature:
            raise ValueError("neural/physical model signatures differ")
        n_geometry = len(physical_model.geometry_names)
        z = np.zeros(n_geometry) if geometry is None else np.asarray(geometry, dtype=float).reshape(-1)
        if z.shape != (n_geometry,) or np.any(~np.isfinite(z)) or np.any(z < -1.0) or np.any(z > 1.0):
            raise ValueError("geometry diagnostics require normalized coordinates in [-1,1]")
        a, z, u = neural_model._check_domain(
            a,
            z,
            u,
            allow_extrapolation=bool(allow_extrapolation),
        )
        context = physical_model.context(physical_model.denormalize(z))
        temperature = geometry_research_temperature_reconstructor(
            physical_model,
            normalized_geometry=True,
        )(a, z)
        return _evaluate_context(
            neural_model=neural_model,
            em=context.em,
            ports=context.ports,
            rhs_map=context.rhs,
            state=a,
            geometry=z,
            operating=u,
            temperature=temperature,
            requested_impedance_error=requested,
        )

    signature = fixed_research_physical_signature(physical_model)
    if neural_model.physical_signature != signature:
        raise ValueError("neural/physical model signatures differ")
    g = np.empty(0, dtype=float)
    if geometry is not None and np.asarray(geometry).size != 0:
        raise ValueError("fixed-model diagnostics do not accept geometry coordinates")
    a, g, u = neural_model._check_domain(
        a,
        g,
        u,
        allow_extrapolation=bool(allow_extrapolation),
    )
    return _evaluate_context(
        neural_model=neural_model,
        em=physical_model.em,
        ports=physical_model.ports,
        rhs_map=physical_model.rhs_map,
        state=a,
        geometry=g,
        operating=u,
        temperature=physical_model.temperature(a),
        requested_impedance_error=requested,
    )


def diagnostics_jsonable(value):
    """JSON-safe representation preserving complex EM outputs explicitly."""
    if is_dataclass(value):
        return diagnostics_jsonable(asdict(value))
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return {
                "real": np.asarray(value.real).tolist(),
                "imag": np.asarray(value.imag).tolist(),
            }
        return value.tolist()
    if isinstance(value, np.generic):
        return diagnostics_jsonable(value.item())
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, dict):
        return {str(key): diagnostics_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [diagnostics_jsonable(item) for item in value]
    return value


__all__ = [
    "NeuralStateEMDiagnostics",
    "diagnose_neural_state",
    "diagnostics_jsonable",
]
