"""Fail-closed physical compatibility signatures for neural ROM artifacts."""
from __future__ import annotations

import hashlib
import json

import numpy as np


def _update_array(digest, name: str, value) -> None:
    array = np.ascontiguousarray(np.asarray(value))
    digest.update(name.encode("utf-8"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())


def _update_json(digest, name: str, value) -> None:
    digest.update(name.encode("utf-8"))
    digest.update(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))


def fixed_research_physical_signature(model) -> str:
    """Hash the fixed geometry, thermal/EM spaces and current parameterization."""
    digest = hashlib.sha256()
    core = model.core
    problem = model.em.problem
    _update_array(digest, "mesh_vertices", core.mesh.vertices)
    _update_array(digest, "mesh_tetrahedra", core.mesh.tetrahedra)
    _update_array(digest, "thermal_phi", core.thermal_model.Phi)
    _update_array(digest, "thermal_lambdas", core.thermal_model.lambdas)
    _update_array(digest, "em_basis", model.em.V)
    _update_array(digest, "rhs_offset", model.rhs_map.offset)
    _update_array(digest, "rhs_matrix", model.rhs_map.matrix)
    _update_array(digest, "reluctivity", problem.reluctivity_tetra)
    _update_array(digest, "temperature_reference_local", problem.temperature_reference_local)
    _update_array(digest, "thermal_modes_local", problem.thermal_modes_local)
    _update_json(digest, "omega", float(problem.omega))
    _update_json(
        digest,
        "constitutive_relative_error_budget",
        float(problem.constitutive_relative_error_budget),
    )
    return digest.hexdigest()


def geometry_research_physical_signature(model) -> str:
    """Extend the fixed reference signature with the affine geometry chart/domain."""
    digest = hashlib.sha256()
    _update_json(digest, "reference_signature", fixed_research_physical_signature(model.reference))
    chart = model.chart
    _update_array(digest, "chart_reference_vertices", chart.reference_vertices)
    _update_array(digest, "chart_tetrahedra", chart.tetrahedra)
    _update_array(digest, "chart_vertex_directions", chart.vertex_directions)
    _update_json(digest, "chart_parameter_names", list(chart.parameter_names))
    _update_array(digest, "geometry_reference", model.geometry_reference)
    _update_array(digest, "geometry_lower", model.lower)
    _update_array(digest, "geometry_upper", model.upper)
    _update_array(digest, "current_offset", model.current_offset)
    _update_array(digest, "current_matrix", model.current_matrix)
    return digest.hexdigest()


__all__ = ["fixed_research_physical_signature", "geometry_research_physical_signature"]
