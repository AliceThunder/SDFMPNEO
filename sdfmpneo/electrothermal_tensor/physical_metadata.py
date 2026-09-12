"""Compact physical/ROM provenance stored with frozen tensor datasets."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib

import numpy as np

from .runtime_metadata import software_environment_summary


def _array_summary(value: np.ndarray) -> dict:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return {
        "kind": "ndarray",
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": digest.hexdigest(),
    }


def compact_summary(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.size <= 32:
            return np.asarray(value).tolist()
        return _array_summary(value)
    if is_dataclass(value):
        return compact_summary(asdict(value))
    if isinstance(value, dict):
        return {str(key): compact_summary(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        if len(value) <= 32:
            return [compact_summary(item) for item in value]
        return {"kind": "sequence", "length": len(value)}
    if hasattr(value, "to_dict"):
        try:
            return compact_summary(value.to_dict())
        except Exception:
            pass
    return {"kind": type(value).__name__}


def _fixed_summary(model) -> dict:
    core = model.core
    em = model.em
    problem = em.problem
    rhs_matrix = np.asarray(model.rhs_map.matrix)
    return {
        "model_kind": "fixed",
        "thermal_rank": int(core.thermal_model.rank),
        "thermal_backend": str(getattr(core, "thermal_spectrum_backend", "unknown")),
        "em_full_dimension": int(problem.n_em),
        "em_reduced_rank": int(em.n_reduced),
        "em_reduction_certificate": compact_summary(getattr(em, "reduction_certificate", None)),
        "frequency_hz": float(problem.omega) / (2.0 * np.pi),
        "constitutive_relative_error_budget": float(problem.constitutive_relative_error_budget),
        "operating_dimension": int(rhs_matrix.shape[1]),
    }


def physical_dataset_metadata(model) -> dict:
    """Return a compact, JSON-safe snapshot-label provenance summary."""
    if hasattr(model, "geometry_names"):
        result = _fixed_summary(model.reference)
        result["model_kind"] = "geometry_family"
        result["geometry_dimension"] = int(len(model.geometry_names))
        result["geometry_names"] = list(model.geometry_names)
        result["geometry_lower"] = np.asarray(model.lower, dtype=float).tolist()
        result["geometry_upper"] = np.asarray(model.upper, dtype=float).tolist()
        result["mesh_certificate"] = compact_summary(getattr(model, "certificate", None))
        result["em_basis_report"] = compact_summary(getattr(model, "em_basis_report", None))
        current_matrix = np.asarray(model.current_matrix)
        result["operating_dimension"] = int(current_matrix.shape[1])
    else:
        result = _fixed_summary(model)
        result["geometry_dimension"] = 0
    result["software_environment"] = software_environment_summary()
    return result


__all__ = ["compact_summary", "physical_dataset_metadata"]
