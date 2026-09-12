"""Frozen audit evidence attached to deployable neural ROM artifacts."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np

from .provenance import canonical_sha256


def _compact(value):
    if is_dataclass(value):
        return _compact(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _compact(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_compact(item) for item in value]
    return value


def gate_report_hash(report) -> str:
    return canonical_sha256(report)


def audit_evidence(report, *, dataset_hash: str | None = None, report_path=None) -> dict:
    """Build a bounded model-metadata summary from a frozen Gate 1--7 report."""
    readiness = report.readiness
    gates = [
        {"name": record.name, "passed": record.passed, "detail": record.detail}
        for record in readiness.gates
    ]
    return {
        "gate_report_hash": gate_report_hash(report),
        "gate_report_path": None if report_path is None else str(Path(report_path)),
        "dataset_hash": None if dataset_hash is None else str(dataset_hash),
        "production_ready": bool(readiness.ready),
        "incomplete": bool(readiness.incomplete),
        "gates": gates,
        "quadratic_identity": _compact(report.quadratic_identity),
        "active_subspace": {
            "method": str(report.active_subspace_method),
            "direction_count": int(report.active_subspace_direction_count),
        },
        "physical_stability": _compact(report.physical_stability),
        "surrogate_test": _compact(report.surrogate),
        "vector_field_test": _compact(report.vector_field),
        "trajectory_test": _compact(report.aggregate_trajectory),
        "vector_field_seconds": report.vector_field_seconds,
        "trajectory_query_seconds": report.trajectory_query_seconds,
    }


def save_audited_model(
    model,
    path: str | Path,
    report,
    *,
    dataset_hash: str | None = None,
    report_path=None,
    require_ready: bool = False,
):
    """Save a copy with frozen audit evidence while preserving training metadata."""
    if require_ready and not bool(report.readiness.ready):
        raise ValueError("cannot create a production-ready certified model from a failing/incomplete audit")
    evidence = audit_evidence(
        report,
        dataset_hash=dataset_hash,
        report_path=report_path,
    )
    return model.save(
        path,
        metadata={"certification": evidence},
    )


__all__ = ["audit_evidence", "gate_report_hash", "save_audited_model"]
