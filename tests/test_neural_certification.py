from dataclasses import dataclass

import pytest

from sdfmpneo.electrothermal_tensor.certification import (
    audit_evidence,
    gate_report_hash,
    save_audited_model,
    training_reproducibility_evidence,
)


@dataclass(frozen=True)
class _Gate:
    name: str
    passed: bool | None
    detail: str


@dataclass(frozen=True)
class _Readiness:
    gates: tuple[_Gate, ...]
    ready: bool

    @property
    def incomplete(self):
        return any(item.passed is None for item in self.gates)


@dataclass(frozen=True)
class _ScalarReport:
    value: float


@dataclass(frozen=True)
class _Report:
    readiness: _Readiness
    quadratic_identity: _ScalarReport
    active_subspace_method: str
    active_subspace_direction_count: int
    physical_stability: _ScalarReport
    surrogate: _ScalarReport
    vector_field: _ScalarReport
    aggregate_trajectory: _ScalarReport
    vector_field_seconds: float | None
    trajectory_query_seconds: float | None


class _FakeModel:
    def __init__(self, artifact_metadata=None):
        self.calls = []
        self.artifact_metadata = {} if artifact_metadata is None else dict(artifact_metadata)

    def save(self, path, *, metadata=None):
        self.calls.append((path, metadata))
        return path


def _report(*, ready: bool):
    gates = (
        _Gate("Gate 1", True, "ok"),
        _Gate("Gate 7", True if ready else None, "timing"),
    )
    return _Report(
        readiness=_Readiness(gates, ready),
        quadratic_identity=_ScalarReport(0.0),
        active_subspace_method="full",
        active_subspace_direction_count=3,
        physical_stability=_ScalarReport(-0.5),
        surrogate=_ScalarReport(1e-4),
        vector_field=_ScalarReport(2e-4),
        aggregate_trajectory=_ScalarReport(3e-4),
        vector_field_seconds=0.001,
        trajectory_query_seconds=0.01,
    )


def _complete_training_metadata(dataset_hash="dataset-123"):
    return {
        "dataset_hash": dataset_hash,
        "training_report": {
            "optimizer": "AdamW",
            "learning_rate_schedule": "constant",
            "training_config": {
                "seed": 7,
                "epochs": 20,
                "batch_size": 8,
                "learning_rate": 1e-3,
                "weight_decay": 1e-6,
                "dtype": "float32",
                "mixed_precision": False,
            },
            "network_config": {"width": 16, "blocks": 2},
            "mixed_precision": False,
            "environment": {
                "git_revision": "abc123",
                "python": "3.12.0",
                "numpy": "2.0.0",
                "scipy": "1.14.0",
                "torch": "2.4.0",
                "requested_device": "cpu",
            },
        },
    }


def test_gate_report_hash_is_deterministic_and_evidence_binds_dataset():
    report = _report(ready=True)
    first = gate_report_hash(report)
    second = gate_report_hash(report)
    assert first == second
    assert len(first) == 64

    evidence = audit_evidence(
        report,
        dataset_hash="dataset-123",
        report_path="frozen-report.json",
    )
    assert evidence["gate_report_hash"] == first
    assert evidence["dataset_hash"] == "dataset-123"
    assert evidence["production_ready"] is True
    assert evidence["incomplete"] is False
    assert evidence["active_subspace"]["method"] == "full"


def test_training_reproducibility_is_derived_from_saved_evidence_and_dataset_hash():
    model = _FakeModel(_complete_training_metadata())
    evidence = training_reproducibility_evidence(
        model,
        expected_dataset_hash="dataset-123",
    )
    assert evidence["complete"] is True
    assert evidence["missing"] == []
    assert evidence["dataset_hash_matches"] is True
    assert evidence["training_seed"] == 7
    assert evidence["optimizer"] == "AdamW"

    mismatch = training_reproducibility_evidence(
        model,
        expected_dataset_hash="different-dataset",
    )
    assert mismatch["complete"] is False
    assert mismatch["dataset_hash_matches"] is False
    assert "dataset_hash_match" in mismatch["missing"]


def test_training_reproducibility_fails_closed_when_provenance_is_missing():
    evidence = training_reproducibility_evidence(_FakeModel(), expected_dataset_hash="dataset-123")
    assert evidence["complete"] is False
    assert "training_report" in evidence["missing"]
    assert "dataset_hash_match" in evidence["missing"]


def test_certified_model_requires_ready_report_but_audited_copy_does_not():
    report = _report(ready=False)
    model = _FakeModel()

    saved = save_audited_model(
        model,
        "audited.npz",
        report,
        dataset_hash="dataset-123",
        report_path="report.json",
        require_ready=False,
    )
    assert saved == "audited.npz"
    assert model.calls[-1][1]["certification"]["production_ready"] is False

    with pytest.raises(ValueError, match="production-ready certified model"):
        save_audited_model(
            model,
            "certified.npz",
            report,
            dataset_hash="dataset-123",
            report_path="report.json",
            require_ready=True,
        )


def test_ready_report_can_create_certified_model_metadata():
    report = _report(ready=True)
    model = _FakeModel()
    reproducibility = {"complete": True, "training_seed": 7}
    saved = save_audited_model(
        model,
        "certified.npz",
        report,
        dataset_hash="dataset-123",
        report_path="report.json",
        require_ready=True,
        reproducibility_evidence=reproducibility,
    )
    assert saved == "certified.npz"
    metadata = model.calls[-1][1]["certification"]
    assert metadata["production_ready"] is True
    assert metadata["dataset_hash"] == "dataset-123"
    assert metadata["training_reproducibility"]["complete"] is True
