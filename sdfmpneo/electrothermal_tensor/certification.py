"""Frozen audit evidence attached to deployable neural ROM artifacts."""
from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .provenance import canonical_sha256


@dataclass(frozen=True)
class PersistenceRoundtripReport:
    passed: bool
    metadata_preserved: bool
    training_domain_preserved: bool
    physical_signature_preserved: bool
    maximum_heat_source_error: float | None
    maximum_vector_field_error: float | None
    detail: str


@dataclass(frozen=True)
class TrainingReproductionReport:
    passed: bool
    attempted: bool
    provenance_complete: bool
    device: str | None
    sample_count: int
    maximum_packed_tensor_absolute_error: float | None
    maximum_packed_tensor_relative_error: float | None
    maximum_heat_source_absolute_error: float | None
    maximum_heat_source_relative_error: float | None
    original_best_epoch: int | None
    reproduced_best_epoch: int | None
    original_epochs_completed: int | None
    reproduced_epochs_completed: int | None
    detail: str


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


def training_reproducibility_evidence(model, *, expected_dataset_hash: str | None = None) -> dict:
    """Check whether the saved model contains the documented training provenance.

    This checks provenance completeness only.  It is deliberately not sufficient
    to pass the formal reproducibility gate; use ``verify_training_reproduction``
    to perform an actual frozen-dataset retraining experiment.
    """
    metadata = dict(getattr(model, "artifact_metadata", {}) or {})
    report = metadata.get("training_report")
    missing: list[str] = []
    if not isinstance(report, dict):
        missing.append("training_report")
        report = {}

    for key in (
        "optimizer",
        "learning_rate_schedule",
        "training_config",
        "network_config",
        "environment",
    ):
        if key not in report:
            missing.append(f"training_report.{key}")

    training_config = report.get("training_config")
    if not isinstance(training_config, dict):
        training_config = {}
    for key in (
        "seed",
        "epochs",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "dtype",
        "mixed_precision",
    ):
        if key not in training_config:
            missing.append(f"training_report.training_config.{key}")

    environment = report.get("environment")
    if not isinstance(environment, dict):
        environment = {}
    for key in ("git_revision", "python", "numpy", "scipy", "torch", "requested_device"):
        if key not in environment:
            missing.append(f"training_report.environment.{key}")

    saved_dataset_hash = metadata.get("dataset_hash")
    dataset_hash_matches = True
    if expected_dataset_hash is not None:
        dataset_hash_matches = str(saved_dataset_hash) == str(expected_dataset_hash)
        if not dataset_hash_matches:
            missing.append("dataset_hash_match")

    return {
        "complete": not missing,
        "missing": missing,
        "dataset_hash": saved_dataset_hash,
        "dataset_hash_matches": bool(dataset_hash_matches),
        "training_seed": training_config.get("seed"),
        "optimizer": report.get("optimizer"),
        "learning_rate_schedule": report.get("learning_rate_schedule"),
        "dtype": training_config.get("dtype"),
        "mixed_precision_requested": training_config.get("mixed_precision"),
        "mixed_precision_used": report.get("mixed_precision"),
        "environment": environment,
    }


def _relative_rows(error: np.ndarray, reference: np.ndarray) -> np.ndarray:
    numerator = np.linalg.norm(error, axis=1)
    denominator = np.maximum(
        np.linalg.norm(reference, axis=1),
        np.finfo(float).tiny,
    )
    return numerator / denominator


def verify_training_reproduction(
    model,
    dataset,
    *,
    device: str | None = None,
    packed_rtol: float = 1e-6,
    packed_atol: float = 1e-7,
    heat_rtol: float = 1e-6,
    heat_atol: float = 1e-7,
    operating_samples: int = 2,
) -> TrainingReproductionReport:
    """Retrain from the frozen dataset and compare basis-independent outputs.

    The POD is refit from the train split at the saved rank and the ordinary MLP
    is retrained with the persisted configuration/seed.  Comparison is performed
    after decoding back to packed ``G`` tensors and Joule heat so POD sign or a
    numerically equivalent basis representation cannot create a false failure.
    """
    manifest = dataset.manifest()
    provenance = training_reproducibility_evidence(
        model,
        expected_dataset_hash=manifest.dataset_hash,
    )
    metadata = dict(getattr(model, "artifact_metadata", {}) or {})
    saved_report = metadata.get("training_report")
    if not provenance["complete"] or not isinstance(saved_report, dict):
        return TrainingReproductionReport(
            passed=False,
            attempted=False,
            provenance_complete=False,
            device=None,
            sample_count=0,
            maximum_packed_tensor_absolute_error=None,
            maximum_packed_tensor_relative_error=None,
            maximum_heat_source_absolute_error=None,
            maximum_heat_source_relative_error=None,
            original_best_epoch=None,
            reproduced_best_epoch=None,
            original_epochs_completed=None,
            reproduced_epochs_completed=None,
            detail="frozen model lacks complete training provenance or matching dataset hash",
        )

    try:
        from .network import ResidualMLPConfig
        from .pod import fit_dataset_pod
        from .trainer import NeuralTrainingConfig, train_tensor_surrogate

        network_config = ResidualMLPConfig(**dict(saved_report["network_config"]))
        training_config = NeuralTrainingConfig(**dict(saved_report["training_config"]))
        resolved_device = str(device or saved_report.get("device") or "cpu")
        pod = fit_dataset_pod(dataset, rank=model.surrogate.pod.rank)
        lower = np.asarray(dataset.metadata["operating_lower"], dtype=float)
        upper = np.asarray(dataset.metadata["operating_upper"], dtype=float)
        reproduced, reproduced_report = train_tensor_surrogate(
            dataset,
            pod,
            operating_lower=lower,
            operating_upper=upper,
            network_config=network_config,
            training_config=training_config,
            device=resolved_device,
        )

        test_ids = dataset.indices("test")
        states = dataset.states[test_ids]
        geometries = dataset.geometries[test_ids]
        original_beta = model.surrogate.predict_coefficients_batch_numpy(states, geometries)
        reproduced_beta = reproduced.predict_coefficients_batch_numpy(states, geometries)
        original_packed = model.surrogate.pod.mean[None, :] + original_beta @ model.surrogate.pod.basis.T
        reproduced_packed = pod.mean[None, :] + reproduced_beta @ pod.basis.T
        packed_error = reproduced_packed - original_packed
        packed_abs = float(np.max(np.abs(packed_error))) if packed_error.size else 0.0
        packed_rel = float(np.max(_relative_rows(packed_error, original_packed))) if len(test_ids) else 0.0
        packed_ok = bool(
            np.allclose(
                reproduced_packed,
                original_packed,
                rtol=float(packed_rtol),
                atol=float(packed_atol),
            )
        )

        rng = np.random.default_rng(int(training_config.seed) + 104729)
        count = max(1, int(operating_samples))
        heat_abs = 0.0
        heat_rel = 0.0
        heat_ok = True
        for _ in range(count):
            operating = rng.uniform(lower, upper, size=(len(test_ids), dataset.current_dimension))
            original_heat = model.surrogate.heat_source_batch_numpy(states, geometries, operating)
            reproduced_heat = reproduced.heat_source_batch_numpy(states, geometries, operating)
            error = reproduced_heat - original_heat
            if error.size:
                heat_abs = max(heat_abs, float(np.max(np.abs(error))))
                heat_rel = max(heat_rel, float(np.max(_relative_rows(error, original_heat))))
            heat_ok = heat_ok and bool(
                np.allclose(
                    reproduced_heat,
                    original_heat,
                    rtol=float(heat_rtol),
                    atol=float(heat_atol),
                )
            )

        original_best = saved_report.get("best_epoch")
        original_completed = saved_report.get("epochs_completed")
        epoch_match = (
            original_best is not None
            and original_completed is not None
            and int(original_best) == int(reproduced_report.best_epoch)
            and int(original_completed) == int(reproduced_report.epochs_completed)
        )
        passed = bool(packed_ok and heat_ok and epoch_match)
        detail = (
            "frozen-dataset retraining reproduced POD-decoded tensors, heat outputs and epoch selection"
            if passed
            else "frozen-dataset retraining changed decoded tensors, heat outputs or epoch selection"
        )
        return TrainingReproductionReport(
            passed=passed,
            attempted=True,
            provenance_complete=True,
            device=resolved_device,
            sample_count=int(len(test_ids)),
            maximum_packed_tensor_absolute_error=packed_abs,
            maximum_packed_tensor_relative_error=packed_rel,
            maximum_heat_source_absolute_error=heat_abs,
            maximum_heat_source_relative_error=heat_rel,
            original_best_epoch=None if original_best is None else int(original_best),
            reproduced_best_epoch=int(reproduced_report.best_epoch),
            original_epochs_completed=(
                None if original_completed is None else int(original_completed)
            ),
            reproduced_epochs_completed=int(reproduced_report.epochs_completed),
            detail=detail,
        )
    except Exception as exc:
        return TrainingReproductionReport(
            passed=False,
            attempted=True,
            provenance_complete=True,
            device=None if device is None else str(device),
            sample_count=0,
            maximum_packed_tensor_absolute_error=None,
            maximum_packed_tensor_relative_error=None,
            maximum_heat_source_absolute_error=None,
            maximum_heat_source_relative_error=None,
            original_best_epoch=None,
            reproduced_best_epoch=None,
            original_epochs_completed=None,
            reproduced_epochs_completed=None,
            detail=f"retraining reproduction failed: {type(exc).__name__}: {exc}",
        )


def _domain_center(model, name: str, dimension: int) -> np.ndarray:
    lower = model.training_domain.get(name + "_lower")
    upper = model.training_domain.get(name + "_upper")
    if lower is None or upper is None:
        return np.zeros(int(dimension), dtype=float)
    lo = np.asarray(lower, dtype=float).reshape(-1)
    hi = np.asarray(upper, dtype=float).reshape(-1)
    if lo.shape != (int(dimension),) or hi.shape != lo.shape:
        raise ValueError(f"saved {name} training domain is incompatible")
    return 0.5 * (lo + hi)


def verify_model_persistence_roundtrip(
    model,
    *,
    device: str = "cpu",
    rtol: float = 1e-12,
    atol: float = 1e-12,
) -> PersistenceRoundtripReport:
    """Actually save, reload and compare a representative in-domain query."""
    try:
        state = _domain_center(model, "state", model.surrogate.state_dimension)
        geometry = _domain_center(model, "geometry", model.surrogate.geometry_dimension)
        operating = _domain_center(model, "operating", model.surrogate.pod.current_dimension)
        model._check_domain(state, geometry, operating, allow_extrapolation=False)
        heat_before = np.asarray(model.field.heat_source(state, geometry, operating), dtype=float)
        field_before = np.asarray(model.field.vector_field(state, geometry, operating), dtype=float)
        metadata_hash_before = canonical_sha256(model.artifact_metadata)
        domain_hash_before = canonical_sha256(
            {key: np.asarray(value, dtype=float) for key, value in model.training_domain.items()}
        )

        with TemporaryDirectory(prefix="sdfmpneo-neural-roundtrip-") as directory:
            path = Path(directory) / "roundtrip.npz"
            model.save(path)
            restored = type(model).load(
                path,
                thermal_operators=model.thermal_operators,
                expected_physical_signature=model.physical_signature,
                device=str(device),
            )
            heat_after = np.asarray(
                restored.field.heat_source(state, geometry, operating),
                dtype=float,
            )
            field_after = np.asarray(
                restored.field.vector_field(state, geometry, operating),
                dtype=float,
            )

        metadata_preserved = canonical_sha256(restored.artifact_metadata) == metadata_hash_before
        domain_preserved = canonical_sha256(
            {key: np.asarray(value, dtype=float) for key, value in restored.training_domain.items()}
        ) == domain_hash_before
        signature_preserved = restored.physical_signature == model.physical_signature
        heat_error = float(np.max(np.abs(heat_after - heat_before))) if heat_before.size else 0.0
        field_error = float(np.max(np.abs(field_after - field_before))) if field_before.size else 0.0
        numeric_ok = bool(
            np.allclose(heat_after, heat_before, rtol=float(rtol), atol=float(atol))
            and np.allclose(field_after, field_before, rtol=float(rtol), atol=float(atol))
        )
        passed = bool(metadata_preserved and domain_preserved and signature_preserved and numeric_ok)
        detail = (
            "save/load preserves metadata, domain, signature and representative heat/vector-field outputs"
            if passed
            else "save/load roundtrip changed metadata, domain, signature or representative numerical outputs"
        )
        return PersistenceRoundtripReport(
            passed=passed,
            metadata_preserved=bool(metadata_preserved),
            training_domain_preserved=bool(domain_preserved),
            physical_signature_preserved=bool(signature_preserved),
            maximum_heat_source_error=heat_error,
            maximum_vector_field_error=field_error,
            detail=detail,
        )
    except Exception as exc:
        return PersistenceRoundtripReport(
            passed=False,
            metadata_preserved=False,
            training_domain_preserved=False,
            physical_signature_preserved=False,
            maximum_heat_source_error=None,
            maximum_vector_field_error=None,
            detail=f"roundtrip failed: {type(exc).__name__}: {exc}",
        )


def audit_evidence(
    report,
    *,
    dataset_hash: str | None = None,
    report_path=None,
    provenance_evidence: dict | None = None,
    training_reproduction: TrainingReproductionReport | None = None,
    persistence_roundtrip: PersistenceRoundtripReport | None = None,
) -> dict:
    """Build a bounded model-metadata summary from a frozen Gate 1--7 report."""
    readiness = report.readiness
    gates = [
        {"name": record.name, "passed": record.passed, "detail": record.detail}
        for record in readiness.gates
    ]
    result = {
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
    if provenance_evidence is not None:
        result["training_provenance"] = _compact(provenance_evidence)
    if training_reproduction is not None:
        result["training_reproduction"] = _compact(training_reproduction)
    if persistence_roundtrip is not None:
        result["persistence_roundtrip"] = _compact(persistence_roundtrip)
    return result


def save_audited_model(
    model,
    path: str | Path,
    report,
    *,
    dataset_hash: str | None = None,
    report_path=None,
    require_ready: bool = False,
    provenance_evidence: dict | None = None,
    training_reproduction: TrainingReproductionReport | None = None,
    persistence_roundtrip: PersistenceRoundtripReport | None = None,
):
    """Save a copy with frozen audit evidence while preserving training metadata."""
    if require_ready and not bool(report.readiness.ready):
        raise ValueError("cannot create a production-ready certified model from a failing/incomplete audit")
    evidence = audit_evidence(
        report,
        dataset_hash=dataset_hash,
        report_path=report_path,
        provenance_evidence=provenance_evidence,
        training_reproduction=training_reproduction,
        persistence_roundtrip=persistence_roundtrip,
    )
    return model.save(
        path,
        metadata={"certification": evidence},
    )


__all__ = [
    "PersistenceRoundtripReport",
    "TrainingReproductionReport",
    "audit_evidence",
    "gate_report_hash",
    "save_audited_model",
    "training_reproducibility_evidence",
    "verify_model_persistence_roundtrip",
    "verify_training_reproduction",
]
