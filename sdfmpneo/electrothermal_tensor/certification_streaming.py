"""Out-of-core formal training-reproduction evidence.

This module mirrors the semantic requirements of ``verify_training_reproduction``
but never materializes the full frozen test tensor matrix or full decoded model
outputs.  It is the preferred implementation for formal certification and works
for both in-memory and directory-backed datasets.
"""
from __future__ import annotations

import numpy as np

from .certification import TrainingReproductionReport, training_reproducibility_evidence


def _row_relative(error: np.ndarray, reference: np.ndarray) -> np.ndarray:
    numerator = np.linalg.norm(error, axis=1)
    denominator = np.maximum(np.linalg.norm(reference, axis=1), np.finfo(float).tiny)
    return numerator / denominator


def verify_training_reproduction_streaming(
    model,
    dataset,
    *,
    device: str | None = None,
    packed_rtol: float = 1e-6,
    packed_atol: float = 1e-7,
    heat_rtol: float = 1e-6,
    heat_atol: float = 1e-7,
    operating_samples: int = 2,
    comparison_batch_size: int = 64,
) -> TrainingReproductionReport:
    """Retrain from the frozen dataset and compare decoded physics in chunks."""
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

        comparison_batch = max(1, int(comparison_batch_size))
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

        test_ids = np.asarray(dataset.indices("test"), dtype=int)
        packed_abs = 0.0
        packed_rel = 0.0
        packed_ok = True
        for start in range(0, len(test_ids), comparison_batch):
            ids = test_ids[start:start + comparison_batch]
            states = np.asarray(dataset.states[ids], dtype=float)
            geometries = np.asarray(dataset.geometries[ids], dtype=float)
            original_beta = model.surrogate.predict_coefficients_batch_numpy(states, geometries)
            reproduced_beta = reproduced.predict_coefficients_batch_numpy(states, geometries)
            original_packed = (
                model.surrogate.pod.mean[None, :]
                + original_beta @ model.surrogate.pod.basis.T
            )
            reproduced_packed = pod.mean[None, :] + reproduced_beta @ pod.basis.T
            error = reproduced_packed - original_packed
            if error.size:
                packed_abs = max(packed_abs, float(np.max(np.abs(error))))
                packed_rel = max(
                    packed_rel,
                    float(np.max(_row_relative(error, original_packed))),
                )
            packed_ok = packed_ok and bool(
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
            operating_all = rng.uniform(
                lower,
                upper,
                size=(len(test_ids), dataset.current_dimension),
            )
            for start in range(0, len(test_ids), comparison_batch):
                ids = test_ids[start:start + comparison_batch]
                states = np.asarray(dataset.states[ids], dtype=float)
                geometries = np.asarray(dataset.geometries[ids], dtype=float)
                operating = operating_all[start:start + len(ids)]
                original_heat = model.surrogate.heat_source_batch_numpy(
                    states,
                    geometries,
                    operating,
                )
                reproduced_heat = reproduced.heat_source_batch_numpy(
                    states,
                    geometries,
                    operating,
                )
                error = reproduced_heat - original_heat
                if error.size:
                    heat_abs = max(heat_abs, float(np.max(np.abs(error))))
                    heat_rel = max(
                        heat_rel,
                        float(np.max(_row_relative(error, original_heat))),
                    )
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
            "streaming frozen-dataset retraining reproduced decoded tensors, heat outputs and epoch selection"
            if passed
            else "streaming frozen-dataset retraining changed decoded tensors, heat outputs or epoch selection"
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
            original_epochs_completed=None if original_completed is None else int(original_completed),
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
            detail=f"streaming retraining reproduction failed: {type(exc).__name__}: {exc}",
        )


__all__ = ["verify_training_reproduction_streaming"]
