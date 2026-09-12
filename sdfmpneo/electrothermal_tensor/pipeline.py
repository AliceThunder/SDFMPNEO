"""High-level construction pipeline for the structure-preserving neural ROM."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .adapters import (
    fixed_research_tensor_factory,
    fixed_research_thermal_family,
    fixed_research_thermal_rhs_forcing,
    fixed_research_vector_field_factory,
    geometry_research_embedded_thermal_family,
    geometry_research_tensor_factory,
    geometry_research_thermal_rhs_forcing,
    geometry_research_vector_field_factory,
)
from .generator import generate_snapshots_resumable
from .model import StructurePreservingNeuralElectroThermalROM
from .network import ResidualMLPConfig
from .physical_metadata import physical_dataset_metadata
from .pod import fit_dataset_pod
from .sampling import (
    SnapshotSamplingResult,
    box_state_geometry_samples,
    hybrid_reachable_state_geometry_samples,
)
from .signatures import fixed_research_physical_signature, geometry_research_physical_signature
from .trainer import NeuralTrainingConfig, train_tensor_surrogate


@dataclass(frozen=True)
class PipelineResult:
    dataset: object
    pod: object
    surrogate: object
    training_report: object
    model: StructurePreservingNeuralElectroThermalROM
    physical_signature: str
    sampling_report: object | None


def _resolved_network_config(network_config, *, input_dimension: int, output_dimension: int):
    if network_config is None:
        return ResidualMLPConfig(input_dimension=input_dimension, output_dimension=output_dimension)
    if isinstance(network_config, ResidualMLPConfig):
        if network_config.input_dimension != input_dimension or network_config.output_dimension != output_dimension:
            raise ValueError("explicit network dimensions do not match dataset/POD")
        return network_config
    settings = dict(network_config)
    settings.pop("input_dimension", None)
    settings.pop("output_dimension", None)
    return ResidualMLPConfig(
        input_dimension=input_dimension,
        output_dimension=output_dimension,
        **settings,
    )


def _resolved_training_config(training_config):
    if training_config is None or isinstance(training_config, NeuralTrainingConfig):
        return training_config
    return NeuralTrainingConfig(**dict(training_config))


def _fit_pod(dataset, *, rank, relative_tail_tolerance, pod_config=None):
    settings = {} if pod_config is None else dict(pod_config)
    allowed = {"exact_svd_max_bytes", "out_of_core_max_rank", "chunk_rows"}
    unknown = sorted(set(settings) - allowed)
    if unknown:
        raise ValueError(f"unknown POD configuration keys: {unknown}")
    return fit_dataset_pod(
        dataset,
        rank=rank,
        relative_tail_tolerance=relative_tail_tolerance,
        **settings,
    )


def _train_from_dataset(
    dataset,
    pod,
    *,
    operating_lower,
    operating_upper,
    network_config,
    training_config,
):
    resolved_network = _resolved_network_config(
        network_config,
        input_dimension=dataset.thermal_rank + dataset.geometry_dimension,
        output_dimension=pod.rank,
    )
    return train_tensor_surrogate(
        dataset,
        pod,
        operating_lower=np.asarray(operating_lower, dtype=float),
        operating_upper=np.asarray(operating_upper, dtype=float),
        network_config=resolved_network,
        training_config=_resolved_training_config(training_config),
    )


def _report_metadata(report) -> dict:
    return {
        "strategy": report.strategy,
        "sample_count": int(report.sample_count),
        "box_sample_count": int(report.box_sample_count),
        "reachable_sample_count": int(report.reachable_sample_count),
        "trajectory_count": int(report.trajectory_count),
        "observed_state_lower": np.asarray(report.observed_state_lower, dtype=float).tolist(),
        "observed_state_upper": np.asarray(report.observed_state_upper, dtype=float).tolist(),
    }


def _sample_locations(
    *,
    physical_vector_field,
    state_lower,
    state_upper,
    geometry_lower,
    geometry_upper,
    operating_lower,
    operating_upper,
    n_samples: int,
    seed: int,
    sampling_config,
) -> SnapshotSamplingResult:
    config = {} if sampling_config is None else dict(sampling_config)
    strategy = str(config.pop("strategy", "box"))
    if strategy == "box":
        if config:
            raise ValueError(f"box sampling does not accept options: {sorted(config)}")
        return box_state_geometry_samples(
            state_lower,
            state_upper,
            geometry_lower,
            geometry_upper,
            n_samples,
            seed=seed,
        )
    if strategy != "hybrid_reachable":
        raise ValueError("sampling strategy must be 'box' or 'hybrid_reachable'")
    if "time_horizon" not in config:
        raise ValueError("hybrid_reachable sampling requires time_horizon")
    return hybrid_reachable_state_geometry_samples(
        physical_vector_field,
        state_lower=state_lower,
        state_upper=state_upper,
        geometry_lower=geometry_lower,
        geometry_upper=geometry_upper,
        operating_lower=operating_lower,
        operating_upper=operating_upper,
        n_samples=n_samples,
        seed=seed,
        **config,
    )


def _dataset_metadata(
    *,
    kind: str,
    signature: str,
    state_lower,
    state_upper,
    geometry_lower,
    geometry_upper,
    operating_lower,
    operating_upper,
    sampling_report,
    physical_provenance: dict,
    extra_metadata: dict | None = None,
) -> dict:
    metadata = {
        "kind": kind,
        "physical_signature": signature,
        "physical_provenance": dict(physical_provenance),
        "state_lower": np.asarray(state_lower, dtype=float).tolist(),
        "state_upper": np.asarray(state_upper, dtype=float).tolist(),
        "geometry_lower": np.asarray(geometry_lower, dtype=float).tolist(),
        "geometry_upper": np.asarray(geometry_upper, dtype=float).tolist(),
        "operating_lower": np.asarray(operating_lower, dtype=float).tolist(),
        "operating_upper": np.asarray(operating_upper, dtype=float).tolist(),
        "sampling": _report_metadata(sampling_report),
    }
    if extra_metadata:
        reserved = set(metadata)
        conflict = sorted(reserved.intersection(extra_metadata))
        if conflict:
            raise ValueError(f"extra dataset metadata cannot override reserved keys: {conflict}")
        metadata.update(dict(extra_metadata))
    return metadata


def _training_domain_from_dataset(dataset) -> dict[str, np.ndarray]:
    meta = dict(dataset.metadata)
    required = (
        "state_lower",
        "state_upper",
        "geometry_lower",
        "geometry_upper",
        "operating_lower",
        "operating_upper",
    )
    missing = [name for name in required if name not in meta]
    if missing:
        raise ValueError(
            "dataset predates complete domain metadata; retraining requires explicit "
            f"domain information, missing {missing}"
        )
    domain = {name: np.asarray(meta[name], dtype=float) for name in required}
    if domain["state_lower"].shape != (dataset.thermal_rank,) or domain["state_upper"].shape != (dataset.thermal_rank,):
        raise ValueError("dataset state-domain metadata is incompatible with thermal rank")
    if domain["geometry_lower"].shape != (dataset.geometry_dimension,) or domain["geometry_upper"].shape != (dataset.geometry_dimension,):
        raise ValueError("dataset geometry-domain metadata is incompatible")
    if domain["operating_lower"].shape != (dataset.current_dimension,) or domain["operating_upper"].shape != (dataset.current_dimension,):
        raise ValueError("dataset operating-domain metadata is incompatible")
    return domain


def _dataset_artifact_metadata(dataset) -> dict:
    """Carry only immutable physical/data provenance into a newly trained model."""
    metadata = dict(dataset.metadata)
    result = {"dataset_hash": dataset.manifest().dataset_hash}
    for key in (
        "physical_provenance",
        "state_domain_report",
        "state_domain_report_hash",
    ):
        if key in metadata:
            result[key] = metadata[key]
    return result


def retrain_neural_rom(
    dataset,
    template_model: StructurePreservingNeuralElectroThermalROM,
    *,
    work_directory: str | Path,
    pod_rank: int | None = None,
    pod_relative_tail_tolerance: float = 1e-4,
    pod_config: dict | None = None,
    network_config: ResidualMLPConfig | dict | None = None,
    training_config: NeuralTrainingConfig | dict | None = None,
    save_model: bool = True,
    model_filename: str = "neural_electrothermal_rom.retrained.npz",
) -> PipelineResult:
    """Retrain POD/MLP from a frozen tensor dataset without any EM rebuild."""
    if dataset.thermal_rank != template_model.surrogate.state_dimension:
        raise ValueError("dataset/template thermal dimensions differ")
    if dataset.geometry_dimension != template_model.surrogate.geometry_dimension:
        raise ValueError("dataset/template geometry dimensions differ")
    if dataset.current_dimension != template_model.surrogate.pod.current_dimension:
        raise ValueError("dataset/template current dimensions differ")
    signature = dataset.metadata.get("physical_signature")
    if signature is None or template_model.physical_signature is None:
        raise ValueError("dataset and template model must both contain a physical signature")
    if str(signature) != str(template_model.physical_signature):
        raise ValueError("dataset/template physical signatures differ")
    domain = _training_domain_from_dataset(dataset)
    pod = _fit_pod(
        dataset,
        rank=pod_rank,
        relative_tail_tolerance=pod_relative_tail_tolerance,
        pod_config=pod_config,
    )
    surrogate, report = _train_from_dataset(
        dataset,
        pod,
        operating_lower=domain["operating_lower"],
        operating_upper=domain["operating_upper"],
        network_config=network_config,
        training_config=training_config,
    )
    artifact_metadata = _dataset_artifact_metadata(dataset)
    model = StructurePreservingNeuralElectroThermalROM(
        surrogate,
        template_model.thermal_operators,
        physical_signature=str(signature),
        training_domain=domain,
        artifact_metadata=artifact_metadata,
        thermal_rhs_forcing=template_model.field.thermal_rhs_forcing,
    )
    work = Path(work_directory)
    work.mkdir(parents=True, exist_ok=True)
    if save_model:
        model.save(
            work / model_filename,
            metadata={
                "training_report": report.__dict__,
                "pod_rank": pod.rank,
                "retrained_without_em": True,
            },
        )
    return PipelineResult(dataset, pod, surrogate, report, model, str(signature), None)


def build_fixed_neural_rom(
    physical_model,
    *,
    state_lower,
    state_upper,
    operating_lower,
    operating_upper,
    n_snapshots: int,
    work_directory: str | Path,
    seed: int = 0,
    pod_rank: int | None = None,
    pod_relative_tail_tolerance: float = 1e-4,
    pod_config: dict | None = None,
    network_config: ResidualMLPConfig | dict | None = None,
    training_config: NeuralTrainingConfig | dict | None = None,
    physical_signature: str | None = None,
    save_model: bool = True,
    snapshot_workers: int = 1,
    checkpoint_every: int = 16,
    snapshot_disk_threshold_bytes: int = 256 << 20,
    sampling_config: dict | None = None,
    dataset_metadata_extra: dict | None = None,
) -> PipelineResult:
    """End-to-end fixed-geometry pipeline without transient supervision labels."""
    work = Path(work_directory)
    work.mkdir(parents=True, exist_ok=True)
    state_lower = np.asarray(state_lower, dtype=float)
    state_upper = np.asarray(state_upper, dtype=float)
    operating_lower = np.asarray(operating_lower, dtype=float)
    operating_upper = np.asarray(operating_upper, dtype=float)
    geometry_lower = np.empty(0)
    geometry_upper = np.empty(0)
    signature = physical_signature or fixed_research_physical_signature(physical_model)
    physical_provenance = physical_dataset_metadata(physical_model)
    sampled = _sample_locations(
        physical_vector_field=fixed_research_vector_field_factory(physical_model),
        state_lower=state_lower,
        state_upper=state_upper,
        geometry_lower=geometry_lower,
        geometry_upper=geometry_upper,
        operating_lower=operating_lower,
        operating_upper=operating_upper,
        n_samples=n_snapshots,
        seed=seed,
        sampling_config=sampling_config,
    )
    dataset = generate_snapshots_resumable(
        sampled.states,
        sampled.geometries,
        fixed_research_tensor_factory(physical_model),
        checkpoint_path=work / "quadratic_joule.partial.npz",
        checkpoint_every=checkpoint_every,
        max_workers=snapshot_workers,
        split_seed=seed,
        metadata=_dataset_metadata(
            kind="fixed",
            signature=signature,
            state_lower=state_lower,
            state_upper=state_upper,
            geometry_lower=geometry_lower,
            geometry_upper=geometry_upper,
            operating_lower=operating_lower,
            operating_upper=operating_upper,
            sampling_report=sampled.report,
            physical_provenance=physical_provenance,
            extra_metadata=dataset_metadata_extra,
        ),
        final_path=work / "quadratic_joule_dataset.npz",
        disk_backed_threshold_bytes=int(snapshot_disk_threshold_bytes),
    )
    pod = _fit_pod(
        dataset,
        rank=pod_rank,
        relative_tail_tolerance=pod_relative_tail_tolerance,
        pod_config=pod_config,
    )
    surrogate, report = _train_from_dataset(
        dataset,
        pod,
        operating_lower=operating_lower,
        operating_upper=operating_upper,
        network_config=network_config,
        training_config=training_config,
    )
    domain = {
        "state_lower": state_lower,
        "state_upper": state_upper,
        "geometry_lower": geometry_lower,
        "geometry_upper": geometry_upper,
        "operating_lower": operating_lower,
        "operating_upper": operating_upper,
    }
    base_metadata = _dataset_artifact_metadata(dataset)
    model = StructurePreservingNeuralElectroThermalROM(
        surrogate,
        fixed_research_thermal_family(physical_model),
        physical_signature=signature,
        training_domain=domain,
        artifact_metadata=base_metadata,
        thermal_rhs_forcing=fixed_research_thermal_rhs_forcing(physical_model),
    )
    if save_model:
        model.save(
            work / "neural_electrothermal_rom.npz",
            metadata={
                "training_report": report.__dict__,
                "pod_rank": pod.rank,
                "sampling": _report_metadata(sampled.report),
            },
        )
    return PipelineResult(dataset, pod, surrogate, report, model, signature, sampled.report)


def build_geometry_neural_rom(
    geometry_model,
    *,
    state_lower,
    state_upper,
    operating_lower,
    operating_upper,
    n_snapshots: int,
    work_directory: str | Path,
    seed: int = 0,
    pod_rank: int | None = None,
    pod_relative_tail_tolerance: float = 1e-4,
    pod_config: dict | None = None,
    network_config: ResidualMLPConfig | dict | None = None,
    training_config: NeuralTrainingConfig | dict | None = None,
    physical_signature: str | None = None,
    save_model: bool = True,
    snapshot_workers: int = 1,
    checkpoint_every: int = 16,
    snapshot_disk_threshold_bytes: int = 256 << 20,
    thermal_cache_size: int = 64,
    sampling_config: dict | None = None,
    dataset_metadata_extra: dict | None = None,
) -> PipelineResult:
    """End-to-end geometry-family pipeline using normalized geometry coordinates."""
    work = Path(work_directory)
    work.mkdir(parents=True, exist_ok=True)
    state_lower = np.asarray(state_lower, dtype=float)
    state_upper = np.asarray(state_upper, dtype=float)
    operating_lower = np.asarray(operating_lower, dtype=float)
    operating_upper = np.asarray(operating_upper, dtype=float)
    signature = physical_signature or geometry_research_physical_signature(geometry_model)
    physical_provenance = physical_dataset_metadata(geometry_model)
    n_geometry = len(geometry_model.geometry_names)
    geometry_lower = -np.ones(n_geometry)
    geometry_upper = np.ones(n_geometry)
    sampled = _sample_locations(
        physical_vector_field=geometry_research_vector_field_factory(
            geometry_model,
            normalized_geometry=True,
        ),
        state_lower=state_lower,
        state_upper=state_upper,
        geometry_lower=geometry_lower,
        geometry_upper=geometry_upper,
        operating_lower=operating_lower,
        operating_upper=operating_upper,
        n_samples=n_snapshots,
        seed=seed,
        sampling_config=sampling_config,
    )
    dataset = generate_snapshots_resumable(
        sampled.states,
        sampled.geometries,
        geometry_research_tensor_factory(geometry_model, normalized_geometry=True),
        checkpoint_path=work / "quadratic_joule.partial.npz",
        checkpoint_every=checkpoint_every,
        max_workers=snapshot_workers,
        split_seed=seed,
        metadata=_dataset_metadata(
            kind="geometry",
            signature=signature,
            state_lower=state_lower,
            state_upper=state_upper,
            geometry_lower=geometry_lower,
            geometry_upper=geometry_upper,
            operating_lower=operating_lower,
            operating_upper=operating_upper,
            sampling_report=sampled.report,
            physical_provenance=physical_provenance,
            extra_metadata=dataset_metadata_extra,
        ),
        final_path=work / "quadratic_joule_dataset.npz",
        disk_backed_threshold_bytes=int(snapshot_disk_threshold_bytes),
    )
    pod = _fit_pod(
        dataset,
        rank=pod_rank,
        relative_tail_tolerance=pod_relative_tail_tolerance,
        pod_config=pod_config,
    )
    surrogate, report = _train_from_dataset(
        dataset,
        pod,
        operating_lower=operating_lower,
        operating_upper=operating_upper,
        network_config=network_config,
        training_config=training_config,
    )
    domain = {
        "state_lower": state_lower,
        "state_upper": state_upper,
        "geometry_lower": geometry_lower,
        "geometry_upper": geometry_upper,
        "operating_lower": operating_lower,
        "operating_upper": operating_upper,
    }
    base_metadata = _dataset_artifact_metadata(dataset)
    model = StructurePreservingNeuralElectroThermalROM(
        surrogate,
        geometry_research_embedded_thermal_family(
            geometry_model,
            cache_size=thermal_cache_size,
        ),
        physical_signature=signature,
        training_domain=domain,
        artifact_metadata=base_metadata,
        thermal_rhs_forcing=geometry_research_thermal_rhs_forcing(geometry_model),
    )
    if save_model:
        model.save(
            work / "neural_electrothermal_rom.npz",
            metadata={
                "training_report": report.__dict__,
                "pod_rank": pod.rank,
                "sampling": _report_metadata(sampled.report),
            },
        )
    return PipelineResult(dataset, pod, surrogate, report, model, signature, sampled.report)


__all__ = [
    "PipelineResult",
    "build_fixed_neural_rom",
    "build_geometry_neural_rom",
    "retrain_neural_rom",
]
