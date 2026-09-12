"""High-level construction pipeline for the structure-preserving neural ROM."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .adapters import (
    fixed_research_tensor_factory,
    fixed_research_thermal_family,
    fixed_research_vector_field_factory,
    geometry_research_embedded_thermal_family,
    geometry_research_tensor_factory,
    geometry_research_vector_field_factory,
)
from .generator import generate_snapshots_resumable
from .model import StructurePreservingNeuralElectroThermalROM
from .network import ResidualMLPConfig
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
    sampling_report: object


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
) -> dict:
    return {
        "kind": kind,
        "physical_signature": signature,
        "state_lower": np.asarray(state_lower, dtype=float).tolist(),
        "state_upper": np.asarray(state_upper, dtype=float).tolist(),
        "geometry_lower": np.asarray(geometry_lower, dtype=float).tolist(),
        "geometry_upper": np.asarray(geometry_upper, dtype=float).tolist(),
        "operating_lower": np.asarray(operating_lower, dtype=float).tolist(),
        "operating_upper": np.asarray(operating_upper, dtype=float).tolist(),
        "sampling": _report_metadata(sampling_report),
    }


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
    network_config: ResidualMLPConfig | dict | None = None,
    training_config: NeuralTrainingConfig | dict | None = None,
    physical_signature: str | None = None,
    save_model: bool = True,
    snapshot_workers: int = 1,
    checkpoint_every: int = 16,
    sampling_config: dict | None = None,
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
        ),
        final_path=work / "quadratic_joule_dataset.npz",
    )
    pod = fit_dataset_pod(
        dataset,
        rank=pod_rank,
        relative_tail_tolerance=pod_relative_tail_tolerance,
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
    model = StructurePreservingNeuralElectroThermalROM(
        surrogate,
        fixed_research_thermal_family(physical_model),
        physical_signature=signature,
        training_domain=domain,
    )
    if save_model:
        model.save(
            work / "neural_electrothermal_rom.npz",
            metadata={
                "dataset_hash": dataset.manifest().dataset_hash,
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
    network_config: ResidualMLPConfig | dict | None = None,
    training_config: NeuralTrainingConfig | dict | None = None,
    physical_signature: str | None = None,
    save_model: bool = True,
    snapshot_workers: int = 1,
    checkpoint_every: int = 16,
    thermal_cache_size: int = 64,
    sampling_config: dict | None = None,
) -> PipelineResult:
    """End-to-end geometry-family pipeline using normalized geometry coordinates."""
    work = Path(work_directory)
    work.mkdir(parents=True, exist_ok=True)
    state_lower = np.asarray(state_lower, dtype=float)
    state_upper = np.asarray(state_upper, dtype=float)
    operating_lower = np.asarray(operating_lower, dtype=float)
    operating_upper = np.asarray(operating_upper, dtype=float)
    signature = physical_signature or geometry_research_physical_signature(geometry_model)
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
        ),
        final_path=work / "quadratic_joule_dataset.npz",
    )
    pod = fit_dataset_pod(
        dataset,
        rank=pod_rank,
        relative_tail_tolerance=pod_relative_tail_tolerance,
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
    model = StructurePreservingNeuralElectroThermalROM(
        surrogate,
        geometry_research_embedded_thermal_family(
            geometry_model,
            cache_size=thermal_cache_size,
        ),
        physical_signature=signature,
        training_domain=domain,
    )
    if save_model:
        model.save(
            work / "neural_electrothermal_rom.npz",
            metadata={
                "dataset_hash": dataset.manifest().dataset_hash,
                "training_report": report.__dict__,
                "pod_rank": pod.rank,
                "sampling": _report_metadata(sampled.report),
            },
        )
    return PipelineResult(dataset, pod, surrogate, report, model, signature, sampled.report)


__all__ = ["PipelineResult", "build_fixed_neural_rom", "build_geometry_neural_rom"]
