"""Minimal construction pipeline for the structure-preserving neural electrothermal ROM."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .adapters import (
    fixed_research_tensor_factory,
    fixed_research_thermal_family,
    fixed_research_thermal_rhs_forcing,
    geometry_research_embedded_thermal_family,
    geometry_research_tensor_factory,
    geometry_research_thermal_rhs_forcing,
)
from .dataset import latin_hypercube_box
from .generator import generate_snapshots_resumable
from .model import StructurePreservingNeuralElectroThermalROM
from .network import ResidualMLPConfig
from .pod import fit_dataset_pod
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


def _network_config(value, *, input_dimension: int, output_dimension: int):
    if value is None:
        return ResidualMLPConfig(input_dimension=input_dimension, output_dimension=output_dimension)
    if isinstance(value, ResidualMLPConfig):
        if value.input_dimension != input_dimension or value.output_dimension != output_dimension:
            raise ValueError("network dimensions do not match dataset/POD")
        return value
    settings = dict(value)
    settings.pop("input_dimension", None)
    settings.pop("output_dimension", None)
    return ResidualMLPConfig(
        input_dimension=input_dimension,
        output_dimension=output_dimension,
        **settings,
    )


def _training_config(value):
    if value is None or isinstance(value, NeuralTrainingConfig):
        return value
    return NeuralTrainingConfig(**dict(value))


def _fit_pod(dataset, *, rank, tolerance, pod_config=None):
    settings = {} if pod_config is None else dict(pod_config)
    allowed = {"exact_svd_max_bytes", "out_of_core_max_rank", "chunk_rows"}
    unknown = sorted(set(settings) - allowed)
    if unknown:
        raise ValueError(f"unknown POD options: {unknown}")
    return fit_dataset_pod(
        dataset,
        rank=rank,
        relative_tail_tolerance=tolerance,
        **settings,
    )


def _train(
    dataset,
    pod,
    operating_lower,
    operating_upper,
    network_config,
    training_config,
    *,
    device: str | None = None,
):
    network = _network_config(
        network_config,
        input_dimension=dataset.thermal_rank + dataset.geometry_dimension,
        output_dimension=pod.rank,
    )
    return train_tensor_surrogate(
        dataset,
        pod,
        operating_lower=np.asarray(operating_lower, dtype=float),
        operating_upper=np.asarray(operating_upper, dtype=float),
        network_config=network,
        training_config=_training_config(training_config),
        device=device,
    )


def _metadata(kind, signature, state_lower, state_upper, geometry_lower, geometry_upper, operating_lower, operating_upper):
    return {
        "kind": kind,
        "physical_signature": str(signature),
        "state_lower": np.asarray(state_lower, dtype=float).tolist(),
        "state_upper": np.asarray(state_upper, dtype=float).tolist(),
        "geometry_lower": np.asarray(geometry_lower, dtype=float).tolist(),
        "geometry_upper": np.asarray(geometry_upper, dtype=float).tolist(),
        "operating_lower": np.asarray(operating_lower, dtype=float).tolist(),
        "operating_upper": np.asarray(operating_upper, dtype=float).tolist(),
    }


def _training_domain(dataset):
    meta = dict(dataset.metadata)
    names = (
        "state_lower",
        "state_upper",
        "geometry_lower",
        "geometry_upper",
        "operating_lower",
        "operating_upper",
    )
    missing = [name for name in names if name not in meta]
    if missing:
        raise ValueError(f"dataset is missing domain metadata: {missing}")
    return {name: np.asarray(meta[name], dtype=float) for name in names}


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
    device: str | None = None,
    save_model: bool = True,
    model_filename: str = "neural_electrothermal_rom.retrained.npz",
) -> PipelineResult:
    """Retrain POD/MLP from an existing tensor dataset without rebuilding EM physics."""
    if dataset.thermal_rank != template_model.surrogate.state_dimension:
        raise ValueError("dataset/template thermal dimensions differ")
    if dataset.geometry_dimension != template_model.surrogate.geometry_dimension:
        raise ValueError("dataset/template geometry dimensions differ")
    if dataset.current_dimension != template_model.surrogate.pod.current_dimension:
        raise ValueError("dataset/template current dimensions differ")
    signature = dataset.metadata.get("physical_signature")
    if signature is None or str(signature) != str(template_model.physical_signature):
        raise ValueError("dataset/template physical signatures differ")

    domain = _training_domain(dataset)
    pod = _fit_pod(
        dataset,
        rank=pod_rank,
        tolerance=pod_relative_tail_tolerance,
        pod_config=pod_config,
    )
    surrogate, report = _train(
        dataset,
        pod,
        domain["operating_lower"],
        domain["operating_upper"],
        network_config,
        training_config,
        device=device,
    )
    model = StructurePreservingNeuralElectroThermalROM(
        surrogate,
        template_model.thermal_operators,
        physical_signature=str(signature),
        training_domain=domain,
        artifact_metadata={"dataset_hash": dataset.manifest().dataset_hash},
        thermal_rhs_forcing=template_model.field.thermal_rhs_forcing,
    )
    work = Path(work_directory)
    work.mkdir(parents=True, exist_ok=True)
    if save_model:
        model.save(
            work / model_filename,
            metadata={"training_report": report.__dict__, "pod_rank": pod.rank},
        )
    return PipelineResult(dataset, pod, surrogate, report, model, str(signature))


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
    device: str | None = None,
    save_model: bool = True,
    snapshot_workers: int = 1,
    checkpoint_every: int = 16,
    snapshot_disk_threshold_bytes: int = 256 << 20,
) -> PipelineResult:
    """Generate quadratic-Joule labels, fit POD/MLP, and build a fixed-geometry neural ROM."""
    work = Path(work_directory)
    work.mkdir(parents=True, exist_ok=True)
    state_lower = np.asarray(state_lower, dtype=float)
    state_upper = np.asarray(state_upper, dtype=float)
    operating_lower = np.asarray(operating_lower, dtype=float)
    operating_upper = np.asarray(operating_upper, dtype=float)
    geometry_lower = np.empty(0)
    geometry_upper = np.empty(0)
    signature = fixed_research_physical_signature(physical_model)

    states = latin_hypercube_box(state_lower, state_upper, n_snapshots, seed=seed)
    geometries = np.empty((int(n_snapshots), 0), dtype=float)
    dataset = generate_snapshots_resumable(
        states,
        geometries,
        fixed_research_tensor_factory(physical_model),
        checkpoint_path=work / "quadratic_joule.partial.npz",
        final_path=work / "quadratic_joule_dataset.npz",
        checkpoint_every=checkpoint_every,
        max_workers=snapshot_workers,
        split_seed=seed,
        disk_backed_threshold_bytes=int(snapshot_disk_threshold_bytes),
        metadata=_metadata(
            "fixed",
            signature,
            state_lower,
            state_upper,
            geometry_lower,
            geometry_upper,
            operating_lower,
            operating_upper,
        ),
    )
    pod = _fit_pod(dataset, rank=pod_rank, tolerance=pod_relative_tail_tolerance, pod_config=pod_config)
    surrogate, report = _train(
        dataset,
        pod,
        operating_lower,
        operating_upper,
        network_config,
        training_config,
        device=device,
    )
    domain = _training_domain(dataset)
    model = StructurePreservingNeuralElectroThermalROM(
        surrogate,
        fixed_research_thermal_family(physical_model),
        physical_signature=signature,
        training_domain=domain,
        artifact_metadata={"dataset_hash": dataset.manifest().dataset_hash},
        thermal_rhs_forcing=fixed_research_thermal_rhs_forcing(physical_model),
    )
    if save_model:
        model.save(
            work / "neural_electrothermal_rom.npz",
            metadata={"training_report": report.__dict__, "pod_rank": pod.rank},
        )
    return PipelineResult(dataset, pod, surrogate, report, model, signature)


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
    device: str | None = None,
    save_model: bool = True,
    snapshot_workers: int = 1,
    checkpoint_every: int = 16,
    snapshot_disk_threshold_bytes: int = 256 << 20,
    thermal_cache_size: int = 64,
) -> PipelineResult:
    """Generate labels over state/geometry boxes and build the geometry-family neural ROM."""
    work = Path(work_directory)
    work.mkdir(parents=True, exist_ok=True)
    state_lower = np.asarray(state_lower, dtype=float)
    state_upper = np.asarray(state_upper, dtype=float)
    operating_lower = np.asarray(operating_lower, dtype=float)
    operating_upper = np.asarray(operating_upper, dtype=float)
    n_geometry = len(geometry_model.geometry_names)
    geometry_lower = -np.ones(n_geometry)
    geometry_upper = np.ones(n_geometry)
    signature = geometry_research_physical_signature(geometry_model)

    states = latin_hypercube_box(state_lower, state_upper, n_snapshots, seed=seed)
    geometries = latin_hypercube_box(geometry_lower, geometry_upper, n_snapshots, seed=seed + 1)
    dataset = generate_snapshots_resumable(
        states,
        geometries,
        geometry_research_tensor_factory(geometry_model, normalized_geometry=True),
        checkpoint_path=work / "quadratic_joule.partial.npz",
        final_path=work / "quadratic_joule_dataset.npz",
        checkpoint_every=checkpoint_every,
        max_workers=snapshot_workers,
        split_seed=seed,
        disk_backed_threshold_bytes=int(snapshot_disk_threshold_bytes),
        metadata=_metadata(
            "geometry",
            signature,
            state_lower,
            state_upper,
            geometry_lower,
            geometry_upper,
            operating_lower,
            operating_upper,
        ),
    )
    pod = _fit_pod(dataset, rank=pod_rank, tolerance=pod_relative_tail_tolerance, pod_config=pod_config)
    surrogate, report = _train(
        dataset,
        pod,
        operating_lower,
        operating_upper,
        network_config,
        training_config,
        device=device,
    )
    domain = _training_domain(dataset)
    model = StructurePreservingNeuralElectroThermalROM(
        surrogate,
        geometry_research_embedded_thermal_family(geometry_model, cache_size=thermal_cache_size),
        physical_signature=signature,
        training_domain=domain,
        artifact_metadata={"dataset_hash": dataset.manifest().dataset_hash},
        thermal_rhs_forcing=geometry_research_thermal_rhs_forcing(geometry_model),
    )
    if save_model:
        model.save(
            work / "neural_electrothermal_rom.npz",
            metadata={"training_report": report.__dict__, "pod_rank": pod.rank},
        )
    return PipelineResult(dataset, pod, surrogate, report, model, signature)


__all__ = [
    "PipelineResult",
    "build_fixed_neural_rom",
    "build_geometry_neural_rom",
    "retrain_neural_rom",
]
