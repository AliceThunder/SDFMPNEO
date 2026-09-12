"""High-level construction pipeline for the structure-preserving neural ROM."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .adapters import (
    fixed_research_tensor_factory,
    fixed_research_thermal_family,
    geometry_research_embedded_thermal_family,
    geometry_research_tensor_factory,
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


def _sample_state_geometry(
    state_lower,
    state_upper,
    geometry_lower,
    geometry_upper,
    n_samples: int,
    *,
    seed: int,
):
    state_lo = np.asarray(state_lower, dtype=float).reshape(-1)
    state_hi = np.asarray(state_upper, dtype=float).reshape(-1)
    geom_lo = np.asarray(geometry_lower, dtype=float).reshape(-1)
    geom_hi = np.asarray(geometry_upper, dtype=float).reshape(-1)
    if state_lo.shape != state_hi.shape or np.any(~np.isfinite(state_lo + state_hi)) or np.any(state_hi <= state_lo):
        raise ValueError("state sampling bounds must be finite and strictly ordered")
    if geom_lo.shape != geom_hi.shape or np.any(~np.isfinite(geom_lo + geom_hi)) or np.any(geom_hi <= geom_lo):
        raise ValueError("geometry sampling bounds must be finite and strictly ordered")
    lower = np.concatenate([state_lo, geom_lo])
    upper = np.concatenate([state_hi, geom_hi])
    sampled = latin_hypercube_box(lower, upper, int(n_samples), seed=int(seed))
    r = state_lo.size
    return sampled[:, :r], sampled[:, r:]


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
) -> PipelineResult:
    """End-to-end fixed-geometry pipeline without trajectory labels."""
    work = Path(work_directory)
    work.mkdir(parents=True, exist_ok=True)
    signature = physical_signature or fixed_research_physical_signature(physical_model)
    states, geometries = _sample_state_geometry(
        state_lower,
        state_upper,
        np.empty(0),
        np.empty(0),
        n_snapshots,
        seed=seed,
    )
    dataset = generate_snapshots_resumable(
        states,
        geometries,
        fixed_research_tensor_factory(physical_model),
        checkpoint_path=work / "quadratic_joule.partial.npz",
        checkpoint_every=checkpoint_every,
        max_workers=snapshot_workers,
        split_seed=seed,
        metadata={"kind": "fixed", "physical_signature": signature},
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
        "state_lower": np.asarray(state_lower, dtype=float),
        "state_upper": np.asarray(state_upper, dtype=float),
        "geometry_lower": np.empty(0),
        "geometry_upper": np.empty(0),
        "operating_lower": np.asarray(operating_lower, dtype=float),
        "operating_upper": np.asarray(operating_upper, dtype=float),
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
            },
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
    network_config: ResidualMLPConfig | dict | None = None,
    training_config: NeuralTrainingConfig | dict | None = None,
    physical_signature: str | None = None,
    save_model: bool = True,
    snapshot_workers: int = 1,
    checkpoint_every: int = 16,
    thermal_cache_size: int = 64,
) -> PipelineResult:
    """End-to-end geometry-family pipeline using normalized geometry coordinates."""
    work = Path(work_directory)
    work.mkdir(parents=True, exist_ok=True)
    signature = physical_signature or geometry_research_physical_signature(geometry_model)
    n_geometry = len(geometry_model.geometry_names)
    geometry_lower = -np.ones(n_geometry)
    geometry_upper = np.ones(n_geometry)
    states, geometries = _sample_state_geometry(
        state_lower,
        state_upper,
        geometry_lower,
        geometry_upper,
        n_snapshots,
        seed=seed,
    )
    dataset = generate_snapshots_resumable(
        states,
        geometries,
        geometry_research_tensor_factory(geometry_model, normalized_geometry=True),
        checkpoint_path=work / "quadratic_joule.partial.npz",
        checkpoint_every=checkpoint_every,
        max_workers=snapshot_workers,
        split_seed=seed,
        metadata={"kind": "geometry", "physical_signature": signature},
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
        "state_lower": np.asarray(state_lower, dtype=float),
        "state_upper": np.asarray(state_upper, dtype=float),
        "geometry_lower": geometry_lower,
        "geometry_upper": geometry_upper,
        "operating_lower": np.asarray(operating_lower, dtype=float),
        "operating_upper": np.asarray(operating_upper, dtype=float),
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
            },
        )
    return PipelineResult(dataset, pod, surrogate, report, model, signature)


__all__ = ["PipelineResult", "build_fixed_neural_rom", "build_geometry_neural_rom"]
