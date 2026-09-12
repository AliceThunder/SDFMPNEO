"""CLI for reproducible physical reachable-state domain design and training."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .cli import (
    _adapters,
    _build_physical_model,
    _jsonable,
    _path_from_config,
    _read_json,
    _write_json,
)
from .domain import probe_reachable_state_domain
from .physical_metadata import physical_dataset_metadata
from .provenance import state_domain_report_hash, verified_state_domain_report


def command_probe(config_file: str | Path) -> int:
    config_path, config = _read_json(config_file)
    physical_config = _path_from_config(config_path, config.get("physical_config"))
    output = _path_from_config(
        config_path,
        config.get("output"),
        "state_domain.report.json",
    )
    physical = _build_physical_model(physical_config)
    adapters = _adapters(physical)
    if adapters["kind"] == "geometry":
        n_geometry = len(physical.geometry_names)
        geometry_lower = -np.ones(n_geometry, dtype=float)
        geometry_upper = np.ones(n_geometry, dtype=float)
    else:
        geometry_lower = np.empty(0, dtype=float)
        geometry_upper = np.empty(0, dtype=float)

    report = probe_reachable_state_domain(
        adapters["vector"],
        initial_lower=np.asarray(config["initial_lower"], dtype=float),
        initial_upper=np.asarray(config["initial_upper"], dtype=float),
        geometry_lower=geometry_lower,
        geometry_upper=geometry_upper,
        operating_lower=np.asarray(config["operating_lower"], dtype=float),
        operating_upper=np.asarray(config["operating_upper"], dtype=float),
        trajectory_count=int(config.get("trajectory_count", 64)),
        samples_per_trajectory=int(config.get("samples_per_trajectory", 24)),
        time_horizon=float(config["time_horizon"]),
        time_min=float(config.get("time_min", 1e-6)),
        seed=int(config.get("seed", 0)),
        rtol=float(config.get("rtol", 1e-8)),
        atol=float(config.get("atol", 1e-10)),
        include_steady_state=bool(config.get("include_steady_state", True)),
        physical_jacobian_factory=adapters["jacobian"],
        steady_residual_tolerance=float(config.get("steady_residual_tolerance", 1e-9)),
        steady_maxfev=int(config.get("steady_maxfev", 400)),
        margin_fraction=float(config.get("margin_fraction", 0.10)),
        absolute_margin=np.asarray(config.get("absolute_margin", 1e-8), dtype=float),
        physical_signature=adapters["signature"],
    )
    report_hash = state_domain_report_hash(report)
    physical_provenance = physical_dataset_metadata(physical)
    _write_json(
        output,
        {
            "kind": "reachable_state_domain",
            "physical_config": str(physical_config),
            "physical_signature": adapters["signature"],
            "report_hash": report_hash,
            "physical_provenance": physical_provenance,
            "report": report,
        },
    )
    print(f"state-domain report saved: {output}")
    print(f"state-domain hash: {report_hash}")
    print(f"successful steady states: {report.successful_steady_states}/{report.trajectory_count}")
    return 0


def command_train(config_file: str | Path) -> int:
    """Train from a frozen, signature-checked and hash-checked state-domain report."""
    config_path, config = _read_json(config_file)
    physical_config = _path_from_config(config_path, config.get("physical_config"))
    domain_report = _path_from_config(config_path, config.get("state_domain_report"))
    _, domain_hash = verified_state_domain_report(domain_report)
    work = _path_from_config(config_path, config.get("work_directory"), "neural_rom_work")
    model_path = _path_from_config(
        config_path,
        config.get("model_path"),
        work / "neural_electrothermal_rom.npz",
    )
    training_report_path = _path_from_config(
        config_path,
        config.get("training_report"),
        work / "training.report.json",
    )
    physical = _build_physical_model(physical_config)
    physical_provenance = physical_dataset_metadata(physical)
    operating_lower = np.asarray(config["operating_lower"], dtype=float)
    operating_upper = np.asarray(config["operating_upper"], dtype=float)
    common = dict(
        operating_lower=operating_lower,
        operating_upper=operating_upper,
        n_snapshots=int(config["n_snapshots"]),
        work_directory=work,
        seed=int(config.get("seed", 0)),
        pod_rank=config.get("pod_rank"),
        pod_relative_tail_tolerance=float(config.get("pod_relative_tail_tolerance", 1e-4)),
        network_config=config.get("network"),
        training_config=config.get("training"),
        save_model=False,
        snapshot_workers=int(config.get("snapshot_workers", 1)),
        checkpoint_every=int(config.get("checkpoint_every", 16)),
        sampling_config=config.get("sampling"),
    )
    from .dataset import QuadraticJouleDataset
    from .domain_pipeline import (
        build_fixed_neural_rom_from_domain_report,
        build_geometry_neural_rom_from_domain_report,
    )

    if hasattr(physical, "geometry_names"):
        result = build_geometry_neural_rom_from_domain_report(
            physical,
            domain_report,
            thermal_cache_size=int(config.get("thermal_cache_size", 64)),
            **common,
        )
    else:
        result = build_fixed_neural_rom_from_domain_report(
            physical,
            domain_report,
            **common,
        )

    dataset_metadata = dict(result.dataset.metadata)
    dataset_metadata.update(
        {
            "state_domain_report": str(domain_report),
            "state_domain_report_hash": domain_hash,
            "physical_provenance": physical_provenance,
        }
    )
    dataset = QuadraticJouleDataset(
        states=result.dataset.states,
        geometries=result.dataset.geometries,
        outputs=result.dataset.outputs,
        split=result.dataset.split,
        thermal_rank=result.dataset.thermal_rank,
        current_dimension=result.dataset.current_dimension,
        metadata=dataset_metadata,
    )
    dataset.save(work / "quadratic_joule_dataset.npz")
    manifest = dataset.manifest()

    result.model.save(
        model_path,
        metadata={
            "dataset_hash": manifest.dataset_hash,
            "pod_rank": result.pod.rank,
            "training_report": _jsonable(result.training_report),
            "sampling": _jsonable(result.sampling_report),
            "state_domain_report": str(domain_report),
            "state_domain_report_hash": domain_hash,
            "physical_provenance": physical_provenance,
            "physical_config": str(physical_config),
        },
    )
    _write_json(
        training_report_path,
        {
            "model": str(model_path),
            "work_directory": str(work),
            "physical_signature": result.physical_signature,
            "state_domain_report": str(domain_report),
            "state_domain_report_hash": domain_hash,
            "physical_provenance": physical_provenance,
            "dataset_manifest": manifest,
            "sampling": result.sampling_report,
            "pod_rank": result.pod.rank,
            "pod_energy_fraction": result.pod.energy_fraction(),
            "training": result.training_report,
        },
    )
    print(f"neural ROM saved: {model_path}")
    print(f"dataset hash: {manifest.dataset_hash}")
    print(f"state-domain hash: {domain_hash}")
    print(f"training report: {training_report_path}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Design and consume a physical reachable thermal-state domain"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    probe = sub.add_parser("probe")
    probe.add_argument("--config", required=True, help="JSON configuration for physical domain probing")
    train = sub.add_parser("train")
    train.add_argument("--config", required=True, help="JSON training configuration with state_domain_report")
    args = parser.parse_args(argv)
    if args.command == "probe":
        return command_probe(args.config)
    if args.command == "train":
        return command_train(args.config)
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
