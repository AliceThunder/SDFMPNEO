"""Formal frozen Gate 1--7 certification CLI for neural electrothermal ROMs."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .certification import audit_evidence, gate_report_hash, save_audited_model
from .cli import _adapters, _build_physical_model, _path_from_config, _read_json, _write_json


def command_audit(config_file: str | Path) -> int:
    config_path, config = _read_json(config_file)
    physical_config = _path_from_config(config_path, config.get("physical_config"))
    model_path = _path_from_config(config_path, config.get("model_path"))
    dataset_path = _path_from_config(config_path, config.get("dataset_path"))
    output_path = _path_from_config(
        config_path,
        config.get("output"),
        "neural_certification_report.json",
    )
    certified_model = config.get("certified_model_output")
    certified_model_path = (
        None if certified_model is None else _path_from_config(config_path, certified_model)
    )
    audited_model = config.get("audited_model_output")
    audited_model_path = (
        None if audited_model is None else _path_from_config(config_path, audited_model)
    )

    physical = _build_physical_model(physical_config)
    adapters = _adapters(physical)

    from .audit import GateSuiteConfig, TrajectoryAuditCase, run_gate_suite
    from .dataset import QuadraticJouleDataset
    from .gates import ProductionBudgets
    from .model import StructurePreservingNeuralElectroThermalROM

    model = StructurePreservingNeuralElectroThermalROM.load(
        model_path,
        expected_physical_signature=adapters["signature"],
        device=str(config.get("device", "cpu")),
    )
    dataset = QuadraticJouleDataset.load(dataset_path)
    manifest = dataset.manifest()
    if manifest.metadata.get("physical_signature") != adapters["signature"]:
        raise ValueError("dataset physical signature does not match certification physics")
    trained_dataset_hash = model.artifact_metadata.get("dataset_hash")
    if trained_dataset_hash is not None and str(trained_dataset_hash) != manifest.dataset_hash:
        raise ValueError("model was trained from a different frozen tensor dataset")

    cases = tuple(
        TrajectoryAuditCase(
            name=str(item["name"]),
            times=np.asarray(item["times"], dtype=float),
            initial_state=np.asarray(item["initial_state"], dtype=float),
            geometry=np.asarray(
                item.get("geometry", [0.0] * model.surrogate.geometry_dimension),
                dtype=float,
            ),
            operating=np.asarray(item["operating"], dtype=float),
            neural_max_step=float(item["neural_max_step"]),
            long_time_check=bool(item.get("long_time_check", False)),
            neural_method=str(item.get("neural_method", "etd2_adaptive")),
            neural_rtol=float(item.get("neural_rtol", 1e-5)),
            neural_atol=float(item.get("neural_atol", 1e-8)),
            neural_initial_step=(
                None if item.get("neural_initial_step") is None
                else float(item["neural_initial_step"])
            ),
        )
        for item in config["trajectory_cases"]
    )
    report = run_gate_suite(
        model=model,
        dataset=dataset,
        pod=model.surrogate.pod,
        tensor_factory=adapters["tensor"],
        direct_heat_factory=adapters["heat"],
        physical_vector_field=adapters["vector"],
        physical_jacobian_factory=adapters["jacobian"],
        thermal_operators=adapters["thermal"],
        operating_lower=np.asarray(config["operating_lower"], dtype=float),
        operating_upper=np.asarray(config["operating_upper"], dtype=float),
        trajectory_cases=cases,
        budgets=ProductionBudgets(**config["budgets"]),
        temperature_reconstructor=adapters["temperature"],
        config=GateSuiteConfig(**config.get("gate_config", {})),
        reproducible_training=bool(config.get("reproducible_training", False)),
        persistence_roundtrip=bool(config.get("persistence_roundtrip", False)),
    )
    report_hash = gate_report_hash(report)
    evidence = audit_evidence(
        report,
        dataset_hash=manifest.dataset_hash,
        report_path=output_path,
    )
    envelope = {
        "kind": "neural_electrothermal_certification",
        "model": str(model_path),
        "physical_config": str(physical_config),
        "physical_signature": adapters["signature"],
        "dataset": str(dataset_path),
        "dataset_hash": manifest.dataset_hash,
        "gate_report_hash": report_hash,
        "production_ready": bool(report.readiness.ready),
        "evidence": evidence,
        "report": report,
    }
    _write_json(output_path, envelope)

    if audited_model_path is not None:
        save_audited_model(
            model,
            audited_model_path,
            report,
            dataset_hash=manifest.dataset_hash,
            report_path=output_path,
            require_ready=False,
        )
        print(f"audited model saved: {audited_model_path}")
    if certified_model_path is not None:
        if report.readiness.ready:
            save_audited_model(
                model,
                certified_model_path,
                report,
                dataset_hash=manifest.dataset_hash,
                report_path=output_path,
                require_ready=True,
            )
            print(f"certified model saved: {certified_model_path}")
        else:
            print("certified model not written: Gate report is failing or incomplete")

    print(f"certification report saved: {output_path}")
    print(f"gate report hash: {report_hash}")
    print(f"production ready: {report.readiness.ready}")
    return 0 if report.readiness.ready else 2


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run frozen Gate 1--7 certification for a neural electrothermal ROM"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("--config", required=True, help="JSON configuration for formal certification")
    args = parser.parse_args(argv)
    if args.command == "audit":
        return command_audit(args.config)
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
