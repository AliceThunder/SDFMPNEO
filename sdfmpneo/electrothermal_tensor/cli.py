"""Command line entry point for the structure-preserving neural electrothermal ROM.

The CLI is intentionally independent from the legacy analytic-response trainer.
It consumes the same physical JSON model configuration for initial tensor-label
training and audits, but electromagnetic physics is absent from neural retraining
and online prediction.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path

import numpy as np


def _jsonable(value):
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _read_json(path: str | Path) -> tuple[Path, dict]:
    resolved = Path(path).expanduser().resolve()
    return resolved, json.loads(resolved.read_text(encoding="utf-8"))


def _path_from_config(config_path: Path, value, default=None) -> Path:
    raw = default if value is None else value
    if raw is None:
        raise ValueError("required path is missing from configuration")
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def _write_json(path: str | Path, value) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return target


def _build_physical_model(config_path: str | Path):
    path = Path(config_path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    geometry = payload.get("geometry_family") or {}
    if bool(geometry.get("enabled", False)):
        from ..geometry_research import geometry_model_from_config

        model, _ = geometry_model_from_config(path)
        return model
    from ..research import model_from_config

    model, _ = model_from_config(path)
    return model


def _adapters(physical_model):
    from .adapters import (
        fixed_research_direct_heat_factory,
        fixed_research_jacobian_factory,
        fixed_research_temperature_reconstructor,
        fixed_research_tensor_factory,
        fixed_research_thermal_family,
        fixed_research_vector_field_factory,
        geometry_research_direct_heat_factory,
        geometry_research_jacobian_factory,
        geometry_research_temperature_reconstructor,
        geometry_research_tensor_factory,
        geometry_research_thermal_family,
        geometry_research_vector_field_factory,
    )
    from .signatures import fixed_research_physical_signature, geometry_research_physical_signature

    if hasattr(physical_model, "geometry_names"):
        return {
            "kind": "geometry",
            "tensor": geometry_research_tensor_factory(physical_model, normalized_geometry=True),
            "heat": geometry_research_direct_heat_factory(physical_model, normalized_geometry=True),
            "vector": geometry_research_vector_field_factory(physical_model, normalized_geometry=True),
            "jacobian": geometry_research_jacobian_factory(physical_model, normalized_geometry=True),
            "temperature": geometry_research_temperature_reconstructor(physical_model, normalized_geometry=True),
            "thermal": geometry_research_thermal_family(physical_model, normalized_geometry=True),
            "signature": geometry_research_physical_signature(physical_model),
        }
    return {
        "kind": "fixed",
        "tensor": fixed_research_tensor_factory(physical_model),
        "heat": fixed_research_direct_heat_factory(physical_model),
        "vector": fixed_research_vector_field_factory(physical_model),
        "jacobian": fixed_research_jacobian_factory(physical_model),
        "temperature": fixed_research_temperature_reconstructor(physical_model),
        "thermal": fixed_research_thermal_family(physical_model),
        "signature": fixed_research_physical_signature(physical_model),
    }


def command_train(config_file: str | Path) -> int:
    config_path, config = _read_json(config_file)
    physical_config = _path_from_config(config_path, config.get("physical_config"))
    physical = _build_physical_model(physical_config)
    work = _path_from_config(config_path, config.get("work_directory"), "neural_rom_work")
    model_path = _path_from_config(config_path, config.get("model_path"), work / "neural_electrothermal_rom.npz")
    report_path = _path_from_config(config_path, config.get("training_report"), work / "training.report.json")

    common = dict(
        state_lower=np.asarray(config["state_lower"], dtype=float),
        state_upper=np.asarray(config["state_upper"], dtype=float),
        operating_lower=np.asarray(config["operating_lower"], dtype=float),
        operating_upper=np.asarray(config["operating_upper"], dtype=float),
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
    from .pipeline import build_fixed_neural_rom, build_geometry_neural_rom

    if hasattr(physical, "geometry_names"):
        result = build_geometry_neural_rom(
            physical,
            thermal_cache_size=int(config.get("thermal_cache_size", 64)),
            **common,
        )
    else:
        result = build_fixed_neural_rom(physical, **common)
    result.model.save(
        model_path,
        metadata={
            "dataset_hash": result.dataset.manifest().dataset_hash,
            "pod_rank": result.pod.rank,
            "training_report": _jsonable(result.training_report),
            "sampling": _jsonable(result.sampling_report),
            "physical_config": str(physical_config),
        },
    )
    _write_json(
        report_path,
        {
            "model": str(model_path),
            "work_directory": str(work),
            "physical_signature": result.physical_signature,
            "dataset_manifest": result.dataset.manifest(),
            "sampling": result.sampling_report,
            "pod_rank": result.pod.rank,
            "pod_energy_fraction": result.pod.energy_fraction(),
            "training": result.training_report,
        },
    )
    print(f"neural ROM saved: {model_path}")
    print(f"training report: {report_path}")
    return 0


def command_retrain(config_file: str | Path) -> int:
    """Retrain POD/MLP from frozen tensors with no physical/EM model construction."""
    config_path, config = _read_json(config_file)
    dataset_path = _path_from_config(config_path, config.get("dataset_path"))
    template_path = _path_from_config(config_path, config.get("template_model_path"))
    work = _path_from_config(config_path, config.get("work_directory"), "neural_rom_retrain")
    model_path = _path_from_config(
        config_path,
        config.get("model_path"),
        work / "neural_electrothermal_rom.retrained.npz",
    )
    report_path = _path_from_config(
        config_path,
        config.get("training_report"),
        work / "retraining.report.json",
    )
    from .dataset import QuadraticJouleDataset
    from .model import StructurePreservingNeuralElectroThermalROM
    from .pipeline import retrain_neural_rom

    dataset = QuadraticJouleDataset.load(dataset_path)
    template = StructurePreservingNeuralElectroThermalROM.load(
        template_path,
        expected_physical_signature=dataset.metadata.get("physical_signature"),
        device=str(config.get("device", "cpu")),
    )
    result = retrain_neural_rom(
        dataset,
        template,
        work_directory=work,
        pod_rank=config.get("pod_rank"),
        pod_relative_tail_tolerance=float(config.get("pod_relative_tail_tolerance", 1e-4)),
        network_config=config.get("network"),
        training_config=config.get("training"),
        save_model=False,
    )
    result.model.save(
        model_path,
        metadata={
            "dataset_hash": dataset.manifest().dataset_hash,
            "pod_rank": result.pod.rank,
            "training_report": _jsonable(result.training_report),
            "retrained_without_em": True,
            "template_model": str(template_path),
        },
    )
    _write_json(
        report_path,
        {
            "model": str(model_path),
            "template_model": str(template_path),
            "dataset": str(dataset_path),
            "dataset_manifest": dataset.manifest(),
            "physical_signature": result.physical_signature,
            "pod_rank": result.pod.rank,
            "pod_energy_fraction": result.pod.energy_fraction(),
            "training": result.training_report,
            "retrained_without_em": True,
        },
    )
    print(f"retrained neural ROM saved: {model_path}")
    print("electromagnetic model was not constructed")
    return 0


def command_predict(config_file: str | Path) -> int:
    config_path, config = _read_json(config_file)
    from .model import StructurePreservingNeuralElectroThermalROM

    model_path = _path_from_config(config_path, config.get("model_path"))
    output_path = _path_from_config(config_path, config.get("output"), "neural_predictions.json")
    model = StructurePreservingNeuralElectroThermalROM.load(
        model_path,
        device=str(config.get("device", "cpu")),
    )
    initial = np.asarray(config["initial_state"], dtype=float)
    geometry = np.asarray(
        config.get("geometry", [0.0] * model.surrogate.geometry_dimension),
        dtype=float,
    )
    operating = np.asarray(config["operating"], dtype=float)
    max_step = float(config["max_step"])
    method = str(config.get("method", "etd2_adaptive"))
    allow_extrapolation = bool(config.get("allow_extrapolation", False))
    rtol = float(config.get("rtol", 1e-5))
    atol = float(config.get("atol", 1e-8))
    initial_step = config.get("initial_step")
    initial_step = None if initial_step is None else float(initial_step)
    max_attempts = int(config.get("max_attempts", 100000))
    results = []
    for value in config["times"]:
        if isinstance(value, str) and value.lower() in {"inf", "infinity"}:
            steady = model.steady_state(
                initial_guess=initial,
                geometry=geometry,
                operating=operating,
                tolerance=float(config.get("steady_tolerance", 1e-10)),
                max_iterations=int(config.get("steady_max_iterations", 40)),
                allow_extrapolation=allow_extrapolation,
            )
            results.append({"time": "inf", "steady": steady})
        else:
            prediction = model.predict(
                float(value),
                initial_state=initial,
                geometry=geometry,
                operating=operating,
                max_step=max_step,
                method=method,
                allow_extrapolation=allow_extrapolation,
                rtol=rtol,
                atol=atol,
                initial_step=initial_step,
                max_attempts=max_attempts,
            )
            results.append(prediction)
    _write_json(
        output_path,
        {
            "model": str(model_path),
            "geometry": geometry,
            "operating": operating,
            "initial_state": initial,
            "method": method,
            "rtol": rtol,
            "atol": atol,
            "results": results,
        },
    )
    print(f"predictions saved: {output_path}")
    return 0


def command_audit(config_file: str | Path) -> int:
    config_path, config = _read_json(config_file)
    physical_config = _path_from_config(config_path, config.get("physical_config"))
    model_path = _path_from_config(config_path, config.get("model_path"))
    dataset_path = _path_from_config(config_path, config.get("dataset_path"))
    output_path = _path_from_config(config_path, config.get("output"), "neural_gate_report.json")
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
    if dataset.manifest().metadata.get("physical_signature") != adapters["signature"]:
        raise ValueError("dataset physical signature does not match audit physics")

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
    _write_json(output_path, report)
    print(f"gate report saved: {output_path}")
    print(f"production ready: {report.readiness.ready}")
    return 0 if report.readiness.ready else 2


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Structure-preserving quadratic-current neural electrothermal ROM"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "retrain", "predict", "audit"):
        command = sub.add_parser(name)
        command.add_argument("--config", required=True, help=f"JSON configuration for {name}")
    args = parser.parse_args(argv)
    if args.command == "train":
        return command_train(args.config)
    if args.command == "retrain":
        return command_retrain(args.config)
    if args.command == "predict":
        return command_predict(args.config)
    if args.command == "audit":
        return command_audit(args.config)
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
