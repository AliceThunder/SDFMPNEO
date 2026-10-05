"""Small CLI for the structure-preserving neural electrothermal ROM."""
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


def _path(config_path: Path, value, default=None) -> Path:
    raw = default if value is None else value
    if raw is None:
        raise ValueError("required path is missing from configuration")
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _build_physical_model(config_path: Path):
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    geometry = payload.get("geometry_family") or {}
    if bool(geometry.get("enabled", False)):
        from ..geometry_research import geometry_model_from_config

        model, _ = geometry_model_from_config(config_path)
        return model
    from ..research import model_from_config

    model, _ = model_from_config(config_path)
    return model


def _dataset_location(dataset, work: Path) -> str:
    directory = getattr(dataset, "directory", None)
    if directory is not None:
        return str(Path(directory))
    return str(work / "quadratic_joule_dataset.npz")


def command_train(config_file: str | Path) -> int:
    config_path, config = _read_json(config_file)
    physical_config = _path(config_path, config.get("physical_config"))
    physical = _build_physical_model(physical_config)
    work = _path(config_path, config.get("work_directory"), "neural_rom_work")
    model_path = _path(config_path, config.get("model_path"), work / "neural_electrothermal_rom.npz")
    report_path = _path(config_path, config.get("training_report"), work / "training.report.json")
    device = None if config.get("device") is None else str(config["device"])

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
        pod_config=config.get("pod"),
        network_config=config.get("network"),
        training_config=config.get("training"),
        device=device,
        save_model=False,
        snapshot_workers=int(config.get("snapshot_workers", 1)),
        checkpoint_every=int(config.get("checkpoint_every", 16)),
        snapshot_disk_threshold_bytes=int(config.get("snapshot_disk_threshold_bytes", 256 << 20)),
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
        },
    )
    _write_json(
        report_path,
        {
            "model": str(model_path),
            "dataset": _dataset_location(result.dataset, work),
            "pod_rank": result.pod.rank,
            "pod_energy_fraction": result.pod.energy_fraction(),
            "training": result.training_report,
        },
    )
    print(f"model: {model_path}")
    print(f"dataset: {_dataset_location(result.dataset, work)}")
    print(f"report: {report_path}")
    return 0


def command_retrain(config_file: str | Path) -> int:
    """Retrain POD/MLP from existing tensor labels without rebuilding EM physics."""
    config_path, config = _read_json(config_file)
    dataset_path = _path(config_path, config.get("dataset_path"))
    template_path = _path(config_path, config.get("template_model_path"))
    work = _path(config_path, config.get("work_directory"), "neural_rom_retrain")
    model_path = _path(config_path, config.get("model_path"), work / "neural_electrothermal_rom.retrained.npz")
    device = None if config.get("device") is None else str(config["device"])

    from .dataset import QuadraticJouleDataset
    from .model import StructurePreservingNeuralElectroThermalROM
    from .pipeline import retrain_neural_rom

    dataset = QuadraticJouleDataset.load(dataset_path)
    template = StructurePreservingNeuralElectroThermalROM.load(
        template_path,
        expected_physical_signature=dataset.metadata.get("physical_signature"),
        device=device or "cpu",
    )
    result = retrain_neural_rom(
        dataset,
        template,
        work_directory=work,
        pod_rank=config.get("pod_rank"),
        pod_relative_tail_tolerance=float(config.get("pod_relative_tail_tolerance", 1e-4)),
        pod_config=config.get("pod"),
        network_config=config.get("network"),
        training_config=config.get("training"),
        device=device,
        save_model=False,
    )
    result.model.save(
        model_path,
        metadata={
            "dataset_hash": dataset.manifest().dataset_hash,
            "pod_rank": result.pod.rank,
            "training_report": _jsonable(result.training_report),
        },
    )
    print(f"model: {model_path}")
    return 0


def command_predict(config_file: str | Path) -> int:
    config_path, config = _read_json(config_file)
    model_path = _path(config_path, config.get("model_path"))
    output_path = _path(config_path, config.get("output"), "neural_predictions.json")

    from .model import StructurePreservingNeuralElectroThermalROM

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
    method = str(config.get("method", "etd2_adaptive"))
    max_step = float(config.get("max_step", 1.0))
    allow_extrapolation = bool(config.get("allow_extrapolation", False))
    results = []

    for value in config["times"]:
        if isinstance(value, str) and value.lower() in {"inf", "infinity"}:
            results.append(
                {
                    "time": "inf",
                    "result": model.steady_state(
                        initial_guess=initial,
                        geometry=geometry,
                        operating=operating,
                        tolerance=float(config.get("steady_tolerance", 1e-10)),
                        max_iterations=int(config.get("steady_max_iterations", 40)),
                        allow_extrapolation=allow_extrapolation,
                    ),
                }
            )
        else:
            results.append(
                model.predict(
                    float(value),
                    initial_state=initial,
                    geometry=geometry,
                    operating=operating,
                    max_step=max_step,
                    method=method,
                    allow_extrapolation=allow_extrapolation,
                    rtol=float(config.get("rtol", 1e-5)),
                    atol=float(config.get("atol", 1e-8)),
                    initial_step=(None if config.get("initial_step") is None else float(config["initial_step"])),
                )
            )

    _write_json(
        output_path,
        {
            "model": str(model_path),
            "initial_state": initial,
            "geometry": geometry,
            "operating": operating,
            "method": method,
            "results": results,
        },
    )
    print(f"predictions: {output_path}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Structure-preserving neural electrothermal ROM")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "retrain", "predict"):
        command = sub.add_parser(name)
        command.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    if args.command == "train":
        return command_train(args.config)
    if args.command == "retrain":
        return command_retrain(args.config)
    if args.command == "predict":
        return command_predict(args.config)
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
