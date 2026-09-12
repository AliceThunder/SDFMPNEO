"""Optional true-EM output diagnostics for neural electrothermal trajectories."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .cli import _build_physical_model, _path_from_config, _read_json
from .diagnostics import diagnose_neural_state, diagnostics_jsonable
from .model import StructurePreservingNeuralElectroThermalROM


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(diagnostics_jsonable(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def command_diagnose(config_file: str | Path) -> int:
    config_path, config = _read_json(config_file)
    physical_config = _path_from_config(config_path, config.get("physical_config"))
    model_path = _path_from_config(config_path, config.get("model_path"))
    output = _path_from_config(
        config_path,
        config.get("output"),
        "neural_em_diagnostics.json",
    )
    physical = _build_physical_model(physical_config)
    from .signatures import fixed_research_physical_signature, geometry_research_physical_signature

    signature = (
        geometry_research_physical_signature(physical)
        if hasattr(physical, "geometry_names")
        else fixed_research_physical_signature(physical)
    )
    model = StructurePreservingNeuralElectroThermalROM.load(
        model_path,
        expected_physical_signature=signature,
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
    rtol = float(config.get("rtol", 1e-5))
    atol = float(config.get("atol", 1e-8))
    initial_step = config.get("initial_step")
    initial_step = None if initial_step is None else float(initial_step)
    allow_extrapolation = bool(config.get("allow_extrapolation", False))
    requested_impedance_error = float(config.get("requested_impedance_error", 1e-6))

    rows = []
    for query_time in config["times"]:
        if isinstance(query_time, str) and query_time.lower() in {"inf", "infinity"}:
            steady = model.steady_state(
                initial_guess=initial,
                geometry=geometry,
                operating=operating,
                tolerance=float(config.get("steady_tolerance", 1e-10)),
                max_iterations=int(config.get("steady_max_iterations", 40)),
                allow_extrapolation=allow_extrapolation,
            )
            state = steady.state
            time_value = "inf"
            prediction = {"steady": steady}
        else:
            finite_time = float(query_time)
            pred = model.predict(
                finite_time,
                initial_state=initial,
                geometry=geometry,
                operating=operating,
                max_step=max_step,
                method=method,
                allow_extrapolation=allow_extrapolation,
                rtol=rtol,
                atol=atol,
                initial_step=initial_step,
            )
            state = pred.state
            time_value = finite_time
            prediction = pred
        diagnostic = diagnose_neural_state(
            model,
            physical,
            state=state,
            geometry=geometry,
            operating=operating,
            requested_impedance_error=requested_impedance_error,
        )
        rows.append(
            {
                "time": time_value,
                "prediction": prediction,
                "em_diagnostics": diagnostic,
            }
        )

    payload = {
        "model": str(model_path),
        "physical_config": str(physical_config),
        "physical_signature": signature,
        "geometry": geometry,
        "operating": operating,
        "requested_impedance_error": requested_impedance_error,
        "results": rows,
    }
    _write(output, payload)
    print(f"EM diagnostics saved: {output}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate true reduced-EM outputs on neural-predicted thermal states"
    )
    parser.add_argument("--config", required=True, help="JSON diagnostic configuration")
    args = parser.parse_args(argv)
    return command_diagnose(args.config)


if __name__ == "__main__":
    raise SystemExit(main())
