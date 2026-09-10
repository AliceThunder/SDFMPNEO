"""python -m sdfmpneo: train, predict/roll out, and independently validate a current model."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass, replace
from pathlib import Path

import numpy as np

from .research import ResearchElectroThermalModel, demo_research_model, model_from_config
from .training.research import ResearchTrainingConfig


def jsonable(value):
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return {"real": value.real.tolist(), "imag": value.imag.tolist()}
        return jsonable(value.tolist())
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and np.isposinf(value):
        return "inf"
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="fit the finite-horizon analytic flow from governing residuals")
    group = train.add_mutually_exclusive_group(required=True)
    group.add_argument("--demo", action="store_true")
    group.add_argument("--config", type=Path)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--residual-tolerance", type=float)

    for command in ("predict", "validate"):
        sub = commands.add_parser(command)
        sub.add_argument("model", type=Path)
        sub.add_argument("--a0", nargs="+", type=float, required=True)
        sub.add_argument("--operating", nargs="*", type=float, default=[])
        sub.add_argument("--times", nargs="+", type=float, required=True)
        sub.add_argument("--output", type=Path)
        if command == "predict":
            sub.add_argument("--state-only", action="store_true", help="skip EM diagnostics")
            sub.add_argument("--allow-extrapolation", action="store_true")
            sub.add_argument("--geometry", type=Path, help="JSON dictionary of saved geometry inputs")
        else:
            sub.add_argument("--rom-reference", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "train":
        if args.demo:
            model = demo_research_model()
            rank = model.core.thermal_model.rank
            config = ResearchTrainingConfig(
                initial_lower=(0.0,) * rank,
                initial_upper=(2.0,) * rank,
                operating_lower=(1000.0, 0.0),
                operating_upper=(3000.0, 1000.0),
                max_response_time=0.25,
                residual_tolerance=0.002,
                sample_count=16,
                validation_count=16,
                semigroup_sample_count=4,
                semigroup_validation_count=4,
                max_network_depth=3,
                max_channels_per_mode=2,
            )
        else:
            raw = json.loads(args.config.read_text(encoding="utf-8"))
            if raw.get("geometry_family", {}).get("enabled", False):
                from .geometry_research import geometry_model_from_config
                model, config = geometry_model_from_config(args.config)
            else:
                model, config = model_from_config(args.config)
        if args.residual_tolerance is not None:
            config = replace(config, residual_tolerance=args.residual_tolerance)
        report = model.train(
            config,
            progress=lambda n, r, m: print(
                f"iteration={n} rms_joint_residual={r:.6g} max_joint_residual={m:.6g}",
                flush=True,
            ),
        )
        model.save(args.output)
        text = json.dumps(jsonable(report), indent=2, allow_nan=False)
        args.output.with_suffix(".training.json").write_text(text + "\n", encoding="utf-8")
        print(text)
        return 0 if report.numerical_tolerance_met else 2

    model = ResearchElectroThermalModel.load(args.model)
    if args.command == "predict":
        geometry_args = {}
        if hasattr(model, "geometry_names"):
            geometry = (
                dict(zip(model.geometry_names, (model.lower + model.upper) / 2))
                if args.geometry is None
                else json.loads(args.geometry.read_text(encoding="utf-8"))
            )
            geometry_args = {"geometry": geometry}
        result = [
            model.predict(
                t,
                a0=args.a0,
                operating=args.operating,
                diagnostics=not args.state_only,
                allow_extrapolation=args.allow_extrapolation,
                **geometry_args,
            )
            for t in args.times
        ]
    else:
        if hasattr(model, "geometry_names"):
            parser.error("trajectory validation is for fixed-geometry models")
        result = model.validate_trajectory(
            args.times,
            a0=args.a0,
            operating=args.operating,
            full_electromagnetics=not args.rom_reference,
        )
    text = json.dumps(jsonable(result), indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
