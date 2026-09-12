"""CLI for reproducible physical reachable-state domain probing."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .cli import _adapters, _build_physical_model, _path_from_config, _read_json, _write_json
from .domain import probe_reachable_state_domain


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
    _write_json(
        output,
        {
            "kind": "reachable_state_domain",
            "physical_config": str(physical_config),
            "physical_signature": adapters["signature"],
            "report": report,
        },
    )
    print(f"state-domain report saved: {output}")
    print(f"successful steady states: {report.successful_steady_states}/{report.trajectory_count}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Probe a neural thermal-state domain using the real reduced physics"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    probe = sub.add_parser("probe")
    probe.add_argument("--config", required=True, help="JSON configuration for physical domain probing")
    args = parser.parse_args(argv)
    if args.command == "probe":
        return command_probe(args.config)
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
