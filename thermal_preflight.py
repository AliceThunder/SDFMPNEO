"""Run only the deterministic thermal-ROM construction/audit and exit.

This diagnostic entry point intentionally skips the expensive EM spatial preflight
so thermal basis changes can be timed independently::

    python thermal_preflight.py

It uses the exact same training/validation geometry sampling, thermal tolerances,
time scales and trajectory audit as ``python run.py --mode train``. It does not
create a production model or mark the EM preflight as certified.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json

import numpy as np

import run as production
from sdfmpneo import unified_runtime as runtime
from sdfmpneo.unified_thermal import build_geometry_aware_thermal_library


def main() -> int:
    settings = production.SETTINGS
    root = Path(settings["ROOT"])
    settings_dir = Path(settings["FILES"]["settings_dir"])
    if not settings_dir.is_absolute():
        settings_dir = root / settings_dir
    settings_dir.mkdir(parents=True, exist_ok=True)

    print("构建 thermal diagnostic background……0%", flush=True)
    background = runtime.build_background(settings)
    print(
        f"背景空间：{background.n_cells} cells；仅执行 geometry-aware thermal ROM diagnostic。",
        flush=True,
    )

    seed = int(settings["TRAINING"].get("seed", 17))
    rng = np.random.default_rng(seed)
    n_basis = int(settings["TRAINING"].get("basis_samples", 8))
    n_basis_val = int(settings["TRAINING"].get("basis_validation_samples", 6))
    basis_geometries = runtime._sample_geometries(settings, n_basis, rng, background)
    validation_geometries = runtime._sample_geometries(
        settings, n_basis_val, rng, background
    )

    _library, report = build_geometry_aware_thermal_library(
        background,
        settings["DEFAULT_GEOMETRY"],
        basis_geometries,
        validation_geometries=validation_geometries,
        target_relative_error=float(
            settings["TRAINING"].get("thermal_basis_energy_tolerance", 5e-2)
        ),
        time_scales=settings["TRAINING"].get(
            "thermal_time_scales", [0.1, 1.0, 10.0]
        ),
        trajectory_times=settings["TRAINING"].get("thermal_trajectory_times"),
        maximum_rank=settings["TRAINING"].get("thermal_basis_max_rank"),
        conditioning_limit=float(
            settings["TRAINING"].get("thermal_basis_conditioning_limit", 1e10)
        ),
        monitor=None,
    )
    payload = asdict(report)
    output = settings_dir / "thermal_preflight.report.json"
    runtime.write_json(output, payload)

    print(
        "thermal diagnostic: "
        f"rank={report.basis_dimension}, train={report.maximum_anchor_relative_energy_error:.3e}, "
        f"validation={report.maximum_validation_relative_energy_error:.3e}, "
        f"trajectory={report.maximum_validation_trajectory_relative_error:.3e}, "
        f"target={report.target_relative_error:.3e}, stop={report.stop_reason}",
        flush=True,
    )
    if report.worst_training_anchor:
        print(
            "worst training anchor: "
            + json.dumps(report.worst_training_anchor, ensure_ascii=False, sort_keys=True),
            flush=True,
        )
    if report.worst_validation_anchor:
        print(
            "worst validation anchor: "
            + json.dumps(report.worst_validation_anchor, ensure_ascii=False, sort_keys=True),
            flush=True,
        )
    if report.worst_validation_trajectory:
        print(
            "worst validation trajectory: "
            + json.dumps(report.worst_validation_trajectory, ensure_ascii=False, sort_keys=True),
            flush=True,
        )
    print(f"报告：{output}", flush=True)
    return 0 if report.converged else 2


if __name__ == "__main__":
    raise SystemExit(main())
