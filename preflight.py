"""Run only the production spatial-truth preflight and exit.

Use this for physics/cost validation without constructing the thermal library,
generating the 96-geometry tensor dataset, or training the neural surrogate::

    python preflight.py

The same SETTINGS, background builder, geometry sampler and installed physics
adapters as ``python run.py --mode train`` are used.
"""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np

import run as production
from sdfmpneo import unified_runtime as runtime


def main() -> int:
    settings = production.SETTINGS
    root = Path(settings["ROOT"])
    settings_dir = Path(settings["FILES"]["settings_dir"])
    if not settings_dir.is_absolute():
        settings_dir = root / settings_dir
    settings_dir.mkdir(parents=True, exist_ok=True)

    print("构建开放边界固定背景物理空间……0%", flush=True)
    background = runtime.build_background(settings)
    print("构建开放边界固定背景物理空间……5%", flush=True)
    print(
        f"背景空间：{background.n_cells} cells，{background.n_edges} Maxwell edge DOFs；"
        "仅执行 spatial truth preflight。",
        flush=True,
    )

    seed = int(settings["TRAINING"].get("seed", 17))
    rng = np.random.default_rng(seed + 65537)
    geometries = runtime._sample_geometries(
        settings,
        runtime._gate_sample_count(settings),
        rng,
        background,
    )
    print("执行 pre-basis spatial truth preflight……6%", flush=True)
    report = runtime.run_truth_preflight(settings, background, geometries, monitor=None)
    output = settings_dir / "preflight.report.json"
    runtime.write_json(output, report)

    if bool(report.get("certified", False)):
        print(f"Spatial truth preflight PASSED；报告：{output}", flush=True)
        return 0

    diagnosis = report.get("failure_diagnosis", {})
    print("Spatial truth preflight FAILED", flush=True)
    print(json.dumps(runtime.jsonable(diagnosis), ensure_ascii=False, sort_keys=True), flush=True)
    print(f"完整报告：{output}", flush=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
