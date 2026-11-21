"""Paired, frozen BFZI V11 blind acceptance (V8 vs V11 ColdNet).

This script consumes newly generated complete BFZI physical histories and
never changes weights, optimizers or validation thresholds.
"""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

KEYS = (
    "cold_z_rel_L2", "dynamic_z_rel_L2",
    "R11_dynamic_rel_L2_self", "X11_dynamic_rel_L2_self",
    "R22_dynamic_rel_L2_self", "X22_dynamic_rel_L2_self",
    "current_rel_L2", "material_mean_rmse_K", "material_mean_max_K",
    "sea_power_rel_L2", "copper_power_rel_L2",
)

OVERALL_LIMITS = {
    "cold_z_rel_L2": 1.03,
    "R11_dynamic_rel_L2_self": 1.02,
    "R22_dynamic_rel_L2_self": 1.02,
    "X11_dynamic_rel_L2_self": 1.03,
    "X22_dynamic_rel_L2_self": 1.03,
    "current_rel_L2": 1.03,
    "material_mean_rmse_K": 1.03,
    "material_mean_max_K": 1.05,
    "sea_power_rel_L2": 1.03,
    "copper_power_rel_L2": 1.03,
}

PER_LIMITS = {
    "cold_z_rel_L2": 1.10,
    "dynamic_z_rel_L2": 1.10,
    "R11_dynamic_rel_L2_self": 1.10,
    "X11_dynamic_rel_L2_self": 1.10,
    "R22_dynamic_rel_L2_self": 1.10,
    "X22_dynamic_rel_L2_self": 1.10,
    "current_rel_L2": 1.10,
    "material_mean_rmse_K": 1.10,
    "material_mean_max_K": 1.15,
    "sea_power_rel_L2": 1.15,
    "copper_power_rel_L2": 1.15,
}

ENERGY_LIMIT_W = 1e-8


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def within(value, base, limit):
    return (math.isfinite(float(value)) and math.isfinite(float(base))
            and float(value) <= limit * max(float(base), 1e-12))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path,
                   default=Path("runs/geomfrac_v11_blind_plan.json"))
    p.add_argument("--manifest", type=Path,
                   default=Path("runs/geomfrac_s5_v11_manifest.json"))
    p.add_argument("--report", type=Path,
                   default=Path("runs/geomfrac_s5_v11_build_report.json"))
    p.add_argument("--script", type=Path,
                   default=Path("D:/work/geomfrac-v8/examples/bfzi_v8_accept.py"))
    p.add_argument("--out", type=Path,
                   default=Path("runs/geomfrac_v11_blind_summary.json"))
    a = p.parse_args()

    plan = read(a.plan)
    mf = read(a.manifest)
    report = read(a.report)
    specs = mf.get("test", [])
    names = [Path(s["anchor"].replace("\\", "/")).stem.split("_")[2]
             for s in specs]
    if (len(specs) != 4 or sorted(names) != sorted(plan["test"])
            or len(set(names)) != 4
            or not all(len(s.get("teacher", [])) == 2 for s in specs)
            or not all(Path(str(f).replace("\\", "/")).is_file()
                       for s in specs for f in s["teacher"])
            or not all(report.get("cases", {}).get(n, {}).get("status")
                       == "complete" for n in names)):
        raise ValueError("incomplete frozen V11 full-physics reference; abort acceptance")

    common = [
        "--plan", str(a.plan), "--manifest", str(a.manifest),
        "--split", "test",
        "--config", "runs/geomfrac_free_rank48_v8_selection.json",
        "--skin-tangent", "runs/geomfrac_rank48_skin_tan_v3.pt",
        "--skin-base", "runs/geomfrac_et_v9_ztune.pt",
        "--parallel-model", "runs/geomfrac_v8_par48.pt",
        "--k-mix", "0.25",
    ]
    models = {
        "V8": ("runs/geomfrac_et_v8_heat_tune.pt",
               "runs/geomfrac_cold_v8_frozen.pt"),
        "V11_cold": ("runs/geomfrac_et_v8_heat_tune.pt",
                     "runs/geomfrac_v11_cold.pt"),
    }
    results = {}
    for name, (heat, cold) in models.items():
        dest = "runs/geomfrac_v11_blind_" + (
            "base" if name == "V8" else "cold") + ".json"
        cmd = [sys.executable, "-u", str(a.script), *common,
               "--model", heat, "--cold-model", cold, "--out", dest]
        subprocess.run(cmd, check=True)
        results[name] = read(dest)
        got = results[name]
        if (got.get("status") != "complete"
                or got.get("geometry_count") != 4
                or got.get("history_count") != 8):
            raise ValueError(f"{name}: incomplete paired acceptance result")
        print("V11_BLIND_MODEL", name,
              json.dumps(got["overall"]), flush=True)

    base = results["V8"]["overall"]
    cur = results["V11_cold"]["overall"]
    delta = {
        k: dict(V8=base[k], V11_cold=cur[k], delta=cur[k] - base[k])
        for k in KEYS
    }
    delta["energy_budget_max_W"] = dict(
        V8=base["energy_budget_max_W"],
        V11_cold=cur["energy_budget_max_W"],
        delta=cur["energy_budget_max_W"] - base["energy_budget_max_W"],
    )

    overall_checks = {
        k: within(cur[k], base[k], lim)
        for k, lim in OVERALL_LIMITS.items()
    }
    overall_checks["dynamic_z_improved"] = (
        math.isfinite(float(cur["dynamic_z_rel_L2"]))
        and math.isfinite(float(base["dynamic_z_rel_L2"]))
        and cur["dynamic_z_rel_L2"] < base["dynamic_z_rel_L2"]
    )
    overall_checks["V8_energy_budget"] = (
        math.isfinite(float(base["energy_budget_max_W"]))
        and base["energy_budget_max_W"] <= ENERGY_LIMIT_W
    )
    overall_checks["V11_energy_budget"] = (
        math.isfinite(float(cur["energy_budget_max_W"]))
        and cur["energy_budget_max_W"] <= ENERGY_LIMIT_W
    )

    per = {}
    per_checks = {}
    for name in names:
        b = results["V8"]["per_geometry"][name]
        c = results["V11_cold"]["per_geometry"][name]
        per[name] = {
            k: dict(V8=b[k], V11_cold=c[k], delta=c[k] - b[k])
            for k in KEYS
        }
        per[name]["energy_budget_max_W"] = dict(
            V8=b["energy_budget_max_W"],
            V11_cold=c["energy_budget_max_W"],
            delta=c["energy_budget_max_W"] - b["energy_budget_max_W"],
        )
        checks = {
            k: within(c[k], b[k], lim)
            for k, lim in PER_LIMITS.items()
        }
        checks["energy_budget"] = (
            math.isfinite(float(c["energy_budget_max_W"]))
            and c["energy_budget_max_W"] <= ENERGY_LIMIT_W
        )
        per_checks[name] = checks

    overall_ok = all(overall_checks.values())
    per_ok = all(all(v.values()) for v in per_checks.values())
    passed = overall_ok and per_ok
    out = dict(
        status="blind_improved" if passed else "blind_not_improved",
        source_test=plan["source_test"],
        geometry_count=4,
        history_count=8,
        train_candidate="runs/geomfrac_v11_cold.pt",
        physics="BFZI V150 S5 ultra matched 60s/300s",
        frozen_before_generation=True,
        changed_module="ColdNet only",
        acceptance=dict(
            passed=passed,
            overall_pass=overall_ok,
            per_geometry_pass=per_ok,
            overall_checks=overall_checks,
            per_geometry_checks=per_checks,
            overall_limits=OVERALL_LIMITS,
            per_geometry_limits=PER_LIMITS,
            energy_budget_limit_W=ENERGY_LIMIT_W,
        ),
        metrics=delta,
        per_geometry=per,
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("V11_BLIND_FINAL", json.dumps({
        k: out[k] for k in ("status", "geometry_count", "history_count",
                            "acceptance", "metrics")
    }), flush=True)


if __name__ == "__main__":
    main()
