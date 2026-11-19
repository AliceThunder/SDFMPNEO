"""Resume and finish V7 cold references across worker and session failures."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import psutil
from examples.bfzi_cold_v7_plan import SOURCES, OLD, category, clear, read
from examples.bfzi_cold_v7_replacements import apply
from examples.bfzi_cold_plan import sample
from examples.bfzi_param import originals, create
from examples.bfzi_physics_recheck import GRIDS

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "runs/geomfrac_cold_v7_plan.json"
PATCH = ROOT / "runs/geomfrac_cold_v7_replacements.json"
OUT = ROOT / "runs/geomfrac_cold_v7"
LOG = ROOT / "runs/geomfrac_cold_v7_watch.log"
BFZI = Path(r"D:\work\geomfrac-s5-BFZI")
PARTS = (("train", 995), ("val", 150), ("test", 150))


def note(*msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S") + " " + " ".join(map(str, msg))
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def save(path, obj):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def waiting():
    """Avoid a second generator writing the same NPZ or progress file."""
    while True:
        found = []
        for p in psutil.process_iter(["pid", "cmdline"]):
            try:
                cmd = p.info.get("cmdline") or []
                if "examples.bfzi_cold_generate" in cmd:
                    found.append(p.pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        if not found:
            return
        note("waiting for active generator", found)
        time.sleep(20)


def run(offset=0, limit=None, retry=False):
    cmd = [
        sys.executable, "-u", "-m", "examples.bfzi_cold_generate",
        "--bfzi-root", str(BFZI), "--plan", str(PLAN),
        "--out", str(OUT), "--grid", "ultra", "--source-order", "5",
        "--timeout", "600", "--offset", str(offset),
    ]
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    if retry:
        cmd.append("--retry-failed")
    log = ROOT / f"runs/geomfrac_cold_v7_watch_{offset}.log"
    with log.open("a", encoding="utf-8") as f:
        f.write(f"\nWATCH_START {time.strftime('%Y-%m-%d %H:%M:%S')} offset={offset} limit={limit} retry={retry}\n")
        f.flush()
        p = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    note("generator", offset, limit, "retry", retry, "exit", p.returncode)
    if p.returncode:
        raise RuntimeError(f"generator exit {p.returncode}; see {log}")


def state():
    plan = apply(read(PLAN))
    store = OUT / "samples"
    missing = [
        (split, name)
        for split, _ in PARTS
        for name in plan[split]
        if not (store / f"{name}.npz").exists()
    ]
    total = sum(len(plan[split]) for split, _ in PARTS)
    return plan, missing, total


def repair(plan, missing):
    """Replace unsolved geometries with unique, native-BFZI-valid peers."""
    patch = read(PATCH)
    used = list(read(PLAN)["cases"].values())
    used.extend(plan["cases"].values())
    for path in list(SOURCES.values()) + [OLD]:
        d = read(ROOT / path)
        for split in ("train", "dev", "val", "test", "bb_diagnostic"):
            used.extend(r["geometry"] for r in d.get(split, []) if "geometry" in r)
    bf, v150 = originals(BFZI)
    xz, y = GRIDS["ultra"]
    chosen = {}
    for trial in range(80):
        if len(chosen) == len(missing):
            break
        seed = 72030 + 37 * len(patch["items"]) + trial
        rows = sample(seed, 250, existing=used, accept=lambda g: clear(bf, g))
        for geom in rows["cases"].values():
            next_item = next(
                ((sp, key) for sp, key in missing if key not in chosen
                 and category(geom) == plan["strata"][key]),
                None,
            )
            if next_item is None:
                continue
            try:
                create(
                    bf, v150, "M4", **geom, sea_void="oriented",
                    physical_box_only=True, grid_xz=xz, grid_y=y,
                    grid_max_nodes=4000, mesh_only=True,
                )
            except (ValueError, RuntimeError):
                continue
            sp, key = next_item
            chosen[key] = (sp, geom, seed)
            used.append(geom)
            if len(chosen) == len(missing):
                break
    if len(chosen) != len(missing):
        raise RuntimeError(f"BFZI-valid replacements unavailable for {len(missing)-len(chosen)} geometries")
    pool = set(plan["cases"])
    for split, name in missing:
        prefix = {"train": "V7T", "val": "V7V", "test": "V7E"}[split]
        number = max(
            (int(n[len(prefix):]) for n in pool
             if n.startswith(prefix) and n[len(prefix):].isdigit()),
            default=0,
        ) + 1
        new = f"{prefix}{number:04d}"
        pool.add(new)
        _, geom, seed = chosen[name]
        patch["items"].append({
            "replaced": name,
            "replacement": {"name": new, "stratum": plan["strata"][name], "geometry": geom},
            "reason": f"Unsolved cold reference; validated with native BFZI mesh_only, seed {seed}",
        })
        note("replaced", name, "with", new, "stratum", plan["strata"][name])
    save(PATCH, patch)


def done(plan):
    data = read(OUT / "data.json")
    got = {sp: len(data.get(sp, [])) for sp, _ in PARTS}
    expected = {sp: n for sp, n in PARTS}
    if got != expected:
        raise RuntimeError(f"assembled split counts {got}, expected {expected}")
    return got


def main():
    note("V7 supervisor started")
    waiting()
    rounds = 0
    while True:
        try:
            plan, missing, total = state()
            note("remaining references", len(missing), "of", total)
            if not missing:
                run()
                note("COMPLETE", done(plan))
                return
            # Re-entering a batch automatically skips successful NPZ files.
            run(800, 400)
            run(1200, 95)
            plan, missing, total = state()
            if missing:
                note("retry missing references", len(missing))
                run(retry=True)
            plan, missing, total = state()
            if missing:
                note("replace unresolved samples", len(missing))
                repair(plan, missing)
            rounds = 0
        except Exception as err:
            rounds += 1
            note("recoverable supervisor error", repr(err), "attempt", rounds)
            time.sleep(min(300, 20 * rounds))


if __name__ == "__main__":
    main()
