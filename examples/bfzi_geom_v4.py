"""Assemble reproducible V4 manifests from completed real S5 teacher files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def manifest(plan, ntrain=5, nval=1, runs=Path("runs")):
    train = plan["train"][:ntrain]
    val = plan["val"][:nval]
    if len(train) != ntrain or len(val) != nval:
        raise ValueError("requested cohort exceeds frozen geometry split")
    if set(train) & set(val) or set(train + val) & set(plan["test"]):
        raise ValueError("geometry leakage")
    output = {"train": [], "val": [], "canonical": True, "rank": 32,
              "duration_s": 600., "geometry_split": {"train": train, "val": val}}
    for section, ids in (("train", train), ("val", val)):
        for name in ids:
            pre = f"bfzi_s5_{name}_ultra"
            if name in ("T01", "T02", "T03", "T04"):
                anchor = runs / f"{pre}_pilot60_anchor.npz"
                teacher = [runs / f"{pre}_pilot60_drive{k}.npz"
                           for k in (1, 2)]
                label = "long600" if name in ("T01", "T04") else "train600"
                teacher.append(runs / f"{pre}_{label}_drive1.npz")
            else:
                anchor = runs / f"{pre}_v4_anchor.npz"
                teacher = [runs / f"{pre}_v4_drive{k}.npz"
                           for k in (1, 2, 3)]
            missing = [str(f) for f in [anchor] + teacher if not f.exists()]
            if missing:
                raise FileNotFoundError(f"{name}: " + ", ".join(missing))
            output[section].append({"anchor":anchor.as_posix(),
                                    "teacher":[f.as_posix() for f in teacher]})
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path,
                   default=Path("runs/geomfrac_s5_v4_plan.json"))
    p.add_argument("--train", type=int, default=5)
    p.add_argument("--val", type=int, default=1)
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()
    obj = manifest(json.loads(a.plan.read_text(encoding="utf-8")),
                   a.train, a.val)
    path = a.out or Path(f"runs/geomfrac_s5_v4_{a.train}g_manifest.json")
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    print("V4_MANIFEST", path, "train", a.train, "val", a.val,
          "histories", sum(len(x["teacher"]) for x in obj["train"]))


if __name__ == "__main__":
    main()
