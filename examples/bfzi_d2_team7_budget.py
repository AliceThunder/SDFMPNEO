"""Compare TEAM-7 h/R changes with the same-observable h10 time-step change.

Both data sets are real Elmer 200 Hz BDF2 and use identical original
SaveLine field definitions at t=0.02 s; measurements are never read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

OBS = ("A1_Bz", "A1_Bxyz", "A2_Bz", "Jy_top", "Jy_bottom")


def calc(space, time):
    out = []
    for axis in ("h", "R"):
        for row in space["runs"]["200"][axis]:
            for key in OBS:
                x = float(row[key + "_rel"])
                dt = float(time["observables"][key]["endpoint"]["rel64_128"])
                out.append({
                    "axis": axis, "pair": row["from"] + "/" + row["to"],
                    "field": key, "spatial_rel": x,
                    "time_h10_64_128_rel": dt,
                    "spatial_over_time": x/dt if dt > 0 else None,
                })
    return {
        "frequency_hz": 200, "time_s": 0.02,
        "source": "Elmer SaveLine 200Hz BDF2; numerical values, no measurements",
        "time_comparison": "h10 64 vs 128 steps, fixed mesh and R",
        "spatial_comparison": "64 steps, fixed dt per independent h or R pair",
        "observations": out, "status": "D2 OPEN",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spatial", type=Path, required=True)
    ap.add_argument("--time", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    res = calc(json.loads(a.spatial.read_text(encoding="utf-8-sig")),
               json.loads(a.time.read_text(encoding="utf-8-sig")))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    for row in res["observations"]:
        print(row["axis"], row["pair"], row["field"],
              f"h/R={100*row['spatial_rel']:.4f}%",
              f"dt={100*row['time_h10_64_128_rel']:.4f}%",
              f"ratio={row['spatial_over_time']:.2f}")
    

if __name__ == "__main__":
    main()
