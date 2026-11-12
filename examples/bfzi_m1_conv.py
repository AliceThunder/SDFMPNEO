"""Compare like-for-like M1 3-D seawater h and R solutions from JSON outputs.

Requires fixed closed-source geometry, frequency, current and corner panels.
The h/R sequences use the exact same physical dual-UMFC + SE(3) scene.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def cmp(a, b):
    """Difference relative to the finer/larger case b."""
    def cx(k):
        return [np.array(d[k])[..., 0] + 1j*np.array(d[k])[..., 1]
                for d in (a, b)]
    def rel(x, y):
        return float(np.linalg.norm(x-y)/max(np.linalg.norm(y), 1e-30))
    z0,z1 = cx("unit_port_Zsea_ohm")
    b0,b1 = cx("unit_B_T")
    e0,e1 = cx("unit_E_V_m")
    j0,j1 = cx("J_A_m2")
    p0,p1 = (d["seawater_power_port_W"] for d in (a,b))
    return {
        "tetra_from": a["tetrahedra"],
        "tetra_to": b["tetrahedra"],
        "port_Z_rel": rel(z0,z1),
        "unit_B_probes_rel": rel(b0,b1),
        "unit_E_probes_rel": rel(e0,e1),
        "J_probes_rel": rel(j0,j1),
        "power_rel": abs(p0-p1)/max(abs(p1), 1e-30),
        "port_power_close_rel": abs(p1-b["seawater_power_volume_W"])/max(abs(p1),1e-30)
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("runs"))
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    cases = ("r052h08","r075h08","r075h06","r120h08")
    data = {n:json.loads((args.root/f"m1_{n}.json").read_text(encoding="utf-8"))
            for n in cases}
    ref = data[cases[0]]
    for name,d in data.items():
        for key in ("frequency_hz","currents_A","probe_m"):
            if d[key] != ref[key]:
                raise ValueError(f"Inconsistent {key} in {name}")
    report = {
        "status": "M1 numerical reference OPEN",
        "geometry": "dual closed UMFC, Rx y-axis rotation 0.18 rad",
        "frequency_hz": ref["frequency_hz"],
        "source_panels_per_corner": 12,
        "seawater_S_m": 5.0,
        "mesh": "physical-tag Gmsh OCC, near-box size controlled; changing R remeshes entire domain",
        "case_settings": {
            "r052h08": {"R_m": .52, "near_h_m": .08, "far_h_m": .25},
            "r075h08": {"R_m": .75, "near_h_m": .08, "far_h_m": .25},
            "r075h06": {"R_m": .75, "near_h_m": .06, "far_h_m": .25},
            "r120h08": {"R_m": 1.20, "near_h_m": .08, "far_h_m": .25}
        },
        "power_W": {name: d["seawater_power_port_W"] for name,d in data.items()},
        "tetra": {name: d["tetrahedra"] for name,d in data.items()},
        "h": {"r075h08_to_r075h06": cmp(data["r075h08"],data["r075h06"])},
        "R": {
            "r052_to_r075": cmp(data["r052h08"],data["r075h08"]),
            "r075_to_r120": cmp(data["r075h08"],data["r120h08"])
        },
        "limit": "Fixed probe set has two wet-seawater points; full 3-D field L2 h/R and source/panel certification are OPEN."
    }
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2),
                            encoding="utf-8")
    print(json.dumps({"h":report["h"],"R":report["R"]}, indent=2))


if __name__ == "__main__":
    main()
