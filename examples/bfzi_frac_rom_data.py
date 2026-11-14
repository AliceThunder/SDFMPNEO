"""Export BFZI V150 skin-source thermal ROM coordinates for fractional learning."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from sdfmpneo_bfzi.heat_rom import HeatROM


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, default=Path("runs"))
    p.add_argument("--poses", nargs="+", default=("P03", "P07"))
    a = p.parse_args()
    summary = {}
    for pose in a.poses:
        key = pose.lower()
        rom = HeatROM.load(a.runs / f"bfzi_heat_rom_{key}.npz")
        with np.load(a.runs / f"bfzi_pose_{key}_skin_source_subcell_extra.npz") as d:
            t = d["time_s"].copy()
            v = d["voltage_v"].copy()
            temp = d["temp_K"].copy()
            z6 = d["z6"].copy()
        coords = (temp - rom.amb) @ (rom.cap[:, None] * rom.basis)
        recon = rom.amb + coords @ rom.basis.T
        out = a.runs / f"bfzi_frac_rom_{key}.npz"
        np.savez_compressed(out, time_s=t, voltage_v=v,
                            thermal_state=coords, delta_z6=z6-z6[0],
                            z0=z6[0], rank=rom.rank)
        summary[pose] = dict(samples=len(t), rank=rom.rank,
                             max_recon_K=float(np.max(abs(recon-temp))),
                             file=str(out))
        print(pose, json.dumps(summary[pose]), flush=True)
    (a.runs / "bfzi_frac_rom_data.json").write_text(
        json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
