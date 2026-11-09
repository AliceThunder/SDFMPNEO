"""D2 BFZI independent continuation sweeps: near-water h, outer R, source chords.

This drives the REAL closed BFZI two-port 3D Maxwell A-phi solver; it
does not run the frozen old G1 or the neural teacher. Each sweep changes
ONE numerical parameter while holding the physical coil, return, boxes,
material and frequency fixed. All full solver reports are saved locally.

    python -m examples.bfzi_d2_convergence --vary h --out runs/d2_h
    python -m examples.bfzi_d2_convergence --vary R --out runs/d2_R
    python -m examples.bfzi_d2_convergence --vary source --out runs/d2_source

A fixed inner core preserves near-field coordinate planes for R variation.
The h sweep uses a padding band wide enough to change conductive wet-side
normal cell layers; h=.08/.06/.04 may exceed a laptop's sparse-direct
memory budget. Use a capable workstation for the final fine level.

These sweeps are NOT official TEAM-7/Elmer or infinite-ocean certificates.
Do not treat a small inter-grid difference as evidence of absolute accuracy.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def _as_complex_2x2(re_im):
    data=np.asarray(re_im,float)
    if data.shape != (2,2,2):
        raise ValueError("D2 BFZI report must contain a full 2x2 complex impedance")
    return data[...,0]+1j*data[...,1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vary",choices=("h","R","source"),required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--frequency",type=float,default=200000.)
    parser.add_argument("--far-step",type=float,default=.30)
    parser.add_argument("--padding",type=float,default=.08)
    parser.add_argument("--box-layers",type=int,default=1)
    parser.add_argument("--inner-core-radius",type=float,default=.24,
                        help="Fix both source and near-field grid planes across R sweep")
    parser.add_argument("--solve-analytic-primary",action="store_true")
    parser.add_argument("--probe",type=float,nargs=3,action="append")
    args=parser.parse_args()
    if args.frequency<=0:
        parser.error("frequency must be positive")
    args.out.mkdir(parents=True,exist_ok=True)

    cases = {
        "h":[("0p08",.08),("0p06",.06),("0p04",.04)],
        "R":[("0p32",.32),("0p40",.40),("0p52",.52)],
        "source":[("2",2),("4",4),("8",8)],
    }[args.vary]

    out=[]
    previous=None
    for key,value in cases:
        near_step = value if args.vary=="h" else .08
        radius = value if args.vary=="R" else .52
        panels = value if args.vary=="source" else 4
        report_path=args.out/f"bfzi_{args.vary}_{key}.json"
        command=[
            sys.executable,"-m","examples.bfzi_d2_aphi",
            "--case","bfzi","--structured-axis-aligned",
            "--frequency",str(args.frequency),
            "--outer-radius",str(radius),
            "--structured-far-step",str(args.far_step),
            "--structured-padding",str(args.padding),
            "--structured-box-layers",str(args.box_layers),
            "--structured-near-step",str(near_step),
            "--structured-inner-core-radius",str(args.inner_core_radius),
            "--panels-per-corner",str(panels),
            "--summary-only","--output",str(report_path),
        ]
        if args.solve_analytic_primary:
            command.append("--solve-analytic-primary")
        if args.probe:
            for probe in args.probe:
                command.extend(["--probe",*(str(x) for x in probe)])
        print(f"D2 {args.vary}={value}: running actual two-port Maxwell A-phi")
        subprocess.run(command,check=True)
        report=json.loads(report_path.read_text(encoding="utf-8"))
        z=_as_complex_2x2(report["material_increment_2port_z_ohm"])
        norm=float(np.linalg.norm(z))
        change=(float(np.linalg.norm(z-previous)/max(norm,1e-30))
                if previous is not None else None)
        phase=report["phase_port_vs_seawater_joule"]
        source_gap={
            name:case.get("corrected_vs_port_absolute_gap_W")
            for name,case in phase.items()
        }
        item=dict(vary=args.vary,value=value,
                  radius_m=radius,near_step_m=near_step,
                  near_padding_m=args.padding,inner_core_radius_m=args.inner_core_radius,
                  panels_per_corner=panels,
                  frequency_hz=args.frequency,
                  tetrahedra=report["tetrahedra"],
                  Z_material_re_im_ohm=report["material_increment_2port_z_ohm"],
                  Z_change_relative_to_previous=change,
                  reciprocity_gap_ohm=report["two_port_reciprocity_gap_ohm"],
                  analytic_source_corrected_vs_port_absolute_gap_W=source_gap,
                  full_result=str(report_path))
        out.append(item)
        previous=z
        print(json.dumps({k:item[k] for k in (
            "tetrahedra","Z_change_relative_to_previous",
            "reciprocity_gap_ohm",
            "analytic_source_corrected_vs_port_absolute_gap_W")},
            indent=2))
        (args.out/"sweep.json").write_text(
            json.dumps({"status":"D2 OPEN; numerical sensitivity only",
                        "vary":args.vary,"results":out},indent=2),
            encoding="utf-8")
    print("Finished sweep; D2 remains OPEN until independent h/R/source "
          "and spatial field convergence plus mature Elmer/TEAM-7 reference.")


if __name__=="__main__":
    main()
