"""Independent 3D A-phi Maxwell low-induction ellipse convergence.

    python -m examples.bfzi_d2_elliptic_limit

The ellipse has NONZERO scalar potential and an analytic leading-order
E, q and average Joule power. This is not a published-table fit, TEAM-7
measurement, or a BFZI coil calculation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sdfmpneo_bfzi.direct3d_elliptic_limit import elliptic_3d_benchmark


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frequency",type=float,default=1000.)
    p.add_argument("--sigma",type=float,default=4.)
    p.add_argument("--axis-a",type=float,default=.12)
    p.add_argument("--axis-b",type=float,default=.07)
    p.add_argument("--height",type=float,default=.1)
    p.add_argument("--B-peak",type=float,default=.001)
    p.add_argument("--output",type=Path)
    args=p.parse_args()
    levels=((8,1,2),(16,2,2),(24,3,3),(32,4,4),(48,6,4))
    rows=[]
    for sectors,radial,axial in levels:
        result=elliptic_3d_benchmark(
            a=args.axis_a,b=args.axis_b,height=args.height,
            sigma=args.sigma,frequency=args.frequency,B_peak=args.B_peak,
            sectors=sectors,radial=radial,axial=axial)
        rows.append({
            "sectors":sectors,"radial":radial,"axial":axial,**result})
    report={
        "status":"Independent low-induction E/phi/Joule method evidence; D2 OPEN",
        "theory":"E = iωB/(a²+b²) * (a² y, -b² x, 0); "
                 "P = σ ω² |B|² π H a³ b³ / [8(a²+b²)]",
        "boundary":"finite all-conductor prism, n cross A_reaction=0",
        "geometry":"inscribed elliptical polygonal prism, not BFZI",
        "validation_limits":"alpha<<1 only; polygon/exterior skin and independent "
                            "3D TEAM-7 still unverified",
        "results":rows,
    }
    out=json.dumps(report,indent=2,ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(out,encoding="utf-8")
        print("Full analytic PDE comparison saved:",args.output)
    print(out)


if __name__=="__main__":
    main()
