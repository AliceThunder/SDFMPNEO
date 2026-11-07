"""Run independent finite-frequency 3D sphere Maxwell–MQS convergence.

From the project root:
    python -m examples.bfzi_d2_sphere_limit --divisions 2 4 6 8 10

Conductor sphere in a uniform harmonic B field: exact modified spherical
Bessel l=1 benchmark. The FE boundary is prescribed from an independent
continuous solution; interior E/B/Joule are solved by the actual Nedelec
A–phi code, not replaced by analytic data after the solve.

This is a method benchmark, not a TEAM-7 or BFZI reference PASS.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sdfmpneo_bfzi.direct3d_sphere_limit import (
    HarmonicConductingSphere, sphere_d2_benchmark,
)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--radius",type=float,default=.05)
    p.add_argument("--sigma",type=float,default=3.526e7)
    p.add_argument("--frequency",type=float,default=50.)
    p.add_argument("--B-mT",type=float,default=1.)
    p.add_argument("--divisions",nargs="+",type=int,default=(2,4,6,8,10))
    p.add_argument("--output",type=Path,
                   help="Save full numerical evidence and mesh/geometry split as JSON")
    args=p.parse_args()
    reference=HarmonicConductingSphere(args.radius,args.sigma,args.frequency,
                                       (0,0,args.B_mT*1e-3))
    records=[sphere_d2_benchmark(reference,n) for n in args.divisions]
    report={
        "reference": "exact time-harmonic conducting-sphere spherical Bessel l=1",
        "time_convention": "exp(+i*omega*t), peak phasors",
        "scope": "Independent magnetic-diffusion FEM E/B/Joule test, D2 OPEN",
        "results": records,
    }
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2),encoding="utf-8")
        print("Saved full 3D sphere convergence evidence:",args.output)
    print("tetra | relative E L2 | relative B L2 | "
          "FEM Joule W | exact polyhedron Joule W | sphere Joule W")
    for v in records:
        print(f'{v["tetrahedra"]:5d} | {v["complex_E_volume_relative_L2"]:.6f} | '
              f'{v["complex_B_volume_relative_L2"]:.6f} | '
              f'{v["joule_fem_W"]:.8g} | '
              f'{v["joule_exact_same_polyhedron_W"]:.8g} | '
              f'{v["joule_exact_whole_sphere_W"]:.8g}')
    print("D2=OPEN: finite-element spatial-field accuracy and independent "
          "TEAM-7/BFZI convergence not certified")


if __name__=="__main__":
    main()
