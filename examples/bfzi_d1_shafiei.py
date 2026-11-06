"""Reproduce Shafiei IEEE Access (2025), Table 2, without modifying BFZI/G1.

Run from the repository root:
    python -m examples.bfzi_d1_shafiei --scenario finite --frequency-khz 400
No FEM, arbitrary SE(3) geometry, or full-wave Joule certification is claimed.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from sdfmpneo_bfzi.layered_helmholtz import paper_case, incremental_impedance
from sdfmpneo_bfzi.layered_grid import secondary_field_grid


PUBLISHED = {
    ("infinite", 85): ((.261, .334, .209), .298),
    ("infinite", 400): ((5.082, 6.505, 3.854), 5.375),
    ("semifinite", 85): ((.133, .301, .152), .231),
    ("semifinite", 400): ((2.668, 6.127, 2.974), 4.497),
    ("finite", 85): ((.104, .133, .090), .132),
    ("finite", 400): ((2.386, 3.054, 2.056), 2.900),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("infinite", "semifinite", "finite"),
                        default="finite")
    parser.add_argument("--frequency-khz", type=int, choices=(85, 400), default=400)
    parser.add_argument("--k-max", type=float, default=4000.)
    parser.add_argument("--panel-width", type=float, default=8.)
    parser.add_argument("--radial-points", type=int, default=151)
    args = parser.parse_args()
    if args.radial_points < 20:
        parser.error("At least 20 radial points are required for this scan")
    case = paper_case(args.scenario, frequency_hz=args.frequency_khz*1000.)
    z = np.r_[np.linspace(-.12, -.0008, 31),
              np.linspace(.0008, .0992, 57),
              np.linspace(.1008, .22, 31)]
    r = np.linspace(0., .5, args.radial_points)
    Z = incremental_impedance(case, k_max=args.k_max, include_displacement=True)
    sec = secondary_field_grid(
        case, r, z, currents=(5., 5.), k_max=args.k_max,
        panel_width=args.panel_width, include_displacement=True)
    ok_z = np.ones(z.shape, dtype=bool)
    if args.scenario == "semifinite":
        ok_z = z > 0.
    elif args.scenario == "finite":
        ok_z = (z > 0.) & (z < case.gap_m)
    field_ampl = np.abs(sec["E_phi_secondary_V_m"][:, ok_z])
    ij = np.unravel_index(int(np.argmax(field_ampl)), field_ampl.shape)
    peak = float(field_ampl[ij])
    pub_r, pub_e = PUBLISHED[(args.scenario, args.frequency_khz)]
    values = [float(Z[0,0].real), float(Z[1,1].real), float(Z[0,1].real)]
    report = {
        "status": "D1 partial - not PASS",
        "case": args.scenario, "frequency_hz": case.frequency_hz,
        "paper": "DOI 10.1109/ACCESS.2025.3539925, Table 2 analytical",
        "geometry": "15/17 separate coaxial sheet loops, 0.1 m gap; NOT BFZI continuous loop",
        "model": "full-wave three-layer planar Helmholtz",
        "k_max_per_m": args.k_max, "panel_width_per_m": args.panel_width,
        "r_grid_points": len(r), "z_grid_points": int(ok_z.sum()),
        "delta_Z_ohm": [[[float(v.real), float(v.imag)] for v in row] for row in Z],
        "added_resistance_ohm": values,
        "published_analytical_resistance_ohm": pub_r,
        "max_secondary_E_V_m": peak,
        "max_secondary_E_location_r_z_m": [float(r[ij[0]]), float(z[ok_z][ij[1]])],
        "published_analytical_E_max_V_m": pub_e,
        "warning": ("The paper does not explicitly define E_max as the vacuum-subtracted "
                    "field; this is a strong numerical interpretation, not independent "
                    "validation or a substitute for 3D A-phi. Joule heating uses E_total.")
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
