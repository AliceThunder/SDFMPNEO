"""TEAM-7 *independent 3-D edge-Galerkin* reference research entry.

Uses the SAME exact Nedelec closed-source edge functional to drive the 3D
A-phi field and read its induced source voltage. Vacuum subtracts the
same finite-box response. No fabricated experimental agreement or G1.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

from sdfmpneo_bfzi.direct3d_galerkin import solve_closed_ports
from sdfmpneo_bfzi.direct3d_probe import sample_cell_fields
from sdfmpneo_bfzi.direct3d_team7 import (
    team7_geo, team7_mesh, team7_stranded_source, official_probe_lines,
    graded_team7_smoke_mesh,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-geo', type=Path)
    parser.add_argument('--mesh', type=Path)
    parser.add_argument('--coarse-smoke', action='store_true')
    parser.add_argument('--refinement-level', type=int, default=1)
    parser.add_argument('--frequency', type=int, choices=(50, 200), default=50)
    parser.add_argument('--mesh-size', type=float, default=.08)
    parser.add_argument('--plate-size', type=float, default=.009)
    parser.add_argument('--width-order', type=int, default=2)
    parser.add_argument('--height-order', type=int, default=2)
    parser.add_argument('--corner-panels', type=int, default=12)
    parser.add_argument('--outer-box', choices=('compumag','elmer'),
                        default='compumag')
    parser.add_argument('--outer-bc', choices=('total_zero',),
                        default='total_zero',
                        help='Galerkin total-A solver uses homogeneous total tangential A')
    parser.add_argument('--current-sign', type=int, choices=(-1, 1), default=1)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--summary-only', action='store_true')
    args = parser.parse_args()

    if args.write_geo:
        bounds = ((-.2,.494),(-.2,.494),(-.2,.394)) if args.outer_box == 'elmer' else (
            (-1.353,1.647),(-1.353,1.647),(-.300,.449))
        args.write_geo.write_text(
            team7_geo(args.mesh_size, args.plate_size, outer_bounds=bounds),
            encoding='utf-8')
        print('Wrote TEAM-7 physical-domain geometry:', args.write_geo)
        if not args.coarse_smoke and args.mesh is None:
            return
    if args.coarse_smoke == bool(args.mesh):
        parser.error('Choose one of --coarse-smoke or --mesh')
    if args.refinement_level < 1 or (not args.coarse_smoke and args.refinement_level != 1):
        parser.error('--refinement-level needs positive level with --coarse-smoke')

    mesh = (graded_team7_smoke_mesh(args.refinement_level)
            if args.coarse_smoke else team7_mesh(args.mesh))
    _, _, strands = team7_stranded_source(
        args.width_order, args.height_order, args.corner_panels,
        return_strands=True, current_sign=args.current_sign)
    reference = solve_closed_ports(mesh, args.frequency, [strands])

    probes = {}
    for name, coords in official_probe_lines().items():
        sample = sample_cell_fields(
            reference.medium, coords, prefer_conductor=('Jy_' in name),
            nearest=256)
        # direct3d_probe's "B_secondary_T" field is the solved curl(A).
        # This solver directly solves TOTAL A (not the old reaction A).
        values = (sample['J_A_m2'][0,:,1] if 'Jy_' in name
                  else sample['B_secondary_T'][0,:,2])
        probes[name] = {
            'unit': 'A/m2' if 'Jy_' in name else 'T',
            'values_real_imag': [
                [float(v.real), float(v.imag)] for v in values],
        }

    z = reference.material_increment_z[0,0]
    volume = reference.conductor_power(np.array([1.+0j]))
    port = reference.port_power(np.array([1.+0j]))
    report = {
        'status':'D2 Galerkin 3D core; TEAM-7 measured accuracy NOT certified',
        'frequency_hz':args.frequency,
        'mesh_case':'local_small_box' if args.coarse_smoke else args.outer_box,
        'boundary':'n cross A_total=0 on finite domain',
        'tetrahedra':len(mesh.tetrahedra),
        'metal_tetrahedra':int(np.count_nonzero(mesh.sigma)),
        'structured_refinement_level': (
            args.refinement_level if args.coarse_smoke else None),
        'source':f'{args.current_sign*2742} A.turn equivalent closed racetrack strands',
        'material_delta_z_real_imag_ohm':[float(z.real),float(z.imag)],
        'vacuum_finite_box_z_real_imag_ohm':[
            float(reference.vacuum_port_z[0,0].real),
            float(reference.vacuum_port_z[0,0].imag)],
        'closed_port_material_power_W_for_1A':port,
        'conducting_volume_joule_W_for_1A':volume,
        'absolute_power_gap_W':abs(port-volume),
        'relative_power_gap':abs(port-volume)/max(abs(volume),1e-30),
        'probes':probes,
        'phasor':'peak e^(+i omega t): phase0=Re, phase90=-Im',
        'limitations':(
            'Physical source/measurement use identical Galerkin line loads. '
            'Power closure verifies discrete assembly, NOT TEAM-7 measured '
            'B/J accuracy. Mesh, coil cross-section/chords, outer boundary '
            'and independent Elmer benchmark still need convergence.'),
    }
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
        print('Saved full D2 TEAM-7 report:', args.output)
    if args.summary_only:
        selected = (
            'status','frequency_hz','mesh_case','tetrahedra',
            'metal_tetrahedra','structured_refinement_level',
            'material_delta_z_real_imag_ohm',
            'closed_port_material_power_W_for_1A',
            'conducting_volume_joule_W_for_1A',
            'absolute_power_gap_W','relative_power_gap')
        print(json.dumps({k:report[k] for k in selected},indent=2))
    else:
        print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
