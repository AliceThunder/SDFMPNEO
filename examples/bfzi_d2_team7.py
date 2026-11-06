"""TEAM-7 3D A-phi research entry (NOT a certified benchmark fit).

Official geometry: https://www.compumag.org/wp/wp-content/uploads/2018/06/problem7.pdf

    python -m examples.bfzi_d2_team7 --coarse-smoke --frequency 50
    python -m examples.bfzi_d2_team7 --write-geo team7.geo
    gmsh team7.geo -3 -format msh2 -o team7.msh
    python -m examples.bfzi_d2_team7 --mesh team7.msh --frequency 50

COARSE-SMOKE has only 48 metal tetrahedra; do not compare its amplitudes
to experimental references. TEAM-7 is an independent FEM calibration case,
not a replacement for the actual BFZI continuous spiral and two OBBs.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

from sdfmpneo_bfzi.direct3d_aphi import solve_aphi
from sdfmpneo_bfzi.direct3d_probe import sample_cell_fields
from sdfmpneo_bfzi.direct3d_team7 import (
    team7_geo, team7_mesh, team7_stranded_source, official_probe_lines,
    coarse_team7_tetra_mesh,
)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-geo',type=Path)
    parser.add_argument('--mesh',type=Path)
    parser.add_argument('--coarse-smoke',action='store_true')
    parser.add_argument('--frequency',type=int,choices=(50,200),default=50)
    parser.add_argument('--mesh-size',type=float,default=.08)
    parser.add_argument('--plate-size',type=float,default=.009)
    parser.add_argument('--width-order',type=int,default=2)
    parser.add_argument('--height-order',type=int,default=2)
    parser.add_argument('--corner-panels',type=int,default=12)
    parser.add_argument('--outer-bc',choices=('secondary_zero','total_zero'),
                        default='secondary_zero')
    parser.add_argument('--outer-box',choices=('compumag','elmer'),default='compumag')
    parser.add_argument('--port-subsegments',type=int,default=2)
    args=parser.parse_args()
    if args.write_geo:
        bounds=((-.2,.494),(-.2,.494),(-.2,.394)) if args.outer_box=='elmer' else (
            (-1.353,1.647),(-1.353,1.647),(-.300,.449))
        args.write_geo.write_text(team7_geo(args.mesh_size,args.plate_size,
                                             outer_bounds=bounds),
                                 encoding='utf-8')
        print('Wrote official-profile TEAM-7 Gmsh geometry:',args.write_geo)
        if not args.coarse_smoke and args.mesh is None:return
    if args.coarse_smoke == bool(args.mesh):
        parser.error('Choose one of --coarse-smoke or --mesh MSH2_PATH')
    mesh=coarse_team7_tetra_mesh() if args.coarse_smoke else team7_mesh(args.mesh)
    A_source,B_source,strands=team7_stranded_source(
        args.width_order,args.height_order,args.corner_panels,return_strands=True)
    sol=solve_aphi(mesh,args.frequency,[A_source],outer_bc=args.outer_bc)
    from sdfmpneo_bfzi.direct3d_port import closed_source_port_impedance
    port_z=closed_source_port_impedance(sol,args.frequency,[strands],
                                        subsegments=args.port_subsegments)
    probes={}
    for name,coords in official_probe_lines().items():
        samples=sample_cell_fields(
            sol,coords,incident_magnetic=[B_source],
            prefer_conductor=('Jy_' in name),nearest=256)
        values=(samples['J_A_m2'][0,:,1] if 'Jy_' in name
                else samples['B_total_T'][0,:,2])
        unit='A/m2' if 'Jy_' in name else 'T'
        probes[name]={'unit':unit,
                      'values_real_imag':[[float(v.real),float(v.imag)] for v in values]}
    report={
        'status':'D2 TEAM-7 initial FEM; no benchmark PASS',
        'geometry':'COMPUMAG Problem 7 Fig.1 (not BFZI)',
        'frequency_hz':args.frequency,'coarse_smoke':args.coarse_smoke,
        'nodes':len(mesh.nodes),'tetrahedra':len(mesh.tetrahedra),
        'metal_tetrahedra':int(np.count_nonzero(mesh.sigma)),
        'source':'2742 AT racetrack, 25x100 mm cross-section Gauss filament approximation',
        'volume_overlap_z_real_imag_ohm':[
            float(sol.delta_z[0,0].real),float(sol.delta_z[0,0].imag)],
        'true_closed_source_reaction_z_real_imag_ohm':[
            float(port_z[0,0].real),float(port_z[0,0].imag)],
        'outer_bc':args.outer_bc,'outer_box':args.outer_box,
        'closed_source_power_W_for_1A':float(.5*port_z[0,0].real),
        'source_voltage_vs_metal_joule_relative_gap':float(
            abs(.5*port_z[0,0].real-.5*sol.power_from_fields[0,0].real)/
            max(abs(.5*sol.power_from_fields[0,0].real),1e-30)),
        'volume_dissipation_W_for_1A':float(.5*sol.power_from_fields[0,0].real),
        'overlap_response_power_W_for_1A':float(.5*sol.power_matrix[0,0].real),
        'probes':probes,
        'time_phasor':'exp(+iwt), phase 0=Re, phase 90=-Im',
        'caveat':('Not experimental agreement; volume overlap is not a genuine '
                  'closed-source voltage at arbitrary finite boundaries. '
                  'The two independent power diagnostics may disagree until '
                  'mesh/source/outer-domain convergence. Inner source quadrature, first-order '
                  'Nedelec space, h, outer-domain truncation and one-sided '
                  'metal surface J probe require independent convergence.'),
    }
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
