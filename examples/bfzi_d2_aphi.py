"""D2 independent 3D edge-Galerkin A-phi reference for REAL closed BFZI ports.

    python -m examples.bfzi_d2_aphi --case smoke
    python -m examples.bfzi_d2_aphi --case bfzi --write-geo bfzi_d2.geo
    gmsh bfzi_d2.geo -3 -format msh2 -o bfzi_d2.msh
    python -m examples.bfzi_d2_aphi --case bfzi --mesh bfzi_d2.msh

Two true continuous BFZI centerlines and explicit return routes are
approximated by exactly CLOSED physical polygonal chords. The *same*
oriented edge load is used for field excitation and port measurement.
Copper self/skin losses and infinite-ocean convergence are NOT certified.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

from sdfmpneo_bfzi.direct3d_aphi import box_tetra_mesh, solve_aphi, closed_polyline_potential
from sdfmpneo_bfzi.direct3d_galerkin import solve_closed_ports


def complex_matrix(a):
    return [[[float(v.real),float(v.imag)] for v in row] for row in a]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', choices=('smoke','bfzi'), default='smoke')
    p.add_argument('--frequency', type=float, default=200000.)
    p.add_argument('--write-geo', type=Path)
    p.add_argument('--mesh', type=Path)
    p.add_argument('--mesh-size', type=float, default=.025)
    p.add_argument('--panels-per-corner', type=int, default=12)
    p.add_argument('--output', type=Path, help='Optional full JSON report')
    p.add_argument('--solve-analytic-primary', action='store_true',
                   help='Additionally solve the Maxwell material PDE with analytic '
                        'closed-port incident A; compare physical E/Joule versus '
                        'Galerkin total-field/post-hoc correction')
    p.add_argument('--probe', type=float, nargs=3, action='append',
                   metavar=('X','Y','Z'),
                   help='3D field query in meters (repeatable, BFZI case only)')
    args = p.parse_args()

    if args.case == 'smoke' and args.probe:
        p.error('--probe is for a physical BFZI closed-port case')
    if args.case == 'smoke':
        if args.mesh or args.write_geo:
            p.error('Smoke uses built-in tetrahedra, not a Gmsh mesh')
        mesh = box_tetra_mesh(
            [[-.2,.2]]*3, [3,3,3],
            lambda x: 5. if np.max(np.abs(x)) < .075 else 0.)
        def uniform_B(p):
            return np.column_stack((-5e-4*p[:,1],5e-4*p[:,0],
                                    np.zeros(len(p))))
        field = solve_aphi(mesh,args.frequency,[uniform_B])
        report = {
            'status':'D2 initial imposed-field smoke ONLY; not a physical port',
            'frequency_hz':args.frequency,
            'tetrahedra':len(mesh.tetrahedra),
            'volume_joule_W_for_unit_imposed_source':
                field.volume_power(np.array([1.+0j])),
            'caveat':'Imposed background B is not a closed source port or copper Z',
        }
    else:
        from examples.bfzi_mqs_demo import build_demo_scene
        from sdfmpneo_bfzi.source_reference import polygonal_port_vertices
        from sdfmpneo_bfzi.direct3d_mesh import (
            gmsh_bfzi_two_boxes, bfzi_mesh_from_gmsh22)
        scene = build_demo_scene()
        if args.write_geo:
            args.write_geo.write_text(
                gmsh_bfzi_two_boxes(scene, mesh_size_m=args.mesh_size),
                encoding='utf-8')
            print('Wrote two independently posed OBBs:',args.write_geo)
            if args.mesh is None:
                return
        if args.mesh is None:
            p.error('BFZI needs a Gmsh MSH 2.2 tetrahedral file (--mesh)')
        mesh = bfzi_mesh_from_gmsh22(args.mesh,scene)
        ports = [
            [(1.,polygonal_port_vertices(
                assembly,panels_per_corner=args.panels_per_corner))]
            for assembly in scene.assemblies]
        reference = solve_closed_ports(mesh,args.frequency,ports)
        physical = None
        if args.solve_analytic_primary:
            from sdfmpneo_bfzi.direct3d_primary_secondary import (
                solve_closed_primary_secondary)
            physical = solve_closed_primary_secondary(
                mesh,args.frequency,ports)
        physical_source_potentials = [
            closed_polyline_potential(port[0][1]) for port in ports]
        corrected_joule_matrix = reference.analytic_source_corrected_joule(
            args.frequency, physical_source_potentials)
        phase_cases = {}
        for name,current in {
            'I10':(1.,0.),'I01':(0.,1.),'I11':(1.,1.),
            'I1minus1':(1.,-1.),'I1plusi':(1.,1j),
        }.items():
            i = np.asarray(current,complex)
            port = reference.port_power(i)
            volume = reference.conductor_power(i)
            corrected = float((.5*np.vdot(i, corrected_joule_matrix@i)).real)
            phase_cases[name] = {
                'PDE_solved_analytic_primary_seawater_joule_W':(
                    physical.volume_power(i) if physical is not None else None),
                'analytic_source_corrected_seawater_joule_W':corrected,
                'corrected_vs_port_absolute_gap_W':abs(port-corrected),
                'material_port_W':port,
                'total_field_seawater_joule_W':volume,
                'absolute_gap_W':abs(port-volume),
            }
        probe_report = None
        if args.probe:
            points = np.asarray(args.probe, dtype=float)
            def carray(a):
                a = np.asarray(a)
                return np.stack((a.real,a.imag),axis=-1).tolist()
            if physical is not None:
                sample = physical.sample_fields(
                    points,prefer_conductor=True)
                e = sample['E_total_V_m']
                b = sample['B_total_T']
            else:
                from sdfmpneo_bfzi.direct3d_fields import (
                    sample_primary_secondary_fields)
                sample = sample_primary_secondary_fields(
                    reference, ports, args.frequency, points,
                    prefer_conductor=True)
                e = sample['E_total_primary_secondary_V_m']
                b = sample['B_total_corrected_T']
            sigma = sample['sigma_S_m']
            phased = {}
            for name,current in {
                'I10':(1.,0.), 'I01':(0.,1.),
                'I11':(1.,1.), 'I1minus1':(1.,-1.),
                'I1plusi':(1.,1j),
            }.items():
                I = np.asarray(current,dtype=complex)
                ei = np.einsum('p,pnc->nc',I,e)
                bi = np.einsum('p,pnc->nc',I,b)
                phased[name] = {
                    'B_T':carray(bi),
                    'E_V_m':carray(ei),
                    'J_A_m2':carray(sigma[:,None]*ei),
                    'q_W_m3':(.5*sigma*np.sum(abs(ei)**2,axis=-1)).tolist(),
                }
            probe_report = {
                'positions_m':points.tolist(),
                'selected_tetrahedra':sample['cell_index'].tolist(),
                'sigma_S_m':sigma.tolist(),
                'corrected_B_per_unit_port_T':carray(b),
                'corrected_E_per_unit_port_V_m':carray(e),
                'material_B_reaction_per_unit_port_T':carray(
                    sample['B_secondary_T'] if physical is not None
                    else sample['B_material_reaction_fem_T']),
                'raw_fem_B_per_unit_port_T':carray(
                    reference.medium.magnetic_secondary[:,sample['cell_index']]
                    if physical is not None else sample['B_total_raw_fem_T']),
                'current_combinations':phased,
                'method':('PDE-solved analytic primary plus FEM reaction'
                          if physical is not None else
                          'Post-hoc corrected coarse total-Galerkin FEM'),
                'warning':('These primary-secondary spatial E/J/q fields '
                           'are independent diagnostics; they are not certified '
                           'by the discrete Galerkin port-power identity. '
                           'No source-wire or outside-domain probes.'),
            }
        report = {
            'status':'D2 3D BFZI closed-port Galerkin core; NOT high-fidelity PASS',
            'frequency_hz':args.frequency,
            'boundary':'finite cube, n cross A_total=0',
            'geometry':'two genuine BFZI continuous spirals and closed returns, '
                       'polygonal centerline approximation, two independent OBBs',
            'tetrahedra':len(mesh.tetrahedra),
            'material_increment_2port_z_ohm':complex_matrix(
                reference.material_increment_z),
            'finite_box_vacuum_2port_z_ohm':complex_matrix(
                reference.vacuum_port_z),
            'analytic_source_corrected_joule_matrix':complex_matrix(
                corrected_joule_matrix),
            'PDE_analytic_primary_volume_overlap_z_ohm':(
                complex_matrix(physical.volume_overlap_z)
                if physical is not None else None),
            'PDE_analytic_primary_closed_source_reaction_z_ohm':(
                complex_matrix(physical.closed_source_reaction_z)
                if physical is not None else None),
            'PDE_analytic_primary_warning':(
                'Analytic primary is imposed INSIDE the Maxwell conductivity '
                'weak form, not substituted after solving. Its finite-domain '
                'closed-path reaction Z and volume-overlap Z can differ by '
                'boundary/discretization terms; neither is a certified '
                'infinite-seawater port until h/R convergence.'
                if physical is not None else None),
            'two_port_reciprocity_gap_ohm':float(abs(
                reference.material_increment_z[0,1]-
                reference.material_increment_z[1,0])),
            'phase_port_vs_seawater_joule':phase_cases,
            'spatial_field_probes':probe_report,
            'limitations':'Discrete source/port Joule equality does NOT certify corrected-source field accuracy. No finite copper self/skin/proximity losses, '
                          'no independently verified outer-domain and h convergence, '
                          'no completed Elmer TEAM-7 experiment agreement',
        }
    text = json.dumps(report,indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(text,encoding='utf-8')
        print('Wrote complete D2 report:', args.output)
    print(text)


if __name__ == '__main__':
    main()
