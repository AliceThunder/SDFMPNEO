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
    p.add_argument('--gauge', choices=('tree','mixed'),default='tree',
                   help='Fast relative tree-cotree gauge or legacy mixed '
                        'A-phi-gauge saddle point for independent comparison')
    p.add_argument('--write-geo', type=Path)
    p.add_argument('--mesh', type=Path)
    p.add_argument('--legacy-single-tag-msh',action='store_true',
                   help='Explicit diagnostic-only import of old .geo mesh '
                        'with all tets tagged 1; NOT usable as Elmer '
                        'material-separated reference')
    p.add_argument('--require-port-manifest',type=Path,
                   help='For imported physical MSH2: verify this exact source '
                        'contract; by default read <mesh>.ports.json')
    p.add_argument('--allow-unverified-source-manifest',action='store_true',
                   help='Diagnostic only: explicitly bypass absent source '
                        'contract on imported MSH2; never certify Elmer '
                        'or D2 under this mode')
    p.add_argument('--export-port-manifest',type=Path,
                   help='Write oriented closed BFZI 1A peak source paths, '
                        'pose, return leads and phasor to JSON')
    p.add_argument('--ports-only',action='store_true',
                   help='Export physical source manifest without needing Gmsh')
    p.add_argument('--mesh-only',action='store_true',
                   help='Only generate physically tagged Gmsh mesh, do not run '
                        'large complex SciPy Maxwell solve')
    p.add_argument('--generate-msh',type=Path,
                   help='Generate directly with Gmsh Python OCC API: '
                        'physical region ids 1=sea, 2=tx dry, 3=rx dry; '
                        'outer physical surface id 101')
    p.add_argument('--structured-axis-aligned',action='store_true',
                   help='Generate a conforming two-box structured tetra mesh '
                        'without Gmsh, ONLY for axis-aligned poses')
    p.add_argument('--structured-far-step',type=float,default=.22)
    p.add_argument('--structured-near-step',type=float,
                   help='Independent near-wall wet/dry mesh step (m), '
                        'not just the interior box_layers')
    p.add_argument('--structured-padding',type=float,default=.025)
    p.add_argument('--structured-box-layers',type=int,default=2)
    p.add_argument('--structured-inner-core-radius',type=float,
                   help='Fixed near-field cube half width for comparable '
                        'outer R sweeps; must enclose padded boxes')
    p.add_argument('--mesh-size', type=float, default=.025)
    p.add_argument('--tx-rotation-vector',type=float,nargs=3,
                   default=(0.,0.,0.),metavar=('RX','RY','RZ'),
                   help='Transmitter additional world-frame rotation vector (rad)')
    p.add_argument('--rx-rotation-vector',type=float,nargs=3,
                   default=(0.,0.,0.),metavar=('RX','RY','RZ'),
                   help='Receiver additional world-frame rotation vector (rad)')
    p.add_argument('--tx-shift',type=float,nargs=3,
                   default=(0.,0.,0.),metavar=('X','Y','Z'),
                   help='Additional transmitter center translation (m)')
    p.add_argument('--rx-shift',type=float,nargs=3,
                   default=(0.,0.,0.),metavar=('X','Y','Z'),
                   help='Additional receiver center translation (m)')
    p.add_argument('--outer-radius',type=float,
                   help='Independent outer CUBE half extent (m); keep BFZI '
                        'ports/OBBs fixed and sweep domain truncation')
    p.add_argument('--near-mesh-size',type=float,
                   help='Near both independently rotated insulating boxes (m)')
    p.add_argument('--near-padding',type=float,
                   help='World-distance padding around each oriented box (m)')
    p.add_argument('--panels-per-corner', type=int, default=12)
    p.add_argument('--output', type=Path, help='Optional full JSON report')
    p.add_argument('--summary-only', action='store_true',
                   help='Print only Z, reciprocity and phase-resolved power; '
                        'write complete fields using --output')
    p.add_argument('--solve-analytic-primary', action='store_true',
                   help='Additionally solve the Maxwell material PDE with analytic '
                        'closed-port incident A; compare physical E/Joule versus '
                        'Galerkin total-field/post-hoc correction')
    p.add_argument('--probe', type=float, nargs=3, action='append',
                   metavar=('X','Y','Z'),
                   help='3D field query in meters (repeatable, BFZI case only)')
    args = p.parse_args()

    if args.structured_axis_aligned and args.case != 'bfzi':
        p.error('--structured-axis-aligned requires --case bfzi')
    if args.structured_axis_aligned and (
            args.mesh or args.write_geo or args.generate_msh):
        p.error('--structured-axis-aligned cannot combine with Gmsh mesh sources')
    if args.ports_only and (
            args.case!='bfzi' or not args.export_port_manifest or
            args.mesh or args.generate_msh or args.write_geo):
        p.error('--ports-only requires --case bfzi and '
                '--export-port-manifest without other mesh operations')
    if args.mesh and args.export_port_manifest:
        p.error('--mesh and --export-port-manifest cannot combine: exporting '
                'would overwrite the imported mesh source evidence. '
                'Use --require-port-manifest instead.')
    if args.mesh_only and not args.generate_msh:
        p.error('--mesh-only requires --generate-msh')
    if args.require_port_manifest and (not args.mesh or
            args.allow_unverified_source_manifest or
            args.legacy_single_tag_msh):
        p.error('--require-port-manifest needs --mesh without bypass')
    if args.allow_unverified_source_manifest and not args.mesh:
        p.error('--allow-unverified-source-manifest requires --mesh')
    if args.legacy_single_tag_msh and (args.generate_msh or not args.mesh):
        p.error('--legacy-single-tag-msh requires --mesh and cannot be '
                'used with physically tagged --generate-msh')
    if args.generate_msh and args.mesh:
        p.error('--generate-msh and --mesh are mutually exclusive')
    if args.generate_msh and args.write_geo:
        p.error('Use --generate-msh alone for physically tagged Elmer-compatible geometry; '
                'legacy --write-geo writes one combined region label')
    if args.case == 'smoke' and args.probe:
        p.error('--probe is for a physical BFZI closed-port case')
    if args.case == 'smoke':
        if args.mesh or args.write_geo or args.generate_msh or args.mesh_only:
            p.error('Smoke uses built-in tetrahedra, not Gmsh mesh sources')
        mesh = box_tetra_mesh(
            [[-.2,.2]]*3, [3,3,3],
            lambda x: 5. if np.max(np.abs(x)) < .075 else 0.)
        def uniform_B(p):
            return np.column_stack((-5e-4*p[:,1],5e-4*p[:,0],
                                    np.zeros(len(p))))
        field = solve_aphi(mesh,args.frequency,[uniform_B],gauge=args.gauge)
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
        from dataclasses import replace
        from scipy.spatial.transform import Rotation
        from sdfmpneo_bfzi.geometry import RigidPose
        # Transform the entire physical assembly, never the enclosure
        # separately from its analytic continuous coil and closed return.
        def move_assembly(assembly, rotvec, displacement):
            rot=Rotation.from_rotvec(np.asarray(rotvec,float)).as_matrix()
            pose=RigidPose(
                rot@assembly.pose.rotation,
                np.asarray(assembly.pose.center,float)+
                np.asarray(displacement,float))
            return replace(assembly,pose=pose)
        scene=replace(
            scene,
            transmitter=move_assembly(
                scene.transmitter,args.tx_rotation_vector,args.tx_shift),
            receiver=move_assembly(
                scene.receiver,args.rx_rotation_vector,args.rx_shift))
        if args.outer_radius is not None:
            from dataclasses import replace
            scene = replace(scene,outer_radius=args.outer_radius)
        explicit_source_manifest=None
        if args.export_port_manifest:
            from sdfmpneo_bfzi.direct3d_port_manifest import (
                write_closed_port_source_manifest)
            explicit_source_manifest=write_closed_port_source_manifest(
                scene,args.export_port_manifest,args.frequency,
                panels_per_corner=args.panels_per_corner)
            print('Wrote actual closed BFZI source contract:',
                  args.export_port_manifest)
        if args.ports_only:
            print('BFZI source SHA256:',
                  explicit_source_manifest['sha256_of_contract_without_hash'])
            return
        if args.write_geo:
            args.write_geo.parent.mkdir(parents=True,exist_ok=True)
            args.write_geo.write_text(
                gmsh_bfzi_two_boxes(scene, mesh_size_m=args.mesh_size,
                                    near_mesh_size_m=args.near_mesh_size,
                                    near_padding_m=args.near_padding),
                encoding='utf-8')
            print('Wrote two independently posed OBBs:',args.write_geo)
            if args.mesh is None and args.generate_msh is None:
                return
        physical_mesh_metadata=None
        source_manifest_verification='not_applicable_structured'
        verified_source_manifest=None
        if args.generate_msh:
            from sdfmpneo_bfzi.direct3d_gmsh_physical import (
                generate_physical_bfzi_msh22)
            physical_mesh_metadata=generate_physical_bfzi_msh22(
                scene,args.generate_msh,mesh_size_m=args.mesh_size,
                near_mesh_size_m=args.near_mesh_size,
                near_padding_m=args.near_padding)
            if explicit_source_manifest is None:
                from sdfmpneo_bfzi.direct3d_port_manifest import (
                    write_closed_port_source_manifest)
                source_path=args.generate_msh.with_suffix('.ports.json')
                source_manifest=write_closed_port_source_manifest(
                    scene,source_path,args.frequency,
                    panels_per_corner=args.panels_per_corner)
            else:
                source_path=args.export_port_manifest
                source_manifest=explicit_source_manifest
            physical_mesh_metadata['closed_port_manifest']=str(source_path)
            physical_mesh_metadata['closed_port_source_sha256']=(
                source_manifest['sha256_of_contract_without_hash'])
            verified_source_manifest=source_manifest
            source_manifest_verification='generated_for_this_mesh'
            if args.mesh_only:
                print(json.dumps(physical_mesh_metadata,indent=2))
                if args.output:
                    args.output.parent.mkdir(parents=True,exist_ok=True)
                    args.output.write_text(
                        json.dumps(physical_mesh_metadata,indent=2),
                        encoding='utf-8')
                return
            mesh=bfzi_mesh_from_gmsh22(args.generate_msh,scene)
            mesh_source="Gmsh OCC physically tagged conforming "
            mesh_source+="SE(3) seawater/tx/rx + outer surface"
        elif args.structured_axis_aligned:
            from sdfmpneo_bfzi.direct3d_structured_bfzi import (
                axis_aligned_bfzi_tetra_mesh)
            mesh = axis_aligned_bfzi_tetra_mesh(
                scene,near_padding_m=args.structured_padding,
                far_step_m=args.structured_far_step,
                box_layers=args.structured_box_layers,
                near_step_m=args.structured_near_step,
                inner_core_radius_m=args.structured_inner_core_radius)
            mesh_source="structured axis-aligned physical OBB; initial D2 smoke only"
        elif args.mesh is not None:
            if args.legacy_single_tag_msh:
                source_manifest_verification='legacy_diagnostic_not_certified'
            else:
                from sdfmpneo_bfzi.direct3d_port_manifest import (
                    verify_closed_port_source_manifest)
                required_source=(args.require_port_manifest or
                                 args.mesh.with_suffix('.ports.json'))
                if required_source.is_file():
                    verified_source_manifest=verify_closed_port_source_manifest(
                        scene,required_source,args.frequency,
                        panels_per_corner=args.panels_per_corner)
                    source_manifest_verification='validated_against_current_run'
                elif args.allow_unverified_source_manifest:
                    source_manifest_verification='unverified_diagnostic_only'
                else:
                    raise ValueError(
                        'Physical MSH2 requires a matching two-port source '
                        'manifest (<mesh>.ports.json or --require-port-manifest). '
                        'Use --allow-unverified-source-manifest only for '
                        'explicit NON-CERTIFIED diagnostics.')
            mesh = bfzi_mesh_from_gmsh22(
                args.mesh,scene,
                allow_legacy_single_tag=args.legacy_single_tag_msh)
            mesh_source=(
                "legacy single-tag Gmsh, centroid material diagnostic ONLY"
                if args.legacy_single_tag_msh else
                "Gmsh OCC physical volumes 1=sea 2=tx 3=rx, "
                "conforming independently posed SE(3) geometry")
        else:
            p.error('BFZI requires either --structured-axis-aligned '
                    'or a physical Gmsh MSH 2.2 tetrahedral file (--mesh)')
        ports = [
            [(1.,polygonal_port_vertices(
                assembly,panels_per_corner=args.panels_per_corner))]
            for assembly in scene.assemblies]
        reference = solve_closed_ports(mesh,args.frequency,ports,
                                       gauge=args.gauge)
        physical = None
        if args.solve_analytic_primary:
            from sdfmpneo_bfzi.direct3d_primary_secondary import (
                solve_closed_primary_secondary)
            physical = solve_closed_primary_secondary(
                mesh,args.frequency,ports,gauge=args.gauge)
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
            'mesh_source':mesh_source,
            'physical_gmsh_metadata':physical_mesh_metadata,
            'source_manifest_verification':source_manifest_verification,
            'closed_port_source_sha256':(
                verified_source_manifest['sha256_of_contract_without_hash']
                if verified_source_manifest is not None else None),
            'gauge':args.gauge,
            'mixed_gauge_multiplier_unknowns':int(reference.medium.n_gauge_dofs),
            'medium_solver_unknowns':int(
                reference.medium.n_edge_dofs+
                reference.medium.n_potential_dofs+
                reference.medium.n_gauge_dofs),
            'vacuum_solver_unknowns':int(
                reference.vacuum.n_edge_dofs+
                reference.vacuum.n_potential_dofs+
                reference.vacuum.n_gauge_dofs),
            'medium_scaled_relative_residual':reference.medium.linear_relative_residual,
            'vacuum_scaled_relative_residual':reference.vacuum.linear_relative_residual,
            'outer_cube_half_extent_m':float(scene.outer_radius),
            'structured_axis_aligned_only':bool(args.structured_axis_aligned),
            'structured_near_step_m':(args.structured_near_step if args.structured_axis_aligned else None),
            'structured_inner_core_radius_m':(
                args.structured_inner_core_radius if args.structured_axis_aligned else None),
            'boundary':'finite cube, n cross A_total=0',
            'transmitter_pose':{
                'center_m':scene.transmitter.pose.center.tolist(),
                'rotation_matrix':scene.transmitter.pose.rotation.tolist()},
            'receiver_pose':{
                'center_m':scene.receiver.pose.center.tolist(),
                'rotation_matrix':scene.receiver.pose.rotation.tolist()},
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
    if args.summary_only and args.case == 'bfzi':
        short = {key: report[key] for key in (
            'status','frequency_hz','mesh_source','tetrahedra',
            'gauge','medium_solver_unknowns','vacuum_solver_unknowns',
            'medium_scaled_relative_residual','vacuum_scaled_relative_residual',
            'material_increment_2port_z_ohm',
            'two_port_reciprocity_gap_ohm','phase_port_vs_seawater_joule')}
        print(json.dumps(short,indent=2))
    elif args.summary_only:
        print(json.dumps({key:report[key] for key in (
            'status','frequency_hz','tetrahedra',
            'volume_joule_W_for_unit_imposed_source')},indent=2))
    else:
        print(text)


if __name__ == '__main__':
    main()
