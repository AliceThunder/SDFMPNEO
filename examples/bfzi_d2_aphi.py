"""Independent D2 A-phi entry: local smoke case, or real BFZI two-box mesh.

Run from repo root:
    python -m examples.bfzi_d2_aphi --case smoke
    python -m examples.bfzi_d2_aphi --case bfzi --write-geo /tmp/bfzi.geo
    gmsh /tmp/bfzi.geo -3 -format msh2 -o /tmp/bfzi.msh
    python -m examples.bfzi_d2_aphi --case bfzi --mesh /tmp/bfzi.msh

This remains UNVALIDATED as an infinite-ocean solution. It is completely
separate from G1 and produces only sea incremental impedance (not copper Z).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

from sdfmpneo_bfzi.direct3d_aphi import (
    box_tetra_mesh, closed_polyline_potential, solve_aphi,
)


def _report(result, case, frequency, *, outer_bc='secondary_zero', closed_path_z=None,
            vacuum_image_z=None, medium_reaction_z=None):
    def to_complex_matrix(a):
        return [[[float(v.real), float(v.imag)] for v in row] for row in a]
    phase_cases={}
    if closed_path_z is not None:
        for key,trial in {
            'I10':(1.,0.), 'I01':(0.,1.), 'I11':(1.,1.),
            'I1minus1':(1.,-1.), 'I1plusi':(1.,1j)
        }.items():
            I=np.asarray(trial,dtype=complex)
            port_w=.5*float(np.vdot(I,closed_path_z@I).real)
            sea_w=result.volume_power(I)
            phase_cases[key]={
                'closed_path_port_W':port_w,
                'total_field_seawater_joule_W':sea_w,
                'absolute_difference_W':abs(port_w-sea_w)
            }
    reciprocity_error=(abs(closed_path_z[0,1]-closed_path_z[1,0])
                       if closed_path_z is not None else None)
    return dict(
        status="D2 early 3D finite-volume-domain A-phi prototype; not TEAM-7 or G1 PASS",
        case=case, frequency_hz=frequency,
        volume_overlap_z_ohm=to_complex_matrix(result.delta_z),
        material_increment_closed_port_z_ohm=(
            to_complex_matrix(closed_path_z) if closed_path_z is not None else None),
        vacuum_image_reaction_z_ohm=(
            to_complex_matrix(vacuum_image_z)
            if vacuum_image_z is not None else None),
        total_medium_reaction_z_ohm=(
            to_complex_matrix(medium_reaction_z)
            if medium_reaction_z is not None else None),
        phase_port_vs_volume=phase_cases,
        closed_source_reciprocity_error_ohm=(
            float(reciprocity_error) if reciprocity_error is not None else None),
        port_loss_from_closed_path_W_for_1A=(
            [float(.5*closed_path_z[p,p].real) for p in range(len(closed_path_z))]
            if closed_path_z is not None else None),
        seawater_power_matrix=to_complex_matrix(result.power_matrix),
        sea_field_power_matrix=to_complex_matrix(result.power_from_fields),
        n_nodes=len(result.mesh.nodes), n_tetrahedra=len(result.mesh.tetrahedra),
        n_nedelec_edges=result.n_edge_dofs,
        n_phi=result.n_potential_dofs, n_gauge=result.n_gauge_dofs,
        boundary=outer_bc + ' on finite cube; outer-radius convergence required',
        caveat=('The volume-overlap Z is NOT guaranteed to be the genuine source '
                'port impedance for finite artificial boundaries. Only the closed-path '
                'reaction line integral measures the induced source voltage; compare '
                'its REAL part against the volume Joule power separately, then refine '
                'mesh/source/exterior. Finite wire copper self Z is missing.'),
    )


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=('smoke','bfzi'),default='smoke')
    parser.add_argument('--frequency',type=float,default=200000.)
    parser.add_argument('--write-geo',type=Path)
    parser.add_argument('--mesh',type=Path)
    parser.add_argument('--mesh-size',type=float,default=.055)
    parser.add_argument('--panels-per-corner',type=int,default=12)
    parser.add_argument('--outer-bc',choices=('secondary_zero','total_zero'),
                        default='secondary_zero')
    args=parser.parse_args()
    if args.case=='smoke':
        if args.mesh or args.write_geo:
            parser.error('smoke case uses built-in tetra mesh')
        mesh=box_tetra_mesh([[-.2,.2]]*3,[3,3,3],
                            lambda p:5. if np.max(np.abs(p)) < .075 else 0.)
        def imposed_uniform_B(p):
            return np.column_stack((-5e-4*p[:,1], 5e-4*p[:,0],
                                    np.zeros(len(p))))
        sol=solve_aphi(mesh,args.frequency,[imposed_uniform_B],
                       outer_bc=args.outer_bc)
        closed_path_z=None
        vacuum_image_z=None
        medium_reaction_z=None
    else:
        from examples.bfzi_mqs_demo import build_demo_scene
        from sdfmpneo_bfzi.source_reference import polygonal_port_vertices
        from sdfmpneo_bfzi.direct3d_mesh import (
            gmsh_bfzi_two_boxes, bfzi_mesh_from_gmsh22,
        )
        scene=build_demo_scene()
        if args.write_geo:
            args.write_geo.write_text(
                gmsh_bfzi_two_boxes(scene,mesh_size_m=args.mesh_size),
                encoding='utf-8')
            print('Wrote OBB-conforming geometry:',args.write_geo)
            if args.mesh is None:
                return
        if args.mesh is None:
            parser.error('BFZI solver needs Gmsh MSH 2.2 file (--mesh)')
        mesh=bfzi_mesh_from_gmsh22(args.mesh,scene)
        sources=[]
        port_paths=[]
        for assembly in scene.assemblies:
            polygon=polygonal_port_vertices(
                assembly,panels_per_corner=args.panels_per_corner)
            sources.append(closed_polyline_potential(polygon))
            port_paths.append([(1.,polygon)])
        sol=solve_aphi(mesh,args.frequency,sources,outer_bc=args.outer_bc)
        from sdfmpneo_bfzi.direct3d_port import medium_vacuum_port_increment
        medium_reaction_z,vacuum_image_z,closed_path_z = medium_vacuum_port_increment(
            sol,args.frequency,sources,port_paths,outer_bc=args.outer_bc)
    print(json.dumps(_report(sol,args.case,args.frequency,
                             outer_bc=args.outer_bc,closed_path_z=closed_path_z,
                             vacuum_image_z=vacuum_image_z,
                             medium_reaction_z=medium_reaction_z),
                     indent=2))


if __name__=='__main__':
    main()
