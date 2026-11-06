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


def _report(result, case, frequency):
    def to_complex_matrix(a):
        return [[[float(v.real), float(v.imag)] for v in row] for row in a]
    return dict(
        status="D2 early 3D finite-volume-domain A-phi prototype; not TEAM-7 or G1 PASS",
        case=case, frequency_hz=frequency,
        delta_z_sw_ohm=to_complex_matrix(result.delta_z),
        seawater_power_matrix=to_complex_matrix(result.power_matrix),
        sea_field_power_matrix=to_complex_matrix(result.power_from_fields),
        n_nodes=len(result.mesh.nodes), n_tetrahedra=len(result.mesh.tetrahedra),
        n_nedelec_edges=result.n_edge_dofs,
        n_phi=result.n_potential_dofs, n_gauge=result.n_gauge_dofs,
        boundary="zero tangential secondary A on finite cube; outer-radius convergence required",
        caveat="Thin closed filament has no finite copper self Z; no TEAM-7 validation",
    )


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=('smoke','bfzi'),default='smoke')
    parser.add_argument('--frequency',type=float,default=200000.)
    parser.add_argument('--write-geo',type=Path)
    parser.add_argument('--mesh',type=Path)
    parser.add_argument('--mesh-size',type=float,default=.055)
    parser.add_argument('--panels-per-corner',type=int,default=12)
    args=parser.parse_args()
    if args.case=='smoke':
        if args.mesh or args.write_geo:
            parser.error('smoke case uses built-in tetra mesh')
        mesh=box_tetra_mesh([[-.2,.2]]*3,[3,3,3],
                            lambda p:5. if np.max(np.abs(p)) < .075 else 0.)
        def imposed_uniform_B(p):
            return np.column_stack((-5e-4*p[:,1], 5e-4*p[:,0],
                                    np.zeros(len(p))))
        sol=solve_aphi(mesh,args.frequency,[imposed_uniform_B])
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
        for assembly in scene.assemblies:
            polygon=polygonal_port_vertices(
                assembly,panels_per_corner=args.panels_per_corner)
            sources.append(closed_polyline_potential(polygon))
        sol=solve_aphi(mesh,args.frequency,sources)
    print(json.dumps(_report(sol,args.case,args.frequency),indent=2))


if __name__=='__main__':
    main()
