"""D2 BFZI: executable Gmsh-to-ElmerGrid-to-two-source A-phi handoff.

This stages physically tagged, conforming independently posed BFZI
geometry and the two REAL unit-current sources on the same tetrahedra.
It never claims that arbitrary elemental J equals the original filament
source or that a standalone BFZI A-phi solve is an independent Elmer solve.

Example on a workstation with ElmerGrid:
    python -m examples.bfzi_d2_elmer_handoff --mesh runs/tilt.msh \
      --rx-rotation-vector .25 -.13 .33 \
      --panels-per-corner 16 --run-elmergrid \
      --elmer-grid-dir runs/tilt_elmer --output runs/tilt_bridge.json

After the *actual* Elmer coil run independently exports two distinct,
mapped, real unit-source J fields, add:
    --two-unit-currents runs/tilt_two_port_element_J.txt \
    --fields-npz runs/tilt_bridge_fields.npz

Even a successful J handoff is a diagnostic until both solvers have the
same impressed source field after any Elmer Jfix projection and equivalent
materials, boundary conditions, terminals and frequency-domain convention.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from scipy.spatial.transform import Rotation

from examples.bfzi_mqs_demo import build_demo_scene
from sdfmpneo_bfzi.geometry import RigidPose
from sdfmpneo_bfzi.direct3d_mesh import (
    bfzi_mesh_from_gmsh22,read_gmsh22)
from sdfmpneo_bfzi.direct3d_elmer_interchange import (
    elmer_to_gmsh_tetra_indices,load_two_unit_element_currents,
    verify_elmer_converted_bfzi_groups)
from sdfmpneo_bfzi.direct3d_port_manifest import (
    verify_closed_port_source_manifest)


def _complex_matrix(values):
    a=np.asarray(values,complex)
    return np.stack((a.real,a.imag),axis=-1).tolist()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mesh",type=Path,required=True,
                   help="Existing genuine three-material Gmsh OCC MSH 2.2")
    p.add_argument("--elmer-grid-dir",type=Path,required=True)
    p.add_argument("--run-elmergrid",action="store_true",
                   help="Run LOCAL ElmerGrid 14 2 -autoclean -names -out")
    p.add_argument("--port-manifest",type=Path,
                   help="Default: <mesh>.ports.json (must match this scene)")
    p.add_argument("--two-unit-currents",type=Path,
                   help="EXPLICIT Elmer-ID + J1xyz J2xyz data; not "
                        "automatically produced by ElmerSolver")
    p.add_argument("--frequency",type=float,default=200000.)
    p.add_argument("--panels-per-corner",type=int,default=12)
    p.add_argument("--gauge",choices=("tree","mixed"),default="tree")
    p.add_argument("--tx-rotation-vector",type=float,nargs=3,
                   default=(0.,0.,0.))
    p.add_argument("--rx-rotation-vector",type=float,nargs=3,
                   default=(0.,0.,0.))
    p.add_argument("--tx-shift",type=float,nargs=3,default=(0.,0.,0.))
    p.add_argument("--rx-shift",type=float,nargs=3,default=(0.,0.,0.))
    p.add_argument("--outer-radius",type=float)
    p.add_argument("--output",type=Path,
                   help="Optional compact JSON result")
    p.add_argument("--fields-npz",type=Path,
                   help="Optional complete 2-port spatial E/B/J/q arrays; "
                        "requires --two-unit-currents")
    args=p.parse_args()
    if args.fields_npz and not args.two_unit_currents:
        p.error("--fields-npz requires --two-unit-currents")
    if not np.isfinite(args.frequency) or args.frequency<=0:
        p.error("--frequency must be positive and finite")

    scene=build_demo_scene()
    def move(assembly,rotvec,shift):
        rotation=Rotation.from_rotvec(np.asarray(rotvec,float)).as_matrix()
        return replace(assembly,pose=RigidPose(
            rotation@assembly.pose.rotation,
            np.asarray(assembly.pose.center,float)+np.asarray(shift,float)))
    scene=replace(scene,
        transmitter=move(
            scene.transmitter,args.tx_rotation_vector,args.tx_shift),
        receiver=move(
            scene.receiver,args.rx_rotation_vector,args.rx_shift))
    if args.outer_radius is not None:
        scene=replace(scene,outer_radius=args.outer_radius)

    manifest=args.port_manifest or args.mesh.with_suffix(".ports.json")
    source=verify_closed_port_source_manifest(
        scene,manifest,args.frequency,
        panels_per_corner=args.panels_per_corner)
    # Both reads must agree on every tetrahedral physical ID and the entire
    # surface 101; no centroid relabeling or legacy tag fallback is allowed.
    mesh=bfzi_mesh_from_gmsh22(args.mesh,scene)
    _,original_tags=read_gmsh22(
        args.mesh,{1:scene.seawater.conductivity,2:0.,3:0.},
        return_physical_tags=True)

    if args.run_elmergrid:
        elmergrid=shutil.which("ElmerGrid")
        if elmergrid is None:
            raise RuntimeError(
                "ElmerGrid not installed on this workstation; cannot "
                "certify a synthetic or unexecuted conversion")
        cmd=[elmergrid,"14","2",str(args.mesh),"-autoclean","-names",
             "-out",str(args.elmer_grid_dir)]
        args.elmer_grid_dir.parent.mkdir(parents=True,exist_ok=True)
        completed=subprocess.run(
            cmd,capture_output=True,text=True,check=False)
        if completed.returncode:
            raise RuntimeError(
                f"ElmerGrid exited {completed.returncode}: "
                +(completed.stderr or completed.stdout)[-3000:])

    mapping=elmer_to_gmsh_tetra_indices(
        mesh,args.elmer_grid_dir)
    groups=verify_elmer_converted_bfzi_groups(
        mesh,original_tags,args.elmer_grid_dir,mapping,scene)
    report={
        "status":"D2_ELMERGRID_MATERIAL_BOUNDARY_MAPPING_PASS_ONLY",
        "source_sha256":source["sha256_of_contract_without_hash"],
        "mesh":str(args.mesh),
        "elmer_grid_directory":str(args.elmer_grid_dir),
        "frequency_hz":float(args.frequency),
        "tetrahedra":len(mesh.tetrahedra),
        "elmer_grid_groups":groups,
        "physical_source_kind":"ideal_filament_manifest_only",
        "independent_elmer_solution":"NOT_RUN_BY_THIS_SCRIPT",
        "finite_volume_source_equivalence":"NOT_ESTABLISHED",
        "D2":"OPEN",
    }
    if args.two_unit_currents:
        from sdfmpneo_bfzi.direct3d_edge_source import (
            elemental_current_edge_load,source_nodal_conservation)
        from sdfmpneo_bfzi.direct3d_galerkin import (
            solve_elemental_current_ports)
        J=load_two_unit_element_currents(
            args.two_unit_currents,mapping,len(mesh.tetrahedra))
        load=elemental_current_edge_load(mesh,J)
        residual=source_nodal_conservation(mesh,load)
        report["volume_source_P1_gradient_defect"]=residual.tolist()
        report["physical_source_kind"]="two_real_elemental_J_basis"
        report["volume_source_1A_flux_and_direction"]="UNVERIFIED"
        report["post_Elmer_Jfix_source_identity"]="UNVERIFIED"
        if np.max(residual)>1e-8:
            report["status"]="D2_VOLUME_SOURCE_CONSERVATION_FAIL"
        else:
            result=solve_elemental_current_ports(
                mesh,args.frequency,J,gauge=args.gauge)
            report.update({
                "status":"D2_SHARED_ELEMENTAL_SOURCE_APHI_ONLY_NOT_ELMER_PASS",
                "material_increment_2port_z_ohm":_complex_matrix(
                    result.material_increment_z),
                "finite_box_vacuum_2port_z_ohm":_complex_matrix(
                    result.vacuum_port_z),
                "finite_box_medium_2port_z_ohm":_complex_matrix(
                    result.medium_port_z),
                "reciprocity_gap_ohm":float(abs(
                    result.material_increment_z[0,1]-
                    result.material_increment_z[1,0])),
                "medium_linear_relative_residual":
                    result.medium.linear_relative_residual,
                "vacuum_linear_relative_residual":
                    result.vacuum.linear_relative_residual,
                "phase_current_power_W":{
                    name:{
                        "material_port":result.port_power(np.asarray(i,complex)),
                        "seawater_volume":result.conductor_power(
                            np.asarray(i,complex))}
                    for name,i in {
                        "I10":(1,0),"I01":(0,1),"I11":(1,1),
                        "I1minus1":(1,-1),"I1plusi":(1,1j)}.items()},
            })
            if args.fields_npz:
                args.fields_npz.parent.mkdir(parents=True,exist_ok=True)
                from sdfmpneo_bfzi.direct3d_aphi import _BARY4
                E=result.medium.electric_at_quadrature
                sigma=mesh.sigma
                quadrature_m=np.einsum(
                    "qi,tij->tqj",_BARY4,mesh.nodes[mesh.tetrahedra])
                # Complete 2x2 Hermitian spatial heating tensor:
                # q(x; I) = I* Q(x) I, including arbitrary port phases.
                # Omitting Q_12 would silently lose interference heating.
                Q=0.5*np.einsum(
                    "ptqc,rtqc,t->tqpr",E.conj(),E,sigma,optimize=True)
                np.savez_compressed(
                    args.fields_npz,
                    nodes_m=mesh.nodes,
                    tetrahedra=mesh.tetrahedra,
                    quadrature_positions_m=quadrature_m,
                    sigma_S_m=sigma,
                    frequency_hz=np.asarray(args.frequency),
                    time_phasor=np.asarray("exp(+i*omega*t)"),
                    source_contract_sha256=np.asarray(
                        source["sha256_of_contract_without_hash"]),
                    E_unit_ports_V_m=E,
                    B_unit_ports_T=result.medium.magnetic_secondary,
                    J_unit_ports_A_m2=sigma[None,:,None,None]*E,
                    q_basis_W_m3=Q[:,:,[0,1],[0,1]].transpose(2,0,1),
                    q_port_cross_W_m3=Q,
                    source_element_J_A_m2=J)
                report["spatial_fields_npz"]=str(args.fields_npz)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(
            json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))
    if report["status"]=="D2_VOLUME_SOURCE_CONSERVATION_FAIL":
        raise SystemExit(2)


if __name__=="__main__":
    main()
