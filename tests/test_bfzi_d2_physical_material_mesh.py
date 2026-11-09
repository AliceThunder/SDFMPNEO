"""D2: physically tagged MSH2 import must preserve actual BFZI material PDE.

The MSH file is written from a known conforming *restricted* BFZI tetra mesh,
not from unavailable OCC/Gmsh. This tests the real MSH2 importer and both
closed-port solves; actual independent SE(3) OCC/Elmer runs remain separate.
"""
from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from examples.bfzi_mqs_demo import build_demo_scene
from sdfmpneo_bfzi.direct3d_mesh import bfzi_mesh_from_gmsh22
from sdfmpneo_bfzi.direct3d_galerkin import solve_closed_ports
from sdfmpneo_bfzi.direct3d_structured_bfzi import axis_aligned_bfzi_tetra_mesh
from sdfmpneo_bfzi.source_reference import polygonal_port_vertices


def _physical_file(tmp_path, scene, *, tag_override=None,
                   surface_override=None):
    mesh=axis_aligned_bfzi_tetra_mesh(
        scene,far_step_m=.30,near_padding_m=.02,box_layers=1)
    centroids=mesh.nodes[mesh.tetrahedra].mean(axis=1)
    tx=scene.transmitter.enclosure.contains(centroids,scene.transmitter.pose)
    rx=scene.receiver.enclosure.contains(centroids,scene.receiver.pose)
    tags=np.where(tx,2,np.where(rx,3,1))
    if tag_override is not None:
        tags=tag_override(tags.copy())
    tets=mesh.tetrahedra
    all_faces=np.concatenate((
        tets[:,(0,1,2)],tets[:,(0,1,3)],
        tets[:,(0,2,3)],tets[:,(1,2,3)]),axis=0)
    faces,counts=np.unique(np.sort(all_faces,axis=1),axis=0,
                           return_counts=True)
    outer_faces=faces[counts==1]
    if surface_override is not None:
        outer_faces=surface_override(outer_faces.copy())
    path=tmp_path/'bfzi_physical123.msh'
    with path.open('w',encoding='utf-8') as fh:
        fh.write('$MeshFormat\n2.2 0 8\n$EndMeshFormat\n')
        fh.write(f'$Nodes\n{len(mesh.nodes)}\n')
        for index,xyz in enumerate(mesh.nodes,1):
            fh.write(f'{index} {xyz[0]:.17g} {xyz[1]:.17g} '
                     f'{xyz[2]:.17g}\n')
        fh.write(f'$EndNodes\n$Elements\n{len(mesh.tetrahedra)+len(outer_faces)}\n')
        for index,tri in enumerate(outer_faces,1):
            fh.write(f'{index} 2 2 101 101 '
                     + ' '.join(str(v+1) for v in tri)+'\n')
        for index,(tet,tag) in enumerate(
                zip(mesh.tetrahedra,tags),len(outer_faces)+1):
            fh.write(f'{index} 4 2 {tag} {tag} '
                     + ' '.join(str(v+1) for v in tet)+'\n')
        fh.write('$EndElements\n')
    return path,mesh


@pytest.mark.parametrize("receiver_orthogonal",[False,True])
def test_three_physical_regions_preserve_real_two_port_E_B_Z_and_joule(
        tmp_path,receiver_orthogonal):
    scene=replace(build_demo_scene(),outer_radius=.32)
    if receiver_orthogonal:
        from sdfmpneo_bfzi.geometry import RigidPose
        rotation=np.array([[1.,0.,0.],[0.,0.,-1.],[0.,1.,0.]])
        scene=replace(scene,receiver=replace(
            scene.receiver,
            pose=RigidPose(rotation,scene.receiver.pose.center)))
    path,original=_physical_file(tmp_path,scene)
    restored=bfzi_mesh_from_gmsh22(path,scene)
    assert_allclose(restored.nodes,original.nodes,rtol=0,atol=0)
    assert_allclose(restored.sigma,original.sigma,rtol=0,atol=0)
    assert len(restored.tetrahedra)==len(original.tetrahedra)==(
        2430 if receiver_orthogonal else 1350)
    ports=[[(1.,polygonal_port_vertices(a,panels_per_corner=2))]
           for a in scene.assemblies]
    before=solve_closed_ports(original,200000.,ports,gauge='tree')
    after=solve_closed_ports(restored,200000.,ports,gauge='tree')
    assert_allclose(after.material_increment_z,before.material_increment_z,
                    rtol=1e-10,atol=1e-12)
    assert_allclose(after.medium.electric_at_quadrature,
                    before.medium.electric_at_quadrature,rtol=1e-10,atol=1e-12)
    assert_allclose(after.medium.magnetic_secondary,
                    before.medium.magnetic_secondary,rtol=1e-10,atol=1e-12)
    current=np.array([1.,1j])
    assert_allclose(after.conductor_power(current),
                    before.conductor_power(current),rtol=1e-10,atol=1e-12)


def test_msh2_rejects_swapped_or_missing_regions_and_wrong_outer_R(tmp_path):
    scene=replace(build_demo_scene(),outer_radius=.32)
    path,_=_physical_file(
        tmp_path,scene,tag_override=lambda tags:np.where(
            tags==2,3,np.where(tags==3,2,tags)))
    with pytest.raises(ValueError,match='physical material tags disagree'):
        bfzi_mesh_from_gmsh22(path,scene)
    path,_=_physical_file(
        tmp_path,scene,tag_override=lambda tags:np.where(tags==3,1,tags))
    with pytest.raises(ValueError,match='exactly physical volumes'):
        bfzi_mesh_from_gmsh22(path,scene)
    path,_=_physical_file(
        tmp_path,scene,tag_override=lambda tags:np.ones_like(tags))
    with pytest.raises(ValueError,match='Single-region legacy'):
        bfzi_mesh_from_gmsh22(path,scene)
    legacy=bfzi_mesh_from_gmsh22(
        path,scene,allow_legacy_single_tag=True)
    assert np.count_nonzero(legacy.sigma==0)>0
    path,_=_physical_file(tmp_path,scene)
    changed=replace(scene,outer_radius=.40)
    with pytest.raises(ValueError,match='outer radius'):
        bfzi_mesh_from_gmsh22(path,changed)


def test_msh2_rejects_incomplete_external_surface_101(tmp_path):
    scene=replace(build_demo_scene(),outer_radius=.32)
    path,_=_physical_file(
        tmp_path,scene,surface_override=lambda faces:faces[1:])
    with pytest.raises(ValueError,match='physical surface 101'):
        bfzi_mesh_from_gmsh22(path,scene)
    path,_=_physical_file(
        tmp_path,scene,surface_override=lambda faces:faces[:0])
    with pytest.raises(ValueError,match='missing physical surface 101'):
        bfzi_mesh_from_gmsh22(path,scene)

def test_occ_rotated_enclosure_must_not_be_clipped_by_outer_cube(tmp_path):
    """World AABB of the actual rotated dry volume must fit inside R."""
    from sdfmpneo_bfzi.direct3d_gmsh_physical import (
        generate_physical_bfzi_msh22)
    from sdfmpneo_bfzi.geometry import RigidPose
    from scipy.spatial.transform import Rotation
    scene=replace(build_demo_scene(),outer_radius=.32)
    receiver=scene.receiver
    rotated=RigidPose(
        Rotation.from_rotvec([0.,0.,np.pi/4]).as_matrix(),
        receiver.pose.center+np.array([.18,0.,0.]))
    bad_receiver=replace(receiver,pose=rotated)
    # The public BfziScene validates a tighter enclosing SPHERE before any
    # mesher runs.  This invalid pose must be rejected there first.
    with pytest.raises(ValueError,match='enclosure touches artificial outer domain'):
        replace(scene,receiver=bad_receiver)
    # Exercise the OCC generator's independent CUBE preflight without
    # requiring gmsh: supply the deliberately invalid geometric inputs
    # directly, bypassing only the already-verified BfziScene constructor.
    from types import SimpleNamespace
    invalid_geometry=SimpleNamespace(
        outer_center=scene.outer_center,
        outer_radius=scene.outer_radius,
        assemblies=(scene.transmitter,bad_receiver))
    with pytest.raises(ValueError,match='inside the finite exterior cube'):
        generate_physical_bfzi_msh22(
            invalid_geometry,tmp_path/'should_not_mesh.msh',mesh_size_m=.08)


def test_real_bfzi_msh_to_elmergrid_handoff_cli_without_solver(
        tmp_path,monkeypatch):
    """Run the actual handoff CLI on BFZI three-material MSH2 and Elmer text.

    The Elmer text is a deterministic fixture built from that same mesh,
    NOT a claim that ElmerGrid or ElmerSolver was executed in this test.
    """
    from sdfmpneo_bfzi.direct3d_mesh import read_gmsh22
    from sdfmpneo_bfzi.direct3d_port_manifest import (
        write_closed_port_source_manifest)
    from examples.bfzi_d2_elmer_handoff import main
    import json
    import sys

    scene=replace(build_demo_scene(),outer_radius=.32)
    path,_=_physical_file(tmp_path,scene)
    mesh,tags=read_gmsh22(
        path,{1:scene.seawater.conductivity,2:0.,3:0.},
        return_physical_tags=True)
    write_closed_port_source_manifest(
        scene,path.with_suffix(".ports.json"),200000.,
        panels_per_corner=2)
    elmer=tmp_path/"synthetic_elmer_text"
    elmer.mkdir()
    (elmer/"mesh.nodes").write_text(
        "".join(f"{i+1} -1 "
                + " ".join(f"{x:.17g}" for x in point)+"\n"
                for i,point in enumerate(mesh.nodes)),
        encoding="utf-8")
    (elmer/"mesh.elements").write_text(
        "".join(f"{i+1} {int(tags[i])} 504 "
                + " ".join(str(int(v)+1) for v in tet)+"\n"
                for i,tet in enumerate(mesh.tetrahedra)),
        encoding="utf-8")
    (elmer/"mesh.names").write_text(
        "! ----- names for bodies -----\n"
        "$ seawater = 1\n"
        "$ transmitter_insulation = 2\n"
        "$ receiver_insulation = 3\n"
        "! ----- names for boundaries -----\n"
        "$ outer_A_tangential_zero = 101\n",
        encoding="utf-8")
    tets=mesh.tetrahedra
    faces=np.concatenate([
        tets[:,(0,1,2)],tets[:,(0,1,3)],
        tets[:,(0,2,3)],tets[:,(1,2,3)]])
    unique,count=np.unique(np.sort(faces,axis=1),axis=0,
                           return_counts=True)
    outer=unique[count==1]
    (elmer/"mesh.boundary").write_text(
        "".join(f"{i+1} 101 1 0 303 "
                + " ".join(str(int(n)+1) for n in tri)+"\n"
                for i,tri in enumerate(outer)),
        encoding="utf-8")
    report_path=tmp_path/"handoff_report.json"
    monkeypatch.setattr(sys,"argv",[
        "bfzi_d2_elmer_handoff","--mesh",str(path),
        "--elmer-grid-dir",str(elmer),"--outer-radius",".32",
        "--frequency","200000","--panels-per-corner","2",
        "--output",str(report_path)])
    main()
    report=json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"]=="D2_ELMERGRID_MATERIAL_BOUNDARY_MAPPING_PASS_ONLY"
    assert report["D2"]=="OPEN"
    assert report["tetrahedra"]==len(mesh.tetrahedra)
    assert report["elmer_grid_groups"]["verified_outer_triangles"]==len(outer)
    assert report["independent_elmer_solution"]=="NOT_RUN_BY_THIS_SCRIPT"

    # Exercise the full explicit two-port table read and conservation gate:
    # an arbitrary elemental constant J must not trigger a Maxwell solve.
    bad_sources=tmp_path/"invalid_elemental_J.txt"
    bad_sources.write_text(
        "".join(f"{i+1} "
                + ("1 0 0 0 1 0" if i==0 else "0 0 0 0 0 0")
                + "\n" for i in range(len(mesh.tetrahedra))),
        encoding="utf-8")
    rejected_report=tmp_path/"rejected_current_report.json"
    monkeypatch.setattr(sys,"argv",[
        "bfzi_d2_elmer_handoff","--mesh",str(path),
        "--elmer-grid-dir",str(elmer),"--outer-radius",".32",
        "--frequency","200000","--panels-per-corner","2",
        "--two-unit-currents",str(bad_sources),
        "--output",str(rejected_report)])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code==2
    rejected=json.loads(rejected_report.read_text(encoding="utf-8"))
    assert rejected["status"]=="D2_VOLUME_SOURCE_CONSERVATION_FAIL"
    assert max(rejected["volume_source_P1_gradient_defect"])>1e-8
