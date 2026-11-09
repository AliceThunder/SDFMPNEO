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


def _physical_file(tmp_path, scene, *, tag_override=None):
    mesh=axis_aligned_bfzi_tetra_mesh(
        scene,far_step_m=.30,near_padding_m=.02,box_layers=1)
    centroids=mesh.nodes[mesh.tetrahedra].mean(axis=1)
    tx=scene.transmitter.enclosure.contains(centroids,scene.transmitter.pose)
    rx=scene.receiver.enclosure.contains(centroids,scene.receiver.pose)
    tags=np.where(tx,2,np.where(rx,3,1))
    if tag_override is not None:
        tags=tag_override(tags.copy())
    path=tmp_path/'bfzi_physical123.msh'
    with path.open('w',encoding='utf-8') as fh:
        fh.write('$MeshFormat\n2.2 0 8\n$EndMeshFormat\n')
        fh.write(f'$Nodes\n{len(mesh.nodes)}\n')
        for index,xyz in enumerate(mesh.nodes,1):
            fh.write(f'{index} {xyz[0]:.17g} {xyz[1]:.17g} '
                     f'{xyz[2]:.17g}\n')
        fh.write(f'$EndNodes\n$Elements\n{len(mesh.tetrahedra)}\n')
        for index,(tet,tag) in enumerate(zip(mesh.tetrahedra,tags),1):
            fh.write(f'{index} 4 2 {tag} {tag} '
                     + ' '.join(str(v+1) for v in tet)+'\n')
        fh.write('$EndElements\n')
    return path,mesh


def test_three_physical_regions_preserve_real_two_port_E_B_Z_and_joule(tmp_path):
    scene=replace(build_demo_scene(),outer_radius=.32)
    path,original=_physical_file(tmp_path,scene)
    restored=bfzi_mesh_from_gmsh22(path,scene)
    assert_allclose(restored.nodes,original.nodes,rtol=0,atol=0)
    assert_allclose(restored.sigma,original.sigma,rtol=0,atol=0)
    assert len(restored.tetrahedra)==len(original.tetrahedra)==1350
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
