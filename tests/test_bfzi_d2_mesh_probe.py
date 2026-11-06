"""D2 mesh and probe smoke tests; no TEAM-7 accuracy certificate."""
import numpy as np
from numpy.testing import assert_allclose
from sdfmpneo_bfzi.direct3d_mesh import read_gmsh22, gmsh_bfzi_two_boxes
from sdfmpneo_bfzi.direct3d_aphi import (
    box_tetra_mesh, solve_aphi,
)
from sdfmpneo_bfzi.direct3d_probe import (
    closed_polyline_magnetic, sample_cell_fields,
)


def test_gmsh22_physical_tetra_material(tmp_path):
    file=tmp_path/'one.msh'
    file.write_text(
        '$MeshFormat\n2.2 0 8\n$EndMeshFormat\n'
        '$Nodes\n4\n1 0 0 0\n2 1 0 0\n3 0 1 0\n4 0 0 1\n$EndNodes\n'
        '$Elements\n1\n1 4 2 9 1 1 2 3 4\n$EndElements\n',
        encoding='utf-8')
    mesh=read_gmsh22(file,{9:3.526e7})
    assert_allclose(mesh.sigma,[3.526e7])
    assert mesh.tetrahedra.shape==(1,4)


def test_cellwise_B_and_J_benchmark_sampling():
    B0=1e-3
    def source(p):
        return np.column_stack((-B0*p[:,1]/2, B0*p[:,0]/2, 0*p[:,0]))
    mesh=box_tetra_mesh([[-.2,.2]]*3,[3,3,3],
                        lambda p: 5. if max(abs(p)) < .075 else 0.)
    solution=solve_aphi(mesh,1e5,[source])
    points=mesh.nodes[mesh.tetrahedra[[0,1,2]]].mean(axis=1)
    fields=sample_cell_fields(solution,points)
    assert fields['J_A_m2'].shape == (1,3,3)
    assert 'B_total_T' not in fields
    assert_allclose(fields['B_secondary_T'],
                    solution.magnetic_secondary[:,fields['cell_index']])


def test_closed_source_biot_savart_orientation():
    a=.05
    vertices=np.array([[-a,-a,0],[a,-a,0],[a,a,0],
                       [-a,a,0],[-a,-a,0]])
    B=closed_polyline_magnetic(vertices)(np.array([[0,0,.1]]))
    assert B[0,2]>0
    assert_allclose(B[:,:2],0,atol=1e-12)


def test_gmsh_geometry_contains_two_independent_box_poses():
    from examples.bfzi_mqs_demo import build_demo_scene
    geo=gmsh_bfzi_two_boxes(build_demo_scene(),mesh_size_m=.07)
    assert 'BooleanFragments' in geo
    assert 'Volume{2,3}' in geo
    assert 'Physical Volume' in geo
    assert 'Mesh.SaveAll' not in geo  # MSH2 would discard physical-region labels
