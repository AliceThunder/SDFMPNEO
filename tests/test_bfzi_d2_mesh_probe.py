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


def test_probe_affine_nedelec_electric_field_exact():
    """Pointwise E/J/q are not cell averages: manufactured affine E."""
    from types import SimpleNamespace
    from sdfmpneo_bfzi.direct3d_aphi import TetraMesh, _BARY4
    vertices=np.array([[0.,0.,0.],[1,0,0],[0,1,0],[0,0,1]])
    mesh=TetraMesh(vertices,np.array([[0,1,2,3]]),np.array([7.]))
    points4=_BARY4 @ vertices
    def field(x):
        return np.column_stack((
            1.+x[:,0]+2*x[:,1],
            3*x[:,1]-x[:,2],
            -2*x[:,0]+5*x[:,2])).astype(complex)
    reference=SimpleNamespace(
        mesh=mesh,electric_at_quadrature=field(points4)[None,None,:,:],
        magnetic_secondary=np.zeros((1,1,3),complex),
        current_at_cells=np.zeros((1,1,3),complex))
    probes=np.array([[.1,.2,.3],[.3,.1,.1]])
    sample=sample_cell_fields(reference,probes)
    expected=field(probes)
    assert_allclose(sample['E_total_V_m'][0],expected,rtol=0,atol=1e-13)
    assert_allclose(sample['J_A_m2'][0],7*expected,rtol=0,atol=1e-12)
    assert_allclose(sample['q_W_m3'][0],3.5*np.sum(abs(expected)**2,axis=1),
                    rtol=0,atol=1e-12)


def test_material_interface_probe_reuses_exact_medium_tetra_in_vacuum():
    """The same surface point has non-unique cells: never mix their traces."""
    from types import SimpleNamespace
    from sdfmpneo_bfzi.direct3d_aphi import TetraMesh
    m=box_tetra_mesh([[-.2,.2]]*3,[4,4,4],
                     lambda x:5. if x[2]<0. else 0.)
    vacuum=TetraMesh(m.nodes,m.tetrahedra,np.zeros_like(m.sigma))
    count=len(m.tetrahedra)
    def sample_result(mesh):
        return SimpleNamespace(
            mesh=mesh,magnetic_secondary=np.zeros((1,count,3),complex),
            current_at_cells=np.zeros((1,count,3),complex),
            electric_at_quadrature=np.ones((1,count,4,3),complex))
    pts=np.array([[.05,.05,0.],[.04,-.03,0.]])
    metal=sample_cell_fields(sample_result(m),pts,prefer_conductor=True)
    vac=sample_cell_fields(sample_result(vacuum),pts,
                           cell_indices=metal['cell_index'])
    assert np.array_equal(metal['cell_index'],vac['cell_index'])
    assert np.all(metal['sigma_S_m']>0)
    assert np.all(vac['sigma_S_m']==0)
