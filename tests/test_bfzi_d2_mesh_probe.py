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


def test_independent_rotated_obb_nearfield_mesh_sizing():
    """Physical OBB rotation changes mesh AABB, not conductor geometry."""
    from dataclasses import replace
    import re
    from examples.bfzi_mqs_demo import build_demo_scene
    from sdfmpneo_bfzi.geometry import RigidPose

    scene=build_demo_scene()
    angle=.35
    rotation=np.array([[np.cos(angle),0,np.sin(angle)],
                       [0,1,0],
                       [-np.sin(angle),0,np.cos(angle)]])
    tilted=replace(scene.receiver,
                   pose=RigidPose(rotation, scene.receiver.pose.center))
    scene=replace(scene,receiver=tilted)
    geo=gmsh_bfzi_two_boxes(
        scene,mesh_size_m=.07,near_mesh_size_m=.008,
        near_padding_m=.025)
    assert 'BooleanFragments' in geo
    assert 'Field[1] = Box;' in geo
    assert 'Field[2] = Box;' in geo
    assert 'Field[3].FieldsList = {1,2};' in geo
    assert 'Background Field = 3;' in geo
    assert 'Mesh.SaveAll' not in geo
    def setting(name):
        found=re.search(re.escape(name)+r'\s*=\s*([-+0-9.eE]+);',geo)
        assert found is not None
        return float(found.group(1))
    assert_allclose(setting("Field[2].VIn"), .008)
    assert_allclose(setting("Field[2].VOut"), .07)
    extent=np.abs(rotation)@np.asarray(tilted.enclosure.half_extents)
    assert_allclose(
        setting("Field[2].XMin"),
        tilted.pose.center[0]-extent[0]-.025,atol=1e-14)
    assert_allclose(
        setting("Field[2].ZMax"),
        tilted.pose.center[2]+extent[2]+.025,atol=1e-14)


def test_real_bfzi_two_axis_aligned_boxes_conforming_structured_mesh():
    """Actual demo OBB volume is represented by complete, non-cut tetrahedra."""
    from dataclasses import replace
    import pytest
    from examples.bfzi_mqs_demo import build_demo_scene
    from sdfmpneo_bfzi.geometry import RigidPose
    from sdfmpneo_bfzi.direct3d_structured_bfzi import axis_aligned_bfzi_tetra_mesh

    scene=build_demo_scene()
    mesh=axis_aligned_bfzi_tetra_mesh(
        scene,far_step_m=.3,near_padding_m=.02,box_layers=1)
    xyz=mesh.nodes[mesh.tetrahedra]
    transform=(xyz[:,1:]-xyz[:,0,None,:]).transpose(0,2,1)
    vol=np.abs(np.linalg.det(transform))/6
    box_volume=sum(8*np.prod(assembly.enclosure.half_extents)
                   for assembly in scene.assemblies)
    assert_allclose(vol[mesh.sigma==0].sum(),box_volume,rtol=0,atol=5e-14)
    assert_allclose(vol.sum(),(2*scene.outer_radius)**3,rtol=0,atol=5e-13)
    assert np.count_nonzero(mesh.sigma==0)>0
    assert np.count_nonzero(mesh.sigma>0)>0
    centroids=xyz.mean(axis=1)
    for assembly in scene.assemblies:
        inside=assembly.enclosure.contains(centroids,assembly.pose)
        assert np.any(inside)
        assert np.all(mesh.sigma[inside]==0)

    angle=.2
    rot=np.array([[np.cos(angle),-np.sin(angle),0.],
                  [np.sin(angle),np.cos(angle),0.],
                  [0.,0.,1.]])
    rotated=replace(scene,receiver=replace(
        scene.receiver,pose=RigidPose(rot,scene.receiver.pose.center)))
    with pytest.raises(ValueError,match="Gmsh"):
        axis_aligned_bfzi_tetra_mesh(rotated)


def test_structured_bfzi_wet_wall_h_is_independent_of_dry_box_layers():
    """Refine actual conductive seawater near both wall faces, not only dry tets."""
    from examples.bfzi_mqs_demo import build_demo_scene
    from sdfmpneo_bfzi.direct3d_structured_bfzi import (
        _axis_coordinates, axis_aligned_bfzi_tetra_mesh)
    import pytest

    scene=build_demo_scene()
    kw=dict(near_padding_m=.02,far_step_m=.30,box_layers=1)
    base=axis_aligned_bfzi_tetra_mesh(scene,**kw)
    fine=axis_aligned_bfzi_tetra_mesh(scene,near_step_m=.04,**kw)
    assert len(fine.tetrahedra)>len(base.tetrahedra)
    assert np.all(np.isin(fine.sigma,[0.,scene.seawater.conductivity]))
    for mesh in (base,fine):
        xyz=mesh.nodes[mesh.tetrahedra]
        det=np.linalg.det((xyz[:,1:]-xyz[:,0,None,:]).transpose(0,2,1))
        volume=np.abs(det)/6
        expected=sum(8*np.prod(a.enclosure.half_extents)
                     for a in scene.assemblies)
        assert_allclose(volume[mesh.sigma==0].sum(),expected,atol=1e-12)
    # The actual wet slab immediately outside a package wall must have
    # cell size <= requested near_step. A higher box_layers alone would
    # not guarantee this for the conductive side.
    z=_axis_coordinates(scene,2,.02,.30,1,.04)
    for center in (-.13,.13):
        upper=center+.015
        wet=np.diff(z[(z>=upper-1e-11)&(z<=upper+.02+1e-11)])
        assert len(wet)>0 and np.max(wet)<=.04+1e-12
    with pytest.raises(ValueError,match='near_step_m'):
        axis_aligned_bfzi_tetra_mesh(scene,near_step_m=.4,**kw)


def test_fixed_core_keeps_nearfield_tetra_planes_identical_for_R_sweep():
    """Changing exterior radius must NOT silently change local OBB h."""
    from dataclasses import replace
    from examples.bfzi_mqs_demo import build_demo_scene
    from sdfmpneo_bfzi.direct3d_structured_bfzi import _axis_coordinates
    import pytest

    base=build_demo_scene()
    for axis in range(3):
        near_coordinates=[]
        for radius in (.32,.40,.52):
            scene=replace(base,outer_radius=radius)
            coords=_axis_coordinates(scene,axis,.08,.30,1,.08,.24)
            near_coordinates.append(
                coords[(coords>=-.24-1e-11)&(coords<=.24+1e-11)])
        for actual in near_coordinates[1:]:
            assert_allclose(actual,near_coordinates[0],rtol=0,atol=1e-12)
    with pytest.raises(ValueError,match='inner_core_radius_m'):
        from sdfmpneo_bfzi.direct3d_structured_bfzi import (
            axis_aligned_bfzi_tetra_mesh)
        axis_aligned_bfzi_tetra_mesh(
            base,near_padding_m=.08,inner_core_radius_m=.12)
