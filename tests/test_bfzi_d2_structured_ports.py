"""Real closed BFZI two-port A-phi integration on conforming dry/wet geometry.

This is the bounded axis-aligned D2 end-to-end entry, NOT a high-fidelity
rotated-SE3 / open-seawater convergence certificate.
"""
from dataclasses import replace

import numpy as np
from numpy.testing import assert_allclose

from examples.bfzi_mqs_demo import build_demo_scene
from sdfmpneo_bfzi.direct3d_structured_bfzi import axis_aligned_bfzi_tetra_mesh
from sdfmpneo_bfzi.direct3d_galerkin import solve_closed_ports
from sdfmpneo_bfzi.source_reference import polygonal_port_vertices


def test_two_real_continuous_bfzi_closed_ports_on_conforming_3d_mesh():
    scene=replace(build_demo_scene(),outer_radius=.32)
    mesh=axis_aligned_bfzi_tetra_mesh(
        scene,near_padding_m=.01,far_step_m=.30,box_layers=1)
    ports=[[(1.,polygonal_port_vertices(
        assembly,panels_per_corner=2))]
        for assembly in scene.assemblies]
    assert all(np.array_equal(p[0][1][0],p[0][1][-1]) for p in ports)
    reference=solve_closed_ports(mesh,200000.,ports)
    delta=reference.material_increment_z
    assert delta.shape==(2,2) and np.isfinite(delta).all()
    assert_allclose(delta,delta.T,rtol=1e-8,atol=2e-10)
    assert np.linalg.norm(delta)>0
    for currents in ((1.,0.),(0.,1.),(1.,1.),(1.,-1.),(1.,1j)):
        I=np.asarray(currents,complex)
        joule=reference.conductor_power(I)
        assert joule>=0
        assert_allclose(reference.port_power(I),joule,rtol=1e-7,atol=1e-10)
    # The source-space Galerkin identity is just an internal consistency
    # check; it is NOT a proof of corrected spatial E/B/J/q accuracy.
    q=reference.joule_density(np.array([1.,1j]))
    assert np.all(np.isfinite(q)) and np.max(q)>0



def test_genuine_bfzi_two_port_global_se3_covariance():
    """Rigidly rotating/translating the entire 3D material+source problem
    must rotate physical E/B and preserve both complex Z port entries.

    This validates the Nedelec source, geometry, solve and field orientations
    under general global SE(3). It does NOT validate *independent* rotations
    of the two boxes; those need a physical Gmsh conforming mesh.
    """
    from sdfmpneo_bfzi.direct3d_aphi import TetraMesh

    scene=replace(build_demo_scene(),outer_radius=.32)
    mesh=axis_aligned_bfzi_tetra_mesh(
        scene,near_padding_m=.02,far_step_m=.30,box_layers=1)
    paths=[polygonal_port_vertices(a,panels_per_corner=2)
           for a in scene.assemblies]
    reference=solve_closed_ports(
        mesh,200000.,[[(1.,path)] for path in paths],gauge='tree')

    # Two non-commuting proper rotations and a translation test actual
    # three-dimensional objectivity, not just in-plane relabeling.
    a,b=.37,.61
    rz=np.array([[np.cos(a),-np.sin(a),0],
                 [np.sin(a),np.cos(a),0],[0,0,1.]])
    rx=np.array([[1,0,0],[0,np.cos(b),-np.sin(b)],
                 [0,np.sin(b),np.cos(b)]])
    q=rz@rx
    displacement=np.array([.21,-.29,.07])
    rotated=TetraMesh(mesh.nodes@q.T+displacement,
                      mesh.tetrahedra,mesh.sigma)
    rotated_ports=[[(1.,p@q.T+displacement)] for p in paths]
    transformed=solve_closed_ports(
        rotated,200000.,rotated_ports,gauge='tree')

    assert_allclose(transformed.material_increment_z,
                    reference.material_increment_z,rtol=1e-8,atol=1e-11)
    assert_allclose(transformed.medium.electric_at_quadrature,
                    reference.medium.electric_at_quadrature@q.T,
                    rtol=1e-8,atol=2e-9)
    assert_allclose(transformed.medium.magnetic_secondary,
                    reference.medium.magnetic_secondary@q.T,
                    rtol=1e-8,atol=2e-11)
    assert_allclose(transformed.medium.power_from_fields,
                    reference.medium.power_from_fields,
                    rtol=1e-8,atol=1e-11)



def test_independent_orthogonal_receiver_pose_two_real_bfzi_ports():
    """Actual receiver-only SE(3) orientation change, NOT global frame change.

    A 90-degree relative rotation is axis-permuting, so the restricted
    3D tetra mesh represents both rotated insulation boundaries exactly.
    Tests a real electromagnetic solve with the physical receiver coil and
    return rotated together; arbitrary OBLIQUE poses still need Gmsh.
    """
    from sdfmpneo_bfzi.geometry import RigidPose

    scene=replace(build_demo_scene(),outer_radius=.32)
    quarter_turn=np.array([[1.,0.,0.],
                           [0.,0.,-1.],
                           [0.,1.,0.]])
    rotated_receiver=replace(
        scene.receiver,
        pose=RigidPose(quarter_turn,scene.receiver.pose.center))
    changed_scene=replace(scene,receiver=rotated_receiver)
    # Rebuild and solve BOTH geometries: do not reuse the initial material
    # mesh or keep the physical receiver winding incorrectly unrotated.
    outputs=[]
    for current_scene in (scene,changed_scene):
        mesh=axis_aligned_bfzi_tetra_mesh(
            current_scene,near_padding_m=.02,
            far_step_m=.30,box_layers=1)
        strands=[]
        for assembly in current_scene.assemblies:
            path=polygonal_port_vertices(assembly,panels_per_corner=2)
            assert np.array_equal(path[0],path[-1])
            assert np.all(assembly.enclosure.contains(
                path,assembly.pose))
            strands.append([(1.,path)])
        result=solve_closed_ports(mesh,200000.,strands,gauge='tree')
        z=result.material_increment_z
        assert_allclose(z,z.T,rtol=1e-8,atol=5e-10)
        I=np.array([1.,1j])
        assert_allclose(result.port_power(I),result.conductor_power(I),
                        rtol=1e-8,atol=1e-10)
        outputs.append((mesh,z))
    assert len(outputs[0][0].tetrahedra)!=len(outputs[1][0].tetrahedra)
    # Mutual coupling must reflect the genuinely different receiver pose.
    assert abs(outputs[1][1][0,1]) < .5*abs(outputs[0][1][0,1])
