"""Minimal BFZI end-to-end physical smoke tests, not a FEM accuracy claim."""
import numpy as np
import pytest

from sdfmpneo_bfzi import (
    BFZICoil, RigidPose, InsulatingBox, BFZIAssembly, Seawater,
    BfziScene, SeaMQSConfig, build_bfzi_source, solve_seawater_reference,
    compress_seawater_response, BFZICurrentControlledEnvelope,
)
from sdfmpneo_bfzi.vnext_bridge import BFZIVNextGeometry


def scene_pair():
    coil = BFZICoil(2, 0.075, 0.045, 0.005, alpha=0.5)
    box = InsulatingBox((0.10, 0.10, 0.015))
    route = ((0.0, 0.0, 0.008), (0.090, 0.0, 0.008))
    first = BFZIAssembly(
        coil, RigidPose(np.eye(3), np.array([0., 0., -.13])),
        box, return_waypoints=route,
    )
    second = BFZIAssembly(
        coil, RigidPose(np.eye(3), np.array([0., 0., .13])),
        box, return_waypoints=route,
    )
    return BfziScene(first, second, Seawater(5.0), outer_radius=0.52)


def test_port_source_closed_and_air_mutual():
    scene = scene_pair()
    source = build_bfzi_source(scene, 200_000., order=3)
    for assembly, line in zip(scene.assemblies, source.lines):
        # The physical spiral is oriented from outer START to inner END.
        # Its separate return starts at inner END and finishes at outer START.
        outer, inner = assembly.coil.endpoints()
        offset = 8 * assembly.coil.turns * 3
        first_return_segment = line.line_elements[offset:offset + 3].sum(axis=0)
        first_waypoint = np.asarray(assembly.return_waypoints[0])
        np.testing.assert_allclose(
            first_return_segment,
            assembly.pose.vector(first_waypoint - inner),
            atol=1e-12,
        )
        # A reversed return route would produce a nonzero total displacement.
        np.testing.assert_allclose(line.line_elements.sum(axis=0), 0., atol=1e-6)
    np.testing.assert_allclose(
        source.inductance_air, source.inductance_air.T, atol=1e-14,
    )
    assert np.all(np.isfinite(source.inductance_air))


def test_same_state_seawater_impedance_and_joule_heat():
    scene = scene_pair()
    source = build_bfzi_source(scene, 200_000., order=3)
    result = solve_seawater_reference(
        source, config=SeaMQSConfig(
            radial_order=5, polar_order=7, azimuthal_order=10,
            field_softening=0.025,
        ),
    )
    z = result.impedance_seawater
    np.testing.assert_allclose(z, z.T, rtol=1e-9, atol=1e-12)
    assert np.min(np.linalg.eigvalsh(0.5 * (z + z.conj().T))) >= -1e-12
    i = np.array([1.2 + 0.1j, -0.4 + 0.8j])
    np.testing.assert_allclose(
        [result.port_seawater_power(i), result.seawater_power(i)],
        [result.quadrature_power(i)] * 2, rtol=3e-8, atol=1e-12,
    )
    assert result.seawater_power(i) >= 0
    prediction = result.as_structured_prediction()
    assert prediction.power_closure_error() < 1e-8
    points = result.integration_points[:5]
    q = result.seawater_loss(points, i)
    assert q.shape == (5,) and np.all(q >= 0)
    current_field = result.current_density(points, i)
    assert current_field.shape == (5, 3)
    # In a true two-port response the current fields superpose, while
    # Joule heat is a quadratic field with cross terms.
    i_a = np.array([i[0], 0j])
    i_b = np.array([0j, i[1]])
    np.testing.assert_allclose(
        current_field,
        result.current_density(points, i_a) + result.current_density(points, i_b),
        rtol=1e-10, atol=1e-12,
    )
    np.testing.assert_allclose(
        q,
        np.sum(np.abs(current_field)**2, axis=1) /
        (2 * scene.seawater.conductivity),
        rtol=1e-10, atol=1e-12,
    )
    # The deterministic projection preserves the same low-dimensional
    # port/volume power identity; it is not yet a trained FAST artifact.
    reduced = compress_seawater_response(
        result, rank=min(2, result.resistance.shape[0]),
    )
    np.testing.assert_allclose(
        reduced.port_seawater_power(i), reduced.quadrature_power(i),
        rtol=1e-7, atol=1e-12,
    )
    assert reduced.seawater_power(i) >= 0


def test_optional_face_guided_basis_nests_complete_port_curls():
    """Physics-informed face modes extend, not replace, the older trial space."""
    from sdfmpneo_bfzi import build_sea_basis

    scene=scene_pair()
    source=build_bfzi_source(scene,200_000.,order=3)
    old=build_sea_basis(scene,sources=source,include_source_guided=True)
    enriched=build_sea_basis(
        scene,sources=source,include_source_guided=True,
        include_source_face_guided=True,
    )
    assert enriched.size==old.size+8
    for previous,new in zip(old.modes,enriched.modes[:old.size]):
        np.testing.assert_array_equal(previous.center,new.center)
        np.testing.assert_array_equal(previous.direction,new.direction)
        assert previous.width==new.width
        assert previous.source_port==new.source_port
    assert {mode.source_port for mode in enriched.modes[-8:]}=={0,1}
    points=np.array([[0.0,0.0,0.0],[0.13,0.0,-0.13]])
    assert scene.in_seawater(points).all()
    modes=enriched.evaluate(points)
    assert modes.shape==(2,enriched.size,3)
    assert np.isfinite(modes).all()
    np.testing.assert_allclose(
        modes[:,:old.size,:],old.evaluate(points),rtol=1e-12,atol=1e-12,
    )


def test_edge_lip_level_five_nests_167_into_199():
    """Experimental edge-lip directions remain independent and nested."""
    from sdfmpneo_bfzi import build_sea_basis

    scene=scene_pair()
    source=build_bfzi_source(scene,200_000.,order=3)
    args=dict(
        modes_per_face=2,refine_thin_faces_only=True,
        sources=source,include_source_guided=True,
        include_source_face_guided=True,
    )
    earlier=build_sea_basis(scene,face_refinement=4,**args)
    expanded=build_sea_basis(scene,face_refinement=5,**args)
    assert earlier.size==167
    assert expanded.size==199

    def signature(mode):
        return (
            tuple(mode.center),tuple(mode.direction),
            mode.width,mode.source_port,
        )
    index={signature(mode):j for j,mode in enumerate(expanded.modes)}
    indices=[index[signature(mode)] for mode in earlier.modes]
    assert len(indices)==len(set(indices))==earlier.size
    points=np.array([[0.,0.,0.],[0.13,0.,-0.13]])
    assert scene.in_seawater(points).all()
    np.testing.assert_allclose(
        expanded.evaluate(points)[:,indices,:],
        earlier.evaluate(points),rtol=1e-12,atol=1e-12,
    )
    assert np.isfinite(expanded.evaluate(points)).all()


def test_anisotropic_face_curls_nest_199_and_follow_rotated_pose():
    """A geometry-rotated anisotropic potential has its exact analytical curl."""
    from dataclasses import replace
    from sdfmpneo_bfzi import build_sea_basis

    scene=scene_pair()
    angle=0.31
    rotation=np.array([
        [np.cos(angle),0.,np.sin(angle)],
        [0.,1.,0.],
        [-np.sin(angle),0.,np.cos(angle)],
    ])
    receiver=replace(
        scene.receiver,
        pose=RigidPose(rotation,scene.receiver.pose.center),
    )
    rotated=replace(scene,receiver=receiver)
    source=build_bfzi_source(rotated,200_000.,order=3)
    args=dict(
        modes_per_face=2,face_refinement=5,
        refine_thin_faces_only=True,
        sources=source,include_source_guided=True,
        include_source_face_guided=True,
    )
    isotropic=build_sea_basis(rotated,**args)
    enriched=build_sea_basis(
        rotated,**args,include_anisotropic_face_guided=True,
    )
    assert isotropic.size==199
    assert enriched.size==207
    from examples.bfzi_convergence import _indices_in_enriched_basis
    indices=_indices_in_enriched_basis(enriched,isotropic)
    assert len(indices)==isotropic.size
    assert len(set(indices))==isotropic.size
    point=receiver.pose.position(np.array([0.04,0.01,0.042]))
    assert rotated.in_seawater(point)
    old_j=isotropic.evaluate(point[None,:])
    new_j=enriched.evaluate(point[None,:])
    np.testing.assert_allclose(
        new_j[:,indices],old_j,rtol=1e-12,atol=1e-12,
    )
    for mode in enriched.modes[-8:]:
        assert mode.gaussian_metric is not None
        np.testing.assert_allclose(
            mode.gaussian_metric,mode.gaussian_metric.T,atol=1e-12,
        )
        assert np.linalg.eigvalsh(mode.gaussian_metric)[0]>0
    # The anisotropic metric must rotate with the package's full pose.
    face_mode=enriched.modes[-2]
    local_metric=rotation.T @ face_mode.gaussian_metric @ rotation
    np.testing.assert_allclose(
        local_metric,np.diag(np.diag(local_metric)),
        rtol=0,atol=1e-12,
    )
    assert local_metric[2,2]>local_metric[0,0]

    # Numerical differentiation checks the physical curl rather than
    # comparing two implementations of the same Gaussian formula.
    h=1e-5
    derivative=[]
    for axis in range(3):
        delta=np.eye(3)[axis]*h
        plus=enriched.potential_primitive((point+delta)[None,:])[0,-1]
        minus=enriched.potential_primitive((point-delta)[None,:])[0,-1]
        derivative.append((plus-minus)/(2*h))
    curl=np.array([
        derivative[1][2]-derivative[2][1],
        derivative[2][0]-derivative[0][2],
        derivative[0][1]-derivative[1][0],
    ])
    np.testing.assert_allclose(
        new_j[0,-1],curl,rtol=3e-4,atol=1e-6,
    )


def test_targeted_global_and_package_corner_modes_preserve_207_parent():
    """Nested geometry-following fields, not a replacement for original coils."""
    from sdfmpneo_bfzi import build_sea_basis
    from examples.bfzi_convergence import _indices_in_enriched_basis

    scene=scene_pair()
    source=build_bfzi_source(scene,200_000.,order=3)
    options=dict(
        modes_per_face=2,face_refinement=5,
        refine_thin_faces_only=True,sources=source,
        include_source_guided=True,
        include_source_face_guided=True,
        include_anisotropic_face_guided=True,
    )
    parent=build_sea_basis(scene,**options)
    near_global=build_sea_basis(
        scene,**options,include_global_multiscale=True,
        global_scale_factors=(0.75,),
    )
    global_trial=build_sea_basis(
        scene,**options,include_global_multiscale=True,
    )
    corner_trial=build_sea_basis(
        scene,**options,include_global_multiscale=True,
        include_corner_source_guided=True,
    )
    assert parent.size==207
    assert near_global.size==210
    assert global_trial.size==213
    assert corner_trial.size==229
    assert len(_indices_in_enriched_basis(near_global,parent))==207
    assert len(_indices_in_enriched_basis(global_trial,near_global))==210
    assert len(_indices_in_enriched_basis(corner_trial,global_trial))==213
    assert len(_indices_in_enriched_basis(global_trial,parent))==207
    indices=_indices_in_enriched_basis(corner_trial,parent)
    assert len(indices)==207 and len(set(indices))==207
    probes=np.array([[0.,0.,0.],[0.13,0.,-0.13]])
    np.testing.assert_allclose(
        corner_trial.evaluate(probes)[:,indices,:],
        parent.evaluate(probes),atol=1e-12,rtol=1e-12,
    )
    for mode in corner_trial.modes[-16:]:
        assert mode.source_port in (0,1)
        assert mode.gaussian_metric is not None
        assert np.linalg.eigvalsh(mode.gaussian_metric)[0]>0

    # Physical current directions must survive an outer-radius change.
    # Their boundary taper may differ, but their center, width, port,
    # and rotated anisotropic Gaussian metric must not change.
    from dataclasses import replace
    expanded_scene = replace(
        scene, outer_radius=scene.outer_radius + 0.25,
    )
    expanded_source = build_bfzi_source(
        expanded_scene, 200_000., order=3,
    )
    expanded = build_sea_basis(
        expanded_scene,
        modes_per_face=2,face_refinement=5,
        refine_thin_faces_only=True,sources=expanded_source,
        include_source_guided=True,
        include_source_face_guided=True,
        include_anisotropic_face_guided=True,
        include_global_multiscale=True,
        include_corner_source_guided=True,
    )
    assert expanded.size == corner_trial.size == 229
    for old, new in zip(corner_trial.modes, expanded.modes):
        np.testing.assert_array_equal(old.center,new.center)
        np.testing.assert_array_equal(old.direction,new.direction)
        assert old.width == new.width
        assert old.source_port == new.source_port
        if old.gaussian_metric is None:
            assert new.gaussian_metric is None
        else:
            np.testing.assert_array_equal(
                old.gaussian_metric,new.gaussian_metric,
            )


def test_bfzi_log_recorder_filters_noise_without_losing_raw_metrics():
    """Keep detailed evidence on disk while showing compact physics deltas."""
    from io import StringIO
    from contextlib import redirect_stdout
    from examples.bfzi_run_record import _LogWriter

    full=StringIO()
    console=StringIO()
    with redirect_stdout(console):
        writer=_LogWriter(full,verbose=False)
        writer.write("  early_probes: q=[0.1, 0.2, 0.3]\n")
        writer.write(
            "  anisotropic_face_curls_to_global_multiscale: "
            "dZ=1.95291, dJ_volume=0.931925, "
            "dJ_ports_volume=1.18121, dq=0.668683, "
            "max_probe_dq=2022.22, dP=0.868147\n"
        )
        writer.write(
            "  rk_far_polar_229_B_vs_A: dR=0.007, dK=0.001, "
            "dJ_volume=0.004, dJ_ports_volume=0.005, "
            "normal_peak_scaled_dq=0.002, response_gate=PASS\n"
        )
        writer.write(
            "  rk_far_azimuth_4_vs_3: dR=0.0003, dK=0.0001, "
            "dJ_volume=0.001, dq=0.002, "
            "normal_peak_scaled_dq=0.001, response_gate=PASS\n"
        )
        writer.write(
            "  domain_step: 0.52->0.7 m, dZ=0.004, dq=0.01, "
            "dP=0.008, dJ_common=0.02, "
            "dJ_ports_common=0.015, dJ_near035=0.006, "
            "dJ_ports_near035=0.007, dq_near035=0.004, "
            "domain_response_gate=FAIL\n"
        )
        writer.write(
            "  domain_operator_step: 1.3->1.6 m, "
            "dR_common=0.02, dK_common=0.01, dL_common=0.03, "
            "dZ_RK_only=0.004, dZ_L_only=0.008, "
            "dJ_near035_RK_only=0.005, "
            "dJ_near035_L_only=0.012\n"
        )
    assert "early_probes: q=" in full.getvalue()
    assert "early_probes" not in console.getvalue()
    assert "dJ_volume=0.931925" in console.getvalue()
    assert "dJ_ports_volume=1.18121" in console.getvalue()
    assert "rk_far_polar_229_B_vs_A" in console.getvalue()
    assert "normal_peak_scaled_dq=0.002" in console.getvalue()
    assert "rk_far_azimuth_4_vs_3" in console.getvalue()
    assert "domain_step: 0.52->0.7" in console.getvalue()
    assert "dJ_ports_common=0.015" in full.getvalue()
    assert "dJ_near035=0.006" in console.getvalue()
    assert "dq_near035=0.004" in console.getvalue()
    assert "domain_response_gate=FAIL" in console.getvalue()
    assert "domain_operator_step: 1.3->1.6" in console.getvalue()
    assert "dZ_L_only=0.008" in console.getvalue()
    assert "dJ_near035_RK_only=0.005" in console.getvalue()


def test_galerkin_volume_current_error_matches_physical_sea_quadrature():
    """The R-weighted metric equals whole-domain J/sigma quadrature."""
    from sdfmpneo_bfzi import build_sea_basis
    from examples.bfzi_convergence import _relative_dissipation_norm

    scene=scene_pair()
    basis=build_sea_basis(scene,modes_per_face=1)
    points,weights=scene.box_shell_quadrature(
        3,4,6,shell_normal_order=2,shell_tangential_order=3,
    )
    modes=basis.evaluate(points)
    inv_sigma_weights=weights/scene.seawater.conductivity
    resistance=np.einsum(
        "n,nic,njc->ij",inv_sigma_weights,modes,modes,optimize=True,
    )
    reference=np.zeros((basis.size,2),dtype=complex)
    reference[0,0]=1.0+0.25j
    reference[1,1]=-0.3+0.2j
    approximate=reference.copy()
    approximate[0,0]+=0.1+0.04j
    approximate[2,1]+=0.02-0.03j
    current_ref=np.einsum(
        "nic,ip->npc",modes,reference,optimize=True,
    )
    current_error=np.einsum(
        "nic,ip->npc",modes,approximate-reference,optimize=True,
    )
    physical=np.sqrt(
        np.sum(inv_sigma_weights[:,None,None]*np.abs(current_error)**2)
        /np.sum(inv_sigma_weights[:,None,None]*np.abs(current_ref)**2)
    )
    np.testing.assert_allclose(
        _relative_dissipation_norm(resistance,approximate,reference),
        physical,rtol=1e-11,atol=1e-12,
    )


def test_parent_energy_bound_and_mode_deletion_match_direct_galerkin():
    """Small complex-symmetric MQS system: bound and removal are physical."""
    from examples.bfzi_convergence import (
        _relative_dissipation_norm, _parent_galerkin_energy_bound,
        _leave_one_out_coefficients,
    )

    r=np.array([[3.,0.3,0.],[0.3,1.5,0.1],[0.,0.1,2.]])
    l=np.array([[2.,0.1,0.],[0.1,1.,0.2],[0.,0.2,1.2]])
    system=r+1j*3.*l
    forcing=np.array([
        [0.5+0.1j,-0.2j],
        [-0.2+0.3j,0.7],
        [0.1,-0.4+0.2j],
    ])
    full=np.linalg.solve(system,-forcing)
    trial=full.copy()
    trial[0]=0
    trial[2]*=0.85
    error=_relative_dissipation_norm(r,trial,full)
    bound=_parent_galerkin_energy_bound(r,system,forcing,trial)
    assert error>0
    assert error<=bound*(1+1e-12)
    assert _parent_galerkin_energy_bound(
        r,system,forcing,full,
    )<1e-12

    inverse=np.linalg.solve(system,np.eye(3))
    for index in (0,1,2):
        dropped=_leave_one_out_coefficients(inverse,full,index)
        keep=[i for i in range(3) if i!=index]
        direct=np.linalg.solve(
            system[np.ix_(keep,keep)],-forcing[keep],
        )
        np.testing.assert_allclose(dropped[index],0.,atol=1e-13)
        np.testing.assert_allclose(
            dropped[keep],direct,rtol=1e-12,atol=1e-13,
        )


def test_reject_overlapping_packages():
    scene = scene_pair()
    with pytest.raises(ValueError, match="overlap"):
        BfziScene(
            scene.transmitter,
            BFZIAssembly(
                scene.receiver.coil, scene.transmitter.pose,
                scene.receiver.enclosure, scene.receiver.return_waypoints,
            ),
            scene.seawater, scene.outer_radius,
        )


def test_vnext_bishop_frame_bridge():
    scene = scene_pair()
    a = scene.transmitter
    adapted = BFZIVNextGeometry(
        a.coil, a.pose, conductor_width=0.0016,
        conductor_thickness=0.0016,
    )
    polyline = adapted.polyline(80)
    assert polyline.points.shape == (81, 3)
    assert np.isfinite(polyline.normal1).all()
    assert polyline.total_length > 0


def test_box_wall_radial_cubature_preserves_sea_volume():
    scene = scene_pair()
    points, weights = scene.quadrature(5, 7, 10)
    plain_points, plain_weights = scene.quadrature(
        5, 7, 10, wall_layers=0,
    )
    assert len(points) > len(plain_points)
    np.testing.assert_allclose(
        weights.sum(), plain_weights.sum(), rtol=1e-12,
    )
    box_volumes = [
        8.0 * np.prod(assembly.enclosure.half_extents)
        for assembly in scene.assemblies
    ]
    exact_volume = (
        4.0 * np.pi * scene.outer_radius**3 / 3.0 - sum(box_volumes)
    )
    # Radial Gauss intervals terminate at actual OBB walls; the original
    # node-rejection rule missed ~0.4% of this same sea volume.
    assert abs(weights.sum() / exact_volume - 1.0) < 8e-4
    assert np.all(weights > 0.0)
    assert scene.in_seawater(points).all()


def test_finite_wire_skin_loss_and_lumped_thermal_coupling():
    from dataclasses import replace
    from sdfmpneo_vnext.scene import ConductorMaterial
    from sdfmpneo_vnext.electrothermal import (
        CoilThermalProperties, build_lumped_coil_thermal_model,
    )
    from sdfmpneo_bfzi import BFZICurrentControlledEnvelope

    original = scene_pair()
    warm_material = ConductorMaterial(
        5.8e7, resistance_temperature_coefficient=0.00393
    )
    tx = replace(original.transmitter, material=warm_material)
    rx = replace(original.receiver, material=warm_material)
    scene = replace(original, transmitter=tx, receiver=rx)
    source = build_bfzi_source(scene, 200_000., order=3)
    impedances = source.conductor_internal_impedance()
    lengths = np.array([
        np.linalg.norm(line.line_elements, axis=1).sum()
        for line in source.lines
    ])
    dc_resistance = lengths / (
        warm_material.conductivity * np.pi * tx.wire_radius**2
    )
    assert np.all(impedances.real > dc_resistance)
    assert np.all(impedances.imag > 0)
    assert np.all(
        source.conductor_internal_impedance([333.15, 333.15]).real
        > impedances.real
    )

    result = solve_seawater_reference(
        source, config=SeaMQSConfig(
            radial_order=4, polar_order=6, azimuthal_order=8,
            field_softening=0.025,
        ),
    )
    currents = np.array([1.0 + 0.0j, 0.2 + 0.1j])
    predicted = result.as_structured_prediction()
    assert predicted.channel_labels == (
        "conductor:0", "conductor:1", "seawater",
    )
    assert predicted.power_closure_error() < 1e-8
    np.testing.assert_allclose(
        predicted.coil_power(currents),
        result.port_conductor_power(currents),
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        predicted.channel_power(currents).sum(),
        result.port_total_power(currents),
        rtol=1e-8,
    )

    thermal = build_lumped_coil_thermal_model(
        (CoilThermalProperties(20.0, 0.7),
         CoilThermalProperties(20.0, 0.7)),
    )
    envelope = BFZICurrentControlledEnvelope(result, thermal)
    initial = np.zeros(2)
    step = envelope.step(initial, currents, 30.0)
    assert step.converged
    assert np.all(step.temperatures >= 293.15)
    assert np.all(step.coil_power >= 0.0)
    np.testing.assert_allclose(
        step.impedance, result.port_impedance_at(step.temperatures),
    )


def test_rotated_box_cubature_and_return_wire_clearance():
    from dataclasses import replace
    scene = scene_pair()
    angle = 0.37
    rotation = np.array([
        [np.cos(angle), 0.0, np.sin(angle)],
        [0.0, 1.0, 0.0],
        [-np.sin(angle), 0.0, np.cos(angle)],
    ])
    rotated = replace(
        scene, receiver=replace(
            scene.receiver,
            pose=RigidPose(rotation, scene.receiver.pose.center),
        ),
    )
    points, weights = rotated.quadrature(4, 9, 14, wall_layers=2)
    assert len(points) > 0
    assert rotated.in_seawater(points).all()
    assert np.all(weights > 0.0)
    with pytest.raises(ValueError, match="return route"):
        replace(
            scene.transmitter,
            return_waypoints=((0., 0., 0.008), (0.11, 0., 0.008)),
        )
    with pytest.raises(ValueError, match="pitch"):
        replace(scene.transmitter, wire_radius=0.003)


def test_voltage_control_recomputes_current_at_warm_port_impedance():
    from dataclasses import replace
    from sdfmpneo_bfzi import BFZIVoltageControlledEnvelope
    from sdfmpneo_vnext.scene import ConductorMaterial
    from sdfmpneo_vnext.electrothermal import (
        CoilThermalProperties, build_lumped_coil_thermal_model,
    )

    initial_scene = scene_pair()
    copper = ConductorMaterial(
        5.8e7, resistance_temperature_coefficient=0.00393,
    )
    scene = replace(
        initial_scene,
        transmitter=replace(initial_scene.transmitter, material=copper),
        receiver=replace(initial_scene.receiver, material=copper),
    )
    source = build_bfzi_source(scene, 200_000., order=3)
    reference = solve_seawater_reference(
        source, config=SeaMQSConfig(
            radial_order=4, polar_order=6, azimuthal_order=8,
            field_softening=0.025,
        ),
    )
    thermal = build_lumped_coil_thermal_model((
        CoilThermalProperties(20.0, 0.7),
        CoilThermalProperties(20.0, 0.7),
    ))
    external = np.diag([0.05 - 0.2j, 0.07 - 0.2j])
    envelope = BFZIVoltageControlledEnvelope(
        reference, thermal, external_impedance=external,
    )
    voltage = np.array([1.0 + 0.2j, 0.0j])
    step = envelope.step(np.zeros(2), voltage, 30.0)
    assert step.converged
    np.testing.assert_allclose(
        (step.impedance + external) @ step.currents, voltage,
        rtol=1e-10, atol=1e-10,
    )
    np.testing.assert_allclose(
        step.coil_power,
        reference.port_conductor_power(step.currents, step.temperatures),
    )
    with pytest.raises(ValueError, match="passive"):
        BFZIVoltageControlledEnvelope(
            reference, thermal, external_impedance=-np.eye(2),
        )


def test_round_wire_joule_profile_and_complete_path_heat_are_consistent():
    scene = scene_pair()
    source = build_bfzi_source(scene, 200_000., order=3)
    currents = np.array([1.1 + 0.2j, -0.4 + 0.5j])
    radius_nodes, radial_weights = np.polynomial.legendre.leggauss(24)
    fraction = 0.5 * (radius_nodes + 1)
    density = source.conductor_cross_section_loss(fraction, currents)
    assert density.shape == (24, 2)
    assert np.all(density >= 0.0)
    assert np.all(
        source.conductor_cross_section_loss(1.0, currents)
        > source.conductor_cross_section_loss(0.0, currents)
    )
    area_integral = np.array([
        np.pi * assembly.wire_radius**2
        * np.sum(radial_weights * fraction * density[:, port])
        for port, assembly in enumerate(scene.assemblies)
    ])
    lengths = np.array([
        np.linalg.norm(line.line_elements, axis=1).sum()
        for line in source.lines
    ])
    expected = 0.5 * source.conductor_internal_impedance().real * np.abs(currents)**2
    np.testing.assert_allclose(
        area_integral * lengths, expected, rtol=1e-8,
    )
    nodes = source.conductor_heat_nodes(currents)
    np.testing.assert_allclose(
        [watts.sum() for _, watts in nodes], expected, rtol=1e-12,
    )
    volumes = source.conductor_volume_heat(
        currents, radial_order=12, angular_order=8,
    )
    np.testing.assert_allclose(
        [np.dot(volume_weights, density)
         for _, volume_weights, density in volumes],
        expected, rtol=1e-8,
    )
    for (positions, volume_weights, density), assembly, length in zip(
        volumes, scene.assemblies, lengths,
    ):
        assert positions.shape == (len(volume_weights), 3)
        assert np.all(volume_weights > 0)
        assert np.all(density >= 0)
        np.testing.assert_allclose(
            volume_weights.sum(), length * np.pi * assembly.wire_radius**2,
            rtol=1e-12,
        )


def test_nested_face_basis_refinement_keeps_original_current_modes():
    from sdfmpneo_bfzi import build_sea_basis

    scene = scene_pair()
    coarse = build_sea_basis(scene, face_refinement=1)
    refined = build_sea_basis(scene, face_refinement=2)
    assert refined.size > coarse.size
    for original in coarse.modes:
        assert any(
            np.array_equal(original.center, candidate.center)
            and np.array_equal(original.direction, candidate.direction)
            and original.width == candidate.width
            for candidate in refined.modes
        )
    points, _ = scene.quadrature(3, 4, 6, wall_layers=1)
    fields = refined.evaluate(points[:5])
    assert fields.shape == (5, refined.size, 3)
    assert np.isfinite(fields).all()



def test_level_three_thin_face_modes_nest_63_into_95():
    """Added spatial directions preserve the 63-mode physical trial space."""
    from sdfmpneo_bfzi import build_sea_basis
    scene = scene_pair()
    source = build_bfzi_source(scene, 200_000., order=3)
    kwargs = dict(
        modes_per_face=2, refine_thin_faces_only=True,
        sources=source, include_source_guided=True,
    )
    original = build_sea_basis(scene, face_refinement=2, **kwargs)
    expanded = build_sea_basis(scene, face_refinement=3, **kwargs)
    candidate = build_sea_basis(scene, face_refinement=4, **kwargs)
    assert original.size == 63 and expanded.size == 95
    assert candidate.size == 159

    def signature(mode):
        return (
            tuple(mode.center), tuple(mode.direction),
            mode.width, mode.source_port,
        )

    assert set(map(signature, original.modes)).issubset(
        set(map(signature, expanded.modes))
    )
    assert set(map(signature, expanded.modes)).issubset(
        set(map(signature, candidate.modes))
    )
    points, _ = scene.box_shell_quadrature(
        3, 4, 6, shell_normal_order=2, shell_tangential_order=3,
    )
    query = points[:8]
    full = expanded.evaluate(query)
    indices = [next(i for i, mode in enumerate(expanded.modes)
                    if signature(mode) == signature(original_mode))
               for original_mode in original.modes]
    np.testing.assert_allclose(
        full[:, indices, :], original.evaluate(query),
        rtol=1e-12, atol=1e-12,
    )
    enlarged = candidate.evaluate(query)
    indices3 = [next(i for i, mode in enumerate(candidate.modes)
                     if signature(mode) == signature(parent))
                for parent in expanded.modes]
    np.testing.assert_allclose(
        enlarged[:, indices3, :], full, rtol=1e-12, atol=1e-12,
    )
    assert np.isfinite(full).all() and np.isfinite(enlarged).all()
    with pytest.raises(ValueError, match="face_refinement"):
        build_sea_basis(scene, face_refinement=6)


def test_reference_refuses_basis_from_a_different_outer_domain():
    from dataclasses import replace
    from sdfmpneo_bfzi import build_sea_basis

    scene = scene_pair()
    sources = build_bfzi_source(scene, 200_000., order=3)
    wrong_basis = build_sea_basis(replace(scene, outer_radius=0.65))
    with pytest.raises(ValueError, match="source scene"):
        solve_seawater_reference(sources, basis=wrong_basis)


def test_symmetric_green_tiles_are_independent_of_chunk_partition():
    scene = scene_pair()
    source = build_bfzi_source(scene, 200_000., order=3)
    base = dict(
        radial_order=4, polar_order=5, azimuthal_order=8,
        wall_layers=1, field_softening=0.025,
    )
    first = solve_seawater_reference(
        source, config=SeaMQSConfig(**base, kernel_chunk_size=17),
    )
    second = solve_seawater_reference(
        source, config=SeaMQSConfig(**base, kernel_chunk_size=29),
    )
    np.testing.assert_allclose(
        first.inductance, second.inductance, rtol=2e-12, atol=1e-14,
    )
    np.testing.assert_allclose(
        first.impedance_seawater, second.impedance_seawater,
        rtol=2e-12, atol=1e-14,
    )
    assert np.min(np.linalg.eigvalsh(first.inductance)) > -1e-12


def test_seawater_heat_nodes_drive_separate_thermal_rom_channel():
    from sdfmpneo_vnext.electrothermal import (
        CoilThermalProperties, build_lumped_coil_thermal_model,
    )

    scene = scene_pair()
    reference = solve_seawater_reference(
        build_bfzi_source(scene, 200_000., order=3),
        config=SeaMQSConfig(
            radial_order=4, polar_order=5, azimuthal_order=8,
            wall_layers=1, field_softening=0.025,
        ),
    )
    currents = np.array([1.3 + 0.2j, -0.3 + 0.4j])
    points, weights, heat = reference.seawater_heat_nodes(currents)
    assert len(points) == len(weights) == len(heat)
    assert scene.in_seawater(points).all()
    assert np.all(weights > 0)
    assert np.all(heat >= 0)
    np.testing.assert_allclose(
        weights @ heat, reference.port_seawater_power(currents),
        rtol=3e-9, atol=1e-12,
    )
    three_channels = reference.thermal_channel_power(currents)
    np.testing.assert_allclose(
        three_channels,
        reference.as_structured_prediction().channel_power(currents),
        rtol=3e-9, atol=1e-12,
    )
    np.testing.assert_allclose(
        three_channels[:2], reference.port_conductor_power(currents),
    )
    np.testing.assert_allclose(three_channels[2], weights @ heat, rtol=3e-9)

    thermal = build_lumped_coil_thermal_model(
        (CoilThermalProperties(20.0, 0.7),
         CoilThermalProperties(20.0, 0.7),
         CoilThermalProperties(100.0, 0.2)),
        mutual_conductance=np.array([
            [0.0, 0.0, 0.04],
            [0.0, 0.0, 0.04],
            [0.04, 0.04, 0.0],
        ]),
    )
    envelope = BFZICurrentControlledEnvelope(reference, thermal)
    step = envelope.step(np.zeros(3), currents, 30.0)
    assert step.converged
    assert step.state.shape == (3,)
    assert step.temperatures.shape == (3,)
    assert step.temperatures[2] >= thermal.reference_temperature
    np.testing.assert_allclose(
        step.coil_power,
        reference.port_conductor_power(currents, step.temperatures[:2]),
    )


def test_air_self_neumann_does_not_depend_on_coarse_port_gauss_nodes():
    scene = scene_pair()
    low = build_bfzi_source(
        scene, 200_000., order=3, air_panel_length_factor=4.0,
    )
    high = build_bfzi_source(
        scene, 200_000., order=5, air_panel_length_factor=4.0,
    )
    np.testing.assert_allclose(
        np.diag(low.inductance_air), np.diag(high.inductance_air),
        rtol=1e-13, atol=1e-16,
    )
    assert np.all(np.diag(low.inductance_air) > 0.0)
    # The mutual term may vary with source Gauss order, as intended.
    np.testing.assert_allclose(
        low.inductance_air, low.inductance_air.T, atol=1e-15,
    )


def test_analytical_straight_panel_self_integral_matches_high_order_quad():
    from scipy.integrate import quad

    length = 0.041
    radius = 0.0008
    analytic = 2 * (
        length * np.arcsinh(length / radius)
        - np.hypot(length, radius) + radius
    )
    # One-dimensional equivalent of the double Neumann panel integral.
    numeric = 2 * quad(
        lambda delta: (length - delta) / np.hypot(delta, radius),
        0.0, length, epsabs=1e-11,
    )[0]
    np.testing.assert_allclose(analytic, numeric, rtol=1e-12)


def test_box_shell_partition_is_positive_and_volume_consistent():
    from dataclasses import replace

    scene = scene_pair()
    points, weights = scene.box_shell_quadrature(
        8, 14, 24, shell_normal_order=3, shell_tangential_order=6,
    )
    assert np.all(weights > 0.0)
    assert scene.in_seawater(points).all()
    exact_volume = (
        4 * np.pi * scene.outer_radius**3 / 3
        - sum(8 * np.prod(a.enclosure.half_extents)
              for a in scene.assemblies)
    )
    # A smooth partition of unity allows positive overlap between the
    # shell and sphere cubatures: their weights must NOT be naively added
    # as unweighted disjoint shell volumes.
    assert abs(weights.sum() / exact_volume - 1) < 0.02

    angle = 0.29
    rotation = np.array([
        [np.cos(angle), 0., np.sin(angle)],
        [0., 1., 0.],
        [-np.sin(angle), 0., np.cos(angle)],
    ])
    rotated = replace(
        scene, receiver=replace(
            scene.receiver,
            pose=RigidPose(rotation, scene.receiver.pose.center),
        ),
    )
    rot_points, rot_weights = rotated.box_shell_quadrature(
        8, 14, 24, shell_normal_order=3, shell_tangential_order=6,
    )
    assert rotated.in_seawater(rot_points).all()
    assert np.all(rot_weights > 0.0)
    assert abs(rot_weights.sum() / exact_volume - 1) < 0.02


def test_box_shell_reference_retains_two_port_energy_identity():
    scene = scene_pair()
    source = build_bfzi_source(scene, 200_000.0, order=3)
    reference = solve_seawater_reference(
        source, config=SeaMQSConfig(
            radial_order=3, polar_order=4, azimuthal_order=6,
            shell_normal_order=2, shell_tangential_order=3,
            field_softening=0.025, quadrature_scheme="box_shell",
        ),
    )
    currents = np.array([1.0 + 0.2j, -0.3 + 0.5j])
    np.testing.assert_allclose(
        reference.port_seawater_power(currents),
        reference.quadrature_power(currents),
        rtol=1e-8, atol=1e-12,
    )
    assert np.all(reference.integration_weights > 0)
    assert scene.in_seawater(reference.integration_points).all()
    assert np.isfinite(reference.impedance_seawater).all()


def test_box_shell_tangential_gauss_and_global_modes_keep_geometry_consistent():
    from dataclasses import replace
    from sdfmpneo_bfzi import build_sea_basis

    scene = scene_pair()
    # Changing the artificial outer sphere must not secretly rescale the
    # underlying global Gaussian trial functions.
    larger = replace(scene, outer_radius=0.70)
    global_a = build_sea_basis(scene).modes[-3:]
    global_b = build_sea_basis(larger).modes[-3:]
    np.testing.assert_allclose(
        [mode.width for mode in global_a],
        [mode.width for mode in global_b],
        rtol=0, atol=1e-14,
    )
    for order in (4, 5):
        points, weights = scene.box_shell_quadrature(
            5, 8, 12, shell_normal_order=3,
            shell_tangential_order=order,
        )
        assert scene.in_seawater(points).all()
        assert np.all(weights > 0.0)
    # The composite tangential rule now splits every physical edge and
    # face center, rather than moving a few global Gauss nodes past them.



def test_restricted_galerkin_subspace_matches_independent_direct_solve():
    """One Green assembly gives the exact same-cubature nested 15-mode solve."""
    from sdfmpneo_bfzi import (
        build_sea_basis, restrict_seawater_response,
    )

    scene = scene_pair()
    source = build_bfzi_source(scene, 200_000., order=3)
    config = SeaMQSConfig(
        radial_order=3, polar_order=4, azimuthal_order=6,
        shell_normal_order=2, shell_tangential_order=3,
        field_softening=0.025,
    )
    enriched_basis = build_sea_basis(scene, modes_per_face=2)
    coarse_basis = build_sea_basis(scene, modes_per_face=1)
    fine = solve_seawater_reference(
        source, basis=enriched_basis, config=config,
    )
    # Every first tangent in the 27-mode hierarchy is one of its
    # original face-center functions. The three global modes also match.
    kept = []
    for mode in coarse_basis.modes:
        matching = [
            i for i, candidate in enumerate(enriched_basis.modes)
            if np.array_equal(candidate.center, mode.center)
            and np.array_equal(candidate.direction, mode.direction)
            and candidate.width == mode.width
        ]
        assert len(matching) == 1
        kept.extend(matching)
    coarse_from_fine = restrict_seawater_response(fine, kept)
    direct = solve_seawater_reference(
        source, basis=coarse_basis, config=config,
    )
    np.testing.assert_allclose(
        coarse_from_fine.impedance_seawater, direct.impedance_seawater,
        rtol=2e-7, atol=1e-12,
    )
    currents = np.array([1.0 + 0.3j, -0.6 + 0.2j])
    np.testing.assert_allclose(
        coarse_from_fine.quadrature_power(currents),
        direct.quadrature_power(currents),
        rtol=2e-7, atol=1e-12,
    )
    probe = np.array([[0., 0., 0.], [0.14, 0.0, 0.0]])
    np.testing.assert_allclose(
        coarse_from_fine.current_density(probe, currents),
        direct.current_density(probe, currents),
        rtol=2e-7, atol=1e-11,
    )


def test_thin_face_refinement_is_strictly_nested_in_original_basis():
    from sdfmpneo_bfzi import build_sea_basis

    scene = scene_pair()
    basic = build_sea_basis(scene, modes_per_face=2)
    patches = build_sea_basis(
        scene, modes_per_face=2, face_refinement=2,
        refine_thin_faces_only=True,
    )
    assert basic.size == 27
    assert patches.size == 59
    for mode in basic.modes:
        assert any(
            np.array_equal(mode.center, candidate.center)
            and np.array_equal(mode.direction, candidate.direction)
            and mode.width == candidate.width
            for candidate in patches.modes
        )



def test_source_guided_curl_modes_match_complete_port_vector_potential():
    """A spatially weighted closed-wire A has the expected solenoidal curl."""
    from sdfmpneo_bfzi import build_sea_basis

    scene = scene_pair()
    source = build_bfzi_source(scene, 200_000., order=3)
    base = build_sea_basis(scene, modes_per_face=2)
    guided = build_sea_basis(
        scene, modes_per_face=2, sources=source,
        include_source_guided=True,
    )
    assert guided.size == base.size + 4
    for i in range(base.size):
        np.testing.assert_allclose(
            guided.modes[i].center, base.modes[i].center
        )
        np.testing.assert_allclose(
            guided.modes[i].direction, base.modes[i].direction
        )
        assert guided.modes[i].width == base.modes[i].width
        assert guided.modes[i].source_port == -1
    assert [m.source_port for m in guided.modes[-4:]] == [0, 0, 1, 1]
    with pytest.raises(ValueError, match="complete port sources"):
        build_sea_basis(scene, include_source_guided=True)

    probe = np.array([0.155, 0.025, 0.17])
    assert scene.in_seawater(probe)
    mode = guided.modes[-2]
    line = guided.source_lines[mode.source_port]

    def potential(x):
        outer = scene.outer_radius
        rel = x - scene.outer_center
        if guided.outer_envelope == "quadratic":
            envelope = 1 - np.dot(rel, rel) / outer**2
        else:
            t = np.clip(
                (outer - np.linalg.norm(rel)) / guided.outer_taper_width,
                0.0, 1.0,
            )
            envelope = t**3 * (10.0 + t * (-15.0 + 6.0 * t))
        for assembly in scene.assemblies:
            dist2, _ = assembly.enclosure.squared_exterior_distance(
                x, assembly.pose,
            )
            d = np.sqrt(dist2)
            envelope *= d / (d + guided.wall_scale)
        offset = x - mode.center
        gaussian = np.exp(-np.dot(offset, offset) / mode.width**2)
        return envelope * gaussian * line.potential_kernel(x)

    h = 1e-5
    derivatives = []
    for axis in range(3):
        delta = np.eye(3)[axis] * h
        derivatives.append(
            (potential(probe + delta) - potential(probe - delta)) / (2 * h)
        )
    curl = np.array([
        derivatives[1][2] - derivatives[2][1],
        derivatives[2][0] - derivatives[0][2],
        derivatives[0][1] - derivatives[1][0],
    ])
    np.testing.assert_allclose(
        guided.evaluate(probe[None, :])[0, -2], curl,
        rtol=2e-5, atol=1e-7,
    )

    response = solve_seawater_reference(
        source, basis=guided, config=SeaMQSConfig(
            radial_order=3, polar_order=4, azimuthal_order=6,
            shell_normal_order=2, shell_tangential_order=3,
            field_softening=0.025,
        ),
    )
    currents = np.array([1.0 + 0.2j, -0.4 + 0.3j])
    assert np.isfinite(response.impedance_seawater).all()
    np.testing.assert_allclose(
        response.port_seawater_power(currents),
        response.quadrature_power(currents), rtol=2e-8, atol=1e-12,
    )
    # Reintegrating R/K and the induced Joule density on the same nodes
    # must recover the operators used by this actual 31-mode solve.
    metrics = response.independent_cubature_metrics(
        response.integration_points, response.integration_weights, currents,
    )
    assert metrics["relative_resistance_difference"] < 1e-8
    assert metrics["relative_coupling_difference"] < 1e-8
    assert metrics["relative_power_difference"] < 1e-8



def test_separate_green_quadrature_preserves_passivity_and_loss_closure():
    """Coarser L integral never compromises accurate dissipative R/K."""
    from dataclasses import replace

    scene = scene_pair()
    source = build_bfzi_source(scene, 200_000.0, order=3)
    config = SeaMQSConfig(
        radial_order=4, polar_order=5, azimuthal_order=8,
        shell_normal_order=2, shell_tangential_order=3,
        field_softening=0.025,
    )
    reference = solve_seawater_reference(source, config=config)
    equivalent = solve_seawater_reference(
        source, config=replace(
            config, inductance_quadrature=(4, 5, 8, 2, 3),
        ),
    )
    np.testing.assert_allclose(
        reference.impedance_seawater, equivalent.impedance_seawater,
        rtol=2e-10, atol=1e-12,
    )
    np.testing.assert_allclose(
        reference.inductance, equivalent.inductance,
        rtol=2e-10, atol=1e-12,
    )
    mixed = solve_seawater_reference(
        source, config=replace(
            config, radial_order=5, polar_order=7, azimuthal_order=10,
            shell_normal_order=3, shell_tangential_order=4,
            inductance_quadrature=(4, 5, 8, 2, 3),
        ),
    )
    assert np.min(np.linalg.eigvalsh(mixed.inductance)) > -1e-10
    assert np.min(np.linalg.eigvalsh(
        0.5 * (mixed.impedance_seawater +
               mixed.impedance_seawater.conj().T)
    )) > -1e-10
    current = np.array([1.2 + 0.1j, -0.3 + 0.5j])
    np.testing.assert_allclose(
        mixed.port_seawater_power(current),
        mixed.quadrature_power(current),
        rtol=1e-8, atol=1e-12,
    )
    with pytest.raises(ValueError, match="five integer orders"):
        SeaMQSConfig(inductance_quadrature=(4, 5, 8))



def test_reintegrated_local_response_reuses_green_and_closes_power():
    scene = scene_pair()
    source = build_bfzi_source(scene, 200_000., order=3)
    reference = solve_seawater_reference(
        source, config=SeaMQSConfig(
            radial_order=3, polar_order=4, azimuthal_order=6,
            shell_normal_order=2, shell_tangential_order=3,
            field_softening=0.025,
        ),
    )
    current = np.array([1.0 + 0.2j, -0.4 + 0.5j])
    probes = np.array([[0., 0., 0.], [0.16, 0., 0.]])
    identical = reference.reintegrate_local_response(
        reference.integration_points, reference.integration_weights,
    )
    np.testing.assert_allclose(
        identical.impedance_seawater, reference.impedance_seawater,
        rtol=1e-8, atol=1e-12,
    )
    metrics = reference.independent_cubature_metrics(
        reference.integration_points, reference.integration_weights,
        current, probes=probes,
    )
    assert metrics["relative_resolved_impedance_change"] < 1e-8
    assert metrics["relative_resolved_power_change"] < 1e-8
    assert metrics["relative_resolved_probe_change"] < 1e-8
    assert metrics["relative_heldout_residual"] < 1e-8

    # Change only R/K and positive heat nodes. L, source, basis and its
    # original normalization are held exactly fixed.
    x, w = scene.box_shell_quadrature(
        3, 4, 6, shell_normal_order=3, shell_tangential_order=4,
        wall_scale=reference.basis.wall_scale,
    )
    refined = reference.reintegrate_local_response(x, w)
    np.testing.assert_allclose(refined.inductance, reference.inductance)
    np.testing.assert_allclose(refined.transform, reference.transform)
    np.testing.assert_allclose(
        refined.port_seawater_power(current),
        refined.quadrature_power(current), rtol=1e-8, atol=1e-12,
    )
    assert refined.port_seawater_power(current) >= 0
    assert np.isfinite(refined.impedance_seawater).all()


def test_green_reintegration_changes_only_L_and_remains_passive():
    """Fixed R/K and heat nodes isolate the nonlocal Green integration error."""
    scene = scene_pair()
    source = build_bfzi_source(scene, 200_000., order=3)
    original = solve_seawater_reference(
        source, config=SeaMQSConfig(
            radial_order=3, polar_order=4, azimuthal_order=6,
            shell_normal_order=2, shell_tangential_order=3,
            field_softening=0.025,
        ),
    )
    currents = np.array([1.0 + 0.2j, -0.4 + 0.5j])
    probes = np.array([[0., 0., 0.], [0.16, 0., 0.]])
    same, sharper = original.reintegrate_green_responses(
        original.integration_points, original.integration_weights,
        softenings=(0.025, 0.018), kernel_chunk_size=17,
    )
    np.testing.assert_allclose(same.inductance, original.inductance,
                               rtol=1e-10, atol=1e-13)
    np.testing.assert_allclose(
        same.impedance_seawater, original.impedance_seawater,
        rtol=1e-10, atol=1e-12,
    )
    assert np.linalg.norm(sharper.inductance - same.inductance) > 0
    for resolved in (same, sharper):
        np.testing.assert_array_equal(resolved.resistance, original.resistance)
        np.testing.assert_array_equal(resolved.coupling, original.coupling)
        np.testing.assert_array_equal(
            resolved.integration_points, original.integration_points,
        )
        assert np.min(np.linalg.eigvalsh(resolved.inductance)) > -1e-10
        np.testing.assert_allclose(
            resolved.port_seawater_power(currents),
            resolved.quadrature_power(currents),
            rtol=1e-8, atol=1e-12,
        )
        assert resolved.port_seawater_power(currents) >= 0.0
    metrics = original.independent_green_metrics(
        original.integration_points, original.integration_weights,
        currents, probes=probes, softenings=(0.025, 0.018),
        kernel_chunk_size=17,
    )
    assert metrics[0]["relative_resolved_impedance_change"] < 1e-8
    assert metrics[0]["relative_resolved_probe_change"] < 1e-8
    assert metrics[0]["relative_heldout_residual"] < 1e-8
    assert metrics[1]["relative_inductance_difference"] > 0
    assert metrics[0]["green_diagonal_trace_fraction"] > 0
    assert np.isfinite(metrics[1]["green_diagonal_trace_fraction"])
    assert metrics[1]["green_diagonal_trace_fraction"] > 0
    assert np.isfinite(metrics[1]["relative_resolved_power_change"])



def test_localized_outer_envelope_keeps_near_pad_modes_invariant():
    """Moving the artificial boundary must not rescale the physical interior."""
    from dataclasses import replace
    from sdfmpneo_bfzi import build_sea_basis

    scene = scene_pair()
    expanded = replace(scene, outer_radius=0.85)
    initial = build_sea_basis(
        scene, modes_per_face=2, face_refinement=2,
        refine_thin_faces_only=True,
    )
    enlarged = build_sea_basis(
        expanded, modes_per_face=2, face_refinement=2,
        refine_thin_faces_only=True,
    )
    probes = np.array([
        [0.0, 0.0, 0.0],
        [0.13, 0.0, 0.0],
        [0.0, 0.13, 0.0],
        [0.0, 0.0, 0.32],
    ])
    assert scene.in_seawater(probes).all()
    assert expanded.in_seawater(probes).all()
    assert initial.outer_envelope == "boundary_layer"
    assert enlarged.outer_envelope == "boundary_layer"
    assert initial.outer_taper_width == enlarged.outer_taper_width
    np.testing.assert_allclose(
        initial.evaluate(probes), enlarged.evaluate(probes),
        rtol=1e-13, atol=1e-13,
    )
    legacy = build_sea_basis(scene, outer_envelope="quadratic")
    old_large = build_sea_basis(expanded, outer_envelope="quadratic")
    assert (
        np.linalg.norm(legacy.evaluate(probes) - old_large.evaluate(probes))
        > 0.01 * np.linalg.norm(legacy.evaluate(probes))
    )
    x, w = scene.box_shell_quadrature(
        3, 4, 6, shell_normal_order=2,
        shell_tangential_order=3,
        outer_taper_width=initial.outer_taper_width,
    )
    assert len(x) == len(w)
    assert np.all(w > 0)
    assert scene.in_seawater(x).all()


def test_localized_outer_taper_rejects_invalid_width():
    from sdfmpneo_bfzi import build_sea_basis

    scene = scene_pair()
    with pytest.raises(ValueError, match="outer_taper_width"):
        build_sea_basis(scene, outer_taper_width=1.0)
    with pytest.raises(ValueError, match="outer_envelope"):
        build_sea_basis(scene, outer_envelope="unknown")


def test_fixed_physical_radial_panel_width_on_growing_outer_domain():
    """Large spheres keep positive physical-resolution outer cubature."""
    from dataclasses import replace

    scene = replace(scene_pair(), outer_radius=1.05)
    base_kwargs = dict(
        radial_order=3, polar_order=4, azimuthal_order=6,
        shell_normal_order=2, shell_tangential_order=3,
    )
    coarse_points, coarse_weights = scene.box_shell_quadrature(**base_kwargs)
    finer_points, finer_weights = scene.box_shell_quadrature(
        **base_kwargs, max_radial_panel_width=0.15,
    )
    assert len(finer_points) > len(coarse_points)
    assert scene.in_seawater(finer_points).all()
    assert np.all(finer_weights > 0)
    assert np.isfinite(finer_weights.sum())
    assert finer_weights.sum() > 0.0
    with pytest.raises(ValueError, match="max_radial_panel_width"):
        SeaMQSConfig(max_radial_panel_width=-0.1)
    with pytest.raises(ValueError, match="max_radial_panel_width"):
        scene.box_shell_quadrature(**base_kwargs,
                                   max_radial_panel_width=0.0)


def test_composite_sphere_angular_gauss_remains_positive_and_symmetric():
    """Angular subpanels preserve weight positivity and seawater geometry."""
    scene = scene_pair()
    base = dict(
        radial_order=3, polar_order=8, azimuthal_order=12,
        shell_normal_order=2, shell_tangential_order=3,
    )
    coarse_x, coarse_w = scene.box_shell_quadrature(**base)
    polar_x, polar_w = scene.box_shell_quadrature(
        **base, polar_panel_count=2,
    )
    gaussian_x, gaussian_w = scene.box_shell_quadrature(
        **base, polar_panel_count=2, azimuthal_panel_order=2,
    )
    assert len(polar_x) > 0
    assert len(gaussian_x) > len(coarse_x)
    for points, weights in (
        (polar_x, polar_w), (gaussian_x, gaussian_w),
    ):
        assert scene.in_seawater(points).all()
        assert np.isfinite(weights).all()
        assert np.all(weights > 0)
        assert np.isfinite(weights.sum())
    with pytest.raises(ValueError, match="polar_panel_count"):
        scene.box_shell_quadrature(**base, polar_panel_count=3)
    with pytest.raises(ValueError, match="azimuthal_panel_order"):
        scene.box_shell_quadrature(**base, azimuthal_panel_order=0)


def test_independent_coupling_resolution_keeps_heat_energy_identity():
    """K-only refinement does not change R, L, or Joule integration nodes."""
    scene = scene_pair()
    source = build_bfzi_source(scene, 200_000., order=3)
    original = solve_seawater_reference(
        source, config=SeaMQSConfig(
            radial_order=3, polar_order=4, azimuthal_order=6,
            shell_normal_order=2, shell_tangential_order=3,
            field_softening=0.025,
        ),
    )
    same = original.resolve_with_coupling(original.coupling)
    np.testing.assert_allclose(
        same.impedance_seawater, original.impedance_seawater,
        rtol=1e-10, atol=1e-13,
    )
    modified = original.resolve_with_coupling(1.02 * original.coupling)
    np.testing.assert_allclose(modified.resistance, original.resistance)
    np.testing.assert_allclose(modified.inductance, original.inductance)
    np.testing.assert_allclose(
        modified.integration_weights, original.integration_weights,
    )
    currents = np.array([1 + 0.2j, -0.4 + 0.3j])
    np.testing.assert_allclose(
        modified.port_seawater_power(currents),
        modified.quadrature_power(currents),
        rtol=1e-10, atol=1e-12,
    )
    assert modified.port_seawater_power(currents) > 0
    with pytest.raises(ValueError, match="refined coupling"):
        original.resolve_with_coupling(np.ones((3, 3)))



def test_package_aligned_polar_panels_are_positive_and_refinable():
    """Project real enclosure corners into fixed positive Gauss mu panels."""
    scene = scene_pair()
    kw = dict(
        radial_order=3, polar_order=6, azimuthal_order=8,
        shell_normal_order=2, shell_tangential_order=3,
    )
    low_x, low_w = scene.box_shell_quadrature(
        **kw, polar_geometry_order=2,
    )
    high_x, high_w = scene.box_shell_quadrature(
        **kw, polar_geometry_order=4,
    )
    assert len(high_x) > len(low_x)
    for x, w in ((low_x, low_w), (high_x, high_w)):
        assert scene.in_seawater(x).all()
        assert np.isfinite(w).all()
        assert np.all(w > 0)
        assert np.isfinite(w.sum())
    with pytest.raises(ValueError, match="polar_geometry_order"):
        scene.box_shell_quadrature(**kw, polar_geometry_order=1)
    with pytest.raises(ValueError, match="polar_geometry_order"):
        scene.box_shell_quadrature(
            **kw, polar_geometry_order=3, polar_panel_count=2,
        )
    with pytest.raises(ValueError, match="polar_geometry_order"):
        SeaMQSConfig(polar_geometry_order=1)
    with pytest.raises(ValueError, match="polar_geometry_order"):
        SeaMQSConfig(polar_geometry_order=3, polar_panel_count=2)


def test_geometry_rk_refinement_does_not_expand_independent_green_l(monkeypatch):
    """A fine K angular rule must not make the separately coarse L quadratic."""
    seen = []
    original = BfziScene.box_shell_quadrature

    def record_quadrature(scene, *args, **kwargs):
        seen.append((args, kwargs))
        return original(scene, *args, **kwargs)

    monkeypatch.setattr(BfziScene, "box_shell_quadrature", record_quadrature)
    scene = scene_pair()
    source = build_bfzi_source(scene, 200_000., order=3)
    config = SeaMQSConfig(
        radial_order=3, polar_order=4, azimuthal_order=6,
        shell_normal_order=2, shell_tangential_order=3,
        polar_geometry_order=2, azimuthal_panel_order=2,
        radial_geometry_cuts=True,
        shell_tangential_max_panel_width=0.025,
        inductance_quadrature=(3, 4, 6, 2, 3),
    )
    response = solve_seawater_reference(source, config=config)
    assert len(seen) == 2
    assert seen[0][1]["polar_geometry_order"] == 2
    assert seen[0][1]["azimuthal_panel_order"] == 2
    assert seen[0][1]["radial_geometry_cuts"] is True
    assert seen[0][1]["shell_tangential_max_panel_width"] == 0.025
    assert seen[1][1].get("radial_geometry_cuts", False) is False
    assert seen[1][1].get("shell_tangential_max_panel_width") is None
    assert seen[1][1].get("polar_geometry_order") is None
    assert seen[1][1].get("azimuthal_panel_order", 1) == 1
    currents = (1.0, -0.3 + 0.4j)
    np.testing.assert_allclose(
        response.port_seawater_power(currents),
        response.quadrature_power(currents),
        rtol=1e-9, atol=1e-12,
    )


def test_curl_dual_source_coupling_is_passive_and_uses_same_r_l():
    """Boundary-dual K = integral F.B retains exactly the same R/L physics."""
    from sdfmpneo_bfzi import build_sea_basis

    scene = scene_pair()
    source = build_bfzi_source(scene, 200_000.0, order=3)
    basis = build_sea_basis(
        scene, sources=source, include_source_guided=True,
    )
    config = SeaMQSConfig(
        radial_order=3, polar_order=4, azimuthal_order=6,
        shell_normal_order=2, shell_tangential_order=3,
        inductance_quadrature=(3, 4, 6, 2, 3),
        coupling_form="curl_dual",
    )
    dual = solve_seawater_reference(source, basis=basis, config=config)
    from dataclasses import replace
    original = solve_seawater_reference(
        source, basis=basis,
        config=replace(config, coupling_form="current_potential"),
    )
    np.testing.assert_allclose(dual.resistance, original.resistance)
    np.testing.assert_allclose(dual.inductance, original.inductance)
    assert np.isfinite(dual.coupling).all()
    assert np.linalg.norm(dual.coupling) > 0
    currents = (1.0 + 0.2j, -0.3 + 0.5j)
    np.testing.assert_allclose(
        dual.port_seawater_power(currents),
        dual.quadrature_power(currents),
        rtol=1e-9, atol=1e-12,
    )
    k_reintegrated = original.reintegrate_source_coupling(
        original.integration_points, original.integration_weights,
    )
    np.testing.assert_allclose(
        k_reintegrated, dual.coupling, rtol=1e-9, atol=1e-12,
    )
    with pytest.raises(ValueError, match="coupling_form"):
        SeaMQSConfig(coupling_form="invalid")


def test_source_coupling_primitives_curl_to_original_current_modes():
    """Verify the new weak-form F is exactly the curl primitive of J."""
    from sdfmpneo_bfzi import build_sea_basis

    scene = scene_pair()
    source = build_bfzi_source(scene, 200_000.0, order=3)
    basis = build_sea_basis(
        scene, sources=source, include_source_guided=True,
    )
    x = np.array([[0.0, 0.0, 0.0], [0.14, 0.0, 0.0]])
    eps = 2e-6
    derivatives = []
    for direction in np.eye(3):
        plus = basis.potential_primitive(x + eps * direction)
        minus = basis.potential_primitive(x - eps * direction)
        derivatives.append((plus - minus) / (2 * eps))
    approx_curl = np.stack((
        derivatives[1][:, :, 2] - derivatives[2][:, :, 1],
        derivatives[2][:, :, 0] - derivatives[0][:, :, 2],
        derivatives[0][:, :, 1] - derivatives[1][:, :, 0],
    ), axis=-1)
    np.testing.assert_allclose(
        approx_curl, basis.evaluate(x), rtol=4e-3, atol=2e-5,
    )



def test_geometry_radial_cuts_and_physical_cap_preserve_positive_wet_nodes():
    """Use actual oriented box extents to split radial Gauss integration."""
    scene = scene_pair()
    orders = dict(
        radial_order=3, polar_order=6, azimuthal_order=8,
        shell_normal_order=2, shell_tangential_order=3,
        polar_geometry_order=3,
    )
    ordinary_x, _ = scene.box_shell_quadrature(**orders)
    cut_x, cut_w = scene.box_shell_quadrature(
        **orders, radial_geometry_cuts=True,
    )
    fine_x, fine_w = scene.box_shell_quadrature(
        **orders, radial_geometry_cuts=True,
        max_radial_panel_width=0.06,
    )
    assert len(cut_x) > len(ordinary_x)
    assert len(fine_x) > len(cut_x)
    for x, w in ((cut_x, cut_w), (fine_x, fine_w)):
        assert scene.in_seawater(x).all()
        assert np.all(np.isfinite(w))
        assert np.all(w > 0)
        assert w.sum() > 0
    with pytest.raises(ValueError, match="radial_geometry_cuts"):
        SeaMQSConfig(radial_geometry_cuts=1)
    with pytest.raises(ValueError, match="radial_geometry_cuts"):
        scene.box_shell_quadrature(**orders, radial_geometry_cuts=1)


def test_sphere_and_box_slab_k_parts_recombine_for_both_forms():
    """Exact node partition adds to K for either continuous weak form."""
    scene = scene_pair()
    kwargs = dict(
        radial_order=3, polar_order=4, azimuthal_order=6,
        shell_normal_order=2, shell_tangential_order=3,
        polar_geometry_order=2, radial_geometry_cuts=True,
    )
    x, w, split = scene.box_shell_quadrature(
        **kwargs, return_sphere_count=True,
    )
    original_x, original_w = scene.box_shell_quadrature(**kwargs)
    np.testing.assert_array_equal(x, original_x)
    np.testing.assert_array_equal(w, original_w)
    assert 0 < split < len(x)
    assert np.all(w > 0)

    source = build_bfzi_source(scene, 200_000.0, order=3)
    response = solve_seawater_reference(
        source, config=SeaMQSConfig(
            radial_order=3, polar_order=4, azimuthal_order=6,
            shell_normal_order=2, shell_tangential_order=3,
            inductance_quadrature=(3, 4, 6, 2, 3),
        ),
    )
    for form in ("current_potential", "curl_dual"):
        full_k = response.reintegrate_source_coupling(
            x, w, coupling_form=form,
        )
        pieces_k = sum((
            response.reintegrate_source_coupling(
                x[:split], w[:split], coupling_form=form,
            ),
            response.reintegrate_source_coupling(
                x[split:], w[split:], coupling_form=form,
            ),
        ))
        np.testing.assert_allclose(
            pieces_k, full_k, rtol=5e-12, atol=1e-12,
        )
    with pytest.raises(ValueError, match="return_sphere_count"):
        scene.box_shell_quadrature(**kwargs, return_sphere_count=1)


def test_composite_near_wall_face_panels_are_positive_and_independent():
    """Refine real face/edge widths without changing the sphere or Green L."""
    scene = scene_pair()
    kw = dict(
        radial_order=3, polar_order=4, azimuthal_order=6,
        shell_normal_order=2, shell_tangential_order=4,
    )
    coarse_x, coarse_w, coarse_sphere = scene.box_shell_quadrature(
        **kw, return_sphere_count=True,
    )
    fine_x, fine_w, fine_sphere = scene.box_shell_quadrature(
        **kw, shell_tangential_max_panel_width=0.025,
        return_sphere_count=True,
    )
    assert len(fine_x) > len(coarse_x)
    assert coarse_sphere == fine_sphere
    np.testing.assert_array_equal(
        fine_x[:fine_sphere], coarse_x[:coarse_sphere],
    )
    np.testing.assert_array_equal(
        fine_w[:fine_sphere], coarse_w[:coarse_sphere],
    )
    assert scene.in_seawater(fine_x).all()
    assert np.isfinite(fine_w).all() and np.all(fine_w > 0)
    with pytest.raises(ValueError, match="shell_tangential_max_panel_width"):
        scene.box_shell_quadrature(
            **kw, shell_tangential_max_panel_width=0,
        )
    with pytest.raises(ValueError, match="shell_tangential_max_panel_width"):
        SeaMQSConfig(shell_tangential_max_panel_width=-0.01)


def test_streamed_wall_operators_match_full_r_and_both_k_forms():
    """Bounded-memory R/K integration matches the existing physical solver."""
    scene = scene_pair()
    source = build_bfzi_source(scene, 200_000.0, order=3)
    reference = solve_seawater_reference(
        source, config=SeaMQSConfig(
            radial_order=3, polar_order=4, azimuthal_order=6,
            shell_normal_order=2, shell_tangential_order=3,
            inductance_quadrature=(3, 4, 6, 2, 3),
        ),
    )
    x, w, split = scene.box_shell_quadrature(
        3, 4, 6, shell_normal_order=2, shell_tangential_order=3,
        shell_tangential_max_panel_width=0.025,
        return_sphere_count=True,
    )
    r, k, kd = reference.reintegrate_local_operators(
        x, w, chunk_size=113,
    )
    full = reference.reintegrate_local_response(x, w)
    np.testing.assert_allclose(r, full.resistance, rtol=5e-9, atol=1e-12)
    np.testing.assert_allclose(k, full.coupling, rtol=5e-9, atol=1e-14)
    np.testing.assert_allclose(
        kd, reference.reintegrate_source_coupling(x, w),
        rtol=5e-9, atol=1e-14,
    )
    sphere = reference.reintegrate_local_operators(
        x[:split], w[:split], chunk_size=113,
    )
    slab = reference.reintegrate_local_operators(
        x[split:], w[split:], chunk_size=113,
    )
    for combined, a, b in zip((r, k, kd), sphere, slab):
        np.testing.assert_allclose(combined, a + b, rtol=5e-9, atol=1e-14)


def test_independent_green_matrices_match_response_and_identify_self_sampling():
    """Operator-only Green L agrees with full solver at identical nodes."""
    scene = scene_pair()
    source = build_bfzi_source(scene, 200_000., order=3)
    baseline = solve_seawater_reference(
        source, config=SeaMQSConfig(
            radial_order=3, polar_order=4, azimuthal_order=6,
            shell_normal_order=2, shell_tangential_order=3,
            inductance_quadrature=(3, 4, 6, 2, 3),
            field_softening=0.025,
        ),
    )
    x, w = scene.box_shell_quadrature(
        3, 4, 6, shell_normal_order=2, shell_tangential_order=3,
        outer_taper_width=baseline.basis.outer_taper_width,
    )
    matrices, diagonal = baseline.reintegrate_green_matrices(
        x, w, softenings=(0.025, 0.018),
        kernel_chunk_size=29,
    )
    resolved = baseline.reintegrate_green_responses(
        x, w, softenings=(0.025, 0.018),
        kernel_chunk_size=17,
    )
    for matrix, answer in zip(matrices, resolved):
        np.testing.assert_allclose(
            matrix, answer.inductance, rtol=1e-10, atol=1e-13,
        )
        assert np.min(np.linalg.eigvalsh(matrix)) >= -1e-12
        assert np.isfinite(matrix).all()
    assert diagonal > 0
    assert diagonal / (0.025 * np.trace(matrices[0])) > 0
    same_cross, same_pair_count = baseline.reintegrate_cross_green_matrices(
        x, w, x, w, softenings=(0.025, 0.018),
        kernel_chunk_size=23,
    )
    assert same_pair_count >= len(x)
    for cross_matrix, ordinary_matrix in zip(same_cross, matrices):
        np.testing.assert_allclose(
            cross_matrix, ordinary_matrix, rtol=2e-10, atol=1e-13,
        )
    split = len(x) // 2
    _, staggered_pair_count = baseline.reintegrate_cross_green_matrices(
        x[:split], w[:split], x[split:], w[split:],
        kernel_chunk_size=17,
    )
    assert staggered_pair_count == 0
    with pytest.raises(ValueError, match="kernel_chunk_size"):
        baseline.reintegrate_green_matrices(x, w, kernel_chunk_size=0)
    with pytest.raises(ValueError, match="kernel_chunk_size"):
        baseline.reintegrate_cross_green_matrices(
            x, w, x, w, kernel_chunk_size=0,
        )


def test_gaussian_volume_green_is_finite_psd_and_chunk_independent():
    """Analytic cloud self interaction removes 1/a singular collocation."""
    scene = scene_pair()
    source = build_bfzi_source(scene, 200_000.0, order=3)
    reference = solve_seawater_reference(
        source, config=SeaMQSConfig(
            radial_order=3, polar_order=4, azimuthal_order=6,
            shell_normal_order=2, shell_tangential_order=3,
            inductance_quadrature=(3, 4, 6, 2, 3),
        ),
    )
    points, weights = scene.box_shell_quadrature(
        3, 4, 6, shell_normal_order=2, shell_tangential_order=3,
        outer_taper_width=reference.basis.outer_taper_width,
    )
    matrices, diagonal = reference.reintegrate_gaussian_green_matrices(
        points, weights, extra_blurs=(0.0, 0.012),
        kernel_chunk_size=17,
    )
    another, another_diag = reference.reintegrate_gaussian_green_matrices(
        points, weights, extra_blurs=(0.0, 0.012),
        kernel_chunk_size=23,
    )
    for a, b, d in zip(matrices, another, diagonal):
        np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-13)
        np.testing.assert_allclose(a, a.T, rtol=1e-12, atol=1e-14)
        assert np.isfinite(a).all()
        assert np.linalg.eigvalsh(a)[0] >= -1e-12
        assert d > 0 and np.trace(a) > 0
    np.testing.assert_allclose(diagonal, another_diag, rtol=1e-12)
    assert np.linalg.norm(matrices[0] - matrices[1]) > 0

    # Multi-width evaluation must match independent single-width calls.
    # Sharing pair distances must not change the Green operator or PSD.
    factors = (0.75, 1.0, 1.25)
    ladder, ladder_self = reference.reintegrate_gaussian_green_matrices(
        points, weights, extra_blurs=(0.0,),
        width_factor=factors, kernel_chunk_size=19,
    )
    for factor, matrix, self_trace in zip(factors, ladder, ladder_self):
        individual, individual_self = (
            reference.reintegrate_gaussian_green_matrices(
                points, weights, extra_blurs=(0.0,),
                width_factor=factor, kernel_chunk_size=23,
            )
        )
        np.testing.assert_allclose(matrix, individual[0], rtol=1e-9, atol=1e-13)
        np.testing.assert_allclose(self_trace, individual_self[0], rtol=1e-12)
        assert np.linalg.eigvalsh(matrix)[0] >= -1e-12
    np.testing.assert_allclose(ladder[1], matrices[0], rtol=1e-9, atol=1e-13)
    # Same public response path, independent physical volume shape.
    ball, ball_self = reference.reintegrate_ball_green_matrix(
        points, weights, kernel_chunk_size=17,
    )
    ball_other, ball_self_other = reference.reintegrate_ball_green_matrix(
        points, weights, kernel_chunk_size=23,
    )
    np.testing.assert_allclose(ball, ball_other, rtol=1e-9, atol=1e-13)
    np.testing.assert_allclose(ball_self, ball_self_other, rtol=1e-12)
    assert np.linalg.eigvalsh(ball)[0] >= -1e-12
    assert ball_self > 0 and np.trace(ball) > 0
    with pytest.raises(ValueError, match="width_factor"):
        reference.reintegrate_gaussian_green_matrices(
            points, weights, width_factor=(0.75, 0.0),
        )

    # Direct analytic single-cloud Coulomb self interaction:
    # K(0)=1/(sqrt(pi)*h), h=equal-volume-radius/sqrt(5).
    from sdfmpneo_bfzi.sea_reference import _assemble_gaussian_green_inductances
    from sdfmpneo_bfzi.source import MU0
    sample_weight = np.array([0.001])
    sample_j = np.array([[[1.0, 0.0, 0.0]]])
    self_m, self_d = _assemble_gaussian_green_inductances(
        np.array([[0.0, 0.0, 0.0]]),
        sample_weight, sample_j, (0.0,), 1, 1.0,
    )
    radius = np.cbrt(3 * sample_weight[0] / (4 * np.pi))
    h = radius / np.sqrt(5)
    expected = MU0 / (4*np.pi) * sample_weight[0]**2 / (np.sqrt(np.pi)*h)
    np.testing.assert_allclose(self_m[0][0, 0], expected, rtol=1e-12)
    np.testing.assert_allclose(self_d[0], expected, rtol=1e-12)
    with pytest.raises(ValueError, match="width_factor"):
        reference.reintegrate_gaussian_green_matrices(
            points, weights, width_factor=0.0,
        )


def test_uniform_ball_coulomb_green_exact_pairs_and_psd():
    """Physical finite ball self, containment and disjoint pair identities."""
    from sdfmpneo_bfzi.sea_reference import (
        _uniform_ball_coulomb_kernel, _assemble_ball_green_inductance,
    )
    radius = np.array([[0.1], [0.1], [0.2]])
    other_radius = np.array([[0.1, 0.05, 0.1]])
    distances = np.array([
        [0.0, 0.0, 0.2],
        [0.1, 0.5, 0.0],
        [0.25, 0.1, 0.5],
    ])
    answer = _uniform_ball_coulomb_kernel(
        distances, radius, other_radius,
    )
    np.testing.assert_allclose(answer[0, 0], 6.0/(5*0.1), rtol=1e-13)
    np.testing.assert_allclose(
        answer[0, 1], 3/(2*0.1) - 3*0.05**2/(10*0.1**3),
        rtol=1e-13,
    )
    np.testing.assert_allclose(answer[0, 2], 1/0.2, rtol=1e-13)
    np.testing.assert_allclose(answer[1, 0], 0.88125/0.1, rtol=1e-13)
    np.testing.assert_allclose(answer[1, 1], 2.0, rtol=1e-13)
    assert np.isfinite(answer).all() and np.all(answer > 0)

    points = np.array([
        [0.0, 0.0, 0.0],
        [0.02, 0.0, 0.0],
        [0.08, 0.1, 0.0],
        [0.2, 0.02, 0.0],
    ])
    weights = np.array([0.001, 0.0003, 0.0008, 0.0005])
    modes = np.array([
        [[1., 0., 0.], [0., 1., 0.]],
        [[0.2, 0.7, 0.], [1., 0.1, 0.]],
        [[0., 1., 0.], [0., 0., 1.]],
        [[0.2, 0., 1.], [0., 1., 0.2]],
    ])
    matrix, self_trace = _assemble_ball_green_inductance(
        points, weights, modes, 2,
    )
    other, other_self = _assemble_ball_green_inductance(
        points, weights, modes, 3,
    )
    np.testing.assert_allclose(matrix, other, rtol=1e-12, atol=1e-16)
    np.testing.assert_allclose(self_trace, other_self, rtol=1e-14)
    np.testing.assert_allclose(matrix, matrix.T, atol=1e-16)
    assert np.linalg.eigvalsh(matrix)[0] >= -1e-14
    assert 0 < self_trace < np.trace(matrix)


def test_ball_green_full_reference_modal_power_and_electrothermal_closure():
    """Ball L is consumed by the actual MQS solver, not just a diagnostic."""
    from dataclasses import replace
    from sdfmpneo_vnext.electrothermal import (
        CoilThermalProperties, build_lumped_coil_thermal_model,
    )
    from sdfmpneo_bfzi import BFZIVoltageControlledEnvelope

    scene = scene_pair()
    source = build_bfzi_source(scene, 200_000.0, order=3)
    cfg = SeaMQSConfig(
        radial_order=3, polar_order=4, azimuthal_order=6,
        shell_normal_order=2, shell_tangential_order=3,
        inductance_quadrature=(3, 4, 6, 2, 3),
        inductance_kernel="uniform_ball", field_softening=0.025,
    )
    reference = solve_seawater_reference(source, config=cfg)
    legacy = solve_seawater_reference(
        source, config=replace(cfg, inductance_kernel="plummer"),
    )
    np.testing.assert_allclose(reference.resistance, legacy.resistance, rtol=1e-12)
    np.testing.assert_allclose(reference.coupling, legacy.coupling, rtol=1e-12)
    assert np.linalg.norm(reference.inductance - legacy.inductance) > 0
    assert np.linalg.eigvalsh(reference.inductance)[0] >= -1e-12
    assert reference.inductance_kernel == "uniform_ball"
    reduced = compress_seawater_response(reference, rank=2)
    assert reduced.inductance_kernel == "uniform_ball"

    gx, gw = scene.box_shell_quadrature(
        3, 4, 6, wall_scale=reference.basis.wall_scale,
        shell_normal_order=2, shell_tangential_order=3,
        outer_taper_width=reference.basis.outer_taper_width,
    )
    independent_l, _ = reference.reintegrate_ball_green_matrix(
        gx, gw, kernel_chunk_size=23,
    )
    np.testing.assert_allclose(
        reference.inductance, independent_l, rtol=1e-10, atol=1e-13,
    )

    i = np.array([1.1 + 0.2j, -0.3 + 0.4j])
    points, weights, heat = reference.seawater_heat_nodes(i)
    assert len(points) == len(weights) == len(heat)
    np.testing.assert_allclose(
        reference.port_seawater_power(i),
        [reference.seawater_power(i), reference.quadrature_power(i),
         weights @ heat], rtol=5e-8, atol=1e-12,
    )
    assert reference.as_structured_prediction().power_closure_error() < 1e-8
    # The response's induced magnetic field must use the same physical
    # uniform-solid-ball cloud model, not the historical Plummer field.
    from sdfmpneo_bfzi.source import MU0
    query = reference.integration_points[:1]
    b = reference.magnetic_field(query, i, chunk_size=1)
    offsets = query[:, None, :] - reference.integration_points[None, :, :]
    radii2 = (3*weights/(4*np.pi))**(2.0/3.0)
    denom = np.maximum(
        np.sum(offsets**2, axis=-1), radii2[None, :]
    )**1.5
    currents_at_nodes = np.einsum(
        "nmc,m->nc", reference.integration_modes,
        reference.unit_current_modes @ i,
    )
    expected = source.magnetic_field(query, i) + (
        MU0/(4*np.pi)*np.sum(
            np.cross(
                (weights[:, None]*currents_at_nodes)[None, :, :],
                offsets,
            ) / denom[..., None], axis=1,
        )
    )
    np.testing.assert_allclose(b, expected, rtol=1e-9, atol=1e-12)
    assert np.isfinite(b).all()

    thermal = build_lumped_coil_thermal_model((
        CoilThermalProperties(20.0, 0.7),
        CoilThermalProperties(20.0, 0.7),
        CoilThermalProperties(100.0, 0.2),
    ))
    current_step = BFZICurrentControlledEnvelope(
        reference, thermal,
    ).step(np.zeros(3), i, 30.0)
    assert current_step.converged
    assert np.isfinite(current_step.temperatures).all()
    external = np.diag([0.05 - 0.15j, 0.05 - 0.15j])
    voltage_step = BFZIVoltageControlledEnvelope(
        reference, thermal, external_impedance=external,
    ).step(np.zeros(3), (reference.port_impedance + external) @ i, 30.0)
    assert voltage_step.converged
    assert np.isfinite(voltage_step.currents).all()
    assert reference.port_seawater_power(voltage_step.currents) >= 0

    with pytest.raises(ValueError, match="inductance_kernel"):
        SeaMQSConfig(inductance_kernel="unsupported")


def test_rank_four_snapshot_rom_reconstructs_parent_and_saves(tmp_path):
    """Two complex port solutions span <=4 real current directions.

    Compression never asserts that the parent spatial Galerkin basis is
    continuum-converged; it only checks exact-in-parent source responses.
    """
    from dataclasses import replace
    from sdfmpneo_bfzi import BFZISnapshotROM, build_sea_basis

    scene = scene_pair()
    sources = build_bfzi_source(scene, 200_000.0, order=3)
    basis = build_sea_basis(
        scene, modes_per_face=2, face_refinement=2,
        refine_thin_faces_only=True,
        sources=sources, include_source_guided=True,
    )
    ref = solve_seawater_reference(
        sources, basis=basis, config=SeaMQSConfig(
            radial_order=3, polar_order=4, azimuthal_order=6,
            shell_normal_order=2, shell_tangential_order=3,
            inductance_kernel="uniform_ball",
            inductance_quadrature=(3, 4, 6, 2, 3),
        ),
    )
    snapshot = BFZISnapshotROM.from_operators(
        sources, basis, ref.transform,
        ref.resistance, ref.inductance, ref.coupling,
    )
    assert snapshot.rank == 4
    assert snapshot.raw_projection.shape == (basis.size, 4)
    np.testing.assert_allclose(
        snapshot.impedance_seawater, ref.impedance_seawater,
        rtol=1e-8, atol=1e-12,
    )
    current = np.array([1.0+0.2j,-0.3+0.4j])
    xyz = ref.integration_points[::max(1,len(ref.integration_points)//9)][:9]
    np.testing.assert_allclose(
        snapshot.current_density(xyz,current),
        ref.current_density(xyz,current), rtol=5e-8, atol=1e-10,
    )
    np.testing.assert_allclose(
        snapshot.seawater_loss(xyz,current),
        ref.seawater_loss(xyz,current), rtol=5e-8, atol=1e-12,
    )
    np.testing.assert_allclose(
        snapshot.port_seawater_power(current),
        ref.port_seawater_power(current),
        rtol=1e-8,atol=1e-12,
    )
    np.testing.assert_allclose(
        snapshot.modal_seawater_power(current),
        snapshot.port_seawater_power(current),
        rtol=1e-8,atol=1e-12,
    )
    file = tmp_path/"bfzi_snapshot.npz"
    snapshot.save(file)
    restored = BFZISnapshotROM.load(file,sources,basis)
    np.testing.assert_allclose(
        restored.current_density(xyz,current),
        snapshot.current_density(xyz,current),
        rtol=1e-12,atol=1e-12,
    )
    np.testing.assert_allclose(
        restored.impedance_seawater,snapshot.impedance_seawater,
        rtol=1e-12,atol=1e-14,
    )
    from sdfmpneo_bfzi import (
        BFZIResponseLabel, geometry_features,
        save_response_labels, load_response_labels,
    )
    label = BFZIResponseLabel.from_snapshot(snapshot)
    np.testing.assert_allclose(
        label.raw_unit_modes,
        ref.transform @ ref.unit_current_modes,
        rtol=2e-8, atol=1e-12,
    )
    assert label.reference_status == "provisional"
    assert label.features.shape == geometry_features(sources).shape
    data_path = tmp_path/"bfzi_labels.npz"
    save_response_labels(data_path, (label,))
    recovered, = load_response_labels(data_path)
    np.testing.assert_array_equal(recovered.features, label.features)
    np.testing.assert_array_equal(
        recovered.raw_unit_modes, label.raw_unit_modes,
    )
    np.testing.assert_array_equal(
        recovered.impedance_seawater, label.impedance_seawater,
    )
    with pytest.raises(ValueError, match="duplicate"):
        save_response_labels(data_path,(label,label))
    wrong_scene = replace(scene,seawater=Seawater(6.0))
    wrong_sources = build_bfzi_source(wrong_scene,200_000.0,order=3)
    wrong_basis = build_sea_basis(
        wrong_scene, modes_per_face=2, face_refinement=2,
        refine_thin_faces_only=True,
        sources=wrong_sources,include_source_guided=True,
    )
    with pytest.raises(ValueError,match="signature mismatch"):
        BFZISnapshotROM.load(file,wrong_sources,wrong_basis)
