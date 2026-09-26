import numpy as np

from sdfmpneo_vnext import (
    AnalyticBaselineArtifact,
    CoilObject,
    CoilThermalProperties,
    ConductorMaterial,
    HomogeneousMedium,
    MeshfreeVNextSystem,
    MQSConfig,
    RigidPose,
    Scene,
    SuperellipseSpiral,
    build_lumped_coil_thermal_model,
)


def _scene():
    copper = ConductorMaterial(
        5.8e7,
        resistance_temperature_coefficient=0.00393,
    )
    first = CoilObject(
        SuperellipseSpiral(
            0.024,
            0.021,
            0.75,
            0.0012,
            0.0012,
            exponent=3.0,
            conductor_width=1.0e-3,
            conductor_thickness=0.8e-3,
        ),
        copper,
        "a",
    )
    second = CoilObject(
        SuperellipseSpiral(
            0.020,
            0.018,
            0.70,
            0.0010,
            0.0010,
            exponent=3.5,
            conductor_width=0.9e-3,
            conductor_thickness=0.7e-3,
            pose=RigidPose(
                np.eye(3),
                np.array([0.004, 0.0, 0.018]),
            ),
        ),
        copper,
        "b",
    )
    return Scene(
        (first, second),
        HomogeneousMedium(),
    )


def _reference_config():
    return MQSConfig(
        segments_per_turn=8,
        min_segments=8,
        section_degree=0,
        radial_order=3,
        angular_order=12,
        line_order=2,
    )


def test_unified_system_fast_reference_and_spatial_share_contract():
    scene = _scene()
    system = MeshfreeVNextSystem(
        AnalyticBaselineArtifact(
            segments_per_coil=24,
        ),
        reference_config=_reference_config(),
    )
    assert system.capabilities.fast_ports
    assert system.capabilities.fast_spatial
    assert system.capabilities.reference
    assert system.capabilities.certified
    assert system.capabilities.electrothermal
    assert (
        system.capabilities.electromagnetic_formulation
        == "magnetoquasistatic_current_potential_charge"
    )
    assert (
        system.capabilities.background_medium
        == "homogeneous_isotropic_unbounded_lossless_or_lossy_ac"
    )
    assert system.capabilities.lossy_background_media
    assert not system.capabilities.heterogeneous_media
    assert not system.capabilities.retardation
    assert system.capabilities.arbitrary_se3_pose
    assert system.capabilities.superelliptic_conductors
    assert system.capabilities.package_geometry
    assert system.capabilities.package_dielectric_sie
    assert system.capabilities.package_em_coupling
    assert system.capabilities.continuous_spatial_loss

    fast = system.fast_ports(
        scene,
        20_000.0,
    )
    reference = system.reference_ports(
        scene,
        20_000.0,
    )
    assert fast.impedance.shape == (
        2,
        2,
    )
    assert reference.impedance.shape == (
        2,
        2,
    )
    assert fast.power_closure_error() < 1e-12
    assert reference.power_closure_error() < 1e-8

    field = system.fast_spatial(
        scene,
        20_000.0,
    )
    assert (
        field.normalization_closure_error
        < 1e-12
    )
    matrix = field.local_dissipation_matrix(
        0,
        0.5,
        (0.0, 0.0),
    )
    assert np.allclose(
        matrix,
        matrix.conj().T,
        atol=1e-12,
    )
    assert (
        np.min(
            np.linalg.eigvalsh(matrix)
        )
        >= -1e-12
    )


def test_unified_system_fast_and_reference_thermal_use_same_drive_contract():
    scene = _scene()
    system = MeshfreeVNextSystem(
        AnalyticBaselineArtifact(
            segments_per_coil=24,
        ),
        reference_config=_reference_config(),
    )
    thermal = build_lumped_coil_thermal_model(
        (
            CoilThermalProperties(8.0, 0.10),
            CoilThermalProperties(7.0, 0.12),
        ),
        mutual_conductance=np.array(
            [
                [0.0, 0.02],
                [0.02, 0.0],
            ]
        ),
    )
    currents = np.array(
        [1.5 + 0j, -0.4 + 0.2j]
    )
    fast_step = system.fast_current_envelope(
        scene,
        20_000.0,
        thermal,
        coupling_tolerance=1e-8,
        max_coupling_iterations=20,
    ).step(
        np.zeros(2),
        currents,
        5.0,
    )
    reference_step = (
        system.reference_current_envelope(
            scene,
            20_000.0,
            thermal,
            coupling_tolerance=1e-8,
            max_coupling_iterations=20,
        ).step(
            np.zeros(2),
            currents,
            5.0,
        )
    )
    assert fast_step.converged
    assert reference_step.converged
    assert np.all(
        fast_step.coil_power >= 0.0
    )
    assert np.all(
        reference_step.coil_power >= 0.0
    )
    assert fast_step.currents.shape == (
        2,
    )
    assert reference_step.currents.shape == (
        2,
    )



def test_fast_rejects_lossy_background_without_explicit_artifact_support():
    scene = _scene()
    lossy = Scene(
        scene.coils,
        HomogeneousMedium(
            relative_permittivity=2.0,
            relative_permeability=1.0,
            conductivity=1e-4,
        ),
    )
    system = MeshfreeVNextSystem(
        AnalyticBaselineArtifact(
            segments_per_coil=24,
        ),
        reference_config=_reference_config(),
    )
    import pytest

    with pytest.raises(
        NotImplementedError,
        match="background-dissipation",
    ):
        system.fast_ports(
            lossy,
            20_000.0,
        )

    # The physical REFERENCE path remains valid at nonzero frequency.
    reference = system.reference_ports(
        lossy,
        20_000.0,
    )
    assert reference.impedance.shape == (2, 2)


def test_reference_spatial_and_continuous_thermal_support_lossy_background():
    scene = _scene()
    lossy = Scene(
        scene.coils,
        HomogeneousMedium(
            relative_permittivity=2.5,
            relative_permeability=1.0,
            conductivity=1e-4,
        ),
    )
    system = MeshfreeVNextSystem(
        AnalyticBaselineArtifact(
            segments_per_coil=24,
        ),
        reference_config=_reference_config(),
    )
    spatial = system.reference_spatial(
        lossy,
        20_000.0,
    )
    assert spatial.background_channel_index == 2
    assert (
        spatial.normalization_closure_error
        < 1e-6
    )
    currents = np.array(
        [1.0 + 0.1j, -0.3 + 0.2j]
    )
    density = spatial.background_joule_density(
        np.array(
            [0.0, 0.0, 0.03]
        ),
        currents,
    )
    assert np.isfinite(
        density
    )
    assert density >= -1e-12

    from sdfmpneo_vnext import (
        HomogeneousThermalMedium,
    )

    thermal = (
        system.reference_continuous_thermal_field(
            lossy,
            20_000.0,
            HomogeneousThermalMedium(
                conductivity=0.6,
                density=1000.0,
                heat_capacity=4200.0,
            ),
            longitudinal_segments=8,
            radial_order=3,
            angular_order=12,
        )
    )
    assert thermal.source.n_channels == 3
    assert (
        thermal.source.normalization_closure_error
        < 1e-6
    )
    temperature = thermal.temperature_step(
        np.array(
            [0.0, 0.0, 0.03]
        ),
        1.0,
        currents,
    )
    assert np.isfinite(
        temperature
    )
    assert (
        temperature
        >= thermal.medium.ambient_temperature
    )
