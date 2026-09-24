import numpy as np

from sdfmpneo_vnext import (
    AnalyticBaselineArtifact,
    CoilObject,
    ConductorMaterial,
    ContinuousThermalGreenArtifact,
    HomogeneousMedium,
    HomogeneousThermalMedium,
    RigidPose,
    Scene,
    SuperellipseSpiral,
    UniformLossFieldDecoder,
    build_thermal_source_quadrature,
    haar_rotation,
)


def _scene(pose=None):
    copper = ConductorMaterial(
        5.8e7
    )
    coil = CoilObject(
        SuperellipseSpiral(
            0.024,
            0.020,
            0.75,
            0.001,
            0.001,
            exponent=3.0,
            conductor_width=1.0e-3,
            conductor_thickness=0.8e-3,
            pose=(
                pose
                or RigidPose.identity()
            ),
        ),
        copper,
        "coil",
    )
    return Scene(
        (coil,),
        HomogeneousMedium(),
    )


def _medium():
    return HomogeneousThermalMedium(
        conductivity=0.6,
        density=1000.0,
        heat_capacity=4200.0,
        ambient_temperature=293.15,
    )


def _artifact():
    return ContinuousThermalGreenArtifact(
        UniformLossFieldDecoder(
            AnalyticBaselineArtifact(
                segments_per_coil=32,
            ),
            length_segments=32,
        ),
        _medium(),
        longitudinal_segments=8,
        radial_order=3,
        angular_order=12,
    )


def test_thermal_source_quadrature_closes_port_loss_channels():
    scene = _scene()
    spatial = UniformLossFieldDecoder(
        AnalyticBaselineArtifact(
            segments_per_coil=32,
        ),
        length_segments=32,
    ).prepare(
        scene,
        40_000.0,
    )
    source = build_thermal_source_quadrature(
        scene,
        spatial,
        longitudinal_segments=8,
        radial_order=3,
        angular_order=12,
    )
    assert source.normalization_closure_error < 1e-10
    assert np.allclose(
        source.integrated_channels(),
        spatial.port_prediction.dissipation_channels,
        rtol=2e-9,
        atol=2e-12,
    )
    currents = np.array(
        [2.0 - 0.3j]
    )
    assert np.allclose(
        source.coil_power(currents),
        spatial.port_prediction.coil_power(
            currents
        ),
        rtol=2e-9,
        atol=2e-12,
    )


def test_continuous_green_temperature_is_common_se3_invariant():
    scene = _scene()
    field = _artifact().prepare(
        scene,
        40_000.0,
    )
    query = np.array(
        [0.006, -0.004, 0.030]
    )
    currents = np.array(
        [2.0 + 0.1j]
    )
    temperature = field.temperature_step(
        query,
        8.0,
        currents,
    )

    rng = np.random.default_rng(
        27
    )
    common = RigidPose(
        haar_rotation(rng),
        np.array(
            [0.3, -0.2, 0.5]
        ),
    )
    moved = _scene(
        common
    )
    moved_field = _artifact().prepare(
        moved,
        40_000.0,
    )
    moved_temperature = (
        moved_field.temperature_step(
            common.apply(
                query
            ),
            8.0,
            currents,
        )
    )
    assert np.isclose(
        temperature,
        moved_temperature,
        rtol=2e-10,
        atol=2e-10,
    )


def test_continuous_green_time_history_matches_step_and_steady_limit():
    field = _artifact().prepare(
        _scene(),
        40_000.0,
    )
    query = np.array(
        [0.0, 0.0, 0.035]
    )
    currents = np.array(
        [1.8 + 0.0j]
    )
    times = np.array(
        [0.5, 2.0, 10.0]
    )
    step = np.asarray(
        [
            field.temperature_step(
                query,
                float(time),
                currents,
            )
            for time in times
        ]
    )
    history = field.temperature_history(
        query,
        np.array(
            [0.0, 10.0]
        ),
        currents[
            None,
            :
        ],
        times,
    )
    assert np.allclose(
        history,
        step,
        rtol=2e-12,
        atol=2e-12,
    )
    assert np.all(
        np.diff(
            step
        )
        > 0.0
    )
    steady = field.steady_temperature(
        query,
        currents,
    )
    assert steady > step[-1]
