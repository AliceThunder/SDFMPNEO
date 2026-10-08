import numpy as np

from sdfmpneo_vnext import (
    CoilObject,
    ConductorMaterial,
    IsotropicMaterial,
    LinearIsotropicTemperatureLaw,
    MeshfreeVNextSystem,
    Scene,
    SuperellipseSpiral,
    TemperatureResolvedChannelCurrentEnvelope,
    ThermalNodeProperties,
    build_lumped_channel_thermal_model,
)


class _PortArtifact:
    supports_packages = False
    supports_tensor_electric = False
    supports_lossy_background = False

    def predict_structured(self, scene, frequency_hz):
        raise AssertionError("factory test must not execute port inference")


def test_system_selects_temperature_resolved_fast_envelope_when_law_is_given():
    scene = Scene(
        (
            CoilObject(
                SuperellipseSpiral(
                    0.02,
                    0.018,
                    0.7,
                    0.001,
                    0.001,
                    conductor_width=8e-4,
                    conductor_thickness=6e-4,
                ),
                ConductorMaterial(5.8e7),
                "coil",
            ),
        ),
        IsotropicMaterial(relative_permittivity=1.5),
        (),
    )
    thermal = build_lumped_channel_thermal_model(
        (ThermalNodeProperties(8.0, 0.1),),
        ambient_temperature=293.15,
    )
    law = LinearIsotropicTemperatureLaw(
        scene.medium,
        permittivity_temperature_coefficient=1.0e-3,
    )

    envelope = MeshfreeVNextSystem(_PortArtifact()).fast_current_envelope(
        scene,
        40_000.0,
        thermal,
        background_temperature_law=law,
        background_temperature_index=0,
    )

    assert isinstance(envelope, TemperatureResolvedChannelCurrentEnvelope)
    warm_scene, temperatures = envelope._scene_at_state(np.array([20.0]))
    assert np.isclose(temperatures[0], 313.15)
    assert warm_scene.medium.relative_permittivity_at(40_000.0).real > 1.5
