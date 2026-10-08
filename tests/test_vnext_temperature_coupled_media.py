import numpy as np
import pytest

from sdfmpneo_vnext.channel_thermal import (
    ThermalNodeProperties,
    build_lumped_channel_thermal_model,
)
from sdfmpneo_vnext.geometry import SuperellipseSpiral
from sdfmpneo_vnext.package_geometry import SuperquadricPackageGeometry
from sdfmpneo_vnext.scene import (
    CoilObject,
    ConductorMaterial,
    IsotropicMaterial,
    PackageObject,
    Scene,
    TensorElectricMaterial,
)
from sdfmpneo_vnext.temperature_coupled_channel import (
    TemperatureResolvedChannelCurrentEnvelope,
)
from sdfmpneo_vnext.thermal_material_laws import (
    LinearIsotropicTemperatureLaw,
    LinearTensorTemperatureLaw,
)


class _UnusedArtifact:
    def predict_structured(self, scene, frequency_hz):
        raise AssertionError("scene reconstruction test must not invoke EM inference")


def test_linear_isotropic_temperature_law_scales_passive_coefficients():
    material = IsotropicMaterial(
        relative_permittivity=3.0,
        relative_permeability=1.2,
        conductivity=2.0e-3,
        thermal_conductivity=0.25,
        density=1200.0,
        heat_capacity=1400.0,
    )
    law = LinearIsotropicTemperatureLaw(
        material,
        reference_temperature=293.15,
        permittivity_temperature_coefficient=1.0e-3,
        loss_temperature_coefficient=2.0e-3,
        permeability_temperature_coefficient=-5.0e-4,
    )
    warm = law.at_temperature(313.15)

    assert np.isclose(warm.relative_permittivity_at(40_000.0).real, 3.0 * 1.02)
    assert np.isclose(warm.loss_conductivity(40_000.0), 2.0e-3 * 1.04)
    assert np.isclose(warm.relative_permeability, 1.2 * 0.99)
    assert np.isclose(warm.thermal_conductivity, 0.25)
    assert np.isclose(warm.density, 1200.0)
    assert np.isclose(warm.heat_capacity, 1400.0)
    assert warm.relative_permittivity_at(40_000.0).imag < 0.0


def test_linear_tensor_temperature_law_preserves_spd_and_shared_axes():
    material = TensorElectricMaterial(
        relative_permittivity_tensor=np.diag([2.0, 3.0, 4.0]),
        conductivity_tensor=np.diag([1.0e-4, 2.0e-4, 3.0e-4]),
        relative_permeability=1.1,
        thermal_conductivity_tensor=np.diag([0.2, 0.3, 0.4]),
        density=1300.0,
        heat_capacity=1100.0,
    )
    law = LinearTensorTemperatureLaw(
        material,
        reference_temperature=300.0,
        permittivity_temperature_coefficient=2.0e-3,
        loss_temperature_coefficient=-1.0e-3,
        permeability_temperature_coefficient=5.0e-4,
    )
    warm = law.at_temperature(320.0)

    assert np.allclose(
        warm.relative_permittivity_tensor,
        material.relative_permittivity_tensor * 1.04,
    )
    assert np.allclose(
        warm.conductivity_tensor,
        material.conductivity_tensor * 0.98,
    )
    assert np.isclose(warm.relative_permeability, 1.1 * 1.01)
    assert np.min(np.linalg.eigvalsh(warm.relative_permittivity_tensor)) > 0.0
    assert np.min(np.linalg.eigvalsh(warm.conductivity_tensor)) >= 0.0
    assert np.allclose(
        warm.relative_permittivity_tensor @ warm.conductivity_tensor,
        warm.conductivity_tensor @ warm.relative_permittivity_tensor,
    )
    assert np.allclose(
        warm.thermal_conductivity_tensor,
        material.thermal_conductivity_tensor,
    )


def test_temperature_resolved_envelope_rebuilds_coil_background_and_package():
    coil = CoilObject(
        SuperellipseSpiral(
            0.02,
            0.018,
            0.7,
            0.001,
            0.001,
            conductor_width=8e-4,
            conductor_thickness=6e-4,
        ),
        ConductorMaterial(
            5.8e7,
            resistance_temperature_coefficient=0.004,
            reference_temperature=293.15,
        ),
        "coil",
    )
    background = IsotropicMaterial(
        relative_permittivity=2.0,
        conductivity=1.0e-3,
    )
    package_material = IsotropicMaterial(
        relative_permittivity=3.0,
        conductivity=2.0e-3,
    )
    package = PackageObject(
        SuperquadricPackageGeometry(np.array([0.03, 0.026, 0.008])),
        package_material,
        "package",
    )
    scene = Scene((coil,), background, (package,))
    thermal = build_lumped_channel_thermal_model(
        tuple(ThermalNodeProperties(10.0, 0.1) for _ in range(3)),
        source_map=np.zeros((3, 2), dtype=float),
        ambient_temperature=293.15,
    )
    envelope = TemperatureResolvedChannelCurrentEnvelope(
        scene,
        40_000.0,
        thermal,
        _UnusedArtifact(),
        coil_temperature_indices=np.array([0]),
        background_temperature_law=LinearIsotropicTemperatureLaw(
            background,
            loss_temperature_coefficient=1.0e-2,
        ),
        background_temperature_index=1,
        package_temperature_laws=(
            LinearIsotropicTemperatureLaw(
                package_material,
                permittivity_temperature_coefficient=2.0e-3,
            ),
        ),
        package_temperature_indices=(2,),
    )

    warm_scene, temperatures = envelope._scene_at_state(np.array([10.0, 20.0, 30.0]))

    assert np.allclose(temperatures, np.array([303.15, 313.15, 323.15]))
    assert warm_scene.coils[0].material.conductivity < coil.material.conductivity
    assert np.isclose(warm_scene.medium.loss_conductivity(40_000.0), 1.2e-3)
    assert np.isclose(
        warm_scene.packages[0].material.relative_permittivity_at(40_000.0).real,
        3.0 * 1.06,
    )


def test_temperature_law_rejects_nonphysical_linear_extrapolation():
    material = IsotropicMaterial(relative_permittivity=2.0)
    law = LinearIsotropicTemperatureLaw(
        material,
        reference_temperature=300.0,
        permittivity_temperature_coefficient=-0.02,
    )
    with pytest.raises(ValueError, match="permittivity"):
        law.at_temperature(360.0)
